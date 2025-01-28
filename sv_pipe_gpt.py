import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from repe import repe_pipeline_registry
from utils import honesty_function_dataset, control_vector_dataset, plot_detection_results
from sklearn.linear_model import LinearRegression
import csv
import os
from datetime import datetime
from tqdm import tqdm

MODEL_PATH = "/opt/extra/avijit/projects/rlof/Meta-Llama-3.1-8B-Instruct"

class SteeringVectorPipeline:
    def __init__(self, model_path, data_path, device="cuda:0"):
        """Initialize the pipeline with model, tokenizer, and dataset."""
        self.device = device
        self.model_path = model_path
        self.data_path = data_path

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast="LlamaForCausalLM" not in self.model.config.architectures, 
            padding_side="left", legacy=False
        )
        self.tokenizer.pad_token_id = 0

        # Register repe pipeline
        repe_pipeline_registry()
        self.rep_pipeline = pipeline("rep-reading", model=self.model, tokenizer=self.tokenizer)
        self.cv_rep_reader = None
        
        self.hidden_layers = list(range(-1, -self.model.config.num_hidden_layers, -1))

    def load_datasets(self):
        """Load the testing and training datasets."""
        self.dataset = honesty_function_dataset(
            self.data_path, self.tokenizer, user_tag="USER:", assistant_tag="ASSISTANT:"
        )

    def load_truthful_qa_dataset(self, csv_path):
        """Load the TruthfulQA dataset."""
        dataset = []
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                question = row['Question']
                correct_answer = row['Correct Answer']
                incorrect_answer = row['Incorrect Answer']

                dataset.append({"question": f"{question} {correct_answer}", "is_correct": 1})
                dataset.append({"question": f"{question} {incorrect_answer}", "is_correct": 0})
        return dataset

    def train_steering_vectors(self, concept_positive, concept_negative):
        """Train steering vectors for given opposite concepts."""
        cv_dataset = control_vector_dataset(
            self.data_path, 
            self.tokenizer, 
            "USER:", 
            "ASSISTANT:", 
            concept_positive, 
            concept_negative
        )

        self.cv_rep_reader = self.rep_pipeline.get_directions(
            cv_dataset['train']['data'], 
            rep_token=-1, 
            hidden_layers=self.hidden_layers, 
            n_difference=1, 
            train_labels=cv_dataset['train']['labels'], 
            direction_method='pca',
            batch_size=32
        )

    def compute_scores(self, text):
        """Compute softmax and steering vector scores for input text."""
        input_ids = self.tokenizer(text, return_tensors='pt').to(self.device)
        outputs = self.model(**input_ids, output_attentions=False, output_hidden_states=True, return_dict=True)

        # Softmax scores
        logits = outputs.logits
        softmax_scores = torch.softmax(logits, dim=-1)[0].max(dim=-1).values.tolist()

        # Steering vector scores
        tokens = self.tokenizer.tokenize(text)
        sv_scores = []
        for token_pos in range(len(tokens)):
            # Compute SV scores for the current token position
            rep_reader_scores = self.rep_pipeline(
                [text],
                rep_reader=self.cv_rep_reader,
                rep_token=-len(tokens) + token_pos,
                hidden_layers=self.hidden_layers
            )
            # Aggregate SV scores across layers for the current token
            token_scores = [
                rep_reader_scores[0][layer][0] * self.cv_rep_reader.direction_signs[layer][0]
                for layer in self.hidden_layers
            ]
            sv_scores.append(np.mean(token_scores))  # Aggregate across layers

        # Normalize SV scores
        sv_scores = self.normalize_scores(sv_scores) # TODO: changed normalization method

        return softmax_scores[:len(tokens)], sv_scores  # Align softmax with tokens

    def normalize_scores(self, scores):
        """Normalize scores to range [0, 1]."""
        min_score = min(scores)
        max_score = max(scores)
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def normalize_scores_gaussian(self, scores): # TODO: pick a normalizer later
        """Normalize scores to a Gaussian distribution with mean 0.5 and std ensuring values in [0, 1]."""
        learned_mu = 0.549685605272083
        learned_std = 0.26435183013946717
        scores = np.array(scores)
        
        # Normalize scores using learned mean and std
        normalized_scores = (scores - learned_mu) / learned_std
        
        # Scale and shift to fit within [0, 1]
        new_mean = 0.5
        new_std = 0.1  # Adjust this value to ensure no values fall below 0 or above 1
        scaled_scores = new_mean + new_std * normalized_scores
        
        # Clip values to ensure they fall within [0, 1]
        scaled_scores = np.clip(scaled_scores, 0, 1)
        
        return scaled_scores

    def process_text(self, text):
        """Process input text to generate tokens, softmax, and SV scores."""
        tokens = self.tokenizer.tokenize(text)
        softmax_scores, sv_scores = self.compute_scores(text)

        return tokens, softmax_scores, sv_scores

    def evaluate_metrics(self, dataset, alpha):
        """Evaluate metrics, save score pairs to a CSV file, and save visuals."""
        # Create a unique folder for this run
        visuals_path = "visuals"
        output_csv = "visuals/score_pairs.csv"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_visuals_path = os.path.join(visuals_path, f"run_{timestamp}")
        os.makedirs(run_visuals_path, exist_ok=True)

        results = []

        for example in tqdm(dataset, desc="testing the model"):
            question, is_correct = example["question"], example["is_correct"]
            tokens, softmax_scores, sv_scores = self.process_text(question)

            # Calculate hybrid scores
            hybrid_scores = [
                (tas + alpha * sv) / (alpha + 1) for tas, sv in zip(softmax_scores, sv_scores)
            ]
            # print(f"type of hybrid scores: {type(hybrid_scores[0])}") # TODO: remove later

            for tas, sv, hybrid in zip(softmax_scores, sv_scores, hybrid_scores):
                results.append({
                    "softmax": tas,
                    "sv": sv,
                    "hybrid": hybrid,
                    "is_correct": is_correct
                })

        # Save results to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)

        # Plot and save distributions
        self.plot_metric_distribution(
            df[df["is_correct"] == 1]["softmax"].tolist(),
            df[df["is_correct"] == 0]["softmax"].tolist(),
            "Softmax Scores",
            os.path.join(run_visuals_path, "softmax_scores.png")
        )
        self.plot_metric_distribution(
            df[df["is_correct"] == 1]["sv"].tolist(),
            df[df["is_correct"] == 0]["sv"].tolist(),
            "Steering Vector Scores",
            os.path.join(run_visuals_path, "sv_scores.png")
        )
        self.plot_metric_distribution(
            df[df["is_correct"] == 1]["hybrid"].tolist(),
            df[df["is_correct"] == 0]["hybrid"].tolist(),
            "Hybrid Scores",
            os.path.join(run_visuals_path, "hybrid_scores.png")
        )
        self.plot_correlation(
            df, "softmax", "sv", "Softmax vs. Steering Vector",
            os.path.join(run_visuals_path, "softmax_vs_sv.png")
        )
        self.plot_correlation(
            df, "softmax", "hybrid", "Softmax vs. Hybrid",
            os.path.join(run_visuals_path, "softmax_vs_hybrid.png")
        )
        self.plot_correlation(
            df, "sv", "hybrid", "Steering Vector vs. Hybrid",
            os.path.join(run_visuals_path, "sv_vs_hybrid.png")
        )

    def plot_metric_distribution(self, correct_scores, incorrect_scores, title, save_path):
        """Plot and save distributions of scores for correct and incorrect answers."""
        plt.figure(figsize=(10, 6))
        plt.hist(correct_scores, bins=30, alpha=0.6, label='Correct', color='blue', density=True)
        plt.hist(incorrect_scores, bins=30, alpha=0.6, label='Incorrect', color='orange', density=True)
        plt.xlabel('Uncertainty Scores')
        plt.ylabel('Density')
        plt.title(f'Distribution of {title}')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def plot_correlation(self, df, col_x, col_y, title, save_path):
        """Plot and save correlation between two metrics."""
        plt.figure(figsize=(8, 6))
        plt.scatter(df[col_x], df[col_y], alpha=0.6, c=df["is_correct"], cmap="coolwarm", label="Correctness")
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.title(f'{title} (Color: Correctness)')
        plt.colorbar(label="Correctness (1=Correct, 0=Incorrect)")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()


# Example usage
def main():
    # model_path = "/opt/extra/avijit/projects/rlof/Meta-Llama-3.1-8B-Instruct"
    data_path = "datasets/facts_true_false.csv"

    pipeline = SteeringVectorPipeline(MODEL_PATH, data_path)
    pipeline.load_datasets()
    pipeline.train_steering_vectors("certain", "uncertain")

    test_text = "Explain how 1+1 is 5."
    tokens, softmax_scores, sv_scores = pipeline.process_text(test_text)

    print("Tokens:", tokens)
    print("Softmax Scores:", softmax_scores)
    print("Steering Vector Scores:", sv_scores)

    # Visualizing the scores for any single example
    # pipeline.visualize_scores(tokens, softmax_scores, "Softmax Scores Across Tokens")
    # plt.savefig('visuals/softmax_scores.png')
    # pipeline.visualize_scores(tokens, sv_scores, "Steering Vector Scores Across Tokens")
    # plt.savefig('visuals/sv_scores.png')

    # Example dataset for alpha optimization
    # test_dataset = [
    #     {"question": "Explain how 1+1 is 2.", "is_correct": 1},
    #     {"question": "Explain how 1+1 is 5.", "is_correct": 0}
    # ]

    def load_truthful_qa_dataset(csv_path):
        dataset = []
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                question = row['Question']
                correct_answer = row['Correct Answer']
                incorrect_answer = row['Incorrect Answer']
                
                dataset.append({"question": f"{question} {correct_answer}", "is_correct": 1})
                dataset.append({"question": f"{question} {incorrect_answer}", "is_correct": 0})
        return dataset

    csv_path = "datasets/truthful_qa_dataset.csv"
    csv_path = "datasets/truthful-qa-500.csv"
    dataset = load_truthful_qa_dataset(csv_path)




    # TODO: Remember to add alpha back in when pipeline is done
    # optimized_alpha = 0.46268226770320137
    # v2 Optimized Alpha: 0.27520127775080766
    # optimized_alpha = pipeline.optimize_alpha(dataset)
    # print("Optimized Alpha:", optimized_alpha)
    optimized_alpha = 0.27520127775080766


    # csv_path = "truthful_qa_dataset.csv"
    # csv_path = "datasets/truthful-qa-test-100.csv"
    # csv_path = "datasets/truthful-qa-validation-100.csv"
    csv_path = "datasets/truthful-qa-test-5.csv"
    test_dataset = load_truthful_qa_dataset(csv_path) # dataset is already defined (innapropriately so)

    # Evaluate metrics with optimized alpha
    pipeline.evaluate_metrics(test_dataset, optimized_alpha)


if __name__ == "__main__":
    main()
