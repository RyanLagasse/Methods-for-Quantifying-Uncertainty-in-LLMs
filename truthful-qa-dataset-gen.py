import csv
from datasets import load_dataset

# Load the dataset
ds = load_dataset("truthfulqa/truthful_qa", "generation")

# Define the number of samples to include in the CSV
num_samples = 100

# Open a CSV file to write the data
with open('truthful_qa_dataset.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Question", "Correct Answer", "Incorrect Answer"])
    
    # Write the data
    for i in range(num_samples):
        question = ds["validation"]["question"][i]
        correct_answer = ds["validation"]["correct_answers"][i][0]
        incorrect_answer = ds["validation"]["incorrect_answers"][i][0]
        writer.writerow([question, correct_answer, incorrect_answer])

print(f"Dataset with {num_samples} samples has been written to 'truthful_qa_dataset.csv'")