import pandas as pd
import random

# Load the dataset
file_path = 'egg_dataset_qa_format_enhanced.csv'
dataset = pd.read_csv(file_path)

# Define multiple templates for context generation to add variety
def generate_varied_context(question, answer):
    templates = [
        f"In this study, we discuss the following: {answer} Additional information includes various related aspects " \
        f"pertaining to the question '{question}'. The details above are central to understanding this topic.",
        
        f"Exploring the topic, we find that: {answer} This information provides insight into '{question}' and " \
        f"other associated details of the subject under discussion.",
        
        f"To address the question '{question}', it's essential to note: {answer} This answer includes key points " \
        f"that elaborate on the subject matter discussed.",
        
        f"Regarding '{question}', one crucial point is: {answer} This context offers clarity and detail on " \
        f"the subject, covering various aspects related to the question."
    ]
    # Randomly choose one of the templates to add variation
    return random.choice(templates)

# Apply the function to create a varied 'context' column
dataset['context'] = dataset.apply(lambda row: generate_varied_context(row['question'], row['answer']), axis=1)

# Define functions to locate the start and end positions of the answer within the generated context
def find_positions(context, answer):
    start_pos = context.find(answer)
    end_pos = start_pos + len(answer) if start_pos != -1 else -1
    return start_pos, end_pos

# Apply these functions to get start and end positions for each answer
dataset[['start_positions', 'end_positions']] = dataset.apply(
    lambda row: find_positions(row['context'], row['answer']), axis=1, result_type="expand"
)

# Save the modified dataset
dataset.to_csv('egg_dataset_qa_format_with_varied_context.csv', index=False)
