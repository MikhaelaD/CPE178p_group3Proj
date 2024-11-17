import pandas as pd
from transformers import BertForQuestionAnswering, BertTokenizer, BertTokenizerFast, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the dataset
data = pd.read_csv("parasitic_egg_contexts.csv")

# Check if start_positions and end_positions are already in the CSV
if 'start_positions' not in data.columns or 'end_positions' not in data.columns:
    # Define a function to calculate start and end positions if they don't exist
    def calculate_positions(row):
        context = row['context']
        answer = row['answer']
        start_pos = context.find(answer)
        end_pos = start_pos + len(answer) if start_pos != -1 else -1
        return pd.Series([start_pos, end_pos], index=['start_positions', 'end_positions'])

    # Apply this function to the dataframe
    data[['start_positions', 'end_positions']] = data.apply(calculate_positions, axis=1)

# Split into train and eval sets
train_data = data.sample(frac=0.8, random_state=42)  # 80% for training
eval_data = data.drop(train_data.index)  # 20% for evaluation

# Custom Dataset Class for BERT QA
class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question = row['question']
        context = row['context']
        answer_text = row['answer']

        # Tokenize question and context
        inputs = self.tokenizer.encode_plus(
            question, 
            context, 
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True
        )

        # Get start and end positions from the dataset
        start_pos = row['start_positions']
        end_pos = row['end_positions']

        # Find token positions within the tokenized context
        tokenized_context = self.tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)
        offsets = tokenized_context['offset_mapping']

        # Initialize the start and end token positions
        start_token_pos, end_token_pos = 0, 0
        for idx, (start_offset, end_offset) in enumerate(offsets):
            if start_offset <= start_pos < end_offset:
                start_token_pos = idx
            if start_offset < end_pos <= end_offset:
                end_token_pos = idx

        # Return the inputs and the start/end token positions
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'token_type_ids': inputs['token_type_ids'].squeeze(0),
            'start_positions': torch.tensor(start_token_pos, dtype=torch.long),
            'end_positions': torch.tensor(end_token_pos, dtype=torch.long)
        }

# Load pre-trained BERT model and tokenizer
model = BertForQuestionAnswering.from_pretrained('D:\_results_6\checkpoint-50')
tokenizer = BertTokenizerFast.from_pretrained('D:\_results_6\checkpoint-50')  # Use the Fast tokenizer
model.to(device)

# Create train and eval datasets
train_dataset = QADataset(train_data, tokenizer)
eval_dataset = QADataset(eval_data, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir='D:\_results_7',          
    evaluation_strategy="epoch",    
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8,  
    num_train_epochs=15,                     
    save_total_limit=2,
    fp16=True,
    report_to="none",                # Disable reporting to third-party platforms like WandB
    no_cuda=True                    # Ensure CUDA is enabled if available     
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=eval_dataset,
    tokenizer=tokenizer  # Important to pass the tokenizer for data processing
)

# Train the model
trainer.train()


