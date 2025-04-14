import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
import os
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up cache directory
def setup_cache_directory():
    try:
        user = os.getenv("USER")
        if not user:
            raise EnvironmentError("The $USER environment variable is not set.")
        
        cache_dir = f"/ocean/projects/cis250010p/{user}/huggingface_cache"
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Cache directory set to: {cache_dir}")
        return cache_dir
    except OSError as e:
        logger.error(f"Failed to set up cache directory: {e}. Ensure there is enough disk space.")
        raise
    except EnvironmentError as e:
        logger.error(f"Environment error: {e}")
        raise
    
# Set seed for reproducibility
def set_seed(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    
set_seed(42)

# Define paths
DATA_DIR = "/jet/home/hlee21/DLNN/HW3/s25-09616-hw3/datafiles/"  
TRAIN_PATH = os.path.join(DATA_DIR, "train_data.csv")
VALID_PATH = os.path.join(DATA_DIR, "val_data.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_data.csv")
OUTPUT_DIR = "models/esm2/"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Protein Dataset class
class ProteinDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512, is_test=False, label_dict=None):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length
        self.is_test = is_test
        self.label_dict = label_dict
        
        if not is_test:
            if label_dict is None:
                # Create label mapping if not provided
                self.labels = sorted(dataframe['family_id'].unique().tolist())
                self.label_dict = {label: idx for idx, label in enumerate(self.labels)}
            else:
                # Use provided label mapping
                self.labels = list(label_dict.keys())
                self.label_dict = label_dict
            
            self.num_labels = len(self.labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Get sequence
        sequence = self.data.iloc[index]['sequence']
        
        # ESM-2 doesn't need spaces between amino acids
        
        # Tokenize sequence
        encoding = self.tokenizer.encode_plus(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Convert to tensors
        inputs = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        # Add labels for training data
        if not self.is_test and 'family_id' in self.data.columns:
            label = self.data.iloc[index]['family_id']
            inputs['labels'] = torch.tensor(self.label_dict[label], dtype=torch.long)
            
        return inputs

# Load data
def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Validation data shape: {valid_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Create label mapping from train data
    all_labels = sorted(train_df['family_id'].unique())
    label_to_id = {label: idx for idx, label in enumerate(all_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    print(f"Number of unique labels: {len(all_labels)}")
    
    return train_df, valid_df, test_df, label_to_id, id_to_label

# Define compute_metrics function for the Trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Main execution
def main():
    cache_dir = setup_cache_directory()
    
    # Load data
    train_df, valid_df, test_df, label_to_id, id_to_label = load_data()
    
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esm2_t30_150M_UR50D", 
        do_lower_case=False,
        cache_dir=cache_dir
    )
    
    # Create datasets
    train_dataset = ProteinDataset(train_df, tokenizer)
    valid_dataset = ProteinDataset(valid_df, tokenizer, label_dict=train_dataset.label_dict)
    test_dataset = ProteinDataset(test_df, tokenizer, is_test=True, label_dict=train_dataset.label_dict)
    
    num_labels = train_dataset.num_labels
    final_label_dict = train_dataset.label_dict
    final_id_to_label = {idx: label for label, idx in final_label_dict.items()}
    print(f"Using {num_labels} labels for classification")
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/esm2_t30_150M_UR50D",
        num_labels=num_labels,
        cache_dir=cache_dir
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        learning_rate=1e-5, 
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        dataloader_num_workers=4,
        save_total_limit=2,
        save_steps= 500,
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"]  
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Train model - Fine-tuning
    trainer.train()
    
    print("Generating predictions with the trainer...")
    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    
    # Convert numeric predictions to family IDs
    predicted_family_ids = [final_id_to_label[pred_idx] for pred_idx in predicted_labels]
    
    # Get sequence names
    sequence_names = test_df['sequence_name'].tolist()
    
    # Verify we're getting multiple different predictions
    unique_preds = set(predicted_family_ids)
    print(f"Number of unique predicted family IDs: {len(unique_preds)}")
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'sequence_name': sequence_names,
        'family_id': predicted_family_ids
    })
    
    # Save submission
    submission_df.to_csv("submission_esm2.csv", index=False)
    print("Submission file created: submission_esm2.csv")

if __name__ == "__main__":
    main()
