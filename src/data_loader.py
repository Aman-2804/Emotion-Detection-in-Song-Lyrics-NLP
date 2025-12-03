"""
Data loading utilities for emotion detection in song lyrics.
PyTorch Dataset and DataLoader for BERT text classification.
"""

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, Optional


class LyricsEmotionDataset(Dataset):
    """
    PyTorch Dataset for BERT text classification with emotion labels.
    Reads CSV with columns 'text' and 'label', tokenizes with BERT tokenizer.
    """
    
    def __init__(
        self,
        csv_path: str,
        label2id: Dict[str, int],
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 256
    ):
        """
        Initialize dataset.
        
        Args:
            csv_path: Path to CSV file with 'text' and 'label' columns
            label2id: Dictionary mapping label strings to integer IDs
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length for tokenization
        """
        self.df = pd.read_csv(csv_path)
        self.label2id = label2id
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a single data sample.
        
        Returns:
            Dictionary with:
            - input_ids: Tokenized input sequence
            - attention_mask: Attention mask
            - labels: Label ID
            - text: Original text (for debugging/interpretability)
        """
        row = self.df.iloc[idx]
        text = str(row["text"])
        label_str = str(row["label"])
        label_id = self.label2id.get(label_str, 0)  # Default to 0 if label not found
        
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": label_id,
            "text": text,  # Keep original text for interpretability
        }


def create_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    label2id: Dict[str, int],
    batch_size: int = 16,
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 256
) -> tuple:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_path: Path to training CSV file
        val_path: Path to validation CSV file
        test_path: Path to test CSV file
        label2id: Dictionary mapping label strings to integer IDs
        batch_size: Batch size for DataLoaders
        tokenizer_name: HuggingFace tokenizer name
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_ds = LyricsEmotionDataset(train_path, label2id, tokenizer_name, max_length)
    val_ds = LyricsEmotionDataset(val_path, label2id, tokenizer_name, max_length)
    test_ds = LyricsEmotionDataset(test_path, label2id, tokenizer_name, max_length)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def create_label_mapping(df: pd.DataFrame, label_column: str = 'label'):
    """
    Create label to ID and ID to label mappings.
    
    Args:
        df: DataFrame with emotion labels
        label_column: Name of the label column
        
    Returns:
        Tuple of (label_to_id, id_to_label) dictionaries
    """
    unique_labels = sorted(df[label_column].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        data_path: Path to CSV file with columns: text, label
        
    Returns:
        DataFrame with lyrics and emotion labels
    """
    df = pd.read_csv(data_path)
    return df
