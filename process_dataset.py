"""
Dataset processing script for GoEmotions dataset.
Downloads, processes, and creates stratified train/val/test splits with 6 emotion categories.
"""

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from pathlib import Path
import requests
from tqdm import tqdm


# GoEmotions 27 emotion labels mapping to 6 categories
EMOTION_MAPPING = {
    # Joy category
    'amusement': 'joy',
    'joy': 'joy',
    'excitement': 'joy',
    'pride': 'joy',
    'relief': 'joy',
    'gratitude': 'joy',
    'approval': 'joy',
    'optimism': 'joy',
    
    # Sadness category
    'sadness': 'sadness',
    'disappointment': 'sadness',
    'grief': 'sadness',
    'embarrassment': 'sadness',
    'remorse': 'sadness',
    
    # Anger category
    'anger': 'anger',
    'annoyance': 'anger',
    'disapproval': 'anger',
    'disgust': 'anger',
    
    # Fear category
    'fear': 'fear',
    'nervousness': 'fear',
    
    # Love category
    'love': 'love',
    'caring': 'love',
    'admiration': 'love',
    'desire': 'love',
    
    # Neutral category (and ambiguous/unclear emotions)
    'neutral': 'neutral',
    'realization': 'neutral',
    'confusion': 'neutral',
    'curiosity': 'neutral',
    'surprise': 'neutral',
}

# Final 6 emotion categories
FINAL_EMOTIONS = ['joy', 'sadness', 'anger', 'fear', 'love', 'neutral']


def download_goemotions_data(output_dir: str = 'data/raw'):
    """
    Download GoEmotions dataset from HuggingFace or GitHub.
    
    Args:
        output_dir: Directory to save raw data files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Option 1: Try to load from HuggingFace datasets
    try:
        from datasets import load_dataset
        print("Loading GoEmotions dataset from HuggingFace...")
        dataset = load_dataset("go_emotions", "simplified")
        
        # Convert to pandas DataFrames
        train_df = dataset['train'].to_pandas()
        val_df = dataset['validation'].to_pandas()
        test_df = dataset['test'].to_pandas()
        
        # Save as CSV
        train_df.to_csv(os.path.join(output_dir, 'goemotions_train.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'goemotions_val.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'goemotions_test.csv'), index=False)
        
        print(f"Dataset downloaded to {output_dir}/")
        return train_df, val_df, test_df
        
    except Exception as e:
        print(f"Could not load from HuggingFace: {e}")
        print("Please manually download GoEmotions dataset or install datasets library:")
        print("  pip install datasets")
        print("\nAlternatively, you can download from:")
        print("  https://github.com/google-research/google-research/tree/master/goemotions")
        return None, None, None


def map_emotion_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map GoEmotions 27 labels to 6 emotion categories.
    GoEmotions dataset from HuggingFace has 'text' and 'labels' columns.
    'labels' is a list of emotion indices or names.
    
    Args:
        df: DataFrame with GoEmotions data (should have 'text' and 'labels' columns)
        
    Returns:
        DataFrame with mapped emotion labels
    """
    processed_data = []
    
    # GoEmotions label names (27 emotions)
    goemotions_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Mapping emotions"):
        text = str(row.get('text', ''))
        
        # Get labels - could be list of indices or list of strings
        labels = row.get('labels', [])
        
        if not isinstance(labels, list):
            labels = [labels] if labels is not None else []
        
        # If labels are indices, map to label names
        if len(labels) > 0 and isinstance(labels[0], (int, float)):
            # Convert indices to label names
            label_names = [goemotions_labels[int(lbl)] for lbl in labels if 0 <= int(lbl) < len(goemotions_labels)]
        elif len(labels) > 0 and isinstance(labels[0], str):
            label_names = labels
        else:
            label_names = ['neutral']
        
        # Get primary emotion (first non-neutral, or neutral if only neutral)
        primary_emotion = 'neutral'
        for lbl in label_names:
            lbl_lower = lbl.lower().strip()
            mapped = EMOTION_MAPPING.get(lbl_lower, None)
            if mapped and mapped != 'neutral':
                primary_emotion = mapped
                break
        else:
            # If all were neutral or unmapped, use neutral
            primary_emotion = 'neutral'
        
        # Only keep if text is valid
        if text and len(text.strip()) > 0:
            processed_data.append({
                'text': text.strip(),
                'label': primary_emotion
            })
    
    result_df = pd.DataFrame(processed_data)
    return result_df


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> tuple:
    """
    Create stratified train/validation/test splits.
    
    Args:
        df: DataFrame with text and label columns
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        stratify=df['label'],
        random_state=random_state
    )
    
    # Second split: val vs test (adjust ratio for remaining data)
    val_size_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size_adjusted),
        stratify=temp_df['label'],
        random_state=random_state
    )
    
    return train_df, val_df, test_df


def process_goemotions_dataset(
    input_dir: str = 'data/raw',
    output_dir: str = 'data',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    download: bool = True
):
    """
    Main function to process GoEmotions dataset.
    
    Args:
        input_dir: Directory containing raw GoEmotions files
        output_dir: Directory to save processed CSV files
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        download: Whether to download dataset if not present
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("PROCESSING GOEMOTIONS DATASET")
    print("=" * 60)
    
    # Download dataset if needed
    if download:
        train_df, val_df, test_df = download_goemotions_data(input_dir)
        
        if train_df is None:
            print("\nPlease download GoEmotions dataset manually.")
            print("Expected files in data/raw/ directory.")
            return
        
        # Process each split
        print("\nProcessing training data...")
        train_processed = map_emotion_labels(train_df)
        
        print("\nProcessing validation data...")
        val_processed = map_emotion_labels(val_df)
        
        print("\nProcessing test data...")
        test_processed = map_emotion_labels(test_df)
        
        print(f"\nProcessed samples:")
        print(f"  Train: {len(train_processed)}")
        print(f"  Val: {len(val_processed)}")
        print(f"  Test: {len(test_processed)}")
        
        # Combine all data for stratified split
        print("\nCombining all data for stratified splitting...")
        all_data = pd.concat([train_processed, val_processed, test_processed], ignore_index=True)
        
        print(f"Total samples: {len(all_data)}")
        print(f"\nLabel distribution:")
        print(all_data['label'].value_counts().sort_index())
        
        # Create stratified splits
        print("\nCreating stratified train/val/test splits...")
        train_final, val_final, test_final = create_stratified_splits(
            all_data,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
    else:
        # Load from existing CSV files
        print(f"Loading data from {input_dir}/...")
        # This assumes you have already processed CSVs
        # For now, we'll just download if files don't exist
        pass
    
    # Save processed splits
    print("\nSaving processed datasets...")
    train_final.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_final.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_final.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"\nProcessed datasets saved to {output_dir}/")
    print(f"  - train.csv: {len(train_final)} samples")
    print(f"  - val.csv: {len(val_final)} samples")
    print(f"  - test.csv: {len(test_final)} samples")
    
    print("\nLabel distribution in splits:")
    print("\nTrain:")
    print(train_final['label'].value_counts().sort_index())
    print("\nValidation:")
    print(val_final['label'].value_counts().sort_index())
    print("\nTest:")
    print(test_final['label'].value_counts().sort_index())
    
    print("\n" + "=" * 60)
    print("DATASET PROCESSING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process GoEmotions dataset')
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data/raw',
        help='Directory containing raw GoEmotions files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='Directory to save processed CSV files'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Proportion for training set'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Proportion for validation set'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Proportion for test set'
    )
    parser.add_argument(
        '--no_download',
        action='store_true',
        help='Skip downloading dataset'
    )
    
    args = parser.parse_args()
    
    process_goemotions_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        download=not args.no_download
    )
