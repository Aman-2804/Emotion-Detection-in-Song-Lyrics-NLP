"""
Utility functions for emotion detection project.
"""

import os
import json
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import numpy as np


def calculate_metrics(
    y_true: List[str],
    y_pred: List[str],
    emotion_labels: List[str],
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for emotion classification.
    
    Args:
        y_true: True emotion labels
        y_pred: Predicted emotion labels
        emotion_labels: List of all emotion categories
        average: Averaging strategy for F1 ('macro', 'micro', 'weighted')
        
    Returns:
        Dictionary with accuracy, macro F1, and per-class metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=emotion_labels, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, labels=emotion_labels, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=emotion_labels, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=emotion_labels, zero_division=0
    )
    
    per_class_metrics = {}
    for i, label in enumerate(emotion_labels):
        per_class_metrics[label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class': per_class_metrics
    }


def save_metrics(metrics: Dict, filepath: str):
    """
    Save metrics to a text file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save metrics file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("EVALUATION METRICS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n")
        f.write(f"Micro F1: {metrics['micro_f1']:.4f}\n")
        f.write(f"Weighted F1: {metrics['weighted_f1']:.4f}\n\n")
        
        f.write("-" * 50 + "\n")
        f.write("PER-CLASS METRICS\n")
        f.write("-" * 50 + "\n\n")
        
        for emotion, scores in metrics['per_class'].items():
            f.write(f"{emotion}:\n")
            f.write(f"  Precision: {scores['precision']:.4f}\n")
            f.write(f"  Recall: {scores['recall']:.4f}\n")
            f.write(f"  F1: {scores['f1']:.4f}\n")
            f.write(f"  Support: {scores['support']}\n\n")


def load_config(config_path: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to JSON config file
        
    Returns:
        Dictionary with configuration parameters
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config: Dict, config_path: str):
    """
    Save configuration to JSON file.
    
    Args:
        config: Dictionary with configuration parameters
        config_path: Path to save config file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
