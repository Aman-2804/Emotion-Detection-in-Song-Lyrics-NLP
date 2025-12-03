"""
BERT model for emotion classification in song lyrics.
Fine-tuning functions for training and evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

# AdamW is in torch.optim for newer PyTorch versions
AdamW = optim.AdamW
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from typing import Tuple, List, Optional


def build_model(num_labels: int = 6, model_name: str = "bert-base-uncased"):
    """
    Build a BERT model for sequence classification.
    
    Args:
        num_labels: Number of emotion classes
        model_name: HuggingFace model name
        
    Returns:
        BERT model configured for classification
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return model


def eval_vanilla_bert(
    data_loader,
    device,
    num_labels: int = 6,
    model_name: str = "bert-base-uncased",
    split_name: str = "Test",
    id_to_label: Optional[dict] = None
):
    """
    Evaluate vanilla (pretrained, non-fine-tuned) BERT model as baseline.
    This evaluates the model without any fine-tuning to measure baseline performance.
    
    Args:
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        num_labels: Number of emotion classes
        model_name: HuggingFace model name
        split_name: Name of the split (for printing)
        id_to_label: Optional mapping from label IDs to label names
        
    Returns:
        Macro F1 score
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING VANILLA BERT BASELINE (No Fine-tuning)")
    print(f"{'='*60}")
    
    # Build model without fine-tuning
    model = build_model(num_labels=num_labels, model_name=model_name)
    model.to(device)
    model.eval()  # Set to eval mode (no training)
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating Vanilla BERT {split_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"\n{split_name} Macro F1 (Vanilla BERT): {macro_f1:.4f}\n")
    
    # Print classification report
    target_names = None
    if id_to_label:
        target_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
    
    print(classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        zero_division=0
    ))
    
    return macro_f1


def eval_bert(
    model,
    data_loader,
    device,
    split_name: str = "Test",
    id_to_label: Optional[dict] = None
):
    """
    Evaluate BERT model on a dataset.
    
    Args:
        model: BERT classification model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        split_name: Name of the split (for printing)
        id_to_label: Optional mapping from label IDs to label names
        
    Returns:
        Macro F1 score
    """
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {split_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"\n{split_name} Macro F1: {macro_f1:.4f}\n")
    
    # Print classification report
    target_names = None
    if id_to_label:
        target_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
    
    print(classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        zero_division=0
    ))
    
    return macro_f1


def calculate_class_weights(train_loader, num_classes):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        train_loader: DataLoader with training data
        num_classes: Number of classes
        
    Returns:
        Class weights tensor
    """
    # Count samples per class
    class_counts = torch.zeros(num_classes)
    
    for batch in train_loader:
        labels = batch["labels"]
        for label in labels:
            class_counts[label] += 1
    
    # Calculate weights: total_samples / (num_classes * class_count)
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)
    
    # Handle division by zero (if a class has 0 samples)
    class_weights[class_counts == 0] = 0.0
    
    return class_weights


def train_bert(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 3,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    id_to_label: Optional[dict] = None,
    use_class_weights: bool = True,
    early_stopping_patience: int = 2
):
    """
    Train BERT model with validation monitoring.
    Keeps the best model based on validation macro F1.
    
    Args:
        model: BERT classification model
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        device: Device to run training on
        epochs: Number of training epochs
        lr: Learning rate
        warmup_ratio: Ratio of warmup steps
        id_to_label: Optional mapping from label IDs to label names
        use_class_weights: Whether to use class weights for imbalanced data
        early_stopping_patience: Number of epochs without improvement before stopping
        
    Returns:
        Trained model (best checkpoint loaded)
    """
    model.to(device)
    
    # Calculate class weights if needed
    num_classes = model.config.num_labels
    class_weights = None
    if use_class_weights:
        print("\nCalculating class weights for imbalanced data...")
        class_weights = calculate_class_weights(train_loader, num_classes).to(device)
        print(f"Class weights: {class_weights.cpu().numpy()}")
    
    optimizer = AdamW(model.parameters(), lr=lr)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps
    )
    
    best_val_f1 = 0.0
    best_state_dict = None
    patience_counter = 0
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {int(warmup_ratio * total_steps)}")
    if early_stopping_patience > 0:
        print(f"Early stopping patience: {early_stopping_patience} epochs\n")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Apply class weights if provided
            if class_weights is not None:
                # Get logits and compute weighted loss manually
                logits = outputs.logits
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                loss = criterion(logits, labels)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} train loss: {avg_loss:.4f}")
        
        # Evaluate on validation set
        val_f1 = eval_bert(model, val_loader, device, split_name="Val", id_to_label=id_to_label)
        
        # Save best model and check early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = model.state_dict().copy()
            patience_counter = 0  # Reset patience
            print(f"✓ New best validation F1: {best_val_f1:.4f}\n")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}\n")
            
            # Early stopping
            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs (no improvement for {early_stopping_patience} epochs)")
                break
    
    print(f"\n{'='*50}")
    print(f"Training complete! Best Val Macro F1: {best_val_f1:.4f}")
    print(f"{'='*50}\n")
    
    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print("Loaded best model checkpoint.\n")
    
    return model


def predict_bert(model, data_loader, device):
    """
    Get predictions from BERT model for analysis.
    
    Args:
        model: Trained BERT classification model
        data_loader: DataLoader for prediction
        device: Device to run prediction on
        
    Returns:
        Tuple of (texts, labels, predictions) as numpy arrays
    """
    model.eval()
    texts, labels, preds = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            batch_preds = outputs.logits.argmax(dim=-1)
            preds.extend(batch_preds.cpu().numpy())
            labels.extend(batch["labels"].numpy())
            texts.extend(batch["text"])
    
    return texts, np.array(labels), np.array(preds)


# Keep interpretability methods as helper functions
def get_attention_weights(model, input_ids, attention_mask, device):
    """
    Extract attention weights for interpretability analysis.
    
    Args:
        model: BERT model (AutoModelForSequenceClassification)
        input_ids: Tokenized input sequences
        attention_mask: Attention mask
        device: Device to run on
        
    Returns:
        Attention weights from all transformer layers
    """
    model.eval()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        # Access the base BERT model to get attention weights
        base_model = model.bert if hasattr(model, 'bert') else model.roberta if hasattr(model, 'roberta') else None
        
        if base_model is not None:
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            return outputs.attentions
        else:
            raise ValueError("Could not access base model for attention weights. Model architecture not recognized.")


def get_embeddings(model, input_ids, attention_mask, device):
    """
    Extract embeddings for visualization.
    
    Note: For AutoModelForSequenceClassification, we need to access the base model.
    This is a helper function for interpretability analysis.
    
    Args:
        model: BERT model (AutoModelForSequenceClassification)
        input_ids: Tokenized input sequences
        attention_mask: Attention mask
        device: Device to run on
        
    Returns:
        [CLS] token embeddings from the base model
    """
    model.eval()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        # Access the base BERT model (the underlying transformer)
        base_model = model.bert if hasattr(model, 'bert') else model.roberta if hasattr(model, 'roberta') else None
        
        if base_model is not None:
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # Get [CLS] token embedding from last hidden state
            hidden_states = outputs.hidden_states
            cls_embeddings = hidden_states[-1][:, 0, :]  # [batch_size, hidden_size]
        else:
            raise ValueError("Could not access base model for embeddings. Model architecture not recognized.")
    
    return cls_embeddings
