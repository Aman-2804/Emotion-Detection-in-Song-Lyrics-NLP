"""
Emotion Detection in Song Lyrics
================================
Combined module containing all functionality for emotion classification:
- Data loading and preprocessing
- BERT model training and evaluation
- LLM evaluation with zero-shot and few-shot prompting
- Utility functions for metrics and configuration
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import csv
import json
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    precision_recall_fscore_support
)
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# AdamW from torch.optim
AdamW = optim.AdamW


# =============================================================================
# CONSTANTS
# =============================================================================
LABELS = ["joy", "sadness", "anger", "fear", "love", "neutral"]

FEW_SHOT_EXAMPLES = [
    ("I'm walking on sunshine and it feels so good", "joy"),
    ("I cry myself to sleep every night", "sadness"),
    ("I hate everything you put me through", "anger"),
    ("Every noise in the dark makes me tremble", "fear"),
    ("I'd cross the ocean just to hold you close", "love"),
    ("It's just another normal day, nothing special", "neutral"),
]

HF_API_URL = "https://api-inference.huggingface.co/models/"
HF_API_TOKEN = os.environ.get("HF_TOKEN", "")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_metrics(
    y_true: List[str],
    y_pred: List[str],
    emotion_labels: List[str],
    average: str = 'macro'
) -> Dict[str, float]:
    """Calculate evaluation metrics for emotion classification."""
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=emotion_labels, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, labels=emotion_labels, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=emotion_labels, average='weighted', zero_division=0)
    
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
    """Save metrics to a text file."""
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
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config: Dict, config_path: str):
    """Save configuration to JSON file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


# =============================================================================
# DATA LOADING
# =============================================================================

class LyricsEmotionDataset(Dataset):
    """PyTorch Dataset for BERT text classification with emotion labels."""
    
    def __init__(
        self,
        csv_path: str,
        label2id: Dict[str, int],
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 256
    ):
        self.df = pd.read_csv(csv_path)
        self.label2id = label2id
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        label_str = str(row["label"])
        label_id = self.label2id.get(label_str, 0)
        
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
            "text": text,
        }


def create_weighted_sampler(dataset: LyricsEmotionDataset, label2id: Dict[str, int]) -> WeightedRandomSampler:
    """Create a weighted sampler for balanced training on imbalanced datasets."""
    labels = [label2id.get(str(row["label"]), 0) for _, row in dataset.df.iterrows()]
    class_counts = np.bincount(labels, minlength=len(label2id))
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.FloatTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def create_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    label2id: Dict[str, int],
    batch_size: int = 16,
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 256,
    use_weighted_sampling: bool = False
) -> tuple:
    """Create DataLoaders for train, validation, and test sets."""
    train_ds = LyricsEmotionDataset(train_path, label2id, tokenizer_name, max_length)
    val_ds = LyricsEmotionDataset(val_path, label2id, tokenizer_name, max_length)
    test_ds = LyricsEmotionDataset(test_path, label2id, tokenizer_name, max_length)
    
    if use_weighted_sampling:
        print("Using weighted random sampling for balanced training")
        sampler = create_weighted_sampler(train_ds, label2id)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                  num_workers=0, pin_memory=True)
    
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader


def create_label_mapping(df: pd.DataFrame, label_column: str = 'label'):
    """Create label to ID and ID to label mappings."""
    unique_labels = sorted(df[label_column].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv(data_path)


# =============================================================================
# BERT MODEL
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        if self.label_smoothing > 0:
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        p = F.softmax(inputs, dim=-1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = focal_weight * alpha_t
        
        loss = focal_weight * ce_loss
        return loss.mean()


def build_model(num_labels: int = 6, model_name: str = "bert-base-uncased"):
    """Build a BERT model for sequence classification."""
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
    """Evaluate vanilla (pretrained, non-fine-tuned) BERT model as baseline."""
    print(f"\n{'='*60}")
    print(f"EVALUATING VANILLA BERT BASELINE (No Fine-tuning)")
    print(f"{'='*60}")
    
    model = build_model(num_labels=num_labels, model_name=model_name)
    model.to(device)
    model.eval()
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating Vanilla BERT {split_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"\n{split_name} Macro F1 (Vanilla BERT): {macro_f1:.4f}\n")
    
    target_names = None
    if id_to_label:
        target_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
    
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    return macro_f1


def eval_bert(
    model,
    data_loader,
    device,
    split_name: str = "Test",
    id_to_label: Optional[dict] = None
):
    """Evaluate BERT model on a dataset."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {split_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_label_ids = sorted(set(all_labels) | set(all_preds))
    macro_f1 = f1_score(all_labels, all_preds, average="macro", labels=all_label_ids)
    print(f"\n{split_name} Macro F1: {macro_f1:.4f}\n")
    
    target_names = None
    if id_to_label:
        target_names = [id_to_label[i] for i in all_label_ids]
    
    print(classification_report(all_labels, all_preds, labels=all_label_ids, target_names=target_names, zero_division=0))
    return macro_f1


def calculate_class_weights(train_loader, num_classes):
    """Calculate class weights for imbalanced datasets."""
    class_counts = torch.zeros(num_classes)
    
    for batch in train_loader:
        labels = batch["labels"]
        if torch.is_tensor(labels):
            labels = labels.tolist()
        for label in labels:
            label_idx = int(label)
            if 0 <= label_idx < num_classes:
                class_counts[label_idx] += 1
    
    total_samples = class_counts.sum()
    class_weights = torch.zeros(num_classes)
    for i in range(num_classes):
        if class_counts[i] > 0:
            class_weights[i] = total_samples / (num_classes * class_counts[i])
    
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
    early_stopping_patience: int = 2,
    gradient_accumulation_steps: int = 4,
    use_mixed_precision: bool = True,
    label_smoothing: float = 0.1
):
    """Train BERT model with validation monitoring."""
    model.to(device)
    
    print(f"Label smoothing: {label_smoothing}")
    
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing for memory savings")
    
    scaler = None
    if use_mixed_precision and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        print("Enabled mixed precision (FP16) training for memory savings")
    
    num_classes = model.config.num_labels
    class_weights = None
    if use_class_weights:
        print("\nCalculating class weights for imbalanced data...")
        class_weights = calculate_class_weights(train_loader, num_classes).to(device)
        print(f"Class weights: {class_weights.cpu().numpy()}")
    
    optimizer = AdamW(model.parameters(), lr=lr)
    
    total_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps
    )
    
    best_val_f1 = 0.0
    best_state_dict = None
    patience_counter = 0
    
    effective_batch_size = train_loader.batch_size * gradient_accumulation_steps
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {train_loader.batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Total optimization steps: {total_steps}")
    print(f"Warmup steps: {int(warmup_ratio * total_steps)}")
    if early_stopping_patience > 0:
        print(f"Early stopping patience: {early_stopping_patience} epochs\n")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Loss: {postfix}')
        pbar.set_postfix_str("--")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    if class_weights is not None:
                        logits = outputs.logits
                        criterion = nn.CrossEntropyLoss(weight=class_weights)
                        loss = criterion(logits, labels)
                    
                    loss = loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                if class_weights is not None:
                    logits = outputs.logits
                    criterion = nn.CrossEntropyLoss(weight=class_weights)
                    loss = criterion(logits, labels)
                
                loss = loss / gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
            
            avg_loss_so_far = total_loss / (batch_idx + 1)
            pbar.set_postfix_str(f"{avg_loss_so_far:.4f}")
            
            if device.type == 'cuda' and (batch_idx + 1) % 100 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} train loss: {avg_loss:.4f}")
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        val_f1 = eval_bert(model, val_loader, device, split_name="Val", id_to_label=id_to_label)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"✓ New best validation F1: {best_val_f1:.4f}\n")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}\n")
            
            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    print(f"\n{'='*50}")
    print(f"Training complete! Best Val Macro F1: {best_val_f1:.4f}")
    print(f"{'='*50}\n")
    
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print("Loaded best model checkpoint.\n")
    
    return model


def predict_bert(model, data_loader, device):
    """Get predictions from BERT model for analysis."""
    model.eval()
    texts, labels, preds = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_preds = outputs.logits.argmax(dim=-1)
            preds.extend(batch_preds.cpu().numpy())
            labels.extend(batch["labels"].numpy())
            texts.extend(batch["text"])
    
    return texts, np.array(labels), np.array(preds)


# =============================================================================
# LLM EVALUATION
# =============================================================================

def call_hf_api(prompt: str, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2") -> str:
    """Call Hugging Face Inference API."""
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    api_url = HF_API_URL + model_name
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 20,
            "temperature": 0.1,
            "return_full_text": False,
            "do_sample": False
        }
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "neutral")
        elif isinstance(result, dict):
            return result.get("generated_text", "neutral")
        else:
            return "neutral"
    except Exception as e:
        print(f"API Error: {e}")
        return "neutral"


def build_prompt(text: str, mode: str = "few-shot") -> str:
    """Build a prompt for emotion classification."""
    labels_str = ", ".join(LABELS)
    
    examples_section = ""
    if mode == "few-shot":
        examples_section = "Here are labeled examples:\n\n"
        for t, lab in FEW_SHOT_EXAMPLES:
            examples_section += f"Lyric: {t}\nEmotion: {lab}\n\n"

    prompt = f"""You are an emotion classifier for song lyrics.

Possible emotions: {labels_str}.

{examples_section}Classify the emotion of the following lyric.
Answer with ONLY one word from this list: {labels_str}.

Lyric: {text}
Emotion:"""
    return prompt.strip()


def call_llm(prompt: str, model=None, tokenizer=None, device='cuda', model_name: str = None, use_api: bool = True) -> str:
    """Call the LLM with a prompt and return the response."""
    if use_api and model_name:
        return call_hf_api(prompt, model_name)
    
    if model is not None and tokenizer is not None:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Emotion:" in generated_text:
            response = generated_text.split("Emotion:")[-1].strip()
        else:
            response = generated_text
        
        return response
    
    if model_name:
        return call_hf_api(prompt, model_name)
    
    return "neutral"


def parse_prediction(raw_output: str) -> str:
    """Extract a label from raw LLM output."""
    text = raw_output.strip().lower()
    for lab in LABELS:
        if lab in text:
            return lab
    return "neutral"


def load_llm_model(model_name: str = "gpt2", device: str = 'cuda'):
    """Load a local LLM model for evaluation."""
    print(f"Loading LLM: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map='auto' if device == 'cuda' else None,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    print(f"LLM loaded successfully!")
    return model, tokenizer


def eval_llm_on_test(
    test_loader,
    id2label: Dict[int, str],
    mode: str,
    output_path: str,
    model=None,
    tokenizer=None,
    device: str = 'cuda',
    max_samples: Optional[int] = None,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    use_api: bool = True
):
    """Evaluate LLM on test set with zero-shot or few-shot prompting."""
    all_true: List[str] = []
    all_pred: List[str] = []
    
    sample_count = 0
    total_samples = len(test_loader.dataset)
    if max_samples:
        total_samples = min(max_samples, total_samples)
    
    print(f"\nEvaluating LLM in {mode.upper()} mode...")
    print(f"Model: {model_name}")
    print(f"Using API: {use_api}")
    print(f"Total samples: {total_samples}")
    
    with open(output_path, "w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "true_label", "pred_label", "raw_output"])
        
        for batch in tqdm(test_loader, desc=f"LLM {mode}"):
            texts = batch["text"]
            labels = batch["labels"].numpy()
            
            for text, lab_id in zip(texts, labels):
                if max_samples and sample_count >= max_samples:
                    break
                
                prompt = build_prompt(text, mode=mode)
                raw_output = call_llm(
                    prompt, 
                    model=model, 
                    tokenizer=tokenizer, 
                    device=device,
                    model_name=model_name,
                    use_api=use_api
                )
                
                pred_label = parse_prediction(raw_output)
                true_label = id2label[int(lab_id)]
                
                all_true.append(true_label)
                all_pred.append(pred_label)
                
                writer.writerow([text[:200], true_label, pred_label, raw_output[:100] if raw_output else ""])
                sample_count += 1
            
            if max_samples and sample_count >= max_samples:
                break
    
    macro_f1 = f1_score(all_true, all_pred, average="macro", labels=LABELS)
    
    print(f"\n{'='*60}")
    print(f"LLM ({mode}) Results - {model_name}")
    print(f"{'='*60}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(all_true, all_pred, labels=LABELS, zero_division=0))
    print(f"\nPredictions saved to: {output_path}")
    
    return macro_f1

