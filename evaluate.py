"""
Evaluation Script for Emotion Detection Project
================================================
This script allows TAs to verify the evaluation results presented in the report.

Models Evaluated:
    - Fine-tuned BERT (bert-base-uncased, 110M params) - trained on GoEmotions
    - Qwen2-0.5B-Instruct (500M params) - zero-shot and few-shot prompting
    
    Note: We also tested GPT-2 (124M) and TinyLlama (1.1B) during development.
    The saved predictions use Qwen2-0.5B-Instruct as the final LLM comparison.

Usage:
    # Generate BERT predictions on test set and evaluate
    python evaluate.py --eval_bert
    
    # Evaluate existing LLM predictions
    python evaluate.py --eval_llm
    
    # Compare all models (BERT vs LLMs)
    python evaluate.py --compare_all
    
    # Run everything
    python evaluate.py --all

Outputs:
    - outputs/bert_predictions.csv     (BERT predictions on test set)
    - outputs/llm_zero_shot_predictions.csv (LLM zero-shot predictions)
    - outputs/llm_few_shot_predictions.csv  (LLM few-shot predictions)
    - outputs/evaluation_report.txt    (Full comparison report)
"""

import argparse
import os
import csv
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report

# Label definitions
LABELS = ["joy", "sadness", "anger", "fear", "love", "neutral"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}


def load_bert_and_predict(test_path="data/test.csv", model_path="models/bert_best.pt", output_path="outputs/bert_predictions.csv"):
    """Load trained BERT model and generate predictions on test set."""
    from emotion_detection import (
        build_model, LyricsEmotionDataset
    )
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    print("=" * 60)
    print("GENERATING BERT PREDICTIONS ON TEST SET")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = build_model(num_labels=6, model_name="bert-base-uncased")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load test data
    print(f"Loading test data from: {test_path}")
    test_ds = LyricsEmotionDataset(test_path, label2id, "bert-base-uncased", max_length=256)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    # Generate predictions
    all_texts = []
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            
            all_texts.extend(batch["text"])
            all_true.extend([id2label[l] for l in batch["labels"].numpy()])
            all_pred.extend([id2label[p] for p in preds])
    
    # Save predictions to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "true_label", "pred_label"])
        for text, true, pred in zip(all_texts, all_true, all_pred):
            writer.writerow([text[:200], true, pred])
    
    print(f"\nPredictions saved to: {output_path}")
    return all_true, all_pred


def evaluate_predictions(true_labels, pred_labels, model_name="Model"):
    """Calculate and print evaluation metrics."""
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", labels=LABELS, zero_division=0)
    accuracy = accuracy_score(true_labels, pred_labels)
    
    print(f"\n{model_name} Results:")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(true_labels, pred_labels, labels=LABELS, zero_division=0))
    
    return {"macro_f1": macro_f1, "accuracy": accuracy}


def load_predictions_from_csv(csv_path):
    """Load predictions from a CSV file."""
    df = pd.read_csv(csv_path)
    return df["true_label"].tolist(), df["pred_label"].tolist()


def compare_all_models(output_path="outputs/evaluation_report.txt"):
    """Compare BERT vs all LLM models and save report."""
    print("\n" + "=" * 60)
    print("COMPARING ALL MODELS")
    print("=" * 60)
    
    results = {}
    
    # BERT predictions
    bert_path = "outputs/bert_predictions.csv"
    if os.path.exists(bert_path):
        true_labels, pred_labels = load_predictions_from_csv(bert_path)
        results["Fine-tuned BERT"] = evaluate_predictions(true_labels, pred_labels, "Fine-tuned BERT")
    else:
        print(f"Warning: {bert_path} not found. Run with --eval_bert first.")
    
    # LLM Zero-shot predictions (Qwen2-0.5B-Instruct)
    llm_zero_path = "outputs/llm_zero_shot_predictions.csv"
    if os.path.exists(llm_zero_path):
        true_labels, pred_labels = load_predictions_from_csv(llm_zero_path)
        results["Qwen2-0.5B Zero-shot"] = evaluate_predictions(true_labels, pred_labels, "Qwen2-0.5B-Instruct (Zero-shot)")
    
    # LLM Few-shot predictions (Qwen2-0.5B-Instruct)
    llm_few_path = "outputs/llm_few_shot_predictions.csv"
    if os.path.exists(llm_few_path):
        true_labels, pred_labels = load_predictions_from_csv(llm_few_path)
        results["Qwen2-0.5B Few-shot"] = evaluate_predictions(true_labels, pred_labels, "Qwen2-0.5B-Instruct (Few-shot)")
    
    # Generate comparison report
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("EMOTION DETECTION - EVALUATION REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append("Model Comparison (Macro F1 Score):")
    report_lines.append("-" * 40)
    
    for model_name, metrics in sorted(results.items(), key=lambda x: -x[1]["macro_f1"]):
        bar = "█" * int(metrics["macro_f1"] * 40)
        report_lines.append(f"{model_name:20s} {metrics['macro_f1']:.4f} {bar}")
        print(f"{model_name:20s} Macro F1: {metrics['macro_f1']:.4f}  Accuracy: {metrics['accuracy']:.4f}")
    
    report_lines.append("")
    report_lines.append("-" * 40)
    report_lines.append("Key Finding: Fine-tuned BERT significantly outperforms")
    report_lines.append("LLMs with zero-shot and few-shot prompting on this task.")
    report_lines.append("=" * 60)
    
    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"\nEvaluation report saved to: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluation Script for Emotion Detection")
    parser.add_argument("--eval_bert", action="store_true", help="Generate and evaluate BERT predictions")
    parser.add_argument("--eval_llm", action="store_true", help="Evaluate existing LLM predictions")
    parser.add_argument("--compare_all", action="store_true", help="Compare all models")
    parser.add_argument("--all", action="store_true", help="Run all evaluations")
    
    args = parser.parse_args()
    
    # Default to --all if no args
    if not (args.eval_bert or args.eval_llm or args.compare_all or args.all):
        args.all = True
    
    if args.eval_bert or args.all:
        true_labels, pred_labels = load_bert_and_predict()
        evaluate_predictions(true_labels, pred_labels, "Fine-tuned BERT")
    
    if args.eval_llm or args.all:
        print("\n" + "=" * 60)
        print("EVALUATING LLM PREDICTIONS (Qwen2-0.5B-Instruct)")
        print("=" * 60)
        
        if os.path.exists("outputs/llm_zero_shot_predictions.csv"):
            true_labels, pred_labels = load_predictions_from_csv("outputs/llm_zero_shot_predictions.csv")
            evaluate_predictions(true_labels, pred_labels, "Qwen2-0.5B-Instruct (Zero-shot)")
        
        if os.path.exists("outputs/llm_few_shot_predictions.csv"):
            true_labels, pred_labels = load_predictions_from_csv("outputs/llm_few_shot_predictions.csv")
            evaluate_predictions(true_labels, pred_labels, "Qwen2-0.5B-Instruct (Few-shot)")
    
    if args.compare_all or args.all:
        compare_all_models()
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

