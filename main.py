"""
Main entry point for emotion detection in song lyrics project.
Orchestrates training, evaluation, and visualization.
"""

import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm

from src.data_loader import create_dataloaders, create_label_mapping, load_data
from src.bert_model import build_model, train_bert, eval_bert, predict_bert, get_attention_weights, get_embeddings, eval_vanilla_bert
from src.llm_eval import LLMEvaluator
from src.interpretability import visualize_attention, visualize_embeddings as vis_embeddings, plot_confusion_matrix
from src.utils import calculate_metrics, save_metrics

# Define label mappings for 6 emotion categories
label2id = {
    "joy": 0,
    "sadness": 1,
    "anger": 2,
    "fear": 3,
    "love": 4,
    "neutral": 5,
}

id2label = {v: k for k, v in label2id.items()}


def eval_llm_on_test(test_path, id2label, output_path, llm_name="meta-llama/Llama-2-7b-hf", zero_shot=True):
    """
    Evaluate LLM on test set with in-context learning.
    
    Args:
        test_path: Path to test CSV file
        id2label: Mapping from label IDs to label names
        output_path: Path to save predictions
        llm_name: HuggingFace model name
        zero_shot: Whether to use zero-shot prompting
    """
    print("=" * 60)
    print("EVALUATING LLM WITH IN-CONTEXT LEARNING")
    print("=" * 60)
    
    # Load test data
    test_df = load_data(test_path)
    print(f"\nTest samples: {len(test_df)}")
    
    # Get emotion categories
    emotion_labels = list(label2id.keys())
    
    # Initialize LLM evaluator
    print(f"\nLoading LLM model: {llm_name}...")
    print(f"Mode: {'Zero-shot' if zero_shot else 'Few-shot'}")
    
    try:
        evaluator = LLMEvaluator(
            model_name=llm_name,
            emotion_categories=emotion_labels
        )
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        print("Please make sure you have access to the model or provide a different model name.")
        return
    
    # Get few-shot examples if not zero-shot
    few_shot_examples = None
    if not zero_shot:
        # Sample a few examples from training data
        train_df = load_data("data/train.csv")
        few_shot_examples = []
        for emotion in emotion_labels[:3]:  # Get examples for first 3 emotions
            examples = train_df[train_df['label'] == emotion].head(1)
            if len(examples) > 0:
                few_shot_examples.append({
                    'lyric': examples.iloc[0]['text'],
                    'emotion': emotion
                })
    
    # Make predictions
    print("\nGenerating predictions...")
    predictions = []
    true_labels = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        text = str(row['text'])
        true_label = str(row['label'])
        
        pred_emotion = evaluator.predict(
            text,
            few_shot_examples=few_shot_examples,
            zero_shot=zero_shot,
            max_new_tokens=10
        )
        
        predictions.append(pred_emotion)
        true_labels.append(true_label)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(true_labels, predictions, emotion_labels)
    
    # Save results
    results_df = pd.DataFrame({
        'text': test_df['text'].values,
        'true_label': true_labels,
        'predicted_label': predictions
    })
    results_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    # Save metrics
    metrics_path = output_path.replace('.csv', '_metrics.txt')
    save_metrics(metrics, metrics_path)
    print(f"Metrics saved to: {metrics_path}")
    
    return metrics


def visualize_attention_example(model, test_loader, id2label, device, model_name="bert-base-uncased", save_path="outputs/attention_example.png"):
    """
    Visualize attention for an example from test set.
    
    Args:
        model: Trained BERT model
        test_loader: Test DataLoader
        id2label: Mapping from label IDs to label names
        device: Device to run on
        model_name: Model name for tokenizer
        save_path: Path to save visualization
    """
    print("\nGenerating attention visualization...")
    
    # Get first batch
    batch = next(iter(test_loader))
    input_ids = batch["input_ids"][:1].to(device)  # Just first example
    attention_mask = batch["attention_mask"][:1].to(device)
    text = batch["text"][0]
    
    # Get attention weights
    attention_weights = get_attention_weights(model, input_ids, attention_mask, device)
    
    # Tokenize to get tokens for visualization
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    
    # Filter out padding tokens
    mask = attention_mask[0].cpu().numpy() == 1
    tokens = [t for i, t in enumerate(tokens) if i < len(tokens) and (i < len(mask) and mask[i])]
    
    # Visualize
    visualize_attention(
        attention_weights,
        tokens,
        layer_idx=-1,  # Last layer
        head_idx=0,  # First head
        save_path=save_path
    )
    print(f"Attention visualization saved to: {save_path}")


def visualize_embeddings(model, test_loader, id2label, device, save_path="outputs/embeddings_umap.png"):
    """
    Visualize embeddings using UMAP.
    
    Args:
        model: Trained BERT model
        test_loader: Test DataLoader
        id2label: Mapping from label IDs to label names
        device: Device to run on
        save_path: Path to save visualization
    """
    print("\nGenerating embedding visualization...")
    
    all_embeddings = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting embeddings"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()
            
            embeddings = get_embeddings(model, input_ids, attention_mask, device)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend([id2label[l] for l in labels])
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    # Visualize
    vis_embeddings(
        embeddings=all_embeddings,
        labels=all_labels,
        method='umap',
        save_path=save_path
    )
    print(f"Embedding visualization saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Emotion Detection in Song Lyrics - Main Orchestration'
    )
    
    parser.add_argument("--train_bert", action="store_true", help="Train BERT model")
    parser.add_argument("--eval_bert", action="store_true", help="Evaluate BERT model on test set")
    parser.add_argument("--eval_vanilla_bert", action="store_true", help="Evaluate vanilla (pretrained) BERT baseline")
    parser.add_argument("--eval_llm", action="store_true", help="Evaluate LLM with in-context learning")
    parser.add_argument("--visualize", action="store_true", help="Generate interpretability visualizations")
    
    # Optional arguments
    parser.add_argument("--train_path", type=str, default="data/train.csv", help="Path to training CSV")
    parser.add_argument("--val_path", type=str, default="data/val.csv", help="Path to validation CSV")
    parser.add_argument("--test_path", type=str, default="data/test.csv", help="Path to test CSV")
    parser.add_argument("--model_path", type=str, default="models/bert_best.pt", help="Path to saved BERT model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", help="BERT model name")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Llama-2-7b-hf", help="LLM model name")
    parser.add_argument("--zero_shot", action="store_true", help="Use zero-shot for LLM (default: few-shot)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataloaders (will be used by multiple operations)
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    # Check if we need to create label mappings from data or use predefined
    # For consistency, we'll use the predefined mappings, but verify data matches
    train_df = load_data(args.train_path)
    val_df = load_data(args.val_path)
    test_df = load_data(args.test_path)
    
    # Verify labels match our predefined categories
    all_labels = set(train_df['label'].unique()) | set(val_df['label'].unique()) | set(test_df['label'].unique())
    if not all_labels.issubset(set(label2id.keys())):
        print(f"Warning: Found labels in data that aren't in predefined categories: {all_labels - set(label2id.keys())}")
        print("Creating label mappings from data instead...")
        label2id_data, id2label_data = create_label_mapping(train_df)
        label2id = label2id_data
        id2label = id2label_data
    else:
        print(f"Using predefined label mappings: {list(label2id.keys())}")
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        args.train_path,
        args.val_path,
        args.test_path,
        label2id,
        batch_size=args.batch_size,
        tokenizer_name=args.bert_model,
        max_length=256
    )
    
    # Train BERT
    if args.train_bert:
        print("\n" + "=" * 60)
        print("TRAINING BERT")
        print("=" * 60)
        
        model = build_model(num_labels=len(label2id), model_name=args.bert_model)
        model = train_bert(
            model,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            id_to_label=id2label
        )
        
        # Save model
        torch.save(model.state_dict(), args.model_path)
        print(f"\nModel saved to: {args.model_path}")
    
    # Evaluate BERT
    if args.eval_bert:
        print("\n" + "=" * 60)
        print("EVALUATING BERT")
        print("=" * 60)
        
        model = build_model(num_labels=len(label2id), model_name=args.bert_model)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        
        # Evaluate on test set
        macro_f1 = eval_bert(model, test_loader, device, split_name="Test", id_to_label=id2label)
        
        # Get predictions for confusion matrix
        texts, labels, preds = predict_bert(model, test_loader, device)
        
        # Plot confusion matrix
        label_names = [id2label[i] for i in sorted(id2label.keys())]
        plot_confusion_matrix(
            [id2label[l] for l in labels],
            [id2label[p] for p in preds],
            label_names,
            save_path=os.path.join(args.output_dir, "confusion_matrix_bert.png")
        )
        
        # Save metrics
        metrics = calculate_metrics(
            [id2label[l] for l in labels],
            [id2label[p] for p in preds],
            label_names
        )
        save_metrics(metrics, os.path.join(args.output_dir, "bert_metrics.txt"))
        print(f"\nMetrics saved to: {args.output_dir}/bert_metrics.txt")
    
    # Evaluate Vanilla BERT Baseline
    if args.eval_vanilla_bert:
        print("\n" + "=" * 60)
        print("EVALUATING VANILLA BERT BASELINE")
        print("=" * 60)
        
        macro_f1 = eval_vanilla_bert(
            test_loader,
            device,
            num_labels=len(label2id),
            model_name=args.bert_model,
            split_name="Test",
            id_to_label=id2label
        )
        
        print(f"\nVanilla BERT Baseline - Test Macro F1: {macro_f1:.4f}")
        print("(Compare this with fine-tuned BERT to measure impact of fine-tuning)")
    
    # Evaluate LLM
    if args.eval_llm:
        eval_llm_on_test(
            args.test_path,
            id2label,
            os.path.join(args.output_dir, "llm_predictions.csv"),
            llm_name=args.llm_name,
            zero_shot=args.zero_shot
        )
    
    # Visualize
    if args.visualize:
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        model = build_model(num_labels=len(label2id), model_name=args.bert_model)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        
        # Attention visualization
        visualize_attention_example(
            model,
            test_loader,
            id2label,
            device,
            model_name=args.bert_model,
            save_path=os.path.join(args.output_dir, "attention_example.png")
        )
        
        # Embedding visualization
        visualize_embeddings(
            model,
            test_loader,
            id2label,
            device,
            save_path=os.path.join(args.output_dir, "embeddings_umap.png")
        )
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()