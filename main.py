
import argparse
import os
import torch

from emotion_detection import (
    # Data loading
    create_dataloaders, create_label_mapping, load_data,
    # BERT model
    build_model, train_bert, eval_bert, predict_bert, eval_vanilla_bert,
    # LLM evaluation
    eval_llm_on_test, load_llm_model,
    # Utilities
    calculate_metrics, save_metrics
)

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


def main():
    global label2id, id2label
    
    parser = argparse.ArgumentParser(
        description='Emotion Detection in Song Lyrics - Main Orchestration'
    )
    
    parser.add_argument("--train_bert", action="store_true", help="Train BERT model")
    parser.add_argument("--eval_bert", action="store_true", help="Evaluate BERT model on test set")
    parser.add_argument("--eval_vanilla_bert", action="store_true", help="Evaluate vanilla (pretrained) BERT baseline")
    parser.add_argument("--eval_llm", action="store_true", help="Evaluate LLM with in-context learning (legacy)")
    parser.add_argument("--eval_llm_zero", action="store_true", help="Evaluate LLM in zero-shot mode")
    parser.add_argument("--eval_llm_few", action="store_true", help="Evaluate LLM in few-shot mode")
    
    # Optional arguments
    parser.add_argument("--train_path", type=str, default="data/train.csv", help="Path to training CSV")
    parser.add_argument("--val_path", type=str, default="data/val.csv", help="Path to validation CSV")
    parser.add_argument("--test_path", type=str, default="data/test.csv", help="Path to test CSV")
    parser.add_argument("--model_path", type=str, default="models/bert_best.pt", help="Path to saved BERT model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (use 2-4 for GPUs with 4GB VRAM)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (1e-5 recommended for fine-tuning)")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", help="Model name (bert-base-uncased recommended)")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length (256 for 4GB VRAM, 512 for 8GB+)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--weighted_sampling", action="store_true", help="Use weighted sampling for class balance")
    parser.add_argument("--llm_name", type=str, default="gpt2", help="LLM model name (gpt2 for testing, or larger models)")
    parser.add_argument("--llm_max_samples", type=int, default=None, help="Limit LLM eval samples (for testing)")
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
        max_length=args.max_length,
        use_weighted_sampling=args.weighted_sampling
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
            id_to_label=id2label,
            early_stopping_patience=args.patience
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
        
        # Get predictions for metrics
        texts, labels, preds = predict_bert(model, test_loader, device)
        
        # Save metrics
        label_names = [id2label[i] for i in sorted(id2label.keys())]
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
    
    # Evaluate LLM - Zero-shot mode (using local model)
    if args.eval_llm_zero:
        print("\n" + "=" * 60)
        print("EVALUATING LLM (ZERO-SHOT)")
        print("=" * 60)
        
        # Load local model
        print(f"Loading local model: {args.llm_name}...")
        llm_model, llm_tokenizer = load_llm_model(args.llm_name, str(device))
        
        eval_llm_on_test(
            test_loader,
            id2label,
            mode="zero-shot",
            output_path=os.path.join(args.output_dir, "llm_zero_shot_predictions.csv"),
            model=llm_model,
            tokenizer=llm_tokenizer,
            device=str(device),
            model_name=args.llm_name,
            use_api=False,
            max_samples=args.llm_max_samples
        )
    
    # Evaluate LLM - Few-shot mode (using local model)
    if args.eval_llm_few:
        print("\n" + "=" * 60)
        print("EVALUATING LLM (FEW-SHOT)")
        print("=" * 60)
        
        # Load local model (reuse if already loaded)
        if not args.eval_llm_zero:
            print(f"Loading local model: {args.llm_name}...")
            llm_model, llm_tokenizer = load_llm_model(args.llm_name, str(device))
        
        eval_llm_on_test(
            test_loader,
            id2label,
            mode="few-shot",
            output_path=os.path.join(args.output_dir, "llm_few_shot_predictions.csv"),
            model=llm_model,
            tokenizer=llm_tokenizer,
            device=str(device),
            model_name=args.llm_name,
            use_api=False,
            max_samples=args.llm_max_samples
        )
    
    # Legacy LLM evaluation (for backward compatibility)
    if args.eval_llm:
        print("\n" + "=" * 60)
        print("EVALUATING LLM via HuggingFace API")
        print("=" * 60)
        
        mode = "zero-shot" if args.zero_shot else "few-shot"
        
        eval_llm_on_test(
            test_loader,
            id2label,
            mode=mode,
            output_path=os.path.join(args.output_dir, "llm_predictions.csv"),
            model_name=args.llm_name,
            use_api=True,
            max_samples=args.llm_max_samples
        )
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
