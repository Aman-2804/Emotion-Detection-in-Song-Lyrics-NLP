# 🚀 Quick Start Guide

## Your Dataset is Ready! ✅

- **Training samples**: ~38,000
- **Validation samples**: ~8,000  
- **Test samples**: ~8,000
- **Format**: CSV with `text,label` columns ✓

## Step-by-Step Training

### 1. Navigate to Project
```bash
cd emotion-lyrics-project
```

### 2. Activate Virtual Environment (if using one)
```bash
source emo_env/bin/activate
# or
python3 -m venv emo_env && source emo_env/bin/activate
pip install -r requirements.txt
```

### 3. Train BERT Model
```bash
python main.py --train_bert --epochs 3 --batch_size 16 --lr 2e-5
```

**What this does:**
- Trains BERT for 3 epochs
- Uses batch size of 16
- Learning rate: 2e-5
- Saves best model to `models/bert_best.pt`
- Shows progress bars and validation F1 scores

### 4. Evaluate on Test Set
```bash
python main.py --eval_bert
```

**What this does:**
- Loads trained model
- Evaluates on test set
- Prints macro F1 and classification report
- Generates confusion matrix
- Saves metrics to `outputs/bert_metrics.txt`

### 5. Compare with Baseline
```bash
# Evaluate vanilla (pretrained) BERT baseline
python main.py --eval_vanilla_bert

# This shows how much fine-tuning helped!
```

### 6. Generate Visualizations
```bash
python main.py --visualize
```

**What this does:**
- Creates attention heatmaps
- Generates embedding visualizations (UMAP)
- Saves to `outputs/` directory

## All-in-One Command

Run everything in sequence:
```bash
python main.py --train_bert --eval_bert --eval_vanilla_bert --visualize
```

## Expected Training Time

- **Training**: ~30-60 minutes (depending on GPU/CPU)
- **Evaluation**: ~5-10 minutes
- **Visualizations**: ~5-10 minutes

## Monitor Training

You'll see:
- Progress bars for each epoch
- Train loss after each epoch
- Validation F1 score
- Best model checkpoint saved

## Troubleshooting

**Out of Memory?**
- Reduce batch size: `--batch_size 8`
- Use smaller model: `--bert_model distilbert-base-uncased`

**Training too slow?**
- Reduce epochs: `--epochs 2`
- Use smaller model: `--bert_model distilbert-base-uncased`

**Need to resume?**
- Model is saved after each epoch
- Just run `--eval_bert` to use saved model

## Next Steps After Training

1. ✅ Check `outputs/bert_metrics.txt` for detailed metrics
2. ✅ View `outputs/confusion_matrix_bert.png`
3. ✅ Compare vanilla vs fine-tuned F1 scores
4. ✅ Analyze attention visualizations
5. ✅ Run LLM evaluation: `python main.py --eval_llm`

## You're Ready! 🎉

Everything is set up. Just run the training command and watch it go!

