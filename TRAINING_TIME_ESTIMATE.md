# Training Time Estimates

## Your Setup
- **Device**: CPU (no GPU detected)
- **Dataset**: ~38,000 training samples, ~8,000 validation samples
- **Model**: BERT-base-uncased (110M parameters)
- **Batch size**: 16
- **Epochs**: 3

## Estimated Training Times

### On CPU (Your Current Setup):
- **Per epoch**: ~2-4 hours
- **Total training time (3 epochs)**: ~6-12 hours
- **Validation**: ~15-30 minutes per epoch

### If You Had GPU:
- **Per epoch**: ~15-30 minutes
- **Total training time (3 epochs)**: ~45-90 minutes

## Why It's Slow on CPU?

BERT-base has 110 million parameters. On CPU, matrix operations are much slower than GPU. Each training step processes 16 samples at a time and updates all parameters.

## Ways to Speed Up:

### 1. Reduce Epochs (Fastest)
```bash
python main.py --train_bert --epochs 2 --batch_size 16 --lr 2e-5
```
- Saves 33% time (4-8 hours instead of 6-12)

### 2. Use Smaller Model
```bash
python main.py --train_bert --epochs 3 --batch_size 16 --lr 2e-5 --bert_model distilbert-base-uncased
```
- DistilBERT is 2x faster, 60% smaller
- Still very good performance
- ~3-6 hours total on CPU

### 3. Increase Batch Size (if you have RAM)
```bash
python main.py --train_bert --epochs 3 --batch_size 32 --lr 2e-5
```
- Processes more samples per step
- Slightly faster (~20-30% speedup)
- Needs more RAM

### 4. Best Combination:
```bash
python main.py --train_bert --epochs 2 --batch_size 32 --lr 2e-5 --bert_model distilbert-base-uncased
```
- **Estimated time**: ~2-3 hours on CPU
- Good balance of speed and performance

## Recommendation:

Since you're on CPU, I'd suggest:

**Option A: Quick Test (2 hours)**
```bash
python main.py --train_bert --epochs 2 --bert_model distilbert-base-uncased --batch_size 32
```

**Option B: Full Training (6-12 hours)**
```bash
python main.py --train_bert --epochs 3 --batch_size 16
```
- Start it before bed or when you have time
- Training will save the best model automatically
- You can stop it early with Ctrl+C and still use the best model saved so far

## Monitoring Progress:

The training will show:
- Progress bars for each epoch
- Train loss after each epoch  
- Validation F1 score
- Best model checkpoint saved

You can safely stop training (Ctrl+C) and the best model so far will be saved.

## After Training:

Evaluation and visualization are much faster:
- **Test evaluation**: ~5-10 minutes
- **Visualizations**: ~5-10 minutes
- **LLM evaluation**: ~30-60 minutes (depends on model)

