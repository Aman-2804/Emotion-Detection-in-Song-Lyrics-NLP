# Emotion Detection in Song Lyrics

A project for emotion detection in song lyrics using BERT fine-tuning and LLM in-context learning, with a focus on interpretability analysis.

## Project Overview

This project compares fine-tuned BERT models against large language models (LLMs) using in-context learning for emotion classification in song lyrics. The project includes:

- **BERT Fine-tuning**: Fine-tuned BERT model for emotion classification
- **LLM Evaluation**: Zero-shot and few-shot prompting with open-source LLMs
- **Interpretability Analysis**: Attention visualization and embedding space analysis

## Setup

### 1. Create Virtual Environment

```bash
cd emotion-lyrics-project
python3 -m venv emo_env
source emo_env/bin/activate  # On Windows: emo_env\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
emotion-lyrics-project/
│
├── data/                    # Dataset files
│   ├── train.csv           # Training data
│   ├── val.csv             # Validation data
│   └── test.csv            # Test data
│
├── models/                  # Saved model checkpoints
│   └── bert_best.pt        # Best BERT model
│
├── outputs/                 # Results and visualizations
│   ├── bert_metrics.txt    # BERT evaluation metrics
│   ├── llm_metrics.txt     # LLM evaluation metrics
│   ├── confusion_matrix_bert.png
│   ├── attention_example.png
│   └── embeddings_umap.png
│
├── src/                     # Source code
│   ├── data_loader.py      # Data loading utilities
│   ├── bert_model.py       # BERT model implementation
│   ├── llm_eval.py         # LLM evaluation
│   ├── interpretability.py # Interpretability analysis
│   ├── utils.py            # Utility functions
│   └── __init__.py
│
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Running BERT Training

To train the BERT model:

```bash
python main.py --mode train --model_type bert
```

The script will:
- Load training and validation data from `data/`
- Fine-tune BERT model for emotion classification
- Save best model to `models/bert_best.pt`
- Generate evaluation metrics in `outputs/bert_metrics.txt`

## Running LLM Evaluation

To evaluate LLMs with in-context learning:

```bash
python main.py --mode eval --model_type llm --llm_name meta-llama/Llama-2-7b-hf
```

Options:
- `--zero_shot`: Use zero-shot prompting (default)
- `--few_shot`: Use few-shot prompting with examples
- `--llm_name`: HuggingFace model name

## Generating Interpretability Plots

To generate attention and embedding visualizations:

```bash
python main.py --mode interpret --model_path models/bert_best.pt
```

This will generate:
- Attention weight heatmaps in `outputs/attention_example.png`
- Embedding visualizations (UMAP/t-SNE) in `outputs/embeddings_umap.png`
- Confusion matrix in `outputs/confusion_matrix_bert.png`

## Data Format

The CSV files in `data/` should have the following format:

```csv
lyric,emotion
"I'm walking on sunshine, and don't it feel good",joy
"Tears falling down, can't stop the pain inside",sadness
```

## Expected Emotion Categories

The project supports 6-8 emotion categories:
- joy
- sadness
- anger
- fear
- love
- surprise
- disgust
- neutral

(Exact categories depend on your dataset)

## Model Configuration

Default hyperparameters:
- **BERT Model**: `bert-base-uncased` or `distilbert-base-uncased`
- **Learning Rate**: 2e-5 to 5e-5
- **Batch Size**: 16-32
- **Max Sequence Length**: 512
- **Epochs**: 3-5

## Evaluation Metrics

Primary metric: **Macro F1-score**

Secondary metrics:
- Accuracy
- Per-class F1, Precision, Recall
- Confusion matrices

## Future Work

- [ ] Add dataset loading utilities
- [ ] Implement training loop
- [ ] Add hyperparameter tuning
- [ ] Expand interpretability analysis
- [ ] Add experiment tracking

## License

[Add your license here]

## Contributors

[Add contributor names]
