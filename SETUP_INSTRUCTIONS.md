# Setup Instructions

## ✅ Virtual Environment Created!

I've created a virtual environment for you. Now follow these steps:

### Step 1: Activate Virtual Environment

**On macOS/Linux:**
```bash
cd emotion-lyrics-project
source emo_env/bin/activate
```

You should see `(emo_env)` at the start of your command prompt.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch
- Transformers (HuggingFace)
- Pandas, scikit-learn
- Matplotlib, seaborn
- UMAP-learn
- Datasets library

### Step 3: Verify Installation

```bash
python -c "import torch; import transformers; print('✅ All packages installed!')"
```

### Step 4: Start Training!

```bash
python main.py --train_bert --epochs 3 --batch_size 16 --lr 2e-5
```

---

## Alternative: Install Without Virtual Environment

If you prefer not to use a virtual environment (not recommended but works):

```bash
cd emotion-lyrics-project
pip3 install -r requirements.txt
python3 main.py --train_bert --epochs 3 --batch_size 16 --lr 2e-5
```

**Note:** This installs packages globally. Use venv if possible to avoid conflicts.

---

## After Each Session

When you're done working, you can deactivate the virtual environment:

```bash
deactivate
```

Next time you work on the project, just activate it again:
```bash
source emo_env/bin/activate
```

---

## Troubleshooting

**If activation fails:**
```bash
# Make sure you're in the project directory
cd emotion-lyrics-project

# Activate again
source emo_env/bin/activate
```

**If pip install fails:**
```bash
# Try upgrading pip first
pip install --upgrade pip
pip install -r requirements.txt
```

**If you get permission errors:**
- Make sure you're not using `sudo` inside the venv
- The venv should handle permissions automatically

