# GitHub Setup Guide

## ✅ Repository Initialized!

I've created:
- ✅ `.gitignore` file (excludes venv, models, outputs)
- ✅ Git repository initialized
- ✅ Empty placeholder files for `models/` and `outputs/` directories

## Steps to Push to GitHub

### Step 1: Add All Files

```bash
cd emotion-lyrics-project
git add .
```

### Step 2: Make Initial Commit

```bash
git commit -m "Initial commit: Emotion detection in song lyrics project"
```

### Step 3: Create GitHub Repository

1. Go to [github.com](https://github.com) and sign in
2. Click the **"+"** button → **"New repository"**
3. Repository name: `emotion-lyrics-detection` (or any name you like)
4. Description: "Emotion detection in song lyrics using BERT fine-tuning and LLM in-context learning"
5. Set to **Public** or **Private** (your choice)
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click **"Create repository"**

### Step 4: Connect and Push

After creating the repo, GitHub will show you commands. Use these:

```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/emotion-lyrics-detection.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

### Alternative: Using SSH (if you have SSH keys set up)

```bash
git remote add origin git@github.com:YOUR_USERNAME/emotion-lyrics-detection.git
git branch -M main
git push -u origin main
```

## What Gets Pushed?

✅ **Will be pushed:**
- All source code (`src/`, `main.py`, etc.)
- Configuration files (`requirements.txt`, etc.)
- Documentation (`.md` files)
- Data processing scripts
- Empty directories (`models/`, `outputs/`)

❌ **Won't be pushed (excluded by .gitignore):**
- Virtual environment (`emo_env/`)
- Trained models (`models/*.pt`)
- Generated outputs (`outputs/*.png`, `outputs/*.txt`)
- Cache files (`__pycache__/`)
- System files (`.DS_Store`)

## Quick Commands Summary

```bash
# Navigate to project
cd emotion-lyrics-project

# Stage all files
git add .

# Commit
git commit -m "Initial commit: Emotion detection in song lyrics project"

# Create repo on GitHub (do this in browser first!)

# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push
git branch -M main
git push -u origin main
```

## Future Updates

After initial push, to update:

```bash
git add .
git commit -m "Your commit message"
git push
```

## Need Help?

If you get authentication errors, you may need to:
1. Use GitHub Personal Access Token (instead of password)
2. Or set up SSH keys
3. Or use GitHub CLI: `gh auth login`

Let me know if you need help with any step!

