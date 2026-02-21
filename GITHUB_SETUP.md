# GitHub Repository Setup Guide

## Method 1: Using GitHub CLI (Recommended - Easiest)

### Step 1: Install GitHub CLI
```bash
# On macOS
brew install gh

# Then authenticate
gh auth login
```

### Step 2: Initialize Git Repository
```bash
# Navigate to your project
cd /path/to/your/project

# Initialize git
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit"
```

### Step 3: Create Repository on GitHub
```bash
# Create public repository
gh repo create facial_recognition --public --source=. --remote=origin --push

# Or create private repository
gh repo create facial_recognition --private --source=. --remote=origin --push
```

**That's it!** The repository is created and your code is pushed.

---

## Method 2: Using Git Commands Only (No GitHub CLI)

### Step 1: Initialize Git Repository
```bash
cd /path/to/your/project
git init
git add .
git commit -m "Initial commit"
```

### Step 2: Create Repository on GitHub
1. Go to https://github.com/new
2. Create a new repository (don't initialize with README)
3. Copy the repository URL (e.g., `https://github.com/yourusername/facial_recognition.git`)

### Step 3: Connect and Push
```bash
# Add remote repository
git remote add origin https://github.com/yourusername/facial_recognition.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

---

## Common Git Commands Reference

### Daily Workflow
```bash
# Check status
git status

# Add files
git add .                    # Add all files
git add filename.py          # Add specific file

# Commit changes
git commit -m "Your message"

# Push to GitHub
git push

# Pull latest changes
git pull
```

### Branch Management
```bash
# Create new branch
git checkout -b feature-name

# Switch branches
git checkout main
git checkout feature-name

# List branches
git branch

# Merge branch
git checkout main
git merge feature-name
```

### Viewing History
```bash
# View commit history
git log

# View changes
git diff

# View specific file changes
git diff filename.py
```

### Undoing Changes
```bash
# Unstage files (keep changes)
git reset HEAD filename.py

# Discard changes in working directory
git checkout -- filename.py

# Undo last commit (keep changes)
git reset --soft HEAD~1
```

---

## Quick Setup Script (Copy & Paste)

```bash
# Initialize repository
git init
git add .
git commit -m "Initial commit"

# If using GitHub CLI:
gh repo create facial_recognition --public --source=. --remote=origin --push

# If NOT using GitHub CLI:
# 1. Create repo on github.com first
# 2. Then run:
git remote add origin https://github.com/YOUR_USERNAME/facial_recognition.git
git branch -M main
git push -u origin main
```
