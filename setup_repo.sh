#!/bin/bash
# GitHub Repository Setup Script
# Run this script to initialize git and create GitHub repository

echo "🚀 Setting up GitHub repository..."

# Step 1: Initialize git repository
echo "📦 Initializing git repository..."
git init

# Step 2: Add all files
echo "➕ Adding files..."
git add .

# Step 3: Make initial commit
echo "💾 Making initial commit..."
git commit -m "Initial commit: Facial recognition project"

# Step 4: Check if GitHub CLI is installed
if command -v gh &> /dev/null; then
    echo "✅ GitHub CLI found!"
    echo "🔐 Make sure you're authenticated: gh auth login"
    echo ""
    read -p "Create repository on GitHub? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Repository name (default: facial_recognition): " repo_name
        repo_name=${repo_name:-facial_recognition}
        
        read -p "Make it private? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            gh repo create $repo_name --private --source=. --remote=origin --push
        else
            gh repo create $repo_name --public --source=. --remote=origin --push
        fi
        echo "✅ Repository created and pushed!"
    fi
else
    echo "⚠️  GitHub CLI not found. Using manual method..."
    echo ""
    echo "📝 Next steps:"
    echo "1. Go to https://github.com/new"
    echo "2. Create a new repository (don't initialize with README)"
    echo "3. Copy the repository URL"
    echo "4. Run these commands:"
    echo ""
    echo "   git remote add origin https://github.com/YOUR_USERNAME/facial_recognition.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
fi

echo ""
echo "✨ Done!"
