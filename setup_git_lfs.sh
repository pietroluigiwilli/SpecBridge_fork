#!/bin/bash
# Setup script for Git LFS to handle large checkpoint files

set -e

echo "Setting up Git LFS..."

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "ERROR: git-lfs is not installed."
    echo "Please install it first:"
    echo "  - On Ubuntu/Debian: sudo apt-get install git-lfs"
    echo "  - On macOS: brew install git-lfs"
    echo "  - On CentOS/RHEL: sudo yum install git-lfs"
    exit 1
fi

# Initialize git-lfs
echo "Initializing Git LFS..."
git lfs install

# Track .pt files
echo "Tracking .pt files with Git LFS..."
git lfs track "*.pt"

# Add .gitattributes
echo "Adding .gitattributes..."
git add .gitattributes

# Check if the large file is in the current commit/staging
LARGE_FILE="runs/specbridge_align_chemberta_pub_v3g_nosample/ckpt_000400.pt"

if git ls-files --error-unmatch "$LARGE_FILE" &> /dev/null; then
    echo "Large file found in git index. Migrating to Git LFS..."
    
    # Remove from index (but keep local file)
    git rm --cached "$LARGE_FILE" 2>/dev/null || true
    
    # Re-add with git-lfs
    git add "$LARGE_FILE"
    
    echo "File migrated to Git LFS."
fi

# Check if file exists in git history
if git log --all --full-history --quiet -- "$LARGE_FILE" &> /dev/null; then
    echo ""
    echo "WARNING: The large file exists in git history!"
    echo "You need to remove it from history before pushing."
    echo ""
    echo "Option 1: If it's only in the most recent commit(s) and not pushed yet:"
    echo "  git reset HEAD~1  # Go back one commit (adjust number as needed)"
    echo "  git add .gitattributes"
    echo "  git add $LARGE_FILE  # This will use git-lfs now"
    echo "  git commit -m 'Migrate checkpoint files to Git LFS'"
    echo ""
    echo "Option 2: If it's in older commits or already pushed, use BFG Repo-Cleaner (recommended):"
    echo "  # Install BFG: https://rtyley.github.io/bfg-repo-cleaner/"
    echo "  java -jar bfg.jar --delete-files ckpt_000400.pt"
    echo "  git reflog expire --expire=now --all && git gc --prune=now --aggressive"
    echo ""
    echo "Option 3: Use git filter-branch (slower but built-in):"
    echo "  git filter-branch --force --index-filter \\"
    echo "    'git rm --cached --ignore-unmatch $LARGE_FILE' \\"
    echo "    --prune-empty --tag-name-filter cat -- --all"
    echo "  git reflog expire --expire=now --all"
    echo "  git gc --prune=now --aggressive"
else
    echo ""
    echo "Git LFS setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Commit the changes:"
    echo "   git commit -m 'Add Git LFS tracking for .pt files'"
    echo ""
    echo "2. Push to remote:"
    echo "   git push origin master"
fi

