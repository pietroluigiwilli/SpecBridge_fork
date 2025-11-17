#!/bin/bash
# Install Git LFS using conda (no sudo required)

echo "Installing Git LFS via conda..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not available. Please activate your conda environment first."
    exit 1
fi

# Install git-lfs from conda-forge
conda install -c conda-forge git-lfs -y

# Verify installation
if command -v git-lfs &> /dev/null; then
    echo "Git LFS installed successfully!"
    git-lfs version
    echo ""
    echo "Now run: git lfs install"
else
    echo "Installation may have failed. Try:"
    echo "  conda install -c conda-forge git-lfs"
fi

