#!/bin/bash
# Setup script for RWKV-Raven-Cog environment

echo "Setting up RWKV-Raven-Cog environment..."

# Make sure git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "Installing git-lfs..."
    # Installation commands would depend on the system
    echo "Please install git-lfs: https://git-lfs.com"
    exit 1
fi

# Initialize git-lfs
echo "Initializing git-lfs..."
git lfs install

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create directories for transformed models
echo "Creating output directories..."
mkdir -p opencog_transformed

# Clone RWKV model (if network access is available)
if ping -c 1 huggingface.co &> /dev/null; then
    echo "Cloning RWKV-4-Raven model..."
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/BlinkDL/rwkv-4-raven
    echo "Model cloned successfully (LFS files are pointers only)"
else
    echo "Network access to huggingface.co not available."
    echo "Using mock model structure for development."
fi

echo "Environment setup complete!"
echo ""
echo "To run the OpenCog transformation:"
echo "  python opencog_transform.py"
echo ""
echo "Or install as a package:"
echo "  pip install -e ."
echo "  rwkv-opencog-transform"