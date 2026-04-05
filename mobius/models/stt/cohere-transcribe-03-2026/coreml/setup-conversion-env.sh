#!/bin/bash
# Setup environment for Cohere Transcribe CoreML conversion
# Works with Python 3.10.x

set -e

echo "=== Setting up Cohere Transcribe CoreML Conversion Environment ==="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "Detected Python version: $PYTHON_VERSION"

if [[ ! "$PYTHON_VERSION" =~ ^3\.10 ]]; then
    echo "⚠️  Warning: Python 3.10.x recommended, but will try with $PYTHON_VERSION"
fi

# Create virtual environment
echo ""
echo "1. Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "   .venv already exists, using existing environment"
else
    python3 -m venv .venv
    echo "   ✓ Created .venv"
fi

# Activate
source .venv/bin/activate

# Install dependencies
echo ""
echo "2. Installing dependencies..."
echo "   Installing PyTorch..."
pip install torch==2.11.0 torchaudio==2.11.0 --quiet

echo "   Installing CoreML Tools..."
pip install coremltools==9.0 --quiet

echo "   Installing transformers and audio libraries..."
pip install transformers soundfile librosa --quiet

echo "   ✓ All dependencies installed"

# Verify installation
echo ""
echo "3. Verifying installation..."
python3 -c "import torch; print(f'   PyTorch: {torch.__version__}')"
python3 -c "import coremltools as ct; print(f'   CoreML Tools: {ct.__version__}')"
python3 -c "import transformers; print(f'   Transformers: {transformers.__version__}')"

echo ""
echo "=== Environment Setup Complete! ==="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the encoder export:"
echo "  python export-ultra-static-encoder.py"
echo ""
echo "To test the full pipeline:"
echo "  python test-full-pipeline-mixed.py"
echo ""
