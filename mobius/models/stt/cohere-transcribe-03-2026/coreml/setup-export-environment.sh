#!/bin/bash
# Setup exact environment matching BarathwajAnandan's export
# They used: torch==2.2.2, coremltools==8.3.0

set -e

echo "=== Setting up BarathwajAnandan Export Environment ==="
echo ""

# Create new venv for export
EXPORT_VENV=".venv-export"

if [ -d "$EXPORT_VENV" ]; then
    echo "Removing existing export venv..."
    rm -rf "$EXPORT_VENV"
fi

echo "Creating new venv: $EXPORT_VENV"
python3.10 -m venv "$EXPORT_VENV"

echo "Activating venv..."
source "$EXPORT_VENV/bin/activate"

echo "Installing exact versions..."
pip install --upgrade pip

# Install torch 2.2.2 (Feb 2024 release)
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

# Install coremltools 8.3.0
pip install coremltools==8.3.0

# Install transformers (need compatible version with torch 2.2.2)
pip install transformers==4.38.2

# Install other dependencies
pip install soundfile librosa numpy

echo ""
echo "=== Environment Ready ==="
echo "Installed versions:"
python -c "import torch; print(f'  torch: {torch.__version__}')"
python -c "import coremltools as ct; print(f'  coremltools: {ct.__version__}')"
python -c "import transformers; print(f'  transformers: {transformers.__version__}')"

echo ""
echo "To use this environment:"
echo "  source $EXPORT_VENV/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
