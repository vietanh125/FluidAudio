#!/usr/bin/env python3
"""Download ONNX models from HuggingFace."""
from huggingface_hub import snapshot_download
import os

print("=== Downloading ONNX Models ===\n")

repo_id = "onnx-community/cohere-transcribe-03-2026-ONNX"
local_dir = "onnx-models"

print(f"Downloading from: {repo_id}")
print(f"Saving to: {local_dir}\n")

# Download only the FP16 models to save space (they're in onnx/ subdirectory)
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    allow_patterns=[
        "onnx/encoder_model_fp16.onnx",
        "onnx/encoder_model_fp16.onnx_data*",
        "onnx/decoder_model_merged_fp16.onnx", 
        "onnx/decoder_model_merged_fp16.onnx_data",
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json"
    ]
)

print("\n✓ Download complete!")
print(f"\nDownloaded files:")
for root, dirs, files in os.walk(local_dir):
    for file in sorted(files):
        if file.startswith('.'):
            continue
        filepath = os.path.join(root, file)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        rel_path = os.path.relpath(filepath, local_dir)
        print(f"  {rel_path}: {size_mb:.2f} MB")
