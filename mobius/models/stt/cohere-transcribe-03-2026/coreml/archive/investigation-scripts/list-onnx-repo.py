#!/usr/bin/env python3
"""List files in ONNX repo."""
from huggingface_hub import list_repo_files

repo_id = "onnx-community/cohere-transcribe-03-2026-ONNX"
files = list_repo_files(repo_id)

print(f"Files in {repo_id}:\n")
for f in sorted(files):
    print(f"  {f}")
