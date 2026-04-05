#!/usr/bin/env python3
"""Inspect Cohere model config to get exact mel parameters."""
from transformers import AutoProcessor
import json

processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)

print("Feature Extractor Config:")
config = processor.feature_extractor.__dict__
for key, value in sorted(config.items()):
    if not key.startswith('_'):
        print(f"  {key}: {value}")
