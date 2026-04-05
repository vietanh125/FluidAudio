#!/usr/bin/env python3
"""Simple test to just load their models."""
import coremltools as ct
import time

models_to_test = [
    ("frontend", "build/barathwaj-models/cohere_frontend.mlpackage"),
    ("encoder", "build/barathwaj-models/cohere_encoder.mlpackage"),
    ("decoder_fullseq", "build/barathwaj-models/cohere_decoder_fullseq_masked.mlpackage"),
    ("decoder_cached", "build/barathwaj-models/cohere_decoder_cached.mlpackage"),
]

for name, path in models_to_test:
    print(f"\nLoading {name}...")
    start = time.time()
    try:
        model = ct.models.MLModel(path)
        elapsed = time.time() - start
        print(f"  ✓ Loaded in {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ❌ Failed after {elapsed:.2f}s: {e}")

print("\n✅ All models loaded successfully!")
