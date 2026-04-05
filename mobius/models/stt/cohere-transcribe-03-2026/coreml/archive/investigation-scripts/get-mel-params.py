#!/usr/bin/env python3
"""Get actual mel parameters from Cohere preprocessor."""
from transformers import AutoModelForSpeechSeq2Seq
import torch

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    dtype=torch.float32,
    trust_remote_code=True
)

# Check if model has preprocessor
if hasattr(model, 'preprocessor'):
    print("Model has preprocessor!")
    preprocessor = model.preprocessor
    print(f"Type: {type(preprocessor)}")
    
    # Get featurizer config
    if hasattr(preprocessor, 'featurizer'):
        featurizer = preprocessor.featurizer
        print("\nFeaturizer attributes:")
        for attr in dir(featurizer):
            if not attr.startswith('_'):
                val = getattr(featurizer, attr, None)
                if not callable(val):
                    print(f"  {attr}: {val}")
else:
    print("No preprocessor in model")
    print("\nModel attributes:")
    for attr in dir(model):
        if not attr.startswith('_') and 'preprocess' in attr.lower():
            print(f"  {attr}")
