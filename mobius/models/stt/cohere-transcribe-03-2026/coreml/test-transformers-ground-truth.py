#!/usr/bin/env python3
"""Get ground truth transcription using transformers library."""
import torch
import soundfile as sf
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

print("=== Ground Truth Test (Transformers) ===\n")

# Load audio
print("Loading audio...")
import librosa
audio, sr = sf.read("test-real-speech.wav")
print(f"  Original: {len(audio)} samples @ {sr} Hz")

# Resample to 16kHz if needed
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sr = 16000
    print(f"  Resampled to: {len(audio)} samples @ {sr} Hz")

# Load model and processor
print("\nLoading Cohere model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    dtype=torch.float32,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)
model.eval()

# Process audio
print("\nProcessing with transformers...")
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

print(f"  Input features shape: {inputs['input_features'].shape}")
print(f"  Input length: {inputs['length']}")

# Generate
print("\nGenerating transcription...")
with torch.no_grad():
    generated_ids = model.generate(
        inputs["input_features"],
        max_new_tokens=50,
        num_beams=1,  # Greedy decoding
        do_sample=False
    )

# Decode
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"\n✅ Ground Truth Transcription:")
print(f'"{transcription}"')

# Also show token IDs
print(f"\nGenerated token IDs: {generated_ids[0].tolist()[:20]}...")
