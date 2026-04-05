#!/usr/bin/env python3
"""Compute cross-attention caches offline and save to pickle.

This script loads the PyTorch model, computes cross-attention K/V caches
from encoder output, and saves them for later use.
"""
import torch
import numpy as np
import pickle
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import soundfile as sf

print("Loading model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    dtype=torch.float32,
    trust_remote_code=True
)
model.eval()

print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)

# Load audio
print("\nLoading audio...")
audio, sr = sf.read("test-audio.wav")
target_length = 480000
if len(audio) < target_length:
    audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')

# Process
print("Processing audio...")
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

# Encode
print("Encoding...")
with torch.no_grad():
    encoder_outputs = model.audio_encoder(
        input_features=inputs["input_features"],
        length=inputs["length"]
    )
    encoder_hidden = encoder_outputs.last_hidden_state

print(f"  Encoder output shape: {encoder_hidden.shape}")

# Project encoder outputs
print("\nComputing cross-attention caches...")
encoder_proj = model.encoder_decoder_proj(encoder_hidden)

B = encoder_proj.shape[0]
enc_seq_len = encoder_proj.shape[1]
H = 8  # num_heads
D = 128  # head_dim

# Compute K/V for all decoder layers
cross_caches = {}
for i, layer in enumerate(model.transf_decoder._decoder.layers):
    cross_attn = layer.second_sub_layer

    key = cross_attn.key_net(encoder_proj).view(B, enc_seq_len, H, D)
    value = cross_attn.value_net(encoder_proj).view(B, enc_seq_len, H, D)

    # Stack as (2, B, enc_seq_len, H, D)
    cross_cache = torch.stack([key, value], dim=0).numpy().astype(np.float32)

    cross_caches[f"cross_cache{i}"] = cross_cache
    print(f"  cross_cache{i}: {cross_cache.shape}")

# Save
output_dir = Path("build")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "cross_caches.pkl"

print(f"\nSaving to {output_path}...")
with open(output_path, "wb") as f:
    pickle.dump(cross_caches, f)

print("✅ Done!")
print("\nUsage:")
print("  with open('build/cross_caches.pkl', 'rb') as f:")
print("      cross_caches = pickle.load(f)")
