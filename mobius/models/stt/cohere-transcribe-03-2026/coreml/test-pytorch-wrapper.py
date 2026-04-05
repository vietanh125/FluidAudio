#!/usr/bin/env python3
"""Test our PyTorch wrapper before CoreML conversion."""

import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq
from cohere_mel_spectrogram import CohereMelSpectrogram
from datasets import load_dataset
import importlib.util
spec = importlib.util.spec_from_file_location("export_decoder_cached", "export-decoder-cached.py")
export_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_module)
SimplifiedCachedDecoderWrapper = export_module.SimplifiedCachedDecoderWrapper

print("="*70)
print("Testing PyTorch Wrapper Before CoreML Conversion")
print("="*70)

# Load model
print("\n[1/4] Loading model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True,
    torch_dtype=torch.float32,
)
model.eval()
print("   ✓ Model loaded")

# Wrap decoder
wrapped_decoder = SimplifiedCachedDecoderWrapper(model, max_seq_len=108)
wrapped_decoder.eval()
print("   ✓ Decoder wrapped")

# Load test audio
print("\n[2/4] Loading test audio...")
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", streaming=True)
sample = next(iter(dataset))
audio = np.array(sample["audio"]["array"], dtype=np.float32)
print(f"   Audio: {len(audio)} samples")

# Get encoder output
print("\n[3/4] Running encoder...")
mel_processor = CohereMelSpectrogram()
mel = mel_processor(audio)
mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3001 - mel.shape[2])), mode='constant', constant_values=0)
mel_tensor = torch.from_numpy(mel_padded).float()

with torch.no_grad():
    encoder_outputs = model.encoder(
        input_features=mel_tensor,
        lengths=torch.tensor([3001], dtype=torch.long),
        return_dict=True
    )
    encoder_hidden = encoder_outputs.last_hidden_state
    if model.encoder_decoder_proj is not None:
        encoder_hidden = model.encoder_decoder_proj(encoder_hidden)

print(f"   Encoder output: {encoder_hidden.shape}")

# Test wrapped decoder
print("\n[4/4] Testing wrapped decoder...")

decoder_start_token_id = 13764
tokens = [decoder_start_token_id]

# Initialize cache
cache_k = torch.zeros(8, 8, 108, 128, dtype=torch.float32)
cache_v = torch.zeros(8, 8, 108, 128, dtype=torch.float32)

for step in range(5):
    input_id = torch.tensor([[tokens[-1]]], dtype=torch.long)
    step_tensor = torch.tensor([step], dtype=torch.int32)
    cross_mask = torch.ones(1, 1, 1, encoder_hidden.shape[1], dtype=torch.float32)

    print(f"\nStep {step}:")
    print(f"  Input token: {tokens[-1]}")

    with torch.no_grad():
        logits, new_cache_k, new_cache_v = wrapped_decoder(
            input_id,
            encoder_hidden,
            cache_k,
            cache_v,
            step_tensor,
            cross_mask,
        )

    next_token = int(torch.argmax(logits[0]))
    tokens.append(next_token)

    # Update cache
    cache_k = new_cache_k
    cache_v = new_cache_v

    print(f"  Output token: {next_token}")
    print(f"  Top-5 tokens: {torch.topk(logits[0], 5).indices.tolist()}")
    print(f"  Cache K shape: {cache_k.shape}, non-zero: {(cache_k != 0).sum().item()}")

print(f"\n{'='*70}")
print(f"Final tokens: {tokens}")
print(f"Expected: [13764, 7, 4, 16, 62, ...]")
print(f"Match: {tokens[:5] == [13764, 7, 4, 16, 62]}")
print("="*70)
