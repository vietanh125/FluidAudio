#!/usr/bin/env python3
"""Debug decoder in PyTorch to understand cache behavior."""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from cohere_mel_spectrogram import CohereMelSpectrogram
from datasets import load_dataset

print("="*70)
print("PyTorch Decoder Debugging")
print("="*70)

# Load model
print("\n[1/4] Loading PyTorch model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True,
    torch_dtype=torch.float32,
)
model.eval()
print("   ✓ Model loaded")

# Load test audio
print("\n[2/4] Loading test audio...")
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", streaming=True)
sample = next(iter(dataset))
audio = np.array(sample["audio"]["array"], dtype=np.float32)
ground_truth = sample["text"].lower()
print(f"   Audio: {len(audio)} samples")
print(f"   Ground truth: {ground_truth}")

# Compute mel and encoder output
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

    # Project to decoder dimension
    if model.encoder_decoder_proj is not None:
        encoder_hidden = model.encoder_decoder_proj(encoder_hidden)

print(f"   Encoder output: {encoder_hidden.shape}")

# Decode step-by-step with detailed logging
print("\n[4/4] Step-by-step decoding with cache inspection...")

decoder_start_token_id = 13764
tokens = [decoder_start_token_id]

# Initialize cache
past_key_values = None

for step in range(5):
    print(f"\n{'='*70}")
    print(f"STEP {step}")
    print(f"{'='*70}")

    input_id = torch.tensor([[tokens[-1]]], dtype=torch.long)
    positions = torch.tensor([[step]], dtype=torch.long)

    print(f"Input token: {tokens[-1]}")
    print(f"Position: {step}")

    # Create attention masks
    if past_key_values is None or step == 0:
        # First step - no cache
        past_len = 0
        self_attention_cache = DynamicCache()
        cross_attention_cache = DynamicCache()
        past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)
    else:
        past_len = step

    total_len = past_len + 1

    # Self-attention mask (causal)
    query_positions = torch.tensor([[past_len]])
    key_positions = torch.arange(total_len)[None, :]
    causal_bool = key_positions > query_positions
    self_attention_mask = torch.zeros((1, 1, 1, total_len), dtype=encoder_hidden.dtype)
    self_attention_mask.masked_fill_(causal_bool[None, None, :, :], float("-inf"))

    # Cross-attention mask (attend to all encoder positions)
    cross_attention_mask = torch.ones((1, encoder_hidden.shape[1]), dtype=encoder_hidden.dtype)

    print(f"Self-attention mask shape: {self_attention_mask.shape}")
    print(f"Cross-attention mask shape: {cross_attention_mask.shape}")

    if past_key_values is not None and past_len > 0:
        print(f"Cache info:")
        if hasattr(past_key_values.self_attention_cache, 'key_cache'):
            for layer_idx in range(min(2, len(past_key_values.self_attention_cache.key_cache))):
                k_cache = past_key_values.self_attention_cache.key_cache[layer_idx]
                v_cache = past_key_values.self_attention_cache.value_cache[layer_idx]
                print(f"  Layer {layer_idx} - K cache: {k_cache.shape}, V cache: {v_cache.shape}")
                print(f"    K stats: min={k_cache.min():.6f}, max={k_cache.max():.6f}, mean={k_cache.mean():.6f}")

    # Call decoder
    with torch.no_grad():
        decoder_outputs, updated_cache = model.transf_decoder(
            input_ids=input_id,
            positions=positions,
            encoder_hidden_states=encoder_hidden,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=cross_attention_mask,
            past_key_values=past_key_values,
            cache_position=None,
            kv_seq_len=None,
        )

        # Get logits
        logits = model.log_softmax(decoder_outputs)
        logits = logits.squeeze(1)  # (1, vocab_size)

        next_token = int(torch.argmax(logits[0]))
        tokens.append(next_token)

        # Update cache for next iteration
        past_key_values = updated_cache

        print(f"\nOutput token: {next_token}")
        print(f"Top-5 tokens: {torch.topk(logits[0], 5).indices.tolist()}")
        print(f"Top-5 logits: {torch.topk(logits[0], 5).values.tolist()}")

        if updated_cache is not None:
            print(f"\nUpdated cache info:")
            if hasattr(updated_cache.self_attention_cache, 'key_cache'):
                for layer_idx in range(min(2, len(updated_cache.self_attention_cache.key_cache))):
                    k_cache = updated_cache.self_attention_cache.key_cache[layer_idx]
                    v_cache = updated_cache.self_attention_cache.value_cache[layer_idx]
                    print(f"  Layer {layer_idx} - K cache: {k_cache.shape}, V cache: {v_cache.shape}")

print(f"\n{'='*70}")
print(f"Final tokens: {tokens}")
print("="*70)
