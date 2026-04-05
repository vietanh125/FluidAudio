#!/usr/bin/env python3
"""Test with proper autoregressive decoding using cached decoder."""

import numpy as np
import soundfile as sf
import coremltools as ct
from transformers import AutoProcessor

print("Testing with AUTOREGRESSIVE DECODING (like PyTorch)")
print("="*70)

# Load tokenizer
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)
tokenizer = processor.tokenizer

# Load audio
audio, sr = sf.read("test-librispeech-real.wav")
if sr != 16000:
    raise ValueError(f"Audio must be 16kHz, got {sr}Hz")

duration = len(audio) / 16000
print(f"Audio: test-librispeech-real.wav ({duration:.2f}s, {len(audio)} samples)")

# Pad to 560k
max_samples = 560000
if len(audio) < max_samples:
    audio_padded = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
else:
    audio_padded = audio[:max_samples]

# Frontend
print("\n[1/3] Frontend...")
frontend = ct.models.MLModel(
    "build/barathwaj-models/cohere_frontend.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)
frontend_output = frontend.predict({
    "audio_samples": audio_padded.astype(np.float32).reshape(1, -1),
    "audio_length": np.array([len(audio_padded)], dtype=np.int32)
})
mel = frontend_output["var_6916"]
print(f"   Mel: {mel.shape}")

# Encoder
print("\n[2/3] Encoder...")
encoder = ct.models.MLModel(
    "build/barathwaj-models/cohere_encoder.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)
encoder_output = encoder.predict({
    "input_features": mel.astype(np.float32),
    "feature_length": np.array([3501], dtype=np.int32)
})

# Find encoder output
hidden_states = None
for key, value in encoder_output.items():
    if hasattr(value, 'shape') and len(value.shape) == 3 and value.shape[1] == 438:
        hidden_states = value
        print(f"   Hidden states: {value.shape}")
        break

if hidden_states is None:
    raise ValueError("Could not find encoder output")

# Decoder - AUTOREGRESSIVE with cached decoder
print("\n[3/3] Autoregressive Decoding...")
decoder = ct.models.MLModel(
    "build/barathwaj-models/cohere_decoder_cached.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)

# Check decoder inputs
spec = decoder.get_spec()
print(f"   Decoder inputs:")
for inp in spec.description.input:
    print(f"     - {inp.name}")

# Start with decoder_start_token_id
decoder_start_token_id = 13764  # from generation_config.json
eos_token_id = 3
max_new_tokens = 100

generated_tokens = [decoder_start_token_id]
past_key_values = None

print(f"   Starting autoregressive generation (max {max_new_tokens} tokens)...")

for step in range(max_new_tokens):
    # Prepare input
    decoder_input = {
        "input_id": np.array([[generated_tokens[-1]]], dtype=np.int32),  # shape [1, 1]
        "encoder_hidden_states": hidden_states.astype(np.float16),
        "step": np.array([step], dtype=np.int32),  # shape [1]
        "cross_attention_mask": np.ones((1, 1, 1, 438), dtype=np.float16),
    }

    # Initialize cache on first step
    if step == 0:
        # Cache shape: [8 layers, 8 heads, 108 max_seq_len, 128 dim_per_head]
        decoder_input["cache_k"] = np.zeros((8, 8, 108, 128), dtype=np.float16)
        decoder_input["cache_v"] = np.zeros((8, 8, 108, 128), dtype=np.float16)
    else:
        # Use previous cache (outputs are var_2894 and var_2897)
        decoder_input["cache_k"] = past_key_values["var_2894"]
        decoder_input["cache_v"] = past_key_values["var_2897"]

    try:
        decoder_output = decoder.predict(decoder_input)
    except Exception as e:
        print(f"   ❌ Decoder error at step {step}: {e}")
        break

    # Extract logits and new cache (outputs: var_2891=logits, var_2894=cache_k, var_2897=cache_v)
    logits = decoder_output.get("var_2891")
    if logits is None:
        print(f"   ❌ Could not find logits (var_2891) in output: {list(decoder_output.keys())}")
        break

    new_cache = {
        "var_2894": decoder_output["var_2894"],  # new cache_k
        "var_2897": decoder_output["var_2897"],  # new cache_v
    }

    # Get next token (logits shape is [1, 16384])
    next_token = int(np.argmax(logits[0]))

    generated_tokens.append(next_token)
    past_key_values = new_cache

    # Stop at EOS
    if next_token == eos_token_id:
        print(f"   ✓ Generated {len(generated_tokens)-1} tokens (stopped at EOS)")
        break

    if (step + 1) % 20 == 0:
        print(f"   ... {step+1} tokens generated")

if generated_tokens[-1] != eos_token_id:
    print(f"   ⚠ Reached max_new_tokens without EOS")

# Decode (skip the start token)
transcription = tokenizer.decode(generated_tokens[1:], skip_special_tokens=True)

print("\n" + "="*70)
print("AUTOREGRESSIVE TRANSCRIPTION")
print("="*70)
print(f'"{transcription}"')
print("="*70)
print(f"\nGenerated {len(generated_tokens)-1} tokens")
print(f"Token IDs (first 20): {generated_tokens[:20]}")
print()
