#!/usr/bin/env python3
"""Test BarathwajAnandan's full pipeline with CPU_AND_GPU (bypassing ANE)."""

import numpy as np
import soundfile as sf
import coremltools as ct
from transformers import AutoProcessor

print("Testing BarathwajAnandan's FULL pipeline with CPU_AND_GPU")
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

# Frontend (CPU_AND_GPU)
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

# Encoder (CPU_AND_GPU - bypasses ANE compilation error)
print("\n[2/3] Encoder (CPU_AND_GPU)...")
encoder = ct.models.MLModel(
    "build/barathwaj-models/cohere_encoder.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)
encoder_output = encoder.predict({
    "input_features": mel.astype(np.float32),
    "feature_length": np.array([3501], dtype=np.int32)
})

# Find encoder output
print(f"   Available outputs: {list(encoder_output.keys())}")
hidden_states = None
for key, value in encoder_output.items():
    if hasattr(value, 'shape'):
        print(f"   {key}: {value.shape}")
        if len(value.shape) == 3 and value.shape[1] == 438 and value.shape[2] == 1024:
            hidden_states = value
            print(f"   ✓ Using {key} as encoder output")

if hidden_states is None:
    raise ValueError("Could not find encoder output with shape (1, 438, 1024)")

# Decoder (CPU_AND_GPU)
print("\n[3/3] Decoder (CPU_AND_GPU)...")
decoder = ct.models.MLModel(
    "build/barathwaj-models/cohere_decoder_fullseq_masked.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)

prompt_tokens = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
padded_prompt = prompt_tokens + [2] * (108 - len(prompt_tokens))
prompt_ids = np.array([padded_prompt], dtype=np.int32)

decoder_attention_mask = np.ones((1, 108), dtype=np.int32)
decoder_attention_mask[0, len(prompt_tokens):] = 0

cross_attention_mask = np.ones((1, 1, 1, 438), dtype=np.float16)

decoder_output = decoder.predict({
    "input_ids": prompt_ids,
    "encoder_hidden_states": hidden_states.astype(np.float16),
    "decoder_attention_mask": decoder_attention_mask,
    "cross_attention_mask": cross_attention_mask,
})

# Find logits output
logits = None
for key, value in decoder_output.items():
    if hasattr(value, 'shape'):
        print(f"   {key}: {value.shape}")
        if len(value.shape) == 3 and value.shape[1] == 108:
            logits = value
            print(f"   ✓ Using {key} as logits")

if logits is None:
    raise ValueError(f"Could not find logits output")

# Extract tokens
all_tokens = np.argmax(logits[0], axis=-1)
generated_tokens = all_tokens[len(prompt_tokens):]

# Find EOS
eos_token_id = 3
eos_indices = np.where(generated_tokens == eos_token_id)[0]
if len(eos_indices) > 0:
    generated_tokens = generated_tokens[:eos_indices[0]]

# Decode
transcription = tokenizer.decode(generated_tokens.tolist(), skip_special_tokens=True)

print("\n" + "="*70)
print("TRANSCRIPTION RESULT")
print("="*70)
print(f'"{transcription}"')
print("="*70)
print(f"\nCompute Units: CPU_AND_GPU (no ANE)")
print(f"File: test-librispeech-real.wav")
print(f"Duration: {duration:.2f}s")
print(f"Tokens: {len(generated_tokens)}")
print()
