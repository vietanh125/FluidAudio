#!/usr/bin/env python3
"""Transcribe test audio with BarathwajAnandan's full pipeline to get ground truth."""

import numpy as np
import soundfile as sf
import coremltools as ct
from transformers import AutoProcessor

print("Transcribing with BarathwajAnandan's FULL pipeline (known working)")
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
frontend = ct.models.MLModel("build/barathwaj-models/cohere_frontend.mlpackage")
frontend_output = frontend.predict({
    "audio_samples": audio_padded.astype(np.float32).reshape(1, -1),
    "audio_length": np.array([len(audio_padded)], dtype=np.int32)
})

# Find mel spectrogram output (shape should be (1, 128, 3501))
mel = None
for key, value in frontend_output.items():
    if hasattr(value, 'shape') and len(value.shape) == 3 and value.shape[1] == 128:
        mel = value
        print(f"   Using {key}: {value.shape}")
        break

if mel is None:
    raise ValueError(f"Could not find mel output. Available keys: {list(frontend_output.keys())}")

# Encoder
print("\n[2/3] Encoder...")
encoder = ct.models.MLModel("build/barathwaj-models/cohere_encoder.mlpackage")
encoder_output = encoder.predict({
    "input_features": mel.astype(np.float32),
    "feature_length": np.array([3501], dtype=np.int32)
})

# Find the encoder output (shape should be (1, 438, 1024))
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
print(f"   Hidden: {hidden_states.shape}")

# Decoder
print("\n[3/3] Decoder...")
decoder = ct.models.MLModel("build/barathwaj-models/cohere_decoder_fullseq_masked.mlpackage")

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

# Find logits output (should be (1, 108, vocab_size))
logits = None
for key, value in decoder_output.items():
    if hasattr(value, 'shape'):
        print(f"   {key}: {value.shape}")
        if len(value.shape) == 3 and value.shape[1] == 108:
            logits = value
            print(f"   ✓ Using {key} as logits")

if logits is None:
    raise ValueError(f"Could not find logits output. Available: {list(decoder_output.keys())}")

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
print("GROUND TRUTH TRANSCRIPTION")
print("="*70)
print(f'"{transcription}"')
print("="*70)
print(f"\nFile: test-librispeech-real.wav")
print(f"Duration: {duration:.2f}s")
print(f"Tokens: {len(generated_tokens)}")
print()
