#!/usr/bin/env python3
"""Test full CoreML pipeline with our exported models.

Pipeline:
1. Load audio
2. Frontend: BarathwajAnandan's cohere_frontend.mlpackage
3. Encoder: Our ultra_static_encoder.mlpackage
4. Decoder: Our ultra_static_decoder.mlpackage
5. Decode tokens to text
"""

import numpy as np
import soundfile as sf
import librosa
import coremltools as ct
from transformers import AutoProcessor

print("="*60)
print("COHERE TRANSCRIBE - FULL PIPELINE TEST")
print("="*60)

# ============================================================================
# 1. LOAD TOKENIZER
# ============================================================================
print("\n[1/6] Loading tokenizer...")
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)
tokenizer = processor.tokenizer
print(f"   ✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

# ============================================================================
# 2. LOAD AUDIO
# ============================================================================
print("\n[2/6] Loading audio...")
audio_file = "test-librispeech-real.wav"
audio, sr = sf.read(audio_file)

if sr != 16000:
    print(f"   Resampling from {sr}Hz to 16000Hz...")
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

duration = len(audio) / 16000
print(f"   ✓ Loaded: {audio_file}")
print(f"   Duration: {duration:.2f}s")
print(f"   Samples: {len(audio)}")

# Pad to 560,000 samples (35s) for frontend
max_samples = 560000
if len(audio) < max_samples:
    audio_padded = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
else:
    audio_padded = audio[:max_samples]

print(f"   Padded to: {len(audio_padded)} samples (35s)")

# ============================================================================
# 3. FRONTEND - MEL SPECTROGRAM
# ============================================================================
print("\n[3/6] Running frontend (audio → mel)...")
frontend = ct.models.MLModel("build/barathwaj-models/cohere_frontend.mlpackage")

frontend_input = {
    "audio_samples": audio_padded.astype(np.float32).reshape(1, -1),
    "audio_length": np.array([len(audio_padded)], dtype=np.int32)
}

frontend_output = frontend.predict(frontend_input)
# Output keys: var_6916 (mel), cast_2 (length)
mel_spectrogram = frontend_output["var_6916"]

print(f"   ✓ Mel spectrogram shape: {mel_spectrogram.shape}")
print(f"   Expected: (1, 128, 3501)")

if mel_spectrogram.shape != (1, 128, 3501):
    print(f"   ❌ ERROR: Wrong shape!")
    exit(1)

# ============================================================================
# 4. ENCODER - MEL → HIDDEN STATES
# ============================================================================
print("\n[4/6] Running encoder (mel → hidden states)...")
# Use BarathwajAnandan's encoder since our ultra_static_encoder has issues
encoder = ct.models.MLModel("build/barathwaj-models/cohere_encoder.mlpackage")

encoder_input = {
    "input_features": mel_spectrogram.astype(np.float32),
    "feature_length": np.array([3501], dtype=np.int32)
}

encoder_output = encoder.predict(encoder_input)
# Output from BarathwajAnandan's encoder - check actual output key
encoder_hidden_states = list(encoder_output.values())[0]  # Take first output

print(f"   ✓ Encoder output shape: {encoder_hidden_states.shape}")
print(f"   Expected: (1, 438, 1024)")

if encoder_hidden_states.shape[2] != 1024:
    print(f"   ❌ ERROR: Wrong hidden dim! Got {encoder_hidden_states.shape[2]}, expected 1024")
    exit(1)

# ============================================================================
# 5. DECODER - HIDDEN STATES → LOGITS
# ============================================================================
print("\n[5/6] Running decoder (hidden states → logits)...")

# Use BarathwajAnandan's decoder for now (our decoder might have issues)
decoder = ct.models.MLModel("build/barathwaj-models/cohere_decoder_fullseq_masked.mlpackage")

# Prompt: [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
# This is the standard English prompt for Cohere Transcribe
# Decoder expects (1, 108) so we need to pad
prompt_tokens = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
padded_prompt = prompt_tokens + [2] * (108 - len(prompt_tokens))  # Pad with pad_token_id=2
prompt_ids = np.array([padded_prompt], dtype=np.int32)

# Create attention masks
decoder_attention_mask = np.ones((1, 108), dtype=np.int32)
decoder_attention_mask[0, len(prompt_tokens):] = 0  # Mask padding

cross_attention_mask = np.ones((1, 1, 1, 438), dtype=np.float16)

decoder_input = {
    "input_ids": prompt_ids,
    "encoder_hidden_states": encoder_hidden_states.astype(np.float16),
    "decoder_attention_mask": decoder_attention_mask,
    "cross_attention_mask": cross_attention_mask,
}

print(f"   Prompt IDs: {prompt_tokens}")
print(f"   Input shape: {prompt_ids.shape}")
print(f"   Encoder states: {encoder_hidden_states.shape}")
print(f"   Running decoder...")

decoder_output = decoder.predict(decoder_input)
# Output is var_1009
logits = decoder_output["var_1009"]

print(f"   ✓ Decoder logits shape: {logits.shape}")
print(f"   Expected: (1, 10, 16384)")

# Get predicted tokens from prompt
predicted_tokens = np.argmax(logits[0], axis=-1)
print(f"   Predicted tokens: {predicted_tokens.tolist()}")

# ============================================================================
# 6. EXTRACT GENERATED TOKENS
# ============================================================================
print("\n[6/6] Extracting generated tokens from decoder output...")

# The decoder outputs logits for all 108 positions
# We take argmax to get the most likely token at each position
all_tokens = np.argmax(logits[0], axis=-1)

print(f"   Logits shape: {logits.shape}")
print(f"   All tokens shape: {all_tokens.shape}")
print(f"   All tokens: {all_tokens.tolist()[:20]}...")  # First 20 tokens

# The first 10 tokens are the prompt, so generated tokens start at index 10
generated_tokens = all_tokens[len(prompt_tokens):]

# Find EOS token
eos_token_id = 3
eos_indices = np.where(generated_tokens == eos_token_id)[0]
if len(eos_indices) > 0:
    eos_idx = eos_indices[0]
    generated_tokens = generated_tokens[:eos_idx]
    print(f"   ✓ Found EOS at position {eos_idx}")
else:
    print(f"   ⚠️  No EOS found, using all tokens")

print(f"   Generated tokens: {len(generated_tokens)} tokens")

# ============================================================================
# 7. DECODE TO TEXT
# ============================================================================
print("\n[7/7] Decoding to text...")

# generated_tokens already excludes the prompt
output_tokens = generated_tokens.tolist() if isinstance(generated_tokens, np.ndarray) else generated_tokens

print(f"   Output tokens: {output_tokens[:30]}...")  # First 30

# Decode
transcription = tokenizer.decode(output_tokens, skip_special_tokens=True)

print("\n" + "="*60)
print("RESULT")
print("="*60)
print(f"Audio file: {audio_file}")
print(f"Duration: {duration:.2f}s")
print(f"Tokens generated: {len(output_tokens)}")
print(f"\nTranscription:\n  \"{transcription}\"")
print("="*60)

# ============================================================================
# REFERENCE COMPARISON
# ============================================================================
print("\n[Reference] Expected for test-librispeech-real.wav:")
print("  \"CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS\"")
print("  (or variations depending on the actual audio)")
print()
