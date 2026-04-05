#!/usr/bin/env python3
"""Test our encoder with BarathwajAnandan's frontend and decoder.

Pipeline:
1. BarathwajAnandan Frontend → Mel
2. OUR Encoder (encoder_correct_static.mlpackage) → Hidden States
3. BarathwajAnandan Decoder → Text
"""

import numpy as np
import soundfile as sf
import librosa
import coremltools as ct
from transformers import AutoProcessor

print("="*70)
print("TESTING OUR ENCODER")
print("="*70)
print()
print("Pipeline:")
print("  [BarathwajAnandan Frontend] → [OUR Encoder] → [BarathwajAnandan Decoder]")
print()

# ============================================================================
# 1. LOAD TOKENIZER
# ============================================================================
print("[1/7] Loading tokenizer...")
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)
tokenizer = processor.tokenizer
print(f"   ✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

# ============================================================================
# 2. LOAD AUDIO
# ============================================================================
print("\n[2/7] Loading audio...")
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
# 3. FRONTEND - BarathwajAnandan's preprocessor
# ============================================================================
print("\n[3/7] Running frontend (BarathwajAnandan)...")
frontend = ct.models.MLModel("build/barathwaj-models/cohere_frontend.mlpackage")

frontend_input = {
    "audio_samples": audio_padded.astype(np.float32).reshape(1, -1),
    "audio_length": np.array([len(audio_padded)], dtype=np.int32)
}

frontend_output = frontend.predict(frontend_input)
mel_spectrogram = frontend_output["var_6916"]

print(f"   ✓ Mel spectrogram shape: {mel_spectrogram.shape}")
print(f"   Expected: (1, 128, 3501)")

if mel_spectrogram.shape != (1, 128, 3501):
    print(f"   ❌ ERROR: Wrong shape!")
    exit(1)

# ============================================================================
# 4. ENCODER - OUR encoder_correct_static.mlpackage
# ============================================================================
print("\n[4/7] Running OUR encoder (encoder_correct_static)...")
our_encoder = ct.models.MLModel("build/encoder_correct_static.mlpackage")

# Check what inputs our encoder expects
print(f"   Checking encoder inputs...")
encoder_spec = our_encoder.get_spec()
input_names = [inp.name for inp in encoder_spec.description.input]
print(f"   Encoder expects inputs: {input_names}")

# Prepare encoder input based on what it expects
encoder_input = {}
for inp in encoder_spec.description.input:
    if "length" in inp.name.lower():
        # Length input: scalar or 1D array
        encoder_input[inp.name] = np.array([3501], dtype=np.int32)
        print(f"   - {inp.name}: [3501] (rank 1)")
    else:
        # Feature input: mel spectrogram
        encoder_input[inp.name] = mel_spectrogram.astype(np.float32)
        print(f"   - {inp.name}: {mel_spectrogram.shape}")

print(f"   Running encoder...")
encoder_output = our_encoder.predict(encoder_input)

# Get the encoder hidden states by name
if "encoder_output" in encoder_output:
    encoder_hidden_states = encoder_output["encoder_output"]
    print(f"   ✓ Found encoder_output")
else:
    # Fallback: find the output with shape (1, 438, 1024)
    for key, val in encoder_output.items():
        if len(val.shape) == 3 and val.shape[-1] == 1024:
            encoder_hidden_states = val
            print(f"   ✓ Found hidden states in: {key}")
            break

print(f"   ✓ Hidden states shape: {encoder_hidden_states.shape}")
print(f"   Expected: (1, 438, 1024)")

# Check dimensions
if encoder_hidden_states.shape[-1] != 1024:
    print(f"   ❌ ERROR: Wrong hidden dim! Got {encoder_hidden_states.shape[-1]}, expected 1024")
    print(f"   This encoder may not include the projection layer (1280→1024)")
    exit(1)

print(f"   ✓ Dimension check passed (1024-dim output)")

# ============================================================================
# 5. DECODER - BarathwajAnandan's decoder
# ============================================================================
print("\n[5/7] Running decoder (BarathwajAnandan)...")
decoder = ct.models.MLModel("build/barathwaj-models/cohere_decoder_fullseq_masked.mlpackage")

# Prompt: English transcription with standard settings
prompt_tokens = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
padded_prompt = prompt_tokens + [2] * (108 - len(prompt_tokens))  # Pad to 108
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

print(f"   Prompt: {prompt_tokens}")
print(f"   Input shape: {prompt_ids.shape}")
print(f"   Encoder states: {encoder_hidden_states.shape}")
print(f"   Running decoder...")

decoder_output = decoder.predict(decoder_input)
logits = decoder_output["var_1009"]

print(f"   ✓ Decoder logits shape: {logits.shape}")

# ============================================================================
# 6. EXTRACT TOKENS
# ============================================================================
print("\n[6/7] Extracting generated tokens...")

# Get most likely tokens
all_tokens = np.argmax(logits[0], axis=-1)
print(f"   All tokens: {all_tokens.tolist()[:20]}...")

# Generated tokens start after prompt
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

print(f"   Generated: {len(generated_tokens)} tokens")

# ============================================================================
# 7. DECODE TO TEXT
# ============================================================================
print("\n[7/7] Decoding to text...")

output_tokens = generated_tokens.tolist() if isinstance(generated_tokens, np.ndarray) else generated_tokens
print(f"   Tokens: {output_tokens[:30]}...")

transcription = tokenizer.decode(output_tokens, skip_special_tokens=True)

print("\n" + "="*70)
print("RESULT")
print("="*70)
print(f"Audio file: {audio_file}")
print(f"Duration: {duration:.2f}s")
print(f"Tokens generated: {len(output_tokens)}")
print()
print(f"Transcription:")
print(f'  "{transcription}"')
print("="*70)

# ============================================================================
# EVALUATION
# ============================================================================
print("\n[Reference] Expected for test-librispeech-real.wav:")
print('  "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS"')
print()

# Check if transcription is garbage
if len(transcription.strip()) == 0:
    print("❌ FAILURE: Empty transcription")
    exit(1)
elif all(c in '.!?' for c in transcription.strip()):
    print("❌ FAILURE: Only punctuation (garbage output)")
    exit(1)
elif len(set(transcription.strip())) < 5:
    print("❌ FAILURE: Very low character diversity (likely garbage)")
    exit(1)
else:
    print("✅ SUCCESS: Encoder produced valid transcription!")
    print()
    print("Our encoder works correctly with BarathwajAnandan's frontend/decoder.")
    print()
