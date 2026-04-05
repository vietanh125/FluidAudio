#!/usr/bin/env python3
"""Test BarathwajAnandan's working CoreML models."""
import coremltools as ct
import numpy as np
import soundfile as sf
import librosa
import json

print("=== Testing BarathwajAnandan's CoreML Models ===\n")

# Load manifest
with open("build/barathwaj-models/coreml_manifest.json") as f:
    manifest = json.load(f)

print(f"Manifest config:")
print(f"  Model: {manifest['model_id']}")
print(f"  Max audio: {manifest['max_audio_seconds']}s")
print(f"  Decoder max length: {manifest['decoder_max_len']}")
print(f"  Prompt IDs: {manifest['prompt_ids']}")
print(f"  EOS token: {manifest['eos_token_id']}")

# Build id_to_token mapping
id_to_token = {i: token for i, token in enumerate(manifest['id_to_token'])}

print(f"\nPrompt sequence:")
for token_id in manifest['prompt_ids']:
    print(f"  {token_id}: {id_to_token[token_id]}")

# Load models
print("\nLoading CoreML models...")
frontend = ct.models.MLModel("build/barathwaj-models/cohere_frontend.mlpackage")
encoder = ct.models.MLModel("build/barathwaj-models/cohere_encoder.mlpackage")
decoder_fullseq = ct.models.MLModel("build/barathwaj-models/cohere_decoder_fullseq_masked.mlpackage")
decoder_cached = ct.models.MLModel("build/barathwaj-models/cohere_decoder_cached.mlpackage")
print("  ✓ All models loaded")

# Load audio
audio_file = "test-librispeech-real.wav"
print(f"\nLoading audio: {audio_file}")
audio, sr = sf.read(audio_file)
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

print(f"  Audio: {len(audio)} samples ({len(audio)/16000:.2f}s)")

# Pad/truncate to max length
max_samples = manifest['max_audio_samples']
if len(audio) < max_samples:
    audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
else:
    audio = audio[:max_samples]

print(f"  Padded to: {len(audio)} samples")

# 1. Frontend (preprocessor)
print("\n1. Running frontend...")
frontend_out = frontend.predict({
    "audio_pcm": audio.astype(np.float32)
})
mel_features = frontend_out["log_mel"]
print(f"   ✓ Mel features: {mel_features.shape}")

# 2. Encoder
print("\n2. Running encoder...")
encoder_out = encoder.predict({
    "log_mel": mel_features
})
encoder_hidden = encoder_out["encoder_output_embeds"]
print(f"   ✓ Encoder output: {encoder_hidden.shape}")

# 3. Decoding
print("\n3. Decoding...")

# Initialize with prompt
tokens = manifest['prompt_ids'].copy()
eos_token_id = manifest['eos_token_id']
max_new_tokens = manifest['default_max_new_tokens']

print(f"   Starting with {len(tokens)} prompt tokens")

# First token using full-sequence decoder
print(f"   a) First token (full-sequence decoder)...")
decoder_input_ids = np.array([tokens], dtype=np.int32)

fullseq_out = decoder_fullseq.predict({
    "input_ids": decoder_input_ids,
    "encoder_output_embeds": encoder_hidden
})

# The output should contain logits for next token
logits_key = list(fullseq_out.keys())[0]
logits = fullseq_out[logits_key]
print(f"      Logits shape: {logits.shape}")

# Get next token (greedy)
next_token = int(np.argmax(logits[0, -1, :]))
print(f"      Next token: {next_token} = '{id_to_token.get(next_token, '<UNK>')}'")

if next_token != eos_token_id:
    tokens.append(next_token)

# Subsequent tokens using cached decoder
print(f"   b) Autoregressive generation (cached decoder)...")

for step in range(max_new_tokens - 1):
    if tokens[-1] == eos_token_id:
        print(f"      ✓ Stopped at EOS (step {step})")
        break

    if len(tokens) >= manifest['decoder_max_len']:
        print(f"      ⚠ Stopped at max length ({manifest['decoder_max_len']})")
        break

    # Run cached decoder with only the last token
    last_token = np.array([[tokens[-1]]], dtype=np.int32)

    try:
        cached_out = decoder_cached.predict({
            "input_ids": last_token,
            "encoder_output_embeds": encoder_hidden
        })

        logits_key = list(cached_out.keys())[0]
        logits = cached_out[logits_key]

        next_token = int(np.argmax(logits[0, 0, :]))

        if step < 5:
            print(f"      Step {step}: {next_token} = '{id_to_token.get(next_token, '<UNK>')}'")

        if next_token == eos_token_id:
            print(f"      ✓ Stopped at EOS (step {step + 1})")
            break

        tokens.append(next_token)
    except Exception as e:
        print(f"      ❌ Error at step {step}: {e}")
        break

# 4. Decode tokens
print(f"\n4. Decoding tokens...")
print(f"   Total tokens: {len(tokens)}")

# Skip prompt tokens
generated_tokens = tokens[len(manifest['prompt_ids']):]
print(f"   Generated: {len(generated_tokens)} tokens")

# Convert to text
text = ""
for token_id in generated_tokens:
    if token_id == eos_token_id:
        break
    if token_id in id_to_token:
        text += id_to_token[token_id]

print(f"\n{'='*60}")
print("TRANSCRIPTION RESULT:")
print('='*60)
print(f'"{text}"')
print('='*60)
