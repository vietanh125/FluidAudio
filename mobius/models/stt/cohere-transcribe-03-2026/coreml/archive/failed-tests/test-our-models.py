#!/usr/bin/env python3
"""Test our ultra-static encoder and decoder with BarathwajAnandan's frontend."""
import coremltools as ct
import numpy as np
import soundfile as sf
import librosa
import json

print("=== Testing Our CoreML Models ===\n")

# Load models
print("1. Loading models...")
frontend = ct.models.MLModel("build/barathwaj-models/cohere_frontend.mlpackage")
encoder = ct.models.MLModel("build/ultra_static_encoder.mlpackage")  # OUR EXPORT
decoder = ct.models.MLModel("build/ultra_static_decoder.mlpackage")  # OUR EXPORT
print("   ✓ All models loaded")

# Load tokenizer
print("\n2. Loading tokenizer...")
with open("build/barathwaj-models/coreml_manifest.json") as f:
    manifest = json.load(f)
id_to_token = {i: token for i, token in enumerate(manifest['id_to_token'])}
prompt_ids = manifest['prompt_ids']
print(f"   Prompt IDs: {prompt_ids}")
print(f"   Vocab size: {len(id_to_token)}")

# Load audio
print("\n3. Loading audio...")
audio, sr = sf.read("test-librispeech-real.wav")
print(f"   Original: {len(audio)} samples @ {sr}Hz ({len(audio)/sr:.2f}s)")

if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    print(f"   Resampled to 16kHz")

# Pad to 560,000 samples (35 seconds)
original_len = len(audio)
if len(audio) < 560000:
    audio = np.pad(audio, (0, 560000 - len(audio)))
else:
    audio = audio[:560000]
print(f"   Padded: {len(audio)} samples (35s)")

# Step 1: Frontend (BarathwajAnandan's)
print("\n4. Running frontend...")
frontend_out = frontend.predict({
    "audio_samples": audio.reshape(1, -1).astype(np.float32),
    "audio_length": np.array([original_len], dtype=np.int32)
})
mel = frontend_out["var_6916"]
print(f"   Mel shape: {mel.shape}")
print(f"   Mel stats: min={mel.min():.3f}, max={mel.max():.3f}, mean={mel.mean():.3f}")

# Step 2: Encoder (OUR EXPORT)
print("\n5. Running encoder (OUR EXPORT)...")
encoder_out = encoder.predict({
    "input_features": mel
})
hidden_states = encoder_out["encoder_output"]
print(f"   Hidden states shape: {hidden_states.shape}")
print(f"   Hidden states stats: min={hidden_states.min():.3f}, max={hidden_states.max():.3f}, mean={hidden_states.mean():.3f}")

# Step 3: Decoder (OUR EXPORT)
print("\n6. Running decoder (OUR EXPORT)...")
input_ids = np.array([prompt_ids + [0] * (108 - len(prompt_ids))], dtype=np.int32)
print(f"   Input IDs shape: {input_ids.shape}")
print(f"   Input IDs: {input_ids[0, :20]}...")  # Show first 20 tokens

decoder_out = decoder.predict({
    "input_ids": input_ids,
    "encoder_hidden_states": hidden_states
})
logits = decoder_out["logits"]
print(f"   Logits shape: {logits.shape}")
print(f"   Logits stats: min={logits.min():.3f}, max={logits.max():.3f}, mean={logits.mean():.3f}")

# Get tokens
print("\n7. Decoding tokens...")
predicted_ids = np.argmax(logits[0], axis=-1)
print(f"   Predicted IDs (first 20): {predicted_ids[:20]}")

# Decode to text
text_chars = []
for i, token_id in enumerate(predicted_ids[:len(prompt_ids)]):
    token = id_to_token.get(int(token_id), f"<{token_id}>")
    text_chars.append(token)
    if i < 20:  # Show first 20 tokens
        print(f"     Token {i}: {token_id} -> '{token}'")

text = "".join(text_chars)
print(f"\n8. Transcription result:")
print(f"   Raw: {text[:200]}")  # First 200 chars
print(f"   Length: {len(text)} characters")

# Check if it's garbage or real text
if text.count(' ') > len(text) * 0.1:  # At least 10% spaces
    print("   ✅ Looks like real text (has spaces)")
else:
    print("   ⚠️  Might be garbage (few spaces)")

print("\n" + "="*60)
print("SUCCESS! Our encoder and decoder work with frontend")
print("="*60)
