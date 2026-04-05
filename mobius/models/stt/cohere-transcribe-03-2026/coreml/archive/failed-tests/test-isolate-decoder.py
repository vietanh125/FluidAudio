#!/usr/bin/env python3
"""Isolation test: BarathwajAnandan Frontend + BarathwajAnandan Encoder + Our Decoder."""
import coremltools as ct
import numpy as np
import soundfile as sf
import librosa
import json

print("=== Isolation Test: Barathwaj Encoder + Our Decoder ===\n")

# Load models
print("1. Loading models...")
frontend = ct.models.MLModel("build/barathwaj-models/cohere_frontend.mlpackage")
encoder = ct.models.MLModel("build/barathwaj-models/cohere_encoder.mlpackage")  # BARATHWAJ
decoder = ct.models.MLModel("build/ultra_static_decoder.mlpackage")  # OUR EXPORT
print("   Frontend: BarathwajAnandan")
print("   Encoder: BarathwajAnandan")
print("   Decoder: OUR EXPORT")

# Load tokenizer
print("\n2. Loading tokenizer...")
with open("build/barathwaj-models/coreml_manifest.json") as f:
    manifest = json.load(f)
id_to_token = {i: token for i, token in enumerate(manifest['id_to_token'])}
prompt_ids = manifest['prompt_ids']

# Load audio
print("\n3. Loading audio...")
audio, sr = sf.read("test-librispeech-real.wav")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

original_len = len(audio)
if len(audio) < 560000:
    audio = np.pad(audio, (0, 560000 - len(audio)))
else:
    audio = audio[:560000]

# Frontend
print("\n4. Running frontend (BarathwajAnandan)...")
frontend_out = frontend.predict({
    "audio_samples": audio.reshape(1, -1).astype(np.float32),
    "audio_length": np.array([original_len], dtype=np.int32)
})
mel = frontend_out["var_6916"]
print(f"   Mel shape: {mel.shape}")

# Encoder (BarathwajAnandan)
print("\n5. Running encoder (BarathwajAnandan)...")
encoder_out = encoder.predict({
    "input_features": mel,
    "feature_length": np.array([3501], dtype=np.int32)
})
hidden_states = encoder_out["var_6733"]
print(f"   Hidden states shape: {hidden_states.shape}")
print(f"   Hidden states stats: min={hidden_states.min():.3f}, max={hidden_states.max():.3f}, mean={hidden_states.mean():.3f}")

# Decoder (OUR EXPORT)
print("\n6. Running decoder (OUR EXPORT)...")
input_ids = np.array([prompt_ids + [0] * (108 - len(prompt_ids))], dtype=np.int32)

decoder_out = decoder.predict({
    "input_ids": input_ids,
    "encoder_hidden_states": hidden_states
})
logits = decoder_out["logits"]
print(f"   Logits shape: {logits.shape}")

# Decode
print("\n7. Decoding tokens...")
predicted_ids = np.argmax(logits[0], axis=-1)

text_chars = []
for i, token_id in enumerate(predicted_ids[:len(prompt_ids)]):
    token = id_to_token.get(int(token_id), f"<{token_id}>")
    text_chars.append(token)

text = "".join(text_chars)
print(f"\n8. Transcription result:")
print(f"   {text}")

# Check quality
if text.count(' ') > len(text) * 0.1:
    print("\n   ✅ LOOKS GOOD - Real text with spaces")
else:
    print("\n   ❌ GARBAGE - Few/no spaces, random tokens")

print("\n" + "="*60)
print("Test complete: Barathwaj Encoder + Our Decoder")
print("="*60)
