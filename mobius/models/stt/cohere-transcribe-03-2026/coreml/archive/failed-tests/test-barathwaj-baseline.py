#!/usr/bin/env python3
"""Test BarathwajAnandan's full pipeline as baseline."""
import coremltools as ct
import numpy as np
import soundfile as sf
import librosa
import json

print("=== Testing BarathwajAnandan's Full Pipeline (Baseline) ===\n")

# Load models
print("1. Loading models...")
frontend = ct.models.MLModel("build/barathwaj-models/cohere_frontend.mlpackage")
encoder = ct.models.MLModel("build/barathwaj-models/cohere_encoder.mlpackage")
decoder = ct.models.MLModel("build/barathwaj-models/cohere_decoder_fullseq_masked.mlpackage")
print("   ✓ All models loaded")

# Load tokenizer
print("\n2. Loading tokenizer...")
with open("build/barathwaj-models/coreml_manifest.json") as f:
    manifest = json.load(f)
id_to_token = {i: token for i, token in enumerate(manifest['id_to_token'])}
prompt_ids = manifest['prompt_ids']
print(f"   Prompt IDs: {prompt_ids}")

# Load audio
print("\n3. Loading audio...")
audio, sr = sf.read("test-librispeech-real.wav")
print(f"   Original: {len(audio)} samples @ {sr}Hz ({len(audio)/sr:.2f}s)")

if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

original_len = len(audio)
if len(audio) < 560000:
    audio = np.pad(audio, (0, 560000 - len(audio)))
else:
    audio = audio[:560000]

# Frontend
print("\n4. Running frontend...")
frontend_out = frontend.predict({
    "audio_samples": audio.reshape(1, -1).astype(np.float32),
    "audio_length": np.array([original_len], dtype=np.int32)
})
mel = frontend_out["var_6916"]
print(f"   Mel shape: {mel.shape}")

# Encoder
print("\n5. Running encoder (BarathwajAnandan's)...")
encoder_out = encoder.predict({
    "input_features": mel,
    "feature_length": np.array([3501], dtype=np.int32)
})
hidden_states = encoder_out["var_6733"]
encoder_len = encoder_out["cast_1"]
print(f"   Hidden states shape: {hidden_states.shape}")
print(f"   Encoder length: {encoder_len}")
print(f"   Hidden states stats: min={hidden_states.min():.3f}, max={hidden_states.max():.3f}, mean={hidden_states.mean():.3f}")

# Decoder
print("\n6. Running decoder (BarathwajAnandan's)...")
input_ids = np.array([prompt_ids + [0] * (108 - len(prompt_ids))], dtype=np.int32)

# Create attention masks as per BarathwajAnandan's configuration
decoder_mask = np.zeros((1, 108), dtype=np.int32)
decoder_mask[0, :len(prompt_ids)] = 1

cross_mask = np.ones((1, 1, 1, 438), dtype=np.int32)

decoder_out = decoder.predict({
    "input_ids": input_ids,
    "decoder_attention_mask": decoder_mask,
    "encoder_hidden_states": hidden_states,
    "cross_attention_mask": cross_mask,
})
logits = decoder_out["var_1009"]
print(f"   Logits shape: {logits.shape}")
print(f"   Logits stats: min={logits.min():.3f}, max={logits.max():.3f}, mean={logits.mean():.3f}")

# Decode
print("\n7. Decoding tokens...")
predicted_ids = np.argmax(logits[0], axis=-1)
print(f"   Predicted IDs (first 20): {predicted_ids[:20]}")

text_chars = []
for i, token_id in enumerate(predicted_ids[:len(prompt_ids)]):
    token = id_to_token.get(int(token_id), f"<{token_id}>")
    text_chars.append(token)
    if i < 20:
        print(f"     Token {i}: {token_id} -> '{token}'")

text = "".join(text_chars)
print(f"\n8. Transcription result:")
print(f"   {text}")

print("\n" + "="*60)
print("Baseline transcription complete")
print("="*60)
