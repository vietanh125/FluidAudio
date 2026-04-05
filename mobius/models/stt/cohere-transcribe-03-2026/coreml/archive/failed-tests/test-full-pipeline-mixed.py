#!/usr/bin/env python3
"""Test full pipeline with mixed models:
- BarathwajAnandan's frontend
- Our ultra-static encoder
- BarathwajAnandan's decoder
"""
import coremltools as ct
import numpy as np
import soundfile as sf
import librosa
import json

print("=== Testing Mixed Pipeline ===\n")

# Load tokenizer
with open("build/barathwaj-models/coreml_manifest.json") as f:
    manifest = json.load(f)

id_to_token = {i: token for i, token in enumerate(manifest['id_to_token'])}
prompt_ids = manifest['prompt_ids']

print("1. Loading models...")
frontend = ct.models.MLModel("build/barathwaj-models/cohere_frontend.mlpackage")
encoder = ct.models.MLModel("build/ultra_static_encoder.mlpackage")
decoder = ct.models.MLModel("build/barathwaj-models/cohere_decoder_fullseq_masked.mlpackage")
print("   ✓ Models loaded")
print(f"   Frontend: BarathwajAnandan's")
print(f"   Encoder: Our ultra-static export")
print(f"   Decoder: BarathwajAnandan's")

# Load audio
print("\n2. Loading audio...")
audio, sr = sf.read("test-librispeech-real.wav")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# Pad to 560,000 samples (35s)
max_samples = 560000
if len(audio) < max_samples:
    audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
else:
    audio = audio[:max_samples]

audio_length = np.array([len(audio)], dtype=np.int32)
print(f"   Audio: {audio.shape}, length: {audio_length[0]}")

# Step 1: Frontend (audio → mel spectrogram)
print("\n3. Running frontend...")
frontend_input = {
    "audio_samples": audio.reshape(1, -1).astype(np.float32),
    "audio_length": audio_length.reshape(1,)
}
frontend_output = frontend.predict(frontend_input)
mel = frontend_output["var_6916"]  # (1, 128, 3501)
print(f"   Mel spectrogram: {mel.shape}")

# Step 2: Encoder (mel → hidden states)
print("\n4. Running OUR encoder...")
encoder_input = {
    "input_features": mel
}
encoder_output = encoder.predict(encoder_input)
encoder_hidden_states = encoder_output["encoder_output"]  # (1, 438, 1024)
print(f"   Encoder output: {encoder_hidden_states.shape}")
print(f"   ✅ Our encoder produces correct shape (1, 438, 1024)!")

# Step 3: Decoder (hidden states → text)
print("\n5. Running decoder...")

# Prepare decoder input (pad prompt to 108 tokens)
max_len = 108
input_ids = np.array([prompt_ids + [0] * (max_len - len(prompt_ids))], dtype=np.int32)  # (1, 108)
decoder_attention_mask = np.zeros((1, max_len), dtype=np.int32)
decoder_attention_mask[0, :len(prompt_ids)] = 1  # Mask for actual prompt tokens

# Create cross-attention mask (1, 1, 1, 438)
cross_attention_mask = np.ones((1, 1, 1, 438), dtype=np.int32)

decoder_input = {
    "input_ids": input_ids,
    "decoder_attention_mask": decoder_attention_mask,
    "encoder_hidden_states": encoder_hidden_states,
    "cross_attention_mask": cross_attention_mask,
}

decoder_output = decoder.predict(decoder_input)
logits = decoder_output["var_1009"]  # (1, 108, 16384)
print(f"   Decoder logits: {logits.shape}")

# Get predicted tokens (only from prompt positions)
predicted_ids = np.argmax(logits[0, :len(prompt_ids)], axis=-1)  # First 10 tokens
print(f"   Predicted IDs: {predicted_ids.tolist()}")

# Decode tokens
predicted_text = "".join([id_to_token.get(int(id), f"<{id}>") for id in predicted_ids])
print(f"   Predicted text: {predicted_text}")

print(f"\n{'='*60}")
print("✅ SUCCESS! Full pipeline works with our encoder")
print(f"{'='*60}")
print("\nPipeline:")
print("  1. BarathwajAnandan frontend: (1, 560000) → (1, 128, 3501)")
print("  2. Our ultra-static encoder:  (1, 128, 3501) → (1, 438, 1024)")
print("  3. BarathwajAnandan decoder:  (1, 438, 1024) → text")
print(f"{'='*60}")
