#!/usr/bin/env python3
"""Quick test of BarathwajAnandan's pipeline - just first token."""
import coremltools as ct
import numpy as np
import soundfile as sf
import librosa
import json
import time

print("=== Testing BarathwajAnandan's Pipeline ===\n")

# Load manifest
with open("build/barathwaj-models/coreml_manifest.json") as f:
    manifest = json.load(f)

id_to_token = {i: token for i, token in enumerate(manifest['id_to_token'])}
prompt_ids = manifest['prompt_ids']
print(f"Prompt: {[id_to_token[i] for i in prompt_ids]}\n")

# Load audio
audio, sr = sf.read("test-librispeech-real.wav")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# Pad to 35s
audio_padded = np.pad(audio, (0, 560000 - len(audio)), mode='constant')
print(f"Audio: {len(audio)} samples ({len(audio)/16000:.2f}s)\n")

# Load and run frontend
print("1. Loading frontend...")
t0 = time.time()
frontend = ct.models.MLModel("build/barathwaj-models/cohere_frontend.mlpackage",
                             compute_units=ct.ComputeUnit.CPU_ONLY)
print(f"   Loaded in {time.time()-t0:.1f}s")

print("   Running frontend...")
t0 = time.time()
frontend_out = frontend.predict({
    "audio_samples": audio_padded.reshape(1, -1).astype(np.float32),
    "audio_length": np.array([len(audio)], dtype=np.int32)
})
print(f"   Frontend output keys: {list(frontend_out.keys())}")
for k, v in frontend_out.items():
    print(f"     {k}: {v.shape}")

# Find the mel features (largest output)
mel_features = max(frontend_out.values(), key=lambda x: x.size)
feature_length = frontend_out["cast_2"][0]
print(f"   Mel features shape: {mel_features.shape}, length: {feature_length}")

# Load and run encoder
print("\n2. Loading encoder...")
t0 = time.time()
encoder = ct.models.MLModel("build/barathwaj-models/cohere_encoder.mlpackage",
                           compute_units=ct.ComputeUnit.CPU_ONLY)
print(f"   Loaded in {time.time()-t0:.1f}s")

print("   Running encoder...")
t0 = time.time()
# Encoder expects: input_features, feature_length
enc_out = encoder.predict({
    "input_features": mel_features,
    "feature_length": np.array([int(feature_length)], dtype=np.int32)
})
encoder_hidden = list(enc_out.values())[0]
print(f"   Done in {time.time()-t0:.1f}s - shape: {encoder_hidden.shape}")

# Load and run decoder (just first token)
print("\n3. Loading decoder (full-seq)...")
t0 = time.time()
decoder_fullseq = ct.models.MLModel("build/barathwaj-models/cohere_decoder_fullseq_masked.mlpackage",
                                   compute_units=ct.ComputeUnit.CPU_ONLY)
print(f"   Loaded in {time.time()-t0:.1f}s")

print("   Running decoder with prompt...")
t0 = time.time()

# Pad prompt to 108 tokens
padded_prompt = prompt_ids + [manifest['pad_token_id']] * (108 - len(prompt_ids))
decoder_input_ids = np.array([padded_prompt], dtype=np.int32)

# Create attention masks (only attend to actual prompt tokens)
decoder_attention_mask = np.array([[1] * len(prompt_ids) + [0] * (108 - len(prompt_ids))], dtype=np.int32)
# Cross-attention mask is rank 4: (1, 1, 1, encoder_seq) - broadcast mask
cross_attention_mask = np.ones((1, 1, 1, encoder_hidden.shape[1]), dtype=np.int32)

dec_out = decoder_fullseq.predict({
    "input_ids": decoder_input_ids,
    "encoder_hidden_states": encoder_hidden,
    "decoder_attention_mask": decoder_attention_mask,
    "cross_attention_mask": cross_attention_mask
})
logits = list(dec_out.values())[0]
print(f"   Done in {time.time()-t0:.1f}s - logits shape: {logits.shape}")

# Get first token
next_token = int(np.argmax(logits[0, -1, :]))
token_str = id_to_token.get(next_token, f"<UNK:{next_token}>")

print(f"\n{'='*60}")
print(f"FIRST GENERATED TOKEN: {next_token} = '{token_str}'")
print('='*60)

if next_token == manifest['eos_token_id']:
    print("❌ Model generated EOS immediately - something is wrong!")
elif next_token < 256:
    print("❌ Model generated special token - may be broken")
else:
    print("✅ Model generated actual text token - pipeline works!")

print(f"\nTop 5 tokens:")
top5 = np.argsort(logits[0, -1, :])[-5:][::-1]
for tok in top5:
    print(f"  {tok}: {id_to_token.get(tok, '<UNK>')} (logit: {logits[0, -1, tok]:.2f})")
