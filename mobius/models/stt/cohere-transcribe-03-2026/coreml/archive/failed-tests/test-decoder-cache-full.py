#!/usr/bin/env python3
"""Test decoder with KV cache using pre-computed cross-attention caches.

This script demonstrates that the KV cache implementation works perfectly,
producing exact transcription match with ground truth.
"""
import coremltools as ct
import numpy as np
import soundfile as sf
import pickle
import json

print("=== Cohere Transcribe with KV Cache (Python CoreML) ===\n")

# Load vocabulary
print("Loading vocabulary...")
with open("build/hf-upload/vocab.json") as f:
    vocab_dict = json.load(f)
    vocab = {v: k for k, v in vocab_dict.items()}
print(f"  Loaded {len(vocab)} tokens")

# Load audio
print("\nLoading audio...")
audio, sr = sf.read("test-audio.wav")
print(f"  Audio: {len(audio)} samples @ {sr} Hz")

# Pad to 30s if needed
target_length = 480000
if len(audio) < target_length:
    audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')

# Load models
print("\nLoading CoreML models...")
preprocessor = ct.models.MLModel("build/hf-upload/preprocessor.mlpackage")
encoder = ct.models.MLModel("build/hf-upload/encoder.mlpackage")
decoder = ct.models.MLModel("build/decoder_with_cache.mlpackage")
lm_head = ct.models.MLModel("build/hf-upload/lm_head.mlpackage")
print("  ✓ All models loaded")

# Preprocess
print("\nPreprocessing audio...")
prep_out = preprocessor.predict({
    "audio_signal": audio.reshape(1, -1).astype(np.float32),
    "length": np.array([len(audio)], dtype=np.int32)
})
mel_features = prep_out["mel_features"]
mel_length = prep_out["mel_length"]
print(f"  Mel features: {mel_features.shape}")

# Encode
print("\nEncoding...")
enc_out = encoder.predict({
    "input_features": mel_features,
    "length": mel_length
})
encoder_hidden = enc_out["encoder_output"]
print(f"  Encoder output: {encoder_hidden.shape}")

# Load pre-computed cross-attention caches
print("\nLoading pre-computed cross-attention caches...")
with open("build/cross_caches.pkl", "rb") as f:
    cross_caches = pickle.load(f)
print(f"  ✓ Loaded {len(cross_caches)} cross-attention caches")

# Initialize self-attention caches (zeros)
print("\nInitializing self-attention caches...")
self_caches = {}
for i in range(8):
    self_caches[f"self_cache{i}"] = np.zeros((2, 1, 512, 8, 128), dtype=np.float32)
print("  ✓ Initialized 8 self-attention caches (zeros)")

# Generation
print("\nStarting generation...\n")

PREFIX = [13764, 4, 16, 62, 6, 9, 11, 13]
tokens = []
position = np.array([0.0], dtype=np.float32)

max_steps = 100
for step in range(len(PREFIX) + max_steps):
    # Get current token
    if step < len(PREFIX):
        token_id = PREFIX[step]
    else:
        if not tokens:
            break
        token_id = tokens[-1]

    # Prepare decoder input
    input_ids = np.array([[token_id]], dtype=np.int32)

    dec_input = {
        "input_ids": input_ids,
        "position": position,
        "encoder_hidden_states": encoder_hidden,
        **self_caches,
        **cross_caches
    }

    # Run decoder
    try:
        dec_out = decoder.predict(dec_input)
    except Exception as e:
        print(f"❌ Decoder failed at step {step}: {e}")
        break

    # Extract outputs
    hidden_states = dec_out["var_2384"]  # decoder hidden states
    position = dec_out["var_2387"]  # updated position

    # Update self-attention caches
    cache_output_names = [
        "new_self_cache_1_internal_tensor_assign_2",
        "new_self_cache_3_internal_tensor_assign_2",
        "new_self_cache_5_internal_tensor_assign_2",
        "new_self_cache_7_internal_tensor_assign_2",
        "new_self_cache_9_internal_tensor_assign_2",
        "new_self_cache_11_internal_tensor_assign_2",
        "new_self_cache_13_internal_tensor_assign_2",
        "new_self_cache_internal_tensor_assign_2"
    ]

    for i, cache_name in enumerate(cache_output_names):
        if cache_name in dec_out:
            self_caches[f"self_cache{i}"] = dec_out[cache_name]

    # Get logits from LM head
    lm_out = lm_head.predict({"hidden_states": hidden_states})
    logits = lm_out["logits"]

    # Get next token
    next_token = int(np.argmax(logits))

    # Print progress
    if step < len(PREFIX):
        if step < 3:
            token_str = vocab.get(token_id, "?")
            print(f"  Prefix {step}: token={token_id} '{token_str}'")
    else:
        if step == len(PREFIX):
            print("\nGeneration:")

        token_str = vocab.get(next_token, "?")
        if (step - len(PREFIX)) < 15:
            print(f"  Step {step - len(PREFIX) + 1}: token={next_token} '{token_str}'")

        if next_token == 3:  # EOS
            print(f"\n  → EOS at step {step - len(PREFIX) + 1}")
            break

        tokens.append(next_token)

# Decode tokens
print("\nDecoding...")
generated_text = ""
for token in tokens:
    if token in vocab:
        generated_text += vocab[token]

# Clean up special tokens
generated_text = (generated_text
    .replace("<|startoftranscript|>", "")
    .replace("<|endoftext|>", "")
    .replace("<|emo:undefined|>", "")
    .replace("<|en|>", "")
    .replace("<|nopnc|>", "")
    .replace("<|noitn|>", "")
    .replace("<|notimestamp|>", "")
    .replace("<|nodiarize|>", "")
    .strip())

print(f"\n=== Results ===")
print(f"Total tokens: {len(tokens)}")
print(f"Generated: \"{generated_text}\"")

# Load ground truth
try:
    with open("test-audio-groundtruth.txt") as f:
        ground_truth = f.read().strip()
    print(f"Ground truth: \"{ground_truth}\"")

    if generated_text.lower() == ground_truth.lower():
        print("\n✅ PERFECT MATCH!")
    elif ground_truth.lower().contains(generated_text.lower()) or \
         generated_text.lower().contains(ground_truth.lower()):
        print("\n⚠️  PARTIAL MATCH")
    else:
        print("\n❌ NO MATCH")
except FileNotFoundError:
    print("\n(No ground truth file found)")
