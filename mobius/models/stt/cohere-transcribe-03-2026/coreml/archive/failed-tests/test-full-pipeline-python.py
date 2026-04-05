#!/usr/bin/env python3
"""Test the full pipeline with KV cache in Python using CoreML models."""
import coremltools as ct
import numpy as np
import soundfile as sf
import json

print("=== Full Pipeline Test (Python CoreML) ===\n")

# Load vocab
with open("build/hf-upload/vocab.json") as f:
    vocab_dict = json.load(f)
    vocab = {v: k for k, v in vocab_dict.items()}

# Load audio
audio, sr = sf.read("test-audio.wav")
target_length = 480000
if len(audio) < target_length:
    audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')

# Load models
print("Loading models...")
preprocessor = ct.models.MLModel("build/fixed-preprocessor/preprocessor.mlpackage")
encoder = ct.models.MLModel("build/hf-upload/encoder.mlpackage")
decoder = ct.models.MLModel("build/decoder_with_cache.mlpackage")
cross_cache_computer = ct.models.MLModel("build/cross_cache_computer.mlpackage")
lm_head = ct.models.MLModel("build/hf-upload/lm_head.mlpackage")

# Preprocess
print("\nPreprocessing...")
prep_out = preprocessor.predict({
    "audio_signal": audio.reshape(1, -1).astype(np.float32),
    "length": np.array([len(audio)], dtype=np.int32)
})
mel_features = prep_out[list(prep_out.keys())[0]]
mel_length = prep_out[list(prep_out.keys())[1]]
print(f"  Mel features: {mel_features.shape}")

# Encode
print("Encoding...")
enc_out = encoder.predict({
    "input_features": mel_features,
    "length": mel_length
})
encoder_hidden = enc_out["encoder_output"]
print(f"  Encoder output: {encoder_hidden.shape}")

# Compute cross-caches
print("Computing cross-attention caches...")
cross_out = cross_cache_computer.predict({"encoder_hidden_states": encoder_hidden})
cross_cache_names = ["var_76", "var_93", "var_110", "var_127", "var_144", "var_161", "var_178", "var_195"]
cross_caches = {f"cross_cache{i}": cross_out[name] for i, name in enumerate(cross_cache_names)}

# Initialize self-caches
self_caches = {f"self_cache{i}": np.zeros((2, 1, 512, 8, 128), dtype=np.float32) for i in range(8)}

# Generation
print("\nGenerating...")
PREFIX = [13764, 4, 16, 62, 6, 9, 11, 13]
tokens = []
position = np.array([[0]], dtype=np.float32)

for step in range(len(PREFIX) + 100):
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
    dec_out = decoder.predict(dec_input)

    # Extract outputs
    hidden_states = dec_out["var_2384"]
    position = dec_out["var_2387"]

    # Update self-caches
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
        self_caches[f"self_cache{i}"] = dec_out[cache_name]

    # Get logits
    lm_out = lm_head.predict({"hidden_states": hidden_states})
    logits = lm_out["logits"]

    # Get next token
    next_token = int(np.argmax(logits))

    if step >= len(PREFIX):
        if next_token == 3:  # EOS
            print(f"  → EOS at step {step - len(PREFIX) + 1}")
            break
        tokens.append(next_token)

# Decode
print("\nDecoding...")
text = "".join([vocab.get(t, "?") for t in tokens])

# Clean up
text = (text
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
print(f"Generated: \"{text}\"")

# Load ground truth
try:
    with open("test-audio-groundtruth.txt") as f:
        ground_truth = f.read().strip()
    print(f"Ground truth: \"{ground_truth}\"")

    if text.lower() == ground_truth.lower():
        print("\n✅ PERFECT MATCH!")
    else:
        print("\n❌ NO MATCH")
except:
    pass
