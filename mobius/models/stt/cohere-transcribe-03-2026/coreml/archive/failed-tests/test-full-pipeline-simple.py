#!/usr/bin/env python3
"""Test full Cohere pipeline with new preprocessor (without KV cache)."""
import coremltools as ct
import numpy as np
import soundfile as sf
import json

print("=== Testing Full Cohere Pipeline ===\n")

# Load vocabulary
print("Loading vocabulary...")
with open("build/hf-upload/vocab.json") as f:
    vocab_dict = json.load(f)
    vocab = {v: k for k, v in vocab_dict.items()}
print(f"  Loaded {len(vocab)} tokens")

# Load audio
print("\nLoading audio...")
audio, sr = sf.read("test-audio.wav")
target_length = 480000
if len(audio) < target_length:
    audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
print(f"  Audio: {len(audio)} samples @ {sr} Hz")

# Load models
print("\nLoading CoreML models...")
preprocessor = ct.models.MLModel("build/hf-upload/preprocessor.mlpackage")
encoder = ct.models.MLModel("build/hf-upload/encoder.mlpackage")
decoder = ct.models.MLModel("build/hf-upload/cohere_decoder.mlpackage")
lm_head = ct.models.MLModel("build/hf-upload/lm_head.mlpackage")
print("  ✓ All models loaded")

# Preprocess
print("\nPreprocessing audio...")
prep_out = preprocessor.predict({
    "audio_signal": audio.reshape(1, -1).astype(np.float32),
    "length": np.array([len(audio)], dtype=np.int32)
})
mel_features = prep_out["mel_features"]
print(f"  Mel features: {mel_features.shape}")

# Encode
print("\nEncoding...")
enc_out = encoder.predict({"input_features": mel_features})
encoder_hidden = enc_out["encoder_output"]
print(f"  Encoder output: {encoder_hidden.shape}")

# Pad encoder output to 1500 (decoder expects this)
if encoder_hidden.shape[1] < 1500:
    padding = np.zeros((1, 1500 - encoder_hidden.shape[1], 1280), dtype=encoder_hidden.dtype)
    encoder_hidden = np.concatenate([encoder_hidden, padding], axis=1)
    print(f"  Padded to: {encoder_hidden.shape}")

# Decode (simple greedy decoding)
print("\nDecoding (greedy)...")
PREFIX = [13764, 2, 13765]  # <|startoftranscript|> <|notimestamps|> <|transcribe|>
EOS = 2

tokens = PREFIX.copy()
max_steps = 50
max_seq_len = 10  # Decoder's fixed max length

for step in range(max_steps):
    # Check if we've reached max sequence length
    if len(tokens) >= max_seq_len:
        print(f"  Stopped at step {step} (max length reached)")
        break

    # Prepare decoder input (pad to fixed size 10)
    seq_len = len(tokens)
    padded_tokens = tokens + [0] * (10 - seq_len)
    input_ids = np.array([padded_tokens], dtype=np.int32)
    positions = np.arange(10, dtype=np.int32).reshape(1, -1)

    # Run decoder
    dec_out = decoder.predict({
        "input_ids": input_ids,
        "positions": positions,
        "encoder_hidden_states": encoder_hidden
    })
    hidden_states = dec_out["hidden_states"]

    # lm_head expects (1, 10, 1024), so pass full hidden_states
    lm_out = lm_head.predict({"hidden_states": hidden_states})
    logits_full = list(lm_out.values())[0]  # (1, 10, vocab_size)

    # Get logits for last actual token (not padding)
    logits = logits_full[:, seq_len-1:seq_len, :]

    # Greedy sampling
    next_token = int(np.argmax(logits[0, 0]))

    if step < 3:
        print(f"  Step {step}: token={next_token} '{vocab.get(next_token, '?')}'")

    if next_token == EOS:
        print(f"  Stopped at step {step} (EOS)")
        break

    tokens.append(next_token)

# Decode
print("\nDecoding tokens...")
generated_text = ""
for token in tokens[len(PREFIX):]:  # Skip prefix
    if token == EOS:
        break
    if token in vocab:
        generated_text += vocab[token]

print(f"\nGenerated: \"{generated_text}\"")
print("\n✅ Pipeline test complete!")
print("\nNote: This uses the standard decoder (no KV cache).")
print("For better performance, the KV cache decoder needs to be debugged.")
