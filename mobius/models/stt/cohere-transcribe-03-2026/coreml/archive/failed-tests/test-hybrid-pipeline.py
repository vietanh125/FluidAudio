#!/usr/bin/env python3
"""
Hybrid pipeline: BarathwajAnandan frontend + our encoder + BarathwajAnandan decoder.

Strategy:
  1. Frontend  → mel spectrogram (BarathwajAnandan)
  2. Encoder   → hidden states   (OUR EXPORT: encoder_correct_static)
  3. Decoder   → autoregressive generation using the cached decoder with explicit KV cache
                 - Prefill:    step through each prompt token to build up KV cache
                 - Generate:   step through generated tokens until EOS or max length
"""
import coremltools as ct
import numpy as np
import soundfile as sf
import librosa
import json

print("=== Hybrid Pipeline: Barathwaj Frontend + Our Encoder + Barathwaj Cached Decoder ===\n")

# ── Config ──────────────────────────────────────────────────────────────────
BASE = "build/barathwaj-models"

# ── Tokenizer ────────────────────────────────────────────────────────────────
print("1. Loading tokenizer...")
with open(f"{BASE}/coreml_manifest.json") as f:
    manifest = json.load(f)

id_to_token         = {i: t for i, t in enumerate(manifest["id_to_token"])}
prompt_ids          = manifest["prompt_ids"]
eos_token_id        = manifest["eos_token_id"]
decoder_max_len     = manifest["decoder_max_len"]          # 108
max_new_tokens      = manifest["default_max_new_tokens"]   # 96
max_encoder_frames  = manifest["max_encoder_frames"]       # 438
num_layers          = manifest["decoder_cached"]["num_layers"]  # 8
num_heads           = manifest["decoder_cached"]["num_heads"]   # 8
head_dim            = manifest["decoder_cached"]["head_dim"]    # 128
logits_out          = manifest["decoder_cached"]["logits_output"]  # var_2891
cache_k_out         = manifest["decoder_cached"]["cache_k_output"] # var_2894
cache_v_out         = manifest["decoder_cached"]["cache_v_output"] # var_2897

print(f"   Prompt IDs    : {prompt_ids}")
print(f"   Prompt tokens : {[id_to_token[i] for i in prompt_ids]}")
print(f"   EOS token ID  : {eos_token_id}")
print(f"   KV cache shape: ({num_layers}, {num_heads}, {decoder_max_len}, {head_dim})")

# ── Models ───────────────────────────────────────────────────────────────────
print("\n2. Loading models...")
frontend       = ct.models.MLModel(f"{BASE}/cohere_frontend.mlpackage")
encoder        = ct.models.MLModel("build/encoder_correct_static.mlpackage")
decoder_cached = ct.models.MLModel(f"{BASE}/cohere_decoder_cached.mlpackage")
print("   Frontend       : BarathwajAnandan (cohere_frontend)")
print("   Encoder        : OUR EXPORT       (encoder_correct_static)")
print("   Decoder        : BarathwajAnandan (cohere_decoder_cached)")

# ── Audio ────────────────────────────────────────────────────────────────────
print("\n3. Loading audio...")
audio, sr = sf.read("test-librispeech-real.wav")
if audio.ndim > 1:
    audio = audio.mean(axis=1)
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

original_len = len(audio)
max_samples  = manifest["max_audio_samples"]  # 560000
print(f"   test-librispeech-real.wav: {original_len} samples @ 16kHz ({original_len/16000:.2f}s)")

if len(audio) < max_samples:
    audio = np.pad(audio, (0, max_samples - len(audio)))
else:
    audio = audio[:max_samples]

# ── Step 1: Frontend → mel ───────────────────────────────────────────────────
print("\n4. Running frontend (mel spectrogram)...")
mel = frontend.predict({
    "audio_samples": audio.reshape(1, -1).astype(np.float32),
    "audio_length":  np.array([original_len], dtype=np.int32),
})["var_6916"]
print(f"   Mel shape: {mel.shape}")

# ── Step 2: Our encoder → hidden states ─────────────────────────────────────
print("\n5. Running our encoder...")
enc_out       = encoder.predict({
    "input_features": mel,
    "feature_length":  np.array([3501], dtype=np.int32),
})
hidden_states  = enc_out["encoder_output"].astype(np.float16)   # (1, 438, 1024)
encoder_length = int(enc_out["encoder_length"][0])
print(f"   Hidden states : {hidden_states.shape}  dtype={hidden_states.dtype}")
print(f"   Encoder length: {encoder_length}")
print(f"   Stats: min={float(hidden_states.min()):.3f}  max={float(hidden_states.max()):.3f}  "
      f"mean={float(hidden_states.mean()):.3f}")

# Cross-attention mask (all 1s = attend to all encoder frames)
cross_mask = np.ones((1, 1, 1, max_encoder_frames), dtype=np.float16)

# ── Step 3: Autoregressive decoding with KV cache ────────────────────────────
print("\n6. Autoregressive decoding (prefill + generate)...")

# Initialise empty KV cache
cache_k = np.zeros((num_layers, num_heads, decoder_max_len, head_dim), dtype=np.float16)
cache_v = np.zeros((num_layers, num_heads, decoder_max_len, head_dim), dtype=np.float16)

# ── 6a. Prefill: run through each prompt token ───────────────────────────────
print(f"   Prefilling {len(prompt_ids)} prompt tokens...")
last_logits = None
for i, token_id in enumerate(prompt_ids):
    out = decoder_cached.predict({
        "encoder_hidden_states": hidden_states,
        "input_id":              np.array([[token_id]], dtype=np.int32),
        "cache_k":               cache_k,
        "cache_v":               cache_v,
        "step":                  np.array([i], dtype=np.int32),
        "cross_attention_mask":  cross_mask,
    })
    cache_k     = out[cache_k_out]
    cache_v     = out[cache_v_out]
    last_logits = out[logits_out]  # (1, 16384) — prediction for position i+1

# First generated token is the argmax of the last prompt-step logits
first_token = int(np.argmax(last_logits[0]))
print(f"   First generated token: {first_token} = '{id_to_token.get(first_token)}'")

# ── 6b. Autoregressive generation ────────────────────────────────────────────
generated = []
current_token = first_token

print("   Generating...")
for step in range(max_new_tokens):
    if current_token == eos_token_id:
        print(f"   EOS reached at step {step}")
        break

    generated.append(current_token)
    seq_pos = len(prompt_ids) + step

    if seq_pos >= decoder_max_len - 1:
        print(f"   Max length ({decoder_max_len}) reached")
        break

    out = decoder_cached.predict({
        "encoder_hidden_states": hidden_states,
        "input_id":              np.array([[current_token]], dtype=np.int32),
        "cache_k":               cache_k,
        "cache_v":               cache_v,
        "step":                  np.array([seq_pos], dtype=np.int32),
        "cross_attention_mask":  cross_mask,
    })
    cache_k       = out[cache_k_out]
    cache_v       = out[cache_v_out]
    logits        = out[logits_out]   # (1, 16384)
    current_token = int(np.argmax(logits[0]))

    if step < 15:
        print(f"     step {step:3d}: pos {seq_pos}  → {current_token:5d} '{id_to_token.get(current_token)}'")

# Decode to text
text = "".join(id_to_token.get(t, f"<{t}>") for t in generated if t != eos_token_id)

print(f"\n{'='*60}")
print("TRANSCRIPTION RESULT (Hybrid Pipeline):")
print("=" * 60)
print(f'"{text}"')
print("=" * 60)
print(f"Generated {len(generated)} tokens from {len(prompt_ids)} prompt tokens")
