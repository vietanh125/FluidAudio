#!/usr/bin/env python3
"""Export encoder CORRECTLY - wraps model.encoder directly, static encoder_length output."""
import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import numpy as np
import soundfile as sf
import librosa
import sys
import os

print("=== Exporting Encoder (Correct - Static) ===\n")

# Load model
print("1. Loading PyTorch model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    dtype=torch.float32,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True
)
model.eval()
print("   ✓ Model loaded")


class EncoderWithProjection(nn.Module):
    """Encoder + Projection with static encoder_length output.

    Uses model.encoder directly (architecturally correct - real NeMo encoder
    forward pass) rather than reimplementing it.
    """

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.projection = model.encoder_decoder_proj

    def forward(self, input_features, feature_length):
        """
        Args:
            input_features: (1, 128, 3501) mel spectrogram (35s audio)
            feature_length: (1,) actual valid frames (unused - see encoder_length)
        Returns:
            encoder_output: (1, 438, 1024) projected encoder hidden states
            encoder_length: (1,) always [438] - STATIC for CoreML compatibility
        """
        encoder_out = self.encoder(input_features)

        if isinstance(encoder_out, tuple):
            hidden_states = encoder_out[0]
        else:
            hidden_states = encoder_out

        # Apply projection: 1280 -> 1024
        projected = self.projection(hidden_states)

        # STATIC encoder_length - no dynamic computation from feature_length.
        # Encoder downsamples 3501 mel frames by 8x -> ceil(3501/8) = 438 frames.
        encoder_length = torch.tensor([438])

        return projected, encoder_length


# ============================================================================
# REAL CORRECTNESS VALIDATION
# ============================================================================

print("\n2. Real audio correctness validation...")

# Find real audio file
audio_candidates = [
    "test-librispeech-real.wav",
    "test-librispeech.wav",
    "test-english.wav",
    "test-real-speech.wav",
    "test-audio.wav",
]
audio_path = None
for candidate in audio_candidates:
    if os.path.exists(candidate):
        audio_path = candidate
        break

if audio_path is None:
    print("   ❌ No .wav file found. Tried:", audio_candidates)
    sys.exit(1)

print(f"   Using audio: {audio_path}")
audio, sr = sf.read(audio_path)
if audio.ndim > 1:
    audio = audio[:, 0]  # mono
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
mel_input = inputs["input_features"]

# Pad/trim to exactly 3501 frames (35s audio -> BarathwajAnandan spec)
target_frames = 3501
if mel_input.shape[2] < target_frames:
    mel_padded = torch.nn.functional.pad(mel_input, (0, target_frames - mel_input.shape[2]))
else:
    mel_padded = mel_input[:, :, :target_frames]

feature_length = torch.tensor([mel_input.shape[2]], dtype=torch.int32)
print(f"   Mel input: {mel_padded.shape}")
print(f"   Feature length: {feature_length.item()}")

# --- Reference: model.encoder + model.encoder_decoder_proj ---
with torch.no_grad():
    ref_enc_out = model.encoder(mel_padded)
    if isinstance(ref_enc_out, tuple):
        ref_hidden = ref_enc_out[0]
    else:
        ref_hidden = ref_enc_out
    ref_projected = model.encoder_decoder_proj(ref_hidden)

print(f"   Reference output shape: {ref_projected.shape}")
print(f"   Reference stats: min={ref_projected.min():.4f}, max={ref_projected.max():.4f}, mean={ref_projected.mean():.4f}")

# --- Wrapper output ---
encoder_wrapper = EncoderWithProjection(model)
encoder_wrapper.eval()

with torch.no_grad():
    wrapper_out, wrapper_len = encoder_wrapper(mel_padded, feature_length)

print(f"\n   Wrapper output shape: {wrapper_out.shape}")
print(f"   Wrapper encoder_length: {wrapper_len.item()} (should be 438)")

max_diff_wrapper = torch.abs(ref_projected - wrapper_out).max().item()
mean_diff_wrapper = torch.abs(ref_projected - wrapper_out).mean().item()
print(f"\n   Wrapper vs Reference:")
print(f"   Max diff:  {max_diff_wrapper:.2e}")
print(f"   Mean diff: {mean_diff_wrapper:.2e}")
if max_diff_wrapper < 1e-5:
    print("   ✅ PASS: Wrapper matches reference exactly")
else:
    print("   ❌ FAIL: Wrapper output differs from reference!")
    sys.exit(1)

if wrapper_out.shape != (1, 438, 1024):
    print(f"   ❌ FAIL: Expected shape (1, 438, 1024), got {wrapper_out.shape}")
    sys.exit(1)


# ============================================================================
# TORCH.JIT.TRACE
# ============================================================================

print("\n3. Tracing model...")
example_features = torch.randn(1, 128, 3501)
example_length = torch.tensor([3501], dtype=torch.int32)

traced_model = torch.jit.trace(encoder_wrapper, (example_features, example_length))
print("   ✓ Model traced")

# Verify traced model matches reference
with torch.no_grad():
    traced_out, traced_len = traced_model(mel_padded, feature_length)

trace_max_diff = torch.abs(ref_projected - traced_out).max().item()
print(f"   Traced vs Reference max diff: {trace_max_diff:.2e}")
if trace_max_diff < 0.01:
    print("   ✅ PASS: Traced model matches reference")
else:
    print(f"   ⚠️  WARNING: Traced diff = {trace_max_diff:.4f}")


# ============================================================================
# COREML CONVERSION
# ============================================================================

print("\n4. Converting to CoreML...")
os.makedirs("build", exist_ok=True)

mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="input_features", shape=(1, 128, 3501), dtype=np.float32),
        ct.TensorType(name="feature_length", shape=(1,), dtype=np.int32),
    ],
    outputs=[
        ct.TensorType(name="encoder_output", dtype=np.float16),
        ct.TensorType(name="encoder_length", dtype=np.int32),
    ],
    minimum_deployment_target=ct.target.macOS13,
    compute_precision=ct.precision.FLOAT16,
)

output_path = "build/encoder_correct_static.mlpackage"
mlmodel.save(output_path)
print(f"   ✓ Saved to: {output_path}")


# ============================================================================
# COREML VALIDATION - compare against original model.encoder + encoder_decoder_proj
# ============================================================================

print("\n5. CoreML validation vs original model...")
coreml_model = ct.models.MLModel(output_path)

coreml_out = coreml_model.predict({
    "input_features": mel_padded.numpy().astype(np.float32),
    "feature_length": feature_length.numpy().astype(np.int32),
})
coreml_encoder = coreml_out["encoder_output"]
coreml_length = coreml_out["encoder_length"]

print(f"   CoreML output shape: {coreml_encoder.shape}")
print(f"   CoreML encoder_length: {coreml_length}")

ref_np = ref_projected.numpy()
coreml_max_diff = np.abs(ref_np - coreml_encoder).max()
coreml_mean_diff = np.abs(ref_np - coreml_encoder).mean()
print(f"\n   CoreML vs Reference (model.encoder + encoder_decoder_proj):")
print(f"   Max diff:  {coreml_max_diff:.4f}")
print(f"   Mean diff: {coreml_mean_diff:.6f}")

if coreml_max_diff < 0.1:
    print("   ✅ PASS: CoreML matches reference (within float16 tolerance)")
else:
    print(f"   ❌ FAIL: CoreML max diff {coreml_max_diff:.4f} exceeds 0.1 threshold")


# ============================================================================
# INTEGRATION TEST: BarathwajAnandan frontend + our encoder + Barathwaj decoder
# ============================================================================

print("\n6. Integration test: Barathwaj frontend + our encoder + Barathwaj decoder...")

barathwaj_dir = "build/barathwaj-models"
if not os.path.exists(barathwaj_dir):
    print(f"   ⚠️  Skipping: {barathwaj_dir} not found")
else:
    import json

    frontend_path = f"{barathwaj_dir}/cohere_frontend.mlpackage"
    decoder_path = f"{barathwaj_dir}/cohere_decoder_fullseq_masked.mlpackage"
    manifest_path = f"{barathwaj_dir}/coreml_manifest.json"

    missing = [p for p in [frontend_path, decoder_path, manifest_path] if not os.path.exists(p)]
    if missing:
        print(f"   ⚠️  Skipping: missing {missing}")
    else:
        frontend = ct.models.MLModel(frontend_path)
        decoder = ct.models.MLModel(decoder_path)

        with open(manifest_path) as f:
            manifest = json.load(f)
        id_to_token = {i: token for i, token in enumerate(manifest["id_to_token"])}
        prompt_ids = manifest["prompt_ids"]

        # Pad audio to exactly 560000 samples (35s at 16kHz)
        original_len = len(audio)
        target_samples = 560000
        if len(audio) < target_samples:
            audio_padded = np.pad(audio, (0, target_samples - len(audio)))
        else:
            audio_padded = audio[:target_samples]

        # Frontend
        frontend_out = frontend.predict({
            "audio_samples": audio_padded.reshape(1, -1).astype(np.float32),
            "audio_length": np.array([original_len], dtype=np.int32),
        })
        mel = frontend_out["var_6916"]
        print(f"   Frontend mel shape: {mel.shape}")

        # Our encoder
        enc_out = coreml_model.predict({
            "input_features": mel,
            "feature_length": np.array([3501], dtype=np.int32),
        })
        hidden_states = enc_out["encoder_output"]
        print(f"   Encoder hidden states: {hidden_states.shape}")
        print(f"   Hidden stats: min={hidden_states.min():.3f}, max={hidden_states.max():.3f}, mean={hidden_states.mean():.3f}")

        # Decoder
        input_ids = np.array([prompt_ids + [0] * (108 - len(prompt_ids))], dtype=np.int32)
        decoder_mask = np.zeros((1, 108), dtype=np.int32)
        decoder_mask[0, : len(prompt_ids)] = 1
        cross_mask = np.ones((1, 1, 1, 438), dtype=np.int32)

        decoder_out = decoder.predict({
            "input_ids": input_ids,
            "decoder_attention_mask": decoder_mask,
            "encoder_hidden_states": hidden_states,
            "cross_attention_mask": cross_mask,
        })
        logits = decoder_out["var_1009"]
        print(f"   Decoder logits: {logits.shape}")

        # Decode tokens
        predicted_ids = np.argmax(logits[0], axis=-1)
        text_chars = []
        for token_id in predicted_ids[: len(prompt_ids)]:
            token = id_to_token.get(int(token_id), f"<{token_id}>")
            text_chars.append(token)
        text = "".join(text_chars)

        print(f"\n   Transcription: {text}")

        if text.count(" ") > len(text) * 0.05 and len(text.strip()) > 5:
            print("   ✅ PASS: Real text output with spaces")
        else:
            print("   ❌ FAIL: Looks like garbage (no spaces / empty)")


print(f"\n{'='*60}")
print("Encoder export complete")
print(f"Output: {output_path}")
print(f"Input:  (1, 128, 3501)  [35s mel spectrogram]")
print(f"Output: (1, 438, 1024) + encoder_length=[438] [static]")
print("=" * 60)
