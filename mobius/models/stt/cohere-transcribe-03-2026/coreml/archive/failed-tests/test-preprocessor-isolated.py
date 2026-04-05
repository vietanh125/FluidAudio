#!/usr/bin/env python3
"""Test CoreML preprocessor with PyTorch encoder/decoder to isolate preprocessing issues."""
import torch
import numpy as np
import soundfile as sf
import librosa
import coremltools as ct
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

print("=== Isolated Preprocessor Test ===\n")

# Load audio
audio, sr = sf.read("test-librispeech-real.wav")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
print(f"Audio: {len(audio)} samples @ 16000Hz ({len(audio)/16000:.2f}s)")

# Load transformers model
print("\nLoading transformers model...")
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

# Test 1: Full Transformers Pipeline (Ground Truth)
print("\n1. Full Transformers Pipeline (Ground Truth):")
with torch.no_grad():
    generated_ids = model.generate(
        processor(audio, sampling_rate=16000, return_tensors="pt")["input_features"],
        max_new_tokens=50,
        num_beams=1,
        do_sample=False
    )
ground_truth = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"   Result: \"{ground_truth}\"")
print(f"   Token IDs: {generated_ids[0].tolist()[:10]}...")

# Test 2: CoreML Preprocessor + PyTorch Encoder/Decoder
print("\n2. CoreML Preprocessor + PyTorch Models:")

# 2a. CoreML Preprocessor
print("   a) Running CoreML preprocessor...")
preprocessor_model = ct.models.MLModel("build/hf-upload/preprocessor.mlpackage")

# Pad audio to 30s
audio_padded = np.pad(audio, (0, 480000 - len(audio)), mode='constant') if len(audio) < 480000 else audio[:480000]
prep_out = preprocessor_model.predict({
    "audio_signal": audio_padded.reshape(1, -1).astype(np.float32),
    "length": np.array([len(audio)], dtype=np.int32)
})
coreml_mel = prep_out["mel_features"]
mel_length = int(prep_out["mel_length"][0])
print(f"      Output: {coreml_mel.shape} (valid length: {mel_length})")

# 2b. Compare with transformers preprocessor
print("   b) Comparing preprocessor outputs...")
transformers_input = processor(audio, sampling_rate=16000, return_tensors="pt")
transformers_mel = transformers_input["input_features"].numpy()
transformers_length = transformers_input["length"].item()

print(f"      Transformers: {transformers_mel.shape} (length: {transformers_length})")
print(f"      CoreML: {coreml_mel.shape} (length: {mel_length})")

# Trim both to valid lengths for comparison
transformers_mel_valid = transformers_mel[:, :, :transformers_length]
coreml_mel_valid = coreml_mel[:, :, :mel_length]

# Pad to same length for comparison
max_len = max(transformers_length, mel_length)
if transformers_mel_valid.shape[2] < max_len:
    transformers_mel_valid = np.pad(transformers_mel_valid, ((0, 0), (0, 0), (0, max_len - transformers_mel_valid.shape[2])))
if coreml_mel_valid.shape[2] < max_len:
    coreml_mel_valid = np.pad(coreml_mel_valid, ((0, 0), (0, 0), (0, max_len - coreml_mel_valid.shape[2])))

diff = np.abs(transformers_mel_valid - coreml_mel_valid)
print(f"      Difference: max={diff.max():.6f}, mean={diff.mean():.6f}, median={np.median(diff):.6f}")

if diff.max() < 0.01:
    print(f"      ✅ Excellent match!")
elif diff.max() < 0.1:
    print(f"      ✅ Good match")
elif diff.max() < 1.0:
    print(f"      ⚠️  Acceptable match")
else:
    print(f"      ❌ Poor match - preprocessor is wrong!")

# 2c. Feed CoreML preprocessor output to PyTorch encoder
print("   c) Running PyTorch encoder with CoreML preprocessor output...")
with torch.no_grad():
    # Convert to torch and trim to valid length
    coreml_mel_torch = torch.from_numpy(coreml_mel[:, :, :mel_length])
    encoder_output = model.encoder(coreml_mel_torch)
    encoder_hidden_states = encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output

print(f"      Encoder output: {encoder_hidden_states.shape}")

# 2d. Run PyTorch decoder
print("   d) Running PyTorch decoder...")
with torch.no_grad():
    # Use generate with the encoder output
    # We need to manually create decoder inputs since we're bypassing the preprocessor
    generated_ids_hybrid = model.generate(
        coreml_mel_torch,
        max_new_tokens=50,
        num_beams=1,
        do_sample=False
    )

hybrid_result = processor.batch_decode(generated_ids_hybrid, skip_special_tokens=True)[0]
print(f"   Result: \"{hybrid_result}\"")
print(f"   Token IDs: {generated_ids_hybrid[0].tolist()[:10]}...")

# Test 3: Comparison
print("\n3. Comparison:")
print(f"   Ground Truth:  \"{ground_truth}\"")
print(f"   Hybrid Result: \"{hybrid_result}\"")

if ground_truth == hybrid_result:
    print(f"   ✅ PERFECT MATCH - CoreML preprocessor is correct!")
elif ground_truth.strip() == hybrid_result.strip():
    print(f"   ✅ Match (minor whitespace difference)")
else:
    print(f"   ❌ MISMATCH - CoreML preprocessor has issues")
    # Show character-level diff
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, ground_truth, hybrid_result)
    ratio = matcher.ratio()
    print(f"   Similarity: {ratio*100:.1f}%")
