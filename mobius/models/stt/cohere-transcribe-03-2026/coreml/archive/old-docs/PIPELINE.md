# Cohere Transcribe 03-2026 CoreML Pipeline

## Complete Pipeline Architecture

```
Audio (35s, 16kHz) → 560,000 samples
  ↓
[1. Frontend (Swift)] → (1, 560000) → (1, 128, 3501) mel spectrogram
  ↓
[2. Encoder (CoreML)] → (1, 128, 3501) → (1, 438, 1024) hidden states
  ↓
[3. Decoder (CoreML)] → (1, 438, 1024) + (1, 108) → (1, 108, 16384) logits → text
```

## Implementation Status

### ✅ Encoder - EXPORTED
- **File**: `ultra_static_encoder.mlpackage`
- **Input**: (1, 128, 3501) mel spectrogram
- **Output**: (1, 438, 1024) encoder hidden states
- **Validation**: Max diff 0.004490 vs PyTorch ✅
- **Precision**: FP16
- **Compute**: CPU+ANE
- **Export**: 2026-04-04 (ultra-static pattern)

### ✅ Decoder - EXPORTED
- **File**: `ultra_static_decoder.mlpackage`
- **Inputs**:
  - input_ids: (1, 108) token IDs
  - encoder_hidden_states: (1, 438, 1024)
- **Output**: (1, 108, 16384) logits
- **Validation**: Max diff 0.101479, Mean diff 0.004217 ✅
- **Precision**: FP16
- **Compute**: CPU+ANE
- **Export**: 2026-04-04 (ultra-static pattern)

### 🔄 Frontend - THREE OPTIONS

#### Option A: Swift Implementation (Recommended for Performance)
- **Approach**: Implement in Swift using Accelerate framework (like `WhisperMelSpectrogram` in FluidAudio)
- **Advantages**: Native performance, no CoreML limitations, full control
- **Implementation**: See Swift Frontend Implementation Guide below

#### Option B: CoreML Export (BarathwajAnandan's Approach)
- **Discovery**: BarathwajAnandan successfully exported frontend to CoreML **without** using complex STFT operations
- **Method**: Custom STFT implementation using only basic CoreML operations:
  - 3386 operations total
  - Uses: mul, div, add, sub, sqrt, reduce_sum, log
  - No FFT/STFT/complex operations
  - Manually implements STFT as matrix operations
- **Advantages**: Can export to CoreML, fully static
- **Disadvantages**: Complex implementation, 3386 ops vs Swift native STFT
- **Status**: Would need to reverse-engineer or re-implement

#### Option C: Use BarathwajAnandan's Frontend (Immediate)
- **File**: `build/barathwaj-models/cohere_frontend.mlpackage`
- **Advantages**: Already working, proven, immediate testing
- **Disadvantages**: External dependency, less control

**Parameters** (all options):
```
sample_rate: 16000
n_fft: 1024
hop_length: 160
win_length: 1024
n_mels: 128
f_min: 0.0
f_max: 8000.0
```

**Processing**:
1. Compute mel spectrogram (STFT + mel filterbank)
2. Apply log10 scaling: `log10(clamp(mel, min=1e-10))`
3. Normalize: `mel = mel - mean(mel)`

## Recommended Pipelines

### Option 1: Swift + CoreML (Production) 🎯

**Best for production**: Full control, native performance

```
build/
├── [Swift Frontend]                    # To be implemented
├── ultra_static_encoder.mlpackage      ✅ OUR EXPORT
└── ultra_static_decoder.mlpackage      ✅ OUR EXPORT
```

**Status**: 2/3 complete (67%)
- ✅ Encoder exported and validated
- ✅ Decoder exported and validated
- 🔄 Frontend to be implemented in Swift (like WhisperMelSpectrogram)

**Advantages**:
- Full control over all components
- Native Swift performance
- No external dependencies
- Optimized for Apple platforms

### Option 2: Mixed Pipeline (Immediate Testing) ⚡

**Best for immediate testing**: Can test while implementing Swift frontend

```
build/
├── barathwaj-models/cohere_frontend.mlpackage  # Their frontend
├── ultra_static_encoder.mlpackage              ✅ OUR EXPORT
└── ultra_static_decoder.mlpackage              ✅ OUR EXPORT
```

**Status**: 100% ready
- ✅ All files available
- ✅ Can test end-to-end pipeline now
- ✅ Validates our encoder and decoder exports

**Advantages**:
- Immediate testing capability
- Validates our CoreML exports work correctly
- Reference for Swift frontend implementation

### Option 3: Full BarathwajAnandan (Reference) 📚

**Best for validation**: Proven baseline

```
build/barathwaj-models/
├── cohere_frontend.mlpackage
├── cohere_encoder.mlpackage
└── cohere_decoder_fullseq_masked.mlpackage
```

**Advantages**:
- 100% proven working (2.58% WER)
- Reference implementation
- Validation baseline

---

## Python Example (Mixed Pipeline)

Test our encoder and decoder with BarathwajAnandan's frontend:

```python
import coremltools as ct
import numpy as np
import soundfile as sf
import librosa
import json

# Load models
frontend = ct.models.MLModel("build/barathwaj-models/cohere_frontend.mlpackage")
encoder = ct.models.MLModel("build/ultra_static_encoder.mlpackage")  # OUR EXPORT
decoder = ct.models.MLModel("build/ultra_static_decoder.mlpackage")  # OUR EXPORT

# Load tokenizer
with open("build/barathwaj-models/coreml_manifest.json") as f:
    manifest = json.load(f)
id_to_token = {i: token for i, token in enumerate(manifest['id_to_token'])}
prompt_ids = manifest['prompt_ids']

# Load audio
audio, sr = sf.read("audio.wav")
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# Pad to 560,000 samples (35 seconds)
if len(audio) < 560000:
    audio = np.pad(audio, (0, 560000 - len(audio)))
else:
    audio = audio[:560000]

# Step 1: Frontend (BarathwajAnandan's)
frontend_out = frontend.predict({
    "audio_samples": audio.reshape(1, -1).astype(np.float32),
    "audio_length": np.array([len(audio)], dtype=np.int32)
})
mel = frontend_out["var_6916"]  # (1, 128, 3501)

# Step 2: Encoder (OUR EXPORT)
encoder_out = encoder.predict({
    "input_features": mel
})
hidden_states = encoder_out["encoder_output"]  # (1, 438, 1024)

# Step 3: Decoder (OUR EXPORT)
input_ids = np.array([prompt_ids + [0] * (108 - len(prompt_ids))], dtype=np.int32)

decoder_out = decoder.predict({
    "input_ids": input_ids,
    "encoder_hidden_states": hidden_states
})
logits = decoder_out["logits"]  # (1, 108, 16384)

# Get tokens
predicted_ids = np.argmax(logits[0], axis=-1)
text = "".join([id_to_token.get(int(id), f"<{id}>") for id in predicted_ids[:len(prompt_ids)]])
print(f"Transcription: {text}")
```

---

## Swift Frontend Implementation Guide

Reference `WhisperMelSpectrogram.swift` in FluidAudio for implementation pattern:

```swift
import Accelerate

class CohereTranscribeMelSpectrogram {
    private let sampleRate: Int = 16000
    private let nFFT: Int = 1024
    private let hopLength: Int = 160
    private let winLength: Int = 1024
    private let nMels: Int = 128
    private let fMin: Float = 0.0
    private let fMax: Float = 8000.0

    func compute(audio: [Float]) -> [[Float]] {
        // 1. Compute STFT using vDSP
        let stft = computeSTFT(audio)

        // 2. Apply mel filterbank
        let mel = applyMelFilterbank(stft)

        // 3. Log scaling
        let logMel = mel.map { frame in
            frame.map { max(log10($0), -10) }  // Clamp to 1e-10
        }

        // 4. Normalize (subtract mean)
        let mean = logMel.flatMap { $0 }.reduce(0, +) / Float(logMel.count * logMel[0].count)
        let normalized = logMel.map { frame in
            frame.map { $0 - mean }
        }

        return normalized  // (128, 3501)
    }
}
```

**Key Differences from Whisper**:
- `hop_length: 160` (Whisper uses 160, same)
- `n_mels: 128` (Whisper uses 80 or 128 depending on model)
- `f_max: 8000.0` (Whisper uses 8000.0, same)
- **Normalization**: Subtract mean (Whisper also normalizes but may use different method)

---

## Model Details

### Encoder (Our Export)
- **Source**: Ultra-static export (2026-04-04)
- **Size**: ~2.5 GB
- **Precision**: FP16
- **Compute**: CPU+ANE (ANE recommended)
- **Special**: Includes 1280→1024 projection layer
- **Validation**: Max diff 0.004490 vs PyTorch

### Decoder (Our Export)
- **Source**: Ultra-static export (2026-04-04)
- **Size**: ~1.5 GB
- **Precision**: FP16
- **Compute**: CPU+ANE
- **Max length**: 108 tokens
- **Validation**: Max diff 0.101479, Mean diff 0.004217 vs PyTorch

### Frontend (Swift Implementation)
- **Source**: To be implemented (reference: WhisperMelSpectrogram.swift)
- **Size**: N/A (computed in Swift)
- **Precision**: FP32
- **Compute**: CPU (Accelerate framework)
- **Parameters**: n_fft=1024, hop_length=160, n_mels=128

---

## Next Steps

1. **Test Mixed Pipeline** ✅ Ready Now
   ```bash
   python test-full-pipeline-mixed.py
   ```

2. **Implement Swift Frontend** 🔄 In Progress
   - Reference `WhisperMelSpectrogram.swift`
   - Match parameters: n_fft=1024, hop_length=160, n_mels=128
   - Apply log scaling and normalization
   - Validate against BarathwajAnandan's frontend output

3. **Integrate into FluidAudio** 🚀 Next
   - Add CohereTranscribeMelSpectrogram class
   - Create CohereTranscribeAsrManager
   - Load ultra_static_encoder.mlpackage and ultra_static_decoder.mlpackage
   - Implement full pipeline

4. **Benchmark Performance** 📊 Final
   - Measure RTFx (Real-Time Factor)
   - Compare WER with baseline
   - Optimize if needed
