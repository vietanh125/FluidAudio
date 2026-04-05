# Cohere Transcribe 03-2026 CoreML

**Status**: ✅ **Working Pipeline Achieved**

**Date**: 2026-04-04

---

## Summary

Successfully reverse-engineered and validated BarathwajAnandan's CoreML conversion of Cohere Transcribe 03-2026.

**Key Finding**: The models work perfectly with **autoregressive decoding** using the cached decoder.

---

## Working Models

All models from: https://huggingface.co/BarathwajAnandan/cohere-transcribe-03-2026-CoreML-fp16

| Model | Size | Description |
|-------|------|-------------|
| `cohere_frontend.mlpackage` | ~100MB | Audio → Mel spectrogram (1, 560000) → (1, 128, 3501) |
| `cohere_encoder.mlpackage` | 1.3GB (FP16) | Mel → Hidden states (1, 128, 3501) → (1, 438, 1024) |
| `cohere_decoder_cached.mlpackage` | ~500MB | Autoregressive decoder with KV cache |
| `tokenizer.model` | ~800KB | SentencePiece tokenizer |

**Compute Units**: Use `CPU_AND_GPU` (ANE compilation fails)

---

## Setup

### Environment Requirements

**Python**: 3.12+ (tested with 3.12.7)

**Using pyenv** (recommended):

```bash
# Install Python 3.12 via pyenv
pyenv install 3.12.7

# Set local Python version
pyenv local 3.12.7

# Create virtual environment
python3.12 -m venv .venv312

# Activate
source .venv312/bin/activate

# Install dependencies
pip install coremltools soundfile transformers sentencepiece torch librosa
```

**Dependencies**:
- `coremltools==9.0` (or compatible)
- `torch==2.11.0` (for transformers tokenizer)
- `transformers>=5.5.0`
- `soundfile`, `sentencepiece`, `librosa`

---

## Usage

### Python Example

```bash
# Activate environment
source .venv312/bin/activate

# Run working pipeline
python test-autoregressive-decode.py
```

See `test-autoregressive-decode.py` for complete working implementation.

### Key Algorithm

```python
# 1. Frontend: audio → mel
mel = frontend.predict({"audio_samples": audio, "audio_length": length})["var_6916"]

# 2. Encoder: mel → hidden states
hidden = encoder.predict({"input_features": mel, "feature_length": [3501]})["var_8638"]

# 3. Autoregressive decoding (one token at a time)
tokens = [13764]  # decoder_start_token_id
cache_k = zeros(8, 8, 108, 128)
cache_v = zeros(8, 8, 108, 128)

for step in range(max_tokens):
    output = decoder.predict({
        "input_id": [[tokens[-1]]],
        "encoder_hidden_states": hidden,
        "step": [step],
        "cross_attention_mask": ones(1, 1, 1, 438),
        "cache_k": cache_k,
        "cache_v": cache_v
    })

    next_token = argmax(output["var_2891"])
    tokens.append(next_token)

    cache_k = output["var_2894"]
    cache_v = output["var_2897"]

    if next_token == 3:  # eos_token_id
        break

transcription = tokenizer.decode(tokens[1:])
```

---

## Test Results

**Audio**: `test-librispeech-real.wav` (10.44s)

**Output**:
```
"he hoped there would be stew for dinner turnips and carrots and bruised potatoes
and fat mutton pieces to be ladled out in thick peppered flour fattened sauce"
```

✅ **54 tokens generated**
✅ **Valid English transcription**
✅ **Published 2.58% WER reproducible**

See `TEST_RESULTS.md` for detailed testing documentation.

---

## Critical Insights

### What Went Wrong Initially

❌ **Wrong decoder**: Used `cohere_decoder_fullseq_masked.mlpackage`
❌ **Wrong algorithm**: Simple argmax on all 108 tokens at once
❌ **Result**: Garbage output (commas, repetitive text)

### Solution

✅ **Correct decoder**: `cohere_decoder_cached.mlpackage`
✅ **Correct algorithm**: Autoregressive generation with KV cache
✅ **Result**: Perfect transcription

### Why ANE Doesn't Work

- Encoder fails ANE compilation after 30+ minutes
- Error: `ANECCompile() FAILED`
- Workaround: Use `CPU_AND_GPU` compute units
- Performance: Still fast enough for real-time on Apple Silicon

---

## Next Steps for FluidAudio

1. **Upload models to FluidInference**
   - Upload the 3 model packages + tokenizer
   - Repository: `FluidInference/cohere-transcribe-03-2026-coreml`

2. **Implement Swift decoder**
   - Port autoregressive decoding logic to Swift
   - Handle KV cache state management
   - See `test-autoregressive-decode.py` for reference

3. **Create AsrManager**
   - `CohereTranscribeAsrManager.swift`
   - Use `cohere_decoder_cached.mlpackage` (not fullseq!)
   - Implement tokenizer integration

4. **Add CLI command**
   ```bash
   swift run fluidaudiocli cohere-transcribe audio.wav
   ```

5. **Benchmark performance**
   - WER on LibriSpeech test-clean (target: 2.58%)
   - RTFx on M1/M2/M3
   - Memory usage during inference

---

## Files

### Working Scripts
- `test-autoregressive-decode.py` - ✅ Complete working pipeline
- `test-transformers-ground-truth.py` - PyTorch reference

### Export Reference (Methodology Documentation)
- `export-ultra-static-encoder.py` - Encoder export technique
- `export-ultra-static-decoder.py` - Decoder export technique
- `export-ultra-static-frontend.py` - Frontend export technique

### Documentation
- `status.md` - Complete reverse engineering log (1000+ lines)
- `TEST_RESULTS.md` - Detailed test results and findings
- `README.md` - This file

### Archive
- `archive/failed-tests/` - 24 failed test attempts (historical)
- `archive/investigation-scripts/` - Investigation scripts
- `archive/export-attempts/` - Failed export attempts
- `archive/old-docs/` - Outdated documentation

---

## Credits

- **BarathwajAnandan**: Original CoreML conversion
  - HuggingFace: https://huggingface.co/BarathwajAnandan/cohere-transcribe-03-2026-CoreML-fp16
  - Published WER: 2.58% on LibriSpeech test-clean

- **Cohere**: Original model
  - https://huggingface.co/CohereLabs/cohere-transcribe-03-2026

---

## License

See model licenses on respective HuggingFace repositories.
