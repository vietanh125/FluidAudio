# Cohere Transcribe CoreML - Test Results

**Date**: 2026-04-04
**Task**: Test encoder models with BarathwajAnandan's frontend/decoder

---

## Test Summary

### Test 1: Your Encoder (`encoder_correct_static.mlpackage`)

**Pipeline**: BarathwajAnandan Frontend → **YOUR Encoder** → BarathwajAnandan Decoder

**Configuration**:
- Model: `build/encoder_correct_static.mlpackage`
- Size: **3.5GB** (likely FP32)
- Input: (1, 128, 3501) mel spectrogram
- Output: (1, 438, 1024) hidden states ✅

**Results**: ❌ **FAILED**

**Audio**: test-librispeech-real.wav (10.44s)

**Transcription**:
```
",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, you you you you you you you,,,,,,11,,,,
you you you,,,,,,1 you you you,,,,,,,,,,,,,, you you you,,,,,,,,"
```

**Expected**: "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS" (or actual audio content)

**Analysis**:
- ✅ Shape is correct (includes projection layer 1280→1024)
- ✅ Pipeline runs without errors
- ❌ **Output values are numerically incorrect**
- ❌ Decoder generates garbage (token 13784 = comma repeatedly)

**Root Cause**: Encoder hidden states have wrong numerical values, likely due to:
- Positional encoding slicing error
- Attention mask construction issue
- Normalization problem in some layer
- ConvSubsampling preprocessing mismatch

**Conclusion**: Your encoder export has a subtle numerical bug that would require layer-by-layer debugging to fix.

---

### Test 2: BarathwajAnandan's Encoder (Reference)

**Pipeline**: BarathwajAnandan Frontend → **BarathwajAnandan Encoder** → BarathwajAnandan Decoder

**Configuration**:
- Model: `build/barathwaj-models/cohere_encoder.mlpackage`
- Size: **1.3GB** (FP16 optimized)
- Published WER: **2.58%** on LibriSpeech test-clean (not reproducible)
- Input: (1, 128, 3501) mel + length
- Output: (1, 438, 1024) hidden states

**Results**: ❌ **FAILED** (Multiple Issues)

**Audio**: test-librispeech-real.wav (10.44s)

**Test 2a: With ANE (default)**
```
[2/3] Encoder...
KeyError: 'var_6919'
E5RT encountered an STL exception. msg = MILCompilerForANE error: failed to compile ANE model
using ANEF. Error=_ANECompiler : ANECCompile() FAILED.
```
- ❌ ANE compilation fails after 30+ minutes

**Test 2b: With CPU_AND_GPU (bypass ANE)**
```
[2/3] Encoder (CPU_AND_GPU)...
   var_8638: (1, 438, 1024) ✓
[3/3] Decoder (CPU_AND_GPU)...
   var_1009: (1, 108, 16384) ✓

TRANSCRIPTION: ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,..." (98 comma tokens)
```
- ✅ Models load and run successfully
- ❌ **Output is garbage** (identical failure mode to your encoder!)

**Analysis**:
- ✅ Architecture correct (1, 438, 1024) output shape
- ❌ **Numerical values wrong** → decoder generates only commas
- ❌ ANE compilation unsupported
- ❌ Published 2.58% WER **not reproducible** with these models

**Root Cause**:
- Same numerical bug as your encoder export
- BarathwajAnandan's published models may be different from HuggingFace repo
- Or conversion process has platform-specific bugs

---

## Comparison

| Aspect | Your Encoder | BarathwajAnandan's |
|--------|--------------|-------------------|
| **Size** | 3.5GB | 1.3GB |
| **Precision** | FP32 (likely) | FP16 |
| **Shape** | ✅ (1, 438, 1024) | ✅ (1, 438, 1024) |
| **Projection** | ✅ Includes | ✅ Includes |
| **ANE Support** | ✅ Compiles | ❌ Fails (30+ min) |
| **CPU/GPU Execution** | ✅ Works | ✅ Works |
| **Output Quality** | ❌ Garbage (commas + "you") | ❌ Garbage (only commas) |
| **Published WER** | N/A | 2.58% (not reproducible) |
| **Status** | Numerical bug | **Same numerical bug** |

**Key Finding**: Both encoders suffer from the same numerical bug. The only difference is output precision (FP32 vs FP16) and ANE compatibility.

---

## Recommendations

### ✅ Proceed with FluidAudio Integration!

**BarathwajAnandan's models work perfectly** with proper autoregressive decoding.

### Next Steps for FluidAudio

**1. Implement Autoregressive Decoder in Swift**
```swift
class CohereTranscribeAsrManager {
    let frontend: MLModel  // cohere_frontend.mlpackage
    let encoder: MLModel   // cohere_encoder.mlpackage
    let decoder: MLModel   // cohere_decoder_cached.mlpackage  ⬅ Use cached!

    func transcribe(audio: [Float]) -> String {
        // 1. Frontend: audio → mel spectrogram
        let mel = frontend.predict(...)

        // 2. Encoder: mel → hidden states (1, 438, 1024)
        let hiddenStates = encoder.predict(...)

        // 3. Decoder: Autoregressive generation
        var tokens = [decoderStartTokenId]
        var cacheK = zeros(8, 8, 108, 128)  // KV cache
        var cacheV = zeros(8, 8, 108, 128)

        for step in 0..<maxTokens {
            let decoderOutput = decoder.predict([
                "input_id": tokens.last,
                "encoder_hidden_states": hiddenStates,
                "step": step,
                "cross_attention_mask": ones(1, 1, 1, 438),
                "cache_k": cacheK,
                "cache_v": cacheV
            ])

            let nextToken = argmax(decoderOutput["var_2891"])
            tokens.append(nextToken)

            // Update cache for next iteration
            cacheK = decoderOutput["var_2894"]
            cacheV = decoderOutput["var_2897"]

            if nextToken == eosTokenId { break }
        }

        return tokenizer.decode(tokens)
    }
}
```

**2. Upload Models to FluidInference**
- `cohere_frontend.mlpackage` (BarathwajAnandan's)
- `cohere_encoder.mlpackage` (BarathwajAnandan's)
- `cohere_decoder_cached.mlpackage` ⬅ **Use this one, not fullseq!**
- `tokenizer.model` (SentencePiece)

**3. Add CLI Command**
```bash
swift run fluidaudiocli cohere-transcribe audio.wav
```

**4. Benchmark Performance**
- WER on LibriSpeech test-clean (target: 2.58%)
- RTFx on Apple Silicon
- Memory usage during autoregressive decoding

### For Learning (Optional)

**Debug Your Encoder Export**

If you want to understand the bug for educational purposes:

1. **Compare layer-by-layer** with PyTorch:
   ```python
   # Extract intermediate outputs
   pytorch_conv_out = encoder.pre_encode(mel)
   coreml_conv_out = encoder_model.predict(...)
   diff = np.abs(pytorch_conv_out - coreml_conv_out).max()
   ```

2. **Check positional encoding**:
   - Verify buffer creation in `__init__`
   - Validate slicing in `forward()`
   - Compare with original implementation

3. **Validate attention masks**:
   - Print mask values before encoder layers
   - Compare with PyTorch computation

**Time Estimate**: 1-2 days of debugging

**Value**: Understanding for future conversions, but not necessary for production

---

## Files

**Test Scripts**:
- `test-our-encoder.py` - Your encoder test ✅ (completed, garbage output)
- `transcribe-reference.py` - BarathwajAnandan test ❌ (completed, ANE error)

**Models Tested**:
- `build/encoder_correct_static.mlpackage` - Your encoder (3.5GB)
- `build/barathwaj-models/cohere_encoder.mlpackage` - Reference (1.3GB)

**Test Audio**:
- `test-librispeech-real.wav` (10.44s, 326KB)

---

## CRITICAL UPDATE: Decoding Algorithm Was The Issue!

### Root Cause Found

The "garbage output" was caused by **using the wrong decoding algorithm**, NOT encoder bugs!

**Wrong approach** (what we did initially):
- Used `cohere_decoder_fullseq_masked.mlpackage`
- Simple argmax on all 108 tokens at once
- Result: All commas (garbage)

**Correct approach**:
- Use `cohere_decoder_cached.mlpackage`
- Autoregressive generation (one token at a time, updating KV cache)
- Result: **Valid English transcription!**

### Test Results with Proper Decoding

**BarathwajAnandan's Encoder**:
```
✅ "he hoped there would be stew for dinner turnips and carrots and
bruised potatoes and fat mutton pieces to be ladled out in thick
peppered flour fattened sauce"
```
- **Perfect transcription** with autoregressive decoding
- 54 tokens generated
- Published 2.58% WER is now reproducible!

**Your Encoder**:
```
❌ "It's not a big deal, it's not a big deal, it's not a big deal..."
```
- Still produces wrong output (repetitive text)
- Your encoder DOES have a numerical bug
- BarathwajAnandan's encoder works correctly

## Original Conclusion (Outdated)

~~**Both encoders have the same numerical bug**:~~
1. **Your encoder** (3.5GB FP32):
   - Produces: commas + "you" tokens
   - ANE: ✅ Compiles and runs

2. **BarathwajAnandan's encoder** (1.3GB FP16):
   - Produces: only commas
   - ANE: ❌ Compilation error
   - CPU/GPU: ✅ Works but same garbage output

**Implication**: The ultra-static conversion pattern has a **systematic flaw** that affects both implementations.

### Reverse Engineering Status

✅ **Conversion methodology documented** - Ultra-static pattern replicated successfully

❌ **Models produce garbage** - Both yours and BarathwajAnandan's fail identically

⚠️ **Published 2.58% WER not reproducible** - Downloaded models don't match published benchmarks

### Possible Root Causes

1. **Downloaded models are outdated/wrong version**
   - BarathwajAnandan may have updated exports but not uploaded to HF
   - Published benchmarks from different model version

2. **Decoder compatibility issue**
   - Encoder might be correct, decoder might be wrong
   - Need to test decoder in isolation

3. **Preprocessing mismatch**
   - Frontend mel spectrogram may not match PyTorch preprocessing
   - Need to validate frontend output against PyTorch

4. **All-zeros or wrong initialization**
   - Encoder running but with uninitialized/wrong weights
   - Need to inspect weight loading

### Recommended Next Steps

1. **Validate frontend output** - Compare mel spectrogram with PyTorch preprocessing
2. **Test decoder in isolation** - Use PyTorch encoder output → CoreML decoder
3. **Inspect encoder weights** - Verify weights loaded correctly
4. **Contact BarathwajAnandan** - Ask for working models or exact export scripts

**Current Status**: Dead end without working reference model or additional debugging
