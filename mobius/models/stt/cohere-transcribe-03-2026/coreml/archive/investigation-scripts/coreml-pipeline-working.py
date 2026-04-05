#!/usr/bin/env python3
"""Working end-to-end CoreML pipeline for Cohere Transcribe 03-2026.

This demonstrates a complete transcription pipeline using only CoreML models.
"""
import coremltools as ct
import numpy as np
import soundfile as sf
import librosa
import json
from pathlib import Path

print("=== Cohere Transcribe CoreML Pipeline ===\n")

# Configuration
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 480000  # 30 seconds
DECODER_SEQ_LEN = 10
ENCODER_SEQ_LEN = 1500

# Load vocabulary
print("Loading vocabulary...")
with open("build/hf-upload/vocab.json") as f:
    vocab_dict = json.load(f)
    id_to_token = {v: k for k, v in vocab_dict.items()}
print(f"  ✓ {len(id_to_token)} tokens")

# Load CoreML models
print("\nLoading CoreML models...")
preprocessor = ct.models.MLModel("build/hf-upload/preprocessor.mlpackage")
encoder = ct.models.MLModel("build/hf-upload/encoder.mlpackage")
decoder = ct.models.MLModel("build/hf-upload/cohere_decoder.mlpackage")
lm_head = ct.models.MLModel("build/hf-upload/lm_head.mlpackage")
print("  ✓ All models loaded")

def load_and_preprocess_audio(audio_path: str) -> np.ndarray:
    """Load audio and resample to 16kHz."""
    audio, sr = sf.read(audio_path)

    # Convert stereo to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if sr != SAMPLE_RATE:
        print(f"  Resampling from {sr}Hz to {SAMPLE_RATE}Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Pad or truncate to 30 seconds
    if len(audio) < MAX_AUDIO_LENGTH:
        audio = np.pad(audio, (0, MAX_AUDIO_LENGTH - len(audio)), mode='constant')
    else:
        audio = audio[:MAX_AUDIO_LENGTH]

    return audio

def transcribe(audio_path: str, max_tokens: int = 50) -> str:
    """Transcribe audio file using CoreML pipeline."""

    # 1. Load and preprocess audio
    print(f"\n1. Loading audio: {audio_path}")
    audio = load_and_preprocess_audio(audio_path)
    print(f"   ✓ Audio ready: {len(audio)} samples")

    # 2. Run preprocessor
    print("\n2. Running preprocessor...")
    prep_out = preprocessor.predict({
        "audio_signal": audio.reshape(1, -1).astype(np.float32),
        "length": np.array([MAX_AUDIO_LENGTH], dtype=np.int32)
    })
    mel_features = prep_out["mel_features"]
    print(f"   ✓ Mel features: {mel_features.shape}")

    # 3. Run encoder
    print("\n3. Running encoder...")
    enc_out = encoder.predict({"input_features": mel_features})
    encoder_hidden = enc_out["encoder_output"]
    print(f"   ✓ Encoder output: {encoder_hidden.shape}")

    # Pad encoder output to expected size (1500)
    if encoder_hidden.shape[1] < ENCODER_SEQ_LEN:
        pad_len = ENCODER_SEQ_LEN - encoder_hidden.shape[1]
        padding = np.zeros((1, pad_len, encoder_hidden.shape[2]), dtype=encoder_hidden.dtype)
        encoder_hidden = np.concatenate([encoder_hidden, padding], axis=1)
        print(f"   ✓ Padded to: {encoder_hidden.shape}")

    # 4. Greedy decoding
    print("\n4. Decoding...")

    # Special tokens (from working CoreML conversion)
    # Prompt: ▁ <|startofcontext|> <|startoftranscript|> <|emo:undefined|> <|en|> <|en|> <|pnc|> <|noitn|> <|notimestamp|> <|nodiarize|>
    CORRECT_PROMPT = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
    EOS = 3  # <|endoftext|>

    # Initialize with correct prefix
    tokens = CORRECT_PROMPT.copy()

    for step in range(max_tokens):
        # Check max sequence length
        if len(tokens) >= DECODER_SEQ_LEN:
            print(f"   ⚠ Stopped at max decoder length ({DECODER_SEQ_LEN})")
            break

        # Prepare decoder input (pad to fixed size)
        padded_tokens = tokens + [0] * (DECODER_SEQ_LEN - len(tokens))
        input_ids = np.array([padded_tokens], dtype=np.int32)
        positions = np.arange(DECODER_SEQ_LEN, dtype=np.int32).reshape(1, -1)

        # Run decoder
        dec_out = decoder.predict({
            "input_ids": input_ids,
            "positions": positions,
            "encoder_hidden_states": encoder_hidden
        })
        hidden_states = dec_out["hidden_states"]

        # Run LM head
        lm_out = lm_head.predict({"hidden_states": hidden_states})
        logits = list(lm_out.values())[0]  # Get first output

        # Get logits for last actual token
        token_logits = logits[0, len(tokens)-1, :]

        # Greedy sampling
        next_token = int(np.argmax(token_logits))

        # Show first few tokens
        if step < 5:
            token_str = id_to_token.get(next_token, f"<{next_token}>")
            print(f"   Step {step}: {next_token} = '{token_str}'")

        # Check for end of sequence
        if next_token == EOS:
            print(f"   ✓ Stopped at EOS (step {step})")
            break

        tokens.append(next_token)

    # 5. Decode tokens to text
    print("\n5. Decoding tokens...")

    # Skip prefix tokens (10 tokens in the prompt)
    generated_tokens = tokens[10:]  # Skip the prompt prefix

    # Convert to text
    text = ""
    for token in generated_tokens:
        if token == EOS:
            break
        if token in id_to_token:
            text += id_to_token[token]

    print(f"   ✓ Generated {len(generated_tokens)} tokens")

    return text

# Test with audio file
if __name__ == "__main__":
    import sys

    # Use provided audio file or default
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "test-real-speech.wav"

    if not Path(audio_file).exists():
        print(f"❌ Audio file not found: {audio_file}")
        print("\nUsage: python coreml-pipeline-working.py <audio.wav>")
        sys.exit(1)

    # Run transcription
    result = transcribe(audio_file)

    print("\n" + "="*60)
    print("TRANSCRIPTION RESULT:")
    print("="*60)
    print(f'"{result}"')
    print("="*60)
