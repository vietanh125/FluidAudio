@preconcurrency import CoreML
import Foundation

/// Canary encoder/decoder weight precision.
///
/// `int4` (per-block-32 symmetric) runs on the Neural Engine and is the
/// smallest build (~573 MB) — but int4 weight payloads require iOS18 / macOS 15.
/// `fp16` is the iOS17 parity fallback (exact match to PyTorch). `int8`
/// (per-channel) decodes correctly only on CPU — it crashes the GPU/ANE MPSGraph
/// backend — so it is a CPU/size-only option.
public enum CanaryPrecision: String, Sendable, CaseIterable {
    case int4
    case fp16
    case int8

    var encoderName: String {
        switch self {
        case .int4: return ModelNames.Canary.encoderInt4
        case .fp16: return ModelNames.Canary.encoder
        case .int8: return ModelNames.Canary.encoderInt8
        }
    }

    var decoderName: String {
        switch self {
        case .int4: return ModelNames.Canary.decoderInt4
        case .fp16: return ModelNames.Canary.decoder
        case .int8: return ModelNames.Canary.decoderInt8
        }
    }

    /// Compute units per precision.
    ///
    /// `int8` only decodes correctly on CPU (the MPSGraph backend crashes on the
    /// per-channel layout). `int4` is documented as ANE-runnable, but the
    /// `EncoderInt4` weight layout hangs the ANE driver indefinitely on at least
    /// some Apple Silicon configurations (observed on M1 Pro / 16 GB / macOS
    /// 26.x): CPU loads in 0.2s, ANE never returns. CPU_AND_GPU is the safe
    /// default — same ANE-bypass effect, still GPU-accelerated, loads in ~2s.
    /// Set `SCRIBION_CANARY_INT4_ANE=1` to opt back into ANE on machines where
    /// it works (M2/M3+).
    var computeUnits: MLComputeUnits {
        switch self {
        case .int8: return .cpuOnly
        case .int4:
            let force = ProcessInfo.processInfo.environment["SCRIBION_CANARY_INT4_ANE"]
            return (force == "1" || force == "true") ? .cpuAndNeuralEngine : .cpuAndGPU
        case .fp16: return .cpuAndNeuralEngine
        }
    }
}

/// Fixed-shape contract for the canary-1b-v2 CoreML pipeline (15 s window).
public enum CanaryConfig {
    public static let sampleRate = 16000
    /// 15 s window — the preprocessor input is fixed at this sample count.
    public static let maxSamples = 240_000
    /// Overlap between adjacent windows when chunking audio longer than 15 s.
    /// 3 s (~19 tokens) gives the seam LCS-merge enough shared context to align
    /// reliably while wasting little recompute. Hop = maxSamples − this.
    public static let chunkOverlapSeconds = 3.0
    public static let chunkOverlapSamples = 48_000
    public static let melDim = 128
    public static let melFrames = 1501
    public static let encoderHidden = 1024
    public static let encoderFrames = 188
    /// Decoder is exported at a fixed `[1, maxDecoderSteps]`. 128 covers a 15 s
    /// window (max observed ~108 tokens incl. prompt) and is ~1.5× faster than 256.
    /// `CanaryManager` reads the real length from the loaded model, so this is just
    /// the contract/fallback value.
    public static let maxDecoderSteps = 128
    public static let vocabSize = 16384

    // Special token ids (the model's real decoder ids — see vocab.json).
    public static let eosId = 3  // <|endoftext|>
    public static let padId = 2  // <pad>
    public static let bosId = 4  // <|startoftranscript|>

    /// canary2 prompt for English transcribe + punctuation/capitalization:
    /// ▁ <|startofcontext|> <|startoftranscript|> <|emo:undefined|> <|en|> <|en|>
    /// <|pnc|> <|noitn|> <|notimestamp|> <|nodiarize|>
    public static let promptEnTranscribePnc: [Int32] = [16053, 7, 4, 16, 64, 64, 5, 9, 11, 13]

    /// canary2 prompt for German transcribe + punctuation/capitalization.
    /// Source and target both <|de|> (id=78); otherwise identical to the English
    /// prompt. Use when transcribing German audio; mixing source/target produces
    /// translation, which is rarely what callers want.
    public static let promptDeTranscribePnc: [Int32] = [16053, 7, 4, 16, 78, 78, 5, 9, 11, 13]

    /// Build a transcribe-PnC prompt for the given ISO-639-1 language code.
    /// Returns the English prompt for unrecognised codes (parity with how
    /// `transcribe(audio:)` without a language hint behaves today).
    public static func promptTranscribePnc(forLanguage language: String) -> [Int32] {
        switch language.lowercased() {
        case "de": return promptDeTranscribePnc
        case "en": return promptEnTranscribePnc
        default:   return promptEnTranscribePnc
        }
    }
}
