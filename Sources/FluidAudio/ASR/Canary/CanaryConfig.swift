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
    /// per-channel layout). `int4` and `fp16` are documented as ANE-runnable,
    /// but large encoder weight layouts hang the ANE driver indefinitely on at
    /// least some Apple Silicon configurations (observed: M1 Pro / 16 GB /
    /// macOS 26.x with both `EncoderInt4.mlmodelc` and the 30-s `Encoder.mlmodelc`).
    /// CPU loads in <1s, ANE never returns. CPU_AND_GPU is the safe default —
    /// same ANE-bypass effect, still GPU-accelerated, loads in ~2s. Opt back
    /// into ANE on machines where it works (M2/M3+) via:
    ///   SCRIBION_CANARY_INT4_ANE=1   for int4
    ///   SCRIBION_CANARY_FP16_ANE=1   for fp16
    var computeUnits: MLComputeUnits {
        func env(_ key: String) -> Bool {
            let v = ProcessInfo.processInfo.environment[key]
            return v == "1" || v == "true"
        }
        switch self {
        case .int8: return .cpuOnly
        case .int4: return env("SCRIBION_CANARY_INT4_ANE") ? .cpuAndNeuralEngine : .cpuAndGPU
        case .fp16: return env("SCRIBION_CANARY_FP16_ANE") ? .cpuAndNeuralEngine : .cpuAndGPU
        }
    }
}

/// Fixed-shape contract for the canary-1b-v2 CoreML pipeline (40 s window).
///
/// Per NVIDIA's Canary technical report (Sec 6.4.1), Canary is trained on
/// 30–40 s utterances; the originally-published FluidInference 15 s variant
/// truncates each window mid-utterance, causing premature EOS and content
/// loss on long-form conversational audio. This re-converted variant ships a
/// 40 s window (the upper end of Sec 6.4.1's range — fewer seams across a
/// long-form clip than 30 s) with 1 s overlap.
/// See `mobius/models/stt/canary-1b-v2/coreml/convert-coreml.py`.
public enum CanaryConfig {
    public static let sampleRate = 16000
    /// 40 s window — preprocessor input is fixed at this sample count.
    public static let maxSamples = 640_000
    /// Overlap between adjacent windows when chunking audio longer than `maxSamples`.
    /// 1 s — NVIDIA Canary tech report Sec 6.4.1. (MLX path uses the same value.)
    public static let chunkOverlapSeconds = 1.0
    public static let chunkOverlapSamples = 16_000
    public static let melDim = 128
    /// 40 s × 16 kHz / hop 160 + 1 = 4001
    public static let melFrames = 4001
    public static let encoderHidden = 1024
    /// FastConformer subsamples mel by 8 → ceil(4001 / 8) = 501
    public static let encoderFrames = 501
    /// 256 covers a 40 s utterance (Canary's training max). `CanaryManager` reads
    /// the real length from the loaded model, so this is just the contract/fallback.
    public static let maxDecoderSteps = 256
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
