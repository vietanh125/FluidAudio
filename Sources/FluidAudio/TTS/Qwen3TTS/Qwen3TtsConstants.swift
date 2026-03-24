import Foundation

/// Constants for the Qwen3-TTS 6-model CoreML pipeline.
public enum Qwen3TtsConstants {

    // MARK: - Audio

    public static let audioSampleRate: Int = 24_000

    /// Audio samples per codec frame (80ms at 24kHz).
    public static let samplesPerFrame: Int = 1_920

    // MARK: - Model dimensions

    public static let hiddenSize: Int = 1024
    public static let numCodebooks: Int = 16
    public static let codecVocabSize: Int = 2048

    // MARK: - CodeDecoder KV cache

    /// Fixed KV cache sequence length for CodeDecoder.
    /// key_cache / value_cache shape: [1, 28672, 1, 256] float16
    public static let cdKvLen: Int = 256

    /// Consolidated KV dimension for CodeDecoder (28 layers).
    public static let cdKvDim: Int = 28_672

    // MARK: - MultiCodeDecoder KV cache

    /// Fixed KV cache sequence length for MultiCodeDecoder.
    /// key_cache / value_cache shape: [1, 5120, 1, 16] float16
    public static let mcdKvLen: Int = 16

    /// Consolidated KV dimension for MultiCodeDecoder (5 layers).
    public static let mcdKvDim: Int = 5_120

    // MARK: - Codec special token IDs

    public static let codecPadId: Int = 2148
    public static let codecBosId: Int = 2149
    public static let codecEosId: Int = 2150
    public static let codecThinkId: Int = 2154
    public static let codecNoThinkId: Int = 2155
    public static let codecThinkBosId: Int = 2156
    public static let codecThinkEosId: Int = 2157

    // MARK: - Language IDs

    public static let languageIds: [String: Int] = [
        "english": 2050,
        "chinese": 2055,
        "german": 2053,
        "italian": 2070,
        "portuguese": 2071,
        "spanish": 2054,
        "japanese": 2058,
        "korean": 2064,
        "french": 2061,
        "russian": 2069,
    ]

    // MARK: - TTS special token IDs

    public static let ttsPadTokenId: Int = 151_671
    public static let ttsBosTokenId: Int = 151_672
    public static let ttsEosTokenId: Int = 151_673

    // MARK: - Role prefix tokens

    /// [im_start, assistant, newline]
    public static let rolePrefixTokens: [Int] = [151_644, 77_091, 198]

    // MARK: - Generation parameters

    public static let maxCodecTokens: Int = 125
    public static let minNewTokens: Int = 2

    // MARK: - CB0 sampling (temperature + top-k, matching Argmax)

    /// Base repetition penalty for CB0 tokens (divides logits of seen tokens).
    /// Set to 1.05 to match Argmax's default.
    public static let cb0RepetitionPenalty: Float = 1.05

    /// Frequency penalty: penalizes tokens proportional to how often they've appeared.
    /// Set to 0.0 (disabled) to match Argmax's approach.
    public static let cb0FrequencyPenalty: Float = 0.0

    /// Presence penalty: applies fixed penalty to any token that appeared at least once.
    /// Set to 0.0 (disabled) to match Argmax's approach.
    public static let cb0PresencePenalty: Float = 0.0

    /// Temperature for CB0 sampling (0.9 matches Argmax's default).
    /// Higher = more diverse output, lower = more deterministic.
    public static let cb0Temperature: Float = 0.9

    /// Top-k for CB0 sampling (50 matches Argmax's default).
    /// Only the top-k highest probability tokens are considered.
    public static let cb0TopK: Int = 50

    /// Random seed for deterministic sampling (nil = random, 42 = Argmax's default).
    /// Set to 42 for reproducible results matching Argmax.
    public static let randomSeed: UInt64? = nil

    // MARK: - CB1-15 sampling (temperature-based for code predictor)

    /// Temperature for code predictor (CB1-15 generation).
    /// Lowered from 0.9 to 0.7 for more focused sampling.
    public static let codeTemperature: Float = 0.7

    /// Top-K filtering for code predictor.
    /// Increased from 50 to 100 for better diversity without excessive randomness.
    public static let codeTopK: Int = 100

    /// Top-P (nucleus sampling) for code predictor.
    /// Keeps tokens with cumulative probability >= topP (0.95 = keep 95% probability mass).
    public static let codeTopP: Float = 0.95

    // MARK: - Audio post-processing

    /// Enable audio post-processing (de-esser, filtering, normalization).
    public static let enablePostProcessing: Bool = true

    /// De-esser threshold in dB (target sibilance above this level).
    public static let deEsserThresholdDb: Float = -20.0

    /// De-esser compression ratio (how much to reduce sibilance).
    public static let deEsserRatio: Float = 3.0

    /// De-esser frequency range (Hz) - targets sibilance frequencies.
    public static let deEsserLowFreq: Float = 5000.0
    public static let deEsserHighFreq: Float = 10_000.0

    /// Low-pass filter cutoff (Hz) - removes harsh high frequencies.
    public static let lowPassCutoffHz: Float = 16_000.0

    /// Target loudness for normalization (LUFS).
    /// -23 LUFS is broadcast standard, -16 LUFS for podcasts/music.
    public static let targetLufs: Float = -23.0

    // MARK: - SpeechDecoder

    /// Fixed input time dimension for SpeechDecoder: [1, 16, 125].
    public static let speechDecoderFrames: Int = 125

    // MARK: - Defaults

    public static let defaultVoice: String = "default"
    public static let defaultLanguage: String = "english"
}

// MARK: - Sampling Configuration

/// Configuration for Qwen3-TTS sampling parameters.
///
/// Allows fine-tuning of:
/// - CB0 penalties (repetition, frequency, presence)
/// - CB1-15 temperature/top-k/top-p
/// - Audio post-processing
public struct Qwen3TtsSamplingConfig: Sendable {
    // CB0 sampling parameters
    public var cb0RepetitionPenalty: Float
    public var cb0FrequencyPenalty: Float
    public var cb0PresencePenalty: Float
    public var cb0Temperature: Float
    public var cb0TopK: Int

    // CB1-15 sampling
    public var codeTemperature: Float
    public var codeTopK: Int
    public var codeTopP: Float

    // Audio post-processing
    public var enablePostProcessing: Bool
    public var deEsserThresholdDb: Float
    public var deEsserRatio: Float
    public var lowPassCutoffHz: Float
    public var targetLufs: Float

    /// Default configuration using values from Qwen3TtsConstants.
    public static let `default` = Qwen3TtsSamplingConfig(
        cb0RepetitionPenalty: Qwen3TtsConstants.cb0RepetitionPenalty,
        cb0FrequencyPenalty: Qwen3TtsConstants.cb0FrequencyPenalty,
        cb0PresencePenalty: Qwen3TtsConstants.cb0PresencePenalty,
        cb0Temperature: Qwen3TtsConstants.cb0Temperature,
        cb0TopK: Qwen3TtsConstants.cb0TopK,
        codeTemperature: Qwen3TtsConstants.codeTemperature,
        codeTopK: Qwen3TtsConstants.codeTopK,
        codeTopP: Qwen3TtsConstants.codeTopP,
        enablePostProcessing: Qwen3TtsConstants.enablePostProcessing,
        deEsserThresholdDb: Qwen3TtsConstants.deEsserThresholdDb,
        deEsserRatio: Qwen3TtsConstants.deEsserRatio,
        lowPassCutoffHz: Qwen3TtsConstants.lowPassCutoffHz,
        targetLufs: Qwen3TtsConstants.targetLufs
    )

    public init(
        cb0RepetitionPenalty: Float = Qwen3TtsConstants.cb0RepetitionPenalty,
        cb0FrequencyPenalty: Float = Qwen3TtsConstants.cb0FrequencyPenalty,
        cb0PresencePenalty: Float = Qwen3TtsConstants.cb0PresencePenalty,
        cb0Temperature: Float = Qwen3TtsConstants.cb0Temperature,
        cb0TopK: Int = Qwen3TtsConstants.cb0TopK,
        codeTemperature: Float = Qwen3TtsConstants.codeTemperature,
        codeTopK: Int = Qwen3TtsConstants.codeTopK,
        codeTopP: Float = Qwen3TtsConstants.codeTopP,
        enablePostProcessing: Bool = Qwen3TtsConstants.enablePostProcessing,
        deEsserThresholdDb: Float = Qwen3TtsConstants.deEsserThresholdDb,
        deEsserRatio: Float = Qwen3TtsConstants.deEsserRatio,
        lowPassCutoffHz: Float = Qwen3TtsConstants.lowPassCutoffHz,
        targetLufs: Float = Qwen3TtsConstants.targetLufs
    ) {
        self.cb0RepetitionPenalty = cb0RepetitionPenalty
        self.cb0FrequencyPenalty = cb0FrequencyPenalty
        self.cb0PresencePenalty = cb0PresencePenalty
        self.cb0Temperature = cb0Temperature
        self.cb0TopK = cb0TopK
        self.codeTemperature = codeTemperature
        self.codeTopK = codeTopK
        self.codeTopP = codeTopP
        self.enablePostProcessing = enablePostProcessing
        self.deEsserThresholdDb = deEsserThresholdDb
        self.deEsserRatio = deEsserRatio
        self.lowPassCutoffHz = lowPassCutoffHz
        self.targetLufs = targetLufs
    }
}
