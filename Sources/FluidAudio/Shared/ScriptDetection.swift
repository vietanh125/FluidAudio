import Foundation

/// Supported languages for script-based filtering
public enum Language: String, Sendable, CaseIterable {
    /// Latin-script languages — Germanic/Romance
    case english = "en"
    case spanish = "es"
    case french = "fr"
    case german = "de"
    case italian = "it"
    case portuguese = "pt"
    case romanian = "ro"  // uses ș, ț (Latin Extended-B)

    /// Latin-script Slavic languages — prone to Cyrillic confusion in multilingual ASR
    case polish = "pl"
    case czech = "cs"
    case slovak = "sk"
    case slovenian = "sl"
    case croatian = "hr"
    case bosnian = "bs"

    /// Cyrillic-script languages
    case russian = "ru"
    case ukrainian = "uk"
    case belarusian = "be"
    case bulgarian = "bg"
    case serbian = "sr"

    /// Returns the writing script used by this language
    public var script: Script {
        switch self {
        case .english, .spanish, .french, .german, .italian, .portuguese, .romanian,
            .polish, .czech, .slovak, .slovenian, .croatian, .bosnian:
            return .latin
        case .russian, .ukrainian, .belarusian, .bulgarian, .serbian:
            return .cyrillic
        }
    }
}

/// Writing script categories
public enum Script: Sendable {
    case latin
    case cyrillic
}

/// Script detection utilities for token filtering
public struct ScriptDetection: Sendable {

    /// Check if text matches a specific script
    ///
    /// - Parameters:
    ///   - text: Text to check
    ///   - script: Target script (Latin or Cyrillic)
    ///
    /// - Returns: True if all characters in the text match the target script
    public static func matches(_ text: String, script: Script) -> Bool {
        // Strip SentencePiece word boundary marker (▁ U+2581) before checking
        // This character is prepended to most tokens but doesn't indicate script
        let cleanedText = text.replacingOccurrences(of: "\u{2581}", with: "")

        // Empty after stripping boundary markers means no actual content to check
        guard !cleanedText.isEmpty else { return false }

        let chars = cleanedText.unicodeScalars
        switch script {
        case .latin:
            return chars.allSatisfy {
                ($0.value >= 0x0020 && $0.value <= 0x007F)  // ASCII
                    || ($0.value >= 0x00A0 && $0.value <= 0x00FF)  // Latin-1
                    || ($0.value >= 0x0100 && $0.value <= 0x017F)  // Latin Extended-A
                    || ($0.value >= 0x0180 && $0.value <= 0x024F)  // Latin Extended-B (Romanian ș ț, etc.)
                    || ($0.value >= 0x1E00 && $0.value <= 0x1EFF)  // Latin Extended Additional (Vietnamese, etc.)
            }
        case .cyrillic:
            return chars.allSatisfy { char in
                let value = char.value
                // Allow Cyrillic characters
                if value >= 0x0400 && value <= 0x04FF {
                    return true
                }
                // Allow spaces, punctuation, and digits (but NOT Latin letters)
                // ASCII letters are 0x41-0x5A (A-Z) and 0x61-0x7A (a-z)
                if value >= 0x0020 && value <= 0x007F {
                    // Reject ASCII letters
                    if (value >= 0x41 && value <= 0x5A) || (value >= 0x61 && value <= 0x7A) {
                        return false
                    }
                    return true  // Allow other ASCII (spaces, punctuation, digits)
                }
                return false
            }
        }
    }

    /// Filter top-K candidates by script and return the highest-probability match.
    ///
    /// The returned probability is computed via softmax over the top-K logits so the
    /// caller receives a value in [0, 1] suitable for use as a confidence score.
    /// This approximates the true probability by assuming the top-K captures most of
    /// the probability mass — a reasonable assumption for K=10 in an 8k vocabulary
    /// when the model is reasonably confident.
    ///
    /// - Parameters:
    ///   - topKIds: Array of token IDs (from `top_k_ids` output).
    ///   - topKLogits: Array of logits (from `top_k_logits` output). Must be the
    ///     same length as `topKIds`.
    ///   - vocabulary: Mapping from token IDs to text.
    ///   - preferredScript: Script to filter for.
    ///
    /// - Returns: Token ID and probability (in [0, 1]) of the highest-probability
    ///   token matching the script, or `nil` if no match is found or the input
    ///   arrays are mismatched/empty.
    public static func filterTopK(
        topKIds: [Int],
        topKLogits: [Float],
        vocabulary: [Int: String],
        preferredScript: Script
    ) -> (tokenId: Int, probability: Float)? {
        // Guard against array length mismatch — CoreML output arrays are read
        // independently in TdtModelInference, so enforce the invariant here.
        let count = min(topKIds.count, topKLogits.count)
        guard count > 0 else { return nil }

        for idx in 0..<count {
            let tokenId = topKIds[idx]
            guard let tokenText = vocabulary[tokenId] else { continue }
            guard matches(tokenText, script: preferredScript) else { continue }

            let probability = softmaxProbability(of: idx, logits: topKLogits, count: count)
            return (tokenId, probability)
        }
        return nil
    }

    /// Compute softmax probability of `index` within a slice of `logits` of length `count`.
    /// Uses the max-logit stability trick to avoid overflow.
    private static func softmaxProbability(of index: Int, logits: [Float], count: Int) -> Float {
        var maxLogit = -Float.infinity
        for i in 0..<count where logits[i] > maxLogit {
            maxLogit = logits[i]
        }
        guard maxLogit.isFinite else { return 0 }

        var sumExp: Float = 0
        for i in 0..<count {
            sumExp += expf(logits[i] - maxLogit)
        }
        guard sumExp > 0 else { return 0 }

        let numerator = expf(logits[index] - maxLogit)
        let prob = numerator / sumExp
        return max(0, min(1, prob))
    }
}
