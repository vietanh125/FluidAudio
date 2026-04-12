import Foundation

/// Supported languages for script-based filtering
public enum Language: String, Sendable, CaseIterable {
    /// Latin-script languages
    case english = "en"
    case polish = "pl"
    case spanish = "es"
    case french = "fr"
    case german = "de"
    case italian = "it"
    case portuguese = "pt"

    /// Cyrillic-script languages
    case russian = "ru"
    case ukrainian = "uk"
    case belarusian = "be"
    case bulgarian = "bg"
    case serbian = "sr"

    /// Returns the writing script used by this language
    public var script: Script {
        switch self {
        case .english, .polish, .spanish, .french, .german, .italian, .portuguese:
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

    /// Filter top-K candidates by script and return the highest-probability match
    ///
    /// - Parameters:
    ///   - topKIds: Array of token IDs (from top_k_ids output)
    ///   - topKLogits: Array of logits (from top_k_logits output)
    ///   - vocabulary: Mapping from token IDs to text
    ///   - preferredScript: Script to filter for
    ///
    /// - Returns: Token ID and logit of the highest-probability token matching the script,
    ///            or nil if no match found
    public static func filterTopK(
        topKIds: [Int],
        topKLogits: [Float],
        vocabulary: [Int: String],
        preferredScript: Script
    ) -> (tokenId: Int, logit: Float)? {
        for (idx, tokenId) in topKIds.enumerated() {
            guard let tokenText = vocabulary[tokenId] else {
                continue
            }

            if matches(tokenText, script: preferredScript) {
                return (tokenId, topKLogits[idx])
            }
        }
        return nil
    }
}
