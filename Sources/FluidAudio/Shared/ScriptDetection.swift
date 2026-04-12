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
        let chars = text.unicodeScalars
        switch script {
        case .latin:
            return chars.allSatisfy {
                ($0.value >= 0x0020 && $0.value <= 0x007F)  // ASCII
                    || ($0.value >= 0x00A0 && $0.value <= 0x00FF)  // Latin-1
                    || ($0.value >= 0x0100 && $0.value <= 0x017F)  // Latin Extended-A
            }
        case .cyrillic:
            return chars.allSatisfy {
                ($0.value >= 0x0400 && $0.value <= 0x04FF)  // Cyrillic
                    || ($0.value >= 0x0020 && $0.value <= 0x007F)  // ASCII (spaces, punctuation)
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
