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

/// Writing script categories.
///
/// The Unicode ranges associated with each case are designed to be
/// **non-overlapping**: no single character can match both `.latin` and
/// `.cyrillic` as letters. This invariant is what lets `matches` use a
/// simple `allSatisfy` per-script check without a "reject the other script"
/// guard on both sides — see `TokenLanguageFilter.matches(_:script:)`.
public enum Script: Sendable {
    case latin
    case cyrillic
}

/// Filters ASR decoder tokens against a target language's writing script.
///
/// Used by the v3 TDT decoder to suppress cross-script leakage (e.g. Parakeet
/// emitting Cyrillic tokens for a short Polish utterance) by walking the CoreML
/// joint decoder's top-K output and picking the highest-ranked candidate whose
/// text is in the expected script. Currently partitions by Unicode script only;
/// per-language token allowlists (e.g. distinguishing Polish from Czech) could
/// plug in here later without changing the call-site API.
///
/// The surface here is intentionally `internal`: `matches` and `filterTopK` are
/// only meaningful when you have raw top-K ids/logits plus a CoreML vocab map,
/// i.e. inside the decoder path. The public language hook is `Language` + the
/// `language:` parameter on the transcribe APIs.
internal struct TokenLanguageFilter: Sendable {

    /// SentencePiece word-boundary marker (▁, U+2581). Prepended to most tokens
    /// as a whitespace indicator; it carries no script information and is
    /// stripped before script checks.
    private static let sentencePieceBoundary: Unicode.Scalar = "\u{2581}"

    /// Check whether every character in `text` is compatible with the given script.
    ///
    /// - Parameters:
    ///   - text: Text to check.
    ///   - script: Target script (Latin or Cyrillic).
    ///
    /// - Returns: `true` if every character (after stripping the SentencePiece
    ///   word-boundary marker) belongs to an allowed range for `script`.
    ///
    /// ## Allowed ranges
    ///
    /// **Latin:** ASCII, Latin-1, Latin Extended-A/B, Latin Extended Additional.
    /// Digits (0x30–0x39), ASCII punctuation, and whitespace fall inside ASCII and
    /// are therefore allowed — treated as script-neutral. The Latin path does **not**
    /// need an explicit "reject Cyrillic" guard because the Cyrillic block
    /// (U+0400–U+04FF) does not overlap any Latin range; `allSatisfy` over the Latin
    /// ranges already fails for any Cyrillic scalar.
    ///
    /// **Cyrillic:** the Cyrillic block (U+0400–U+04FF) plus script-neutral ASCII
    /// (space, digits, punctuation) — but ASCII letters A–Z/a–z are explicitly
    /// rejected. This asymmetry exists because ASCII overlaps with the Latin range,
    /// so Cyrillic *does* need an explicit Latin-letter rejection to avoid
    /// accidentally accepting tokens like "cat".
    static func matches(_ text: String, script: Script) -> Bool {
        // Strip SentencePiece word boundary marker before checking — it carries
        // no script information and would otherwise fail every range check.
        let cleanedText = text.replacingOccurrences(of: String(sentencePieceBoundary), with: "")

        // Empty after stripping boundary markers means no actual content to check
        guard !cleanedText.isEmpty else { return false }

        let chars = cleanedText.unicodeScalars
        switch script {
        case .latin:
            // No reverse Cyrillic guard needed: the Cyrillic block is outside every
            // Latin range below, so `allSatisfy` naturally rejects Cyrillic characters.
            return chars.allSatisfy {
                ($0.value >= 0x0020 && $0.value <= 0x007F)  // ASCII (incl. digits, punct)
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
                // Allow script-neutral ASCII (digits, punctuation, whitespace) but
                // explicitly reject ASCII letters — unlike Cyrillic, ASCII overlaps
                // with the Latin range, so the asymmetric guard is required.
                // ASCII letters are 0x41-0x5A (A-Z) and 0x61-0x7A (a-z).
                if value >= 0x0020 && value <= 0x007F {
                    if (value >= 0x41 && value <= 0x5A) || (value >= 0x61 && value <= 0x7A) {
                        return false
                    }
                    return true
                }
                return false
            }
        }
    }

    /// Filter top-K candidates by script and return the highest-logit match.
    ///
    /// Walks every entry in `topKIds`/`topKLogits`, keeps the one with the greatest
    /// logit whose text matches `preferredScript`, and returns it together with a
    /// softmax probability. The scan is explicit argmax — we do **not** assume the
    /// CoreML top-K output is sorted by logit.
    ///
    /// ## Probability semantics
    ///
    /// The returned probability is softmax **over the top-K logits only**, not over
    /// the full vocabulary. Because the denominator excludes ~vocab-size−K terms,
    /// this value is systematically larger than a full-vocab softmax. It is
    /// intended for relative ranking inside the filtered path and for bounded
    /// confidence-score consumers, not as a drop-in replacement for a full-vocab
    /// probability. For K=64 in an 8k vocabulary, the top-K typically captures
    /// most of the probability mass when the model is reasonably confident, so the
    /// approximation is close but not equal to the true probability.
    ///
    /// - Parameters:
    ///   - topKIds: Array of token IDs (from `top_k_ids` output).
    ///   - topKLogits: Array of logits (from `top_k_logits` output). Must be the
    ///     same length as `topKIds`.
    ///   - vocabulary: Mapping from token IDs to text.
    ///   - preferredScript: Script to filter for.
    ///
    /// - Returns: Token ID and top-K softmax probability (in [0, 1]) of the
    ///   highest-logit in-script candidate, or `nil` if no in-script match exists
    ///   or the input arrays are mismatched/empty.
    static func filterTopK(
        topKIds: [Int],
        topKLogits: [Float],
        vocabulary: [Int: String],
        preferredScript: Script
    ) -> (tokenId: Int, probability: Float)? {
        // Guard against array length mismatch — CoreML output arrays are read
        // independently in TdtModelInference, so enforce the invariant here.
        let count = min(topKIds.count, topKLogits.count)
        guard count > 0 else { return nil }

        // Argmax over in-script candidates. No assumption that CoreML returned
        // top-K in sorted order — scan all K and keep the highest-logit match.
        var bestIdx: Int = -1
        var bestLogit: Float = -.infinity
        for idx in 0..<count {
            let tokenId = topKIds[idx]
            guard let tokenText = vocabulary[tokenId] else { continue }
            guard matches(tokenText, script: preferredScript) else { continue }

            let logit = topKLogits[idx]
            if logit > bestLogit {
                bestLogit = logit
                bestIdx = idx
            }
        }
        guard bestIdx >= 0 else { return nil }

        let probability = softmaxProbability(of: bestIdx, logits: topKLogits, count: count)
        return (topKIds[bestIdx], probability)
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
