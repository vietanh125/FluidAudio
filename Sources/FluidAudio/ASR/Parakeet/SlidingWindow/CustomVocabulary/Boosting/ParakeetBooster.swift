import Foundation

/// Aho-Corasick boosting tree + its global strength `α`, packaged so callers
/// (ScribionCore, CLI, tests) pass around a single value rather than re-plumbing
/// the boost scheme through every transcribe call.
///
/// Construct once per vocabulary (e.g. per retrieved subset, or once for a
/// static list); pass to `AsrManager.transcribe(..., booster:)`. Thread-safe and
/// `Sendable`: all stored state is immutable after init.
public struct ParakeetBooster: Sendable {

    public let trie: TokenBoostTrie
    /// Global boost strength: the per-step vector is `α · tree.advance(state)`.
    public let alpha: Float

    public init(trie: TokenBoostTrie, alpha: Float = BoostConstants.defaultAlpha) {
        self.trie = trie
        self.alpha = alpha
    }

    /// Build a booster directly from a pre-tokenized `{word: [token_ids]}` map.
    ///
    /// Lets callers ship tokenization offline (e.g. via a SentencePiece script)
    /// and load the result as a bundle resource — no `BpeTokenizer` needed at
    /// runtime. Recommended for the v3 CoreML bundle (which ships no `tokenizer.json`).
    public static func fromTokenMap(
        _ map: [String: [Int]],
        vocabSize: Int,
        alpha: Float = BoostConstants.defaultAlpha
    ) -> ParakeetBooster {
        let entries: [(tokens: [Int], word: String)] = map
            .compactMap { key, value in value.isEmpty ? nil : (tokens: value, word: key) }
        let trie = TokenBoostTrie(terms: entries, vocabSize: vocabSize)
        return ParakeetBooster(trie: trie, alpha: alpha)
    }

    /// Build a booster from raw term strings using a BPE tokenizer.
    ///
    /// - Parameters:
    ///   - terms: vocabulary terms (one per line). Blank lines / `#` comments ignored.
    ///   - tokenizer: Parakeet's BPE tokenizer (same instance used by the ASR
    ///     model it biases). Emits BOTH the canonical and all-lowercase form.
    ///   - vocabSize: token-logit vocabulary size (e.g. 8193 for v3 = 8192 + blank).
    public static func build(
        terms: [String],
        tokenizer: BpeTokenizer,
        vocabSize: Int,
        alpha: Float = BoostConstants.defaultAlpha
    ) -> ParakeetBooster {
        var entries: [(tokens: [Int], word: String)] = []
        var seenTokens: Set<[Int]> = []

        for raw in terms {
            let term = raw.trimmingCharacters(in: .whitespaces)
            guard !term.isEmpty, !term.hasPrefix("#") else { continue }
            for form in Set([term, term.lowercased()]) {
                let ids = tokenizer.encode(form)
                guard !ids.isEmpty, seenTokens.insert(ids).inserted else { continue }
                entries.append((tokens: ids, word: form))
            }
        }

        let trie = TokenBoostTrie(terms: entries, vocabSize: vocabSize)
        return ParakeetBooster(trie: trie, alpha: alpha)
    }

    /// Hot-path query used by the TDT decoder at each joint step.
    @inline(__always)
    public func boostLogprobs(previousTokens: ArraySlice<Int>) -> [Float] {
        trie.boostLogprobs(previousTokens: previousTokens, alpha: alpha)
    }

    public var wordCount: Int { trie.wordCount }
    public var vocabSize: Int { trie.vocabSize }
}
