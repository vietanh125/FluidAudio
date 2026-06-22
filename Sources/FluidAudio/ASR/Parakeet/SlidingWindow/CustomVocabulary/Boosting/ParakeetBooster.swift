import Foundation

/// Trie-based vocabulary booster for Parakeet TDT decoding.
///
/// Wraps a `TokenBoostTrie` together with boost hyperparameters so callers
/// (ScribionCore, CLI, tests) can just pass around a single value rather
/// than re-plumbing bb/sb/lb through every transcribe call.
///
/// Construct once per app lifetime (or per vocabulary change); pass through
/// to `AsrManager.transcribe(..., booster:)`. Thread-safe and `Sendable`:
/// all stored state is immutable after init.
public struct ParakeetBooster: Sendable {

    public let trie: TokenBoostTrie
    public let baseBoost: Float
    public let sequenceBoost: Float
    public let maxPrefixLen: Int

    public init(
        trie: TokenBoostTrie,
        baseBoost: Float = BoostConstants.defaultBaseBoost,
        sequenceBoost: Float = BoostConstants.defaultSequenceBoost,
        maxPrefixLen: Int = BoostConstants.defaultMaxPrefixLen
    ) {
        self.trie = trie
        self.baseBoost = baseBoost
        self.sequenceBoost = sequenceBoost
        self.maxPrefixLen = maxPrefixLen
    }

    /// Build a booster directly from a pre-tokenized `{word: [token_ids]}` map.
    ///
    /// Lets callers ship the tokenization step offline (e.g. via
    /// `pymlx/tokenize_boost_words.py`) and load the result as a bundle
    /// resource — no `BpeTokenizer` needed at runtime. Recommended for
    /// stable vocabularies where terms change rarely.
    public static func fromTokenMap(
        _ map: [String: [Int]],
        vocabSize: Int,
        baseBoost: Float = BoostConstants.defaultBaseBoost,
        sequenceBoost: Float = BoostConstants.defaultSequenceBoost,
        maxPrefixLen: Int = BoostConstants.defaultMaxPrefixLen
    ) -> ParakeetBooster {
        let entries: [(tokens: [Int], word: String)] = map
            .compactMap { key, value in value.isEmpty ? nil : (tokens: value, word: key) }
        let trie = TokenBoostTrie(terms: entries, vocabSize: vocabSize)
        return ParakeetBooster(
            trie: trie,
            baseBoost: baseBoost,
            sequenceBoost: sequenceBoost,
            maxPrefixLen: maxPrefixLen
        )
    }

    /// Build a booster from raw term strings using a BPE tokenizer.
    ///
    /// - Parameters:
    ///   - terms: vocabulary terms (one per line, whitespace trimmed).
    ///     Blank lines and `#` comments are ignored.
    ///   - tokenizer: Parakeet's BPE tokenizer (same instance used by the
    ///     ASR model it biases). Terms are lowercased + NFKC-normalized by
    ///     BpeTokenizer.encode; we emit BOTH the canonical and the
    ///     all-lowercase form so the trie catches both cased spellings
    ///     that might appear in other corpora.
    ///   - vocabSize: token-logit vocabulary size (e.g. 8193 for v3 =
    ///     8192 BPE vocab + 1 blank).
    public static func build(
        terms: [String],
        tokenizer: BpeTokenizer,
        vocabSize: Int,
        baseBoost: Float = BoostConstants.defaultBaseBoost,
        sequenceBoost: Float = BoostConstants.defaultSequenceBoost,
        maxPrefixLen: Int = BoostConstants.defaultMaxPrefixLen
    ) -> ParakeetBooster {
        var entries: [(tokens: [Int], word: String)] = []
        var seenTokens: Set<[Int]> = []

        for raw in terms {
            let term = raw.trimmingCharacters(in: .whitespaces)
            guard !term.isEmpty, !term.hasPrefix("#") else { continue }

            let forms = [term, term.lowercased()]
            for form in Set(forms) {
                let ids = tokenizer.encode(form)
                guard !ids.isEmpty, seenTokens.insert(ids).inserted else { continue }
                entries.append((tokens: ids, word: form))
            }
        }

        let trie = TokenBoostTrie(terms: entries, vocabSize: vocabSize)
        return ParakeetBooster(
            trie: trie,
            baseBoost: baseBoost,
            sequenceBoost: sequenceBoost,
            maxPrefixLen: maxPrefixLen
        )
    }

    /// Hot-path query used by the TDT decoder at each joint step.
    @inline(__always)
    public func boostLogprobs(previousTokens: ArraySlice<Int>) -> [Float] {
        trie.boostLogprobs(
            previousTokens: previousTokens,
            baseBoost: baseBoost,
            sequenceBoost: sequenceBoost,
            maxPrefixLen: maxPrefixLen
        )
    }

    public var wordCount: Int { trie.wordCount }
    public var vocabSize: Int { trie.vocabSize }
}
