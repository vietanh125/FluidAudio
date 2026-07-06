import Foundation

/// Vocabulary booster for Parakeet TDT decoding.
///
/// Wraps one of two scoring backends together with its hyperparameters so
/// callers (ScribionCore, CLI, tests) can pass around a single value:
///
/// - `.heuristic` — the original `TokenBoostTrie` scheme: additive
///   `baseBoost` for term-initial tokens, `sequenceBoost * (1 + 0.25·lookback)`
///   for continuations, no score retraction.
/// - `.phraseTree` — NeMo GPU-PB (`PhraseBoostingTree`): Aho-Corasick trie
///   with depth-scaled arc scores and backoff retraction. Requires the
///   decoder to preserve the blank/non-blank category from the unbiased
///   argmax (NeMo two-stage selection), signalled via
///   `preservesBlankCategory`.
///
/// Construct once per app lifetime (or per vocabulary change); pass through
/// to `AsrManager.transcribe(..., booster:)`. Thread-safe and `Sendable`:
/// all stored state is immutable after init.
public struct ParakeetBooster: Sendable {

    public enum Scoring: Sendable {
        case heuristic(trie: TokenBoostTrie, baseBoost: Float, sequenceBoost: Float, maxPrefixLen: Int)
        case phraseTree(PhraseBoostingTree)
    }

    public let scoring: Scoring

    /// GPU-PB scoring assigns negative retraction scores to non-continuing
    /// tokens, which must not tip the blank/non-blank decision: the decoder
    /// runs NeMo's two-stage selection (unbiased argmax picks the category;
    /// boost re-ranks non-blank tokens only). The heuristic scheme keeps the
    /// original single-argmax behavior.
    public var preservesBlankCategory: Bool {
        if case .phraseTree = scoring { return true }
        return false
    }

    public init(
        trie: TokenBoostTrie,
        baseBoost: Float = BoostConstants.defaultBaseBoost,
        sequenceBoost: Float = BoostConstants.defaultSequenceBoost,
        maxPrefixLen: Int = BoostConstants.defaultMaxPrefixLen
    ) {
        self.scoring = .heuristic(
            trie: trie, baseBoost: baseBoost, sequenceBoost: sequenceBoost, maxPrefixLen: maxPrefixLen)
    }

    public init(phraseTree: PhraseBoostingTree) {
        self.scoring = .phraseTree(phraseTree)
    }

    /// Build a heuristic booster directly from a pre-tokenized `{word: [token_ids]}` map.
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
        let trie = TokenBoostTrie(terms: Self.entries(fromTokenMap: map), vocabSize: vocabSize)
        return ParakeetBooster(
            trie: trie,
            baseBoost: baseBoost,
            sequenceBoost: sequenceBoost,
            maxPrefixLen: maxPrefixLen
        )
    }

    /// Build a GPU-PB (NeMo phrase-boosting tree) booster from a pre-tokenized
    /// `{word: [token_ids]}` map.
    public static func gpuPB(
        tokenMap: [String: [Int]],
        vocabSize: Int,
        config: PhraseBoostingTreeConfig
    ) -> ParakeetBooster {
        let tree = PhraseBoostingTree(
            terms: Self.entries(fromTokenMap: tokenMap), vocabSize: vocabSize, config: config)
        return ParakeetBooster(phraseTree: tree)
    }

    private static func entries(fromTokenMap map: [String: [Int]]) -> [(tokens: [Int], word: String)] {
        map.compactMap { key, value in value.isEmpty ? nil : (tokens: value, word: key) }
    }

    /// Build a heuristic booster from raw term strings using a BPE tokenizer.
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
        switch scoring {
        case .heuristic(let trie, let baseBoost, let sequenceBoost, let maxPrefixLen):
            return trie.boostLogprobs(
                previousTokens: previousTokens,
                baseBoost: baseBoost,
                sequenceBoost: sequenceBoost,
                maxPrefixLen: maxPrefixLen
            )
        case .phraseTree(let tree):
            return tree.boostLogprobs(previousTokens: previousTokens)
        }
    }

    public var wordCount: Int {
        switch scoring {
        case .heuristic(let trie, _, _, _): return trie.wordCount
        case .phraseTree(let tree): return tree.phraseCount
        }
    }

    public var vocabSize: Int {
        switch scoring {
        case .heuristic(let trie, _, _, _): return trie.vocabSize
        case .phraseTree(let tree): return tree.vocabSize
        }
    }
}
