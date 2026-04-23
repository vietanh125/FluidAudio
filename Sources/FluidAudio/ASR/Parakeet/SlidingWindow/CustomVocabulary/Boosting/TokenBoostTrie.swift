import Foundation

/// Token-level trie for Parakeet vocabulary biasing.
///
/// Built once from a list of BPE token sequences. At decoding time, for each
/// step the decoder queries `boostLogprobs(previousTokens:)` and the returned
/// vector is added to token logits before argmax.
///
/// Uses a flat node array (`Node` values stored in `[Node]`) rather than a
/// tree of reference nodes so the instance is naturally immutable and
/// `Sendable` without `@unchecked`. Build cost is still O(Σ|term|), query
/// cost is O(maxPrefixLen) per step.
public final class TokenBoostTrie: Sendable {

    /// Total distinct terms inserted.
    public let wordCount: Int

    /// Number of distinct first tokens across all inserted terms.
    /// Useful as a sanity check (should be < firstTokenCount upper bound of
    /// vocab size; if nearly equal to vocab size the trie is too broad).
    public var firstTokenCount: Int { firstTokens.count }

    /// Token-logit vocabulary size of the target model (e.g. 8193 for
    /// Parakeet v3: 8192 vocab + 1 blank). Boost vectors are sized to match.
    public let vocabSize: Int

    private let nodes: [Node]  // nodes[0] is root
    private let firstTokens: Set<Int>

    // MARK: - Build

    public init(terms: [(tokens: [Int], word: String)], vocabSize: Int) {
        precondition(vocabSize > 0, "vocabSize must be positive")

        var builder: [BuilderNode] = [BuilderNode()]
        var first = Set<Int>()
        var count = 0

        for (tokens, word) in terms {
            guard !tokens.isEmpty else { continue }
            var idx = 0
            for tok in tokens {
                if let next = builder[idx].children[tok] {
                    idx = next
                    continue
                }
                let newIdx = builder.count
                builder.append(BuilderNode())
                builder[idx].children[tok] = newIdx
                idx = newIdx
            }
            builder[idx].isTerminal = true
            builder[idx].word = word
            first.insert(tokens[0])
            count += 1
        }

        self.nodes = builder.map { Node(children: $0.children, isTerminal: $0.isTerminal, word: $0.word) }
        self.firstTokens = first
        self.wordCount = count
        self.vocabSize = vocabSize
    }

    // MARK: - Query

    /// Produce an additive boost vector over the full token-logit vocabulary.
    ///
    /// Mirrors `CanaryBoostProcessor.getBoostLogprobs` semantics exactly so
    /// the hyperparameters from the Python sweep transfer unchanged:
    /// - Every token in `firstTokens` receives `baseBoost` unconditionally.
    /// - For each lookback in 1…min(|previousTokens|, maxPrefixLen), walk the
    ///   trie over the suffix. If the walk succeeds, award `sequenceBoost *
    ///   (1 + lookback * 0.25)` to every continuation, taking the max when a
    ///   token is eligible via multiple paths.
    public func boostLogprobs(
        previousTokens: ArraySlice<Int>,
        baseBoost: Float = BoostConstants.defaultBaseBoost,
        sequenceBoost: Float = BoostConstants.defaultSequenceBoost,
        maxPrefixLen: Int = BoostConstants.defaultMaxPrefixLen
    ) -> [Float] {
        var boost = [Float](repeating: 0, count: vocabSize)

        for tid in firstTokens where tid >= 0 && tid < vocabSize {
            boost[tid] = baseBoost
        }

        guard !previousTokens.isEmpty, maxPrefixLen > 0 else { return boost }

        let maxLookback = min(previousTokens.count, maxPrefixLen)
        for lookback in 1...maxLookback {
            // previousTokens is an ArraySlice, `.suffix` is O(1) and returns a slice.
            let prefix = previousTokens.suffix(lookback)
            var nodeIdx = 0
            var matched = true
            for tok in prefix {
                guard let next = nodes[nodeIdx].children[tok] else {
                    matched = false
                    break
                }
                nodeIdx = next
            }
            guard matched else { continue }

            let strength = sequenceBoost * (1.0 + Float(lookback) * 0.25)
            for tid in nodes[nodeIdx].children.keys where tid >= 0 && tid < vocabSize {
                if strength > boost[tid] {
                    boost[tid] = strength
                }
            }
        }
        return boost
    }

    /// Convenience when the caller has `[Int]` rather than a slice.
    public func boostLogprobs(
        previousTokens: [Int],
        baseBoost: Float = BoostConstants.defaultBaseBoost,
        sequenceBoost: Float = BoostConstants.defaultSequenceBoost,
        maxPrefixLen: Int = BoostConstants.defaultMaxPrefixLen
    ) -> [Float] {
        boostLogprobs(
            previousTokens: previousTokens[...],
            baseBoost: baseBoost,
            sequenceBoost: sequenceBoost,
            maxPrefixLen: maxPrefixLen
        )
    }

    // MARK: - Private types

    private struct Node: Sendable {
        let children: [Int: Int]
        let isTerminal: Bool
        let word: String?
    }

    private struct BuilderNode {
        var children: [Int: Int] = [:]
        var isTerminal: Bool = false
        var word: String? = nil
    }
}
