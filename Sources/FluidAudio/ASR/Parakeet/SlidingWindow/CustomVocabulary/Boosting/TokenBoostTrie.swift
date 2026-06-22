import Foundation

/// Aho-Corasick boosting tree for Parakeet vocabulary biasing — a faithful port
/// of NeMo's greedy `boosting_tree` (TurboBias, arXiv:2508.07014).
///
/// Built once from a list of BPE token sequences. At each decode step the
/// decoder queries `boostLogprobs(previousTokens:alpha:)`; the returned DENSE
/// vector is added to the token logits before argmax.
///
/// Each node carries a depth-scaled `tokenScore` and an accumulated `nodeScore`
/// (root→node). Failure links (Aho-Corasick) let a broken partial match back
/// off to the longest matching suffix. The per-step vector is
/// `α · advance(state)`:
///   - continuation token(s) → positive forward score `tokenScore(child)`;
///   - every other non-blank token → negative **baseline** `= Σ backoff_w`
///     (telescopes to `−nodeScore(state)`), which *retracts* the accumulated
///     boost when the phrase doesn't continue;
///   - other phrases' first-tokens → backoff + re-entry (often nets ~0).
/// This negative-on-all-others baseline is the cancellation that lets the tree
/// boost hard (high α) without flooding false inserts — the one thing the prior
/// positive-only additive trie lacked.
///
/// Flat node array (`[Node]` values) → naturally immutable and `Sendable`.
/// Build cost O(Σ|term|); per-step query O(maxDepth + |arcs on the fail chain|).
///
/// Verified token-for-token against `docs/asr-research/nemo-boosting-tree-golden.json`.
public final class TokenBoostTrie: Sendable {

    /// Total distinct terms inserted.
    public let wordCount: Int

    /// Token-logit vocabulary size of the target model (e.g. 8193 for Parakeet
    /// v3: 8192 BPE vocab + 1 blank). Boost vectors are sized to match; the
    /// blank slot (assumed last, `vocabSize − 1` for v3) always stays 0 so the
    /// negative baseline never suppresses blank.
    public let vocabSize: Int

    private let nodes: [Node]  // nodes[0] is root
    private let maxDepth: Int  // deepest node level → bounds the suffix re-walk
    private let contextScore: Double
    private let depthScaling: Double

    // MARK: - Build

    public init(
        terms: [(tokens: [Int], word: String)],
        vocabSize: Int,
        contextScore: Double = BoostConstants.contextScore,
        depthScaling: Double = BoostConstants.depthScaling
    ) {
        precondition(vocabSize > 0, "vocabSize must be positive")
        self.vocabSize = vocabSize
        self.contextScore = contextScore
        self.depthScaling = depthScaling

        var builder: [BuilderNode] = [BuilderNode()]  // root
        var count = 0
        var deepest = 0

        for (tokens, _) in terms {
            guard !tokens.isEmpty else { continue }
            var idx = 0
            for (i, tok) in tokens.enumerated() {
                let ts = i == 0 ? contextScore : contextScore * depthScaling + log(Double(i + 1))
                if let next = builder[idx].children[tok] {
                    // Shared prefix: token_score = max(contextScore, existing).
                    builder[next].tokenScore = max(contextScore, builder[next].tokenScore)
                    idx = next
                } else {
                    let newIdx = builder.count
                    var n = BuilderNode()
                    n.tokenScore = ts
                    n.nodeScore = builder[idx].nodeScore + ts
                    n.level = builder[idx].level + 1
                    builder.append(n)
                    builder[idx].children[tok] = newIdx
                    idx = newIdx
                }
            }
            builder[idx].isEnd = true
            deepest = max(deepest, builder[idx].level)
            count += 1
        }

        // Aho-Corasick failure links via BFS over the trie.
        var queue: [Int] = []
        for (_, c) in builder[0].children { builder[c].failIdx = 0; queue.append(c) }
        var qi = 0
        while qi < queue.count {
            let u = queue[qi]; qi += 1
            // Snapshot children to avoid mutating while iterating.
            for (tok, child) in builder[u].children {
                var f = builder[u].failIdx
                while f != 0 && builder[f].children[tok] == nil { f = builder[f].failIdx }
                if let fc = builder[f].children[tok], fc != child {
                    builder[child].failIdx = fc
                } else {
                    builder[child].failIdx = 0
                }
                queue.append(child)
            }
        }

        self.nodes = builder.map {
            Node(children: $0.children, tokenScore: $0.tokenScore, nodeScore: $0.nodeScore,
                 isEnd: $0.isEnd, failIdx: $0.failIdx)
        }
        self.maxDepth = max(deepest, 1)
        self.wordCount = count
    }

    // MARK: - Query

    /// Aho-Corasick goto: next state after emitting `token` from `state`.
    @inline(__always)
    private func goto(_ state: Int, _ token: Int) -> Int {
        var cur = state
        while cur != 0 && nodes[cur].children[token] == nil { cur = nodes[cur].failIdx }
        return nodes[cur].children[token] ?? 0
    }

    /// Reconstruct the current automaton state from the emitted token history.
    /// Only the last `maxDepth` tokens matter (the longest matchable suffix is
    /// ≤ maxDepth), so the re-walk is O(maxDepth) per step, not O(history).
    private func state(after previousTokens: ArraySlice<Int>) -> Int {
        let tail = previousTokens.suffix(maxDepth)
        var s = 0
        for t in tail { s = goto(s, t) }
        return s
    }

    /// Dense per-step boost vector `α · advance(state)` (size `vocabSize`).
    /// The blank slot (last index, v3) is left at 0.
    public func boostLogprobs(previousTokens: ArraySlice<Int>, alpha: Float) -> [Float] {
        let s = state(after: previousTokens)

        // advance(state): baseline (telescopes to −nodeScore) + per-token exceptions.
        var exceptions: [Int: Double] = [:]
        var processed = Set<Int>()
        var acc = 0.0
        var cur = s
        while true {
            for (tok, child) in nodes[cur].children where !processed.contains(tok) {
                exceptions[tok] = acc + nodes[child].tokenScore
                processed.insert(tok)
            }
            if cur == 0 { break }
            acc += nodes[nodes[cur].failIdx].nodeScore - nodes[cur].nodeScore
            cur = nodes[cur].failIdx
        }
        let baseline = Float(alpha) * Float(acc)

        // Dense vector: baseline on all non-blank tokens, exceptions override,
        // blank (last slot) stays 0.
        let lastNonBlank = vocabSize - 1  // blank assumed last (v3: 8192)
        var boost = [Float](repeating: baseline, count: vocabSize)
        boost[lastNonBlank] = 0  // never push/suppress blank
        let a = Float(alpha)
        for (tok, score) in exceptions where tok >= 0 && tok < lastNonBlank {
            boost[tok] = a * Float(score)
        }
        return boost
    }

    /// Convenience when the caller has `[Int]` rather than a slice.
    public func boostLogprobs(previousTokens: [Int], alpha: Float) -> [Float] {
        boostLogprobs(previousTokens: previousTokens[...], alpha: alpha)
    }

    // MARK: - Private types

    private struct Node: Sendable {
        let children: [Int: Int]
        let tokenScore: Double
        let nodeScore: Double
        let isEnd: Bool
        let failIdx: Int
    }

    private struct BuilderNode {
        var children: [Int: Int] = [:]
        var tokenScore: Double = 0
        var nodeScore: Double = 0
        var isEnd: Bool = false
        var failIdx: Int = 0
        var level: Int = 0
    }
}
