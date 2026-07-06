import Foundation

/// Configuration for `PhraseBoostingTree` (NeMo GPU-PB parameter names).
public struct PhraseBoostingTreeConfig: Sendable {
    /// Base score for the first token of a phrase (NeMo `context_score`).
    public var contextScore: Float
    /// Depth multiplier for tokens after the first: the arc into a node at
    /// depth d > 1 scores `contextScore * depthScaling + ln(d)`
    /// (NeMo `depth_scaling`; 2.0 recommended for CTC/RNN-T/TDT, 1.0 for AED).
    public var depthScaling: Float
    /// Score for tokens that do not start or continue any phrase, taken from
    /// the root self-loop (NeMo `unk_score`).
    public var unkScore: Float
    /// Bonus added to the EOS score when the current state completes a phrase
    /// (NeMo `final_eos_score`; AED decoders only).
    public var finalEosScore: Float
    /// Shallow-fusion weight applied to the whole returned score vector
    /// (NeMo decoder-side `boosting_tree_alpha`).
    public var alpha: Float

    public init(
        contextScore: Float = 1.0,
        depthScaling: Float = 2.0,
        unkScore: Float = 0.0,
        finalEosScore: Float = 0.0,
        alpha: Float = 1.0
    ) {
        self.contextScore = contextScore
        self.depthScaling = depthScaling
        self.unkScore = unkScore
        self.finalEosScore = finalEosScore
        self.alpha = alpha
    }
}

/// NeMo GPU-PB phrase-boosting tree ("TurboBias", arXiv:2508.07014).
///
/// An Aho-Corasick token trie with NeMo's modified weight distribution. The
/// arc into a depth-1 node scores `contextScore`; the arc into a node at
/// depth d > 1 scores `contextScore * depthScaling + ln(d)`. Every node has
/// a fail (backoff) transition to the longest proper suffix of its path that
/// is also a trie path; taking it costs `fail.nodeScore - node.nodeScore`,
/// so abandoning a partially matched phrase exactly retracts the boost
/// already granted. Nodes that complete a phrase keep their earned score
/// (backoff weight 0). Tokens that start no phrase score `unkScore`.
///
/// Port of NeMo's `ContextGraph` + `GPUBoostingTreeModel` scalar semantics
/// (`context_graph_universal.py`, `boosting_graph_batched.py`,
/// `ngram_lm_batched.py:_advance_pytorch`), validated against the golden
/// vectors in NeMo's `test_boosting_tree.py`. The batched GPU machinery is
/// irrelevant at batch size 1; per step this is a short backoff-chain walk.
///
/// State tracking is stateless-by-history: the automaton state after a token
/// history is the longest suffix of that history that forms a trie path,
/// which only depends on the last `maxDepth` tokens.
public final class PhraseBoostingTree: Sendable {

    private struct Node: Sendable {
        var children: [Int: Int] = [:]  // token id -> node index
        var arcWeight: [Int: Float] = [:]  // token id -> weight of the arc to children[token]
        var nodeScore: Float = 0  // accumulated arc weights root -> node
        var isEnd: Bool = false
        var level: Int = 0
        var fail: Int = 0  // Aho-Corasick fail link (backoff target)
        var backoffWeight: Float = 0  // 0 for root and phrase-final nodes
    }

    private let nodes: [Node]
    public let vocabSize: Int
    public let config: PhraseBoostingTreeConfig
    /// Longest phrase length in tokens; bounds the history window needed to
    /// recover the automaton state.
    public let maxDepth: Int
    public let phraseCount: Int

    public init(terms: [(tokens: [Int], word: String)], vocabSize: Int, config: PhraseBoostingTreeConfig) {
        precondition(vocabSize > 0, "vocabSize must be positive")
        self.vocabSize = vocabSize
        self.config = config

        var builder: [Node] = [Node()]
        var count = 0
        var depth = 0

        // Trie insertion — mirrors NeMo ContextGraph.build (non-uniform weights).
        for (tokens, _) in terms {
            guard !tokens.isEmpty else { continue }
            var cur = 0
            for (i, token) in tokens.enumerated() {
                let isLast = i == tokens.count - 1
                if let child = builder[cur].children[token] {
                    // Shared node: score is the max across phrases (NeMo compares
                    // against the raw contextScore here, not the depth-scaled value).
                    let tokenScore = max(config.contextScore, builder[cur].arcWeight[token] ?? 0)
                    builder[cur].arcWeight[token] = tokenScore
                    builder[child].nodeScore = builder[cur].nodeScore + tokenScore
                    builder[child].isEnd = isLast || builder[child].isEnd
                    cur = child
                } else {
                    let tokenScore: Float
                    if i > 0 {
                        tokenScore = config.contextScore * config.depthScaling + Foundation.log(Float(i + 1))
                    } else {
                        tokenScore = config.contextScore
                    }
                    var node = Node()
                    node.nodeScore = builder[cur].nodeScore + tokenScore
                    node.isEnd = isLast
                    node.level = i + 1
                    builder.append(node)
                    let idx = builder.count - 1
                    builder[cur].children[token] = idx
                    builder[cur].arcWeight[token] = tokenScore
                    cur = idx
                }
            }
            count += 1
            depth = max(depth, tokens.count)
        }
        self.phraseCount = count
        self.maxDepth = depth

        // Fail links via BFS (Aho-Corasick), then backoff weights.
        var queue: [Int] = []
        for (_, child) in builder[0].children {
            builder[child].fail = 0
            queue.append(child)
        }
        var head = 0
        while head < queue.count {
            let u = queue[head]
            head += 1
            for (token, v) in builder[u].children {
                var fail = builder[u].fail
                if let next = builder[fail].children[token] {
                    fail = next
                } else {
                    fail = builder[fail].fail
                    while builder[fail].children[token] == nil, fail != 0 {
                        fail = builder[fail].fail
                    }
                    if let next = builder[fail].children[token] {
                        fail = next
                    }
                }
                builder[v].fail = fail
                queue.append(v)
            }
        }
        for i in 1..<builder.count {
            // Phrase-final nodes keep their earned score (no retraction after a
            // completed match); all others pay back the partial boost.
            builder[i].backoffWeight =
                builder[i].isEnd ? 0 : builder[builder[i].fail].nodeScore - builder[i].nodeScore
        }

        self.nodes = builder
    }

    /// Aho-Corasick transition: state after consuming `token` from `state`.
    public func transition(from state: Int, token: Int) -> Int {
        var cur = state
        while true {
            if let next = nodes[cur].children[token] { return next }
            if cur == 0 { return 0 }
            cur = nodes[cur].fail
        }
    }

    /// Automaton state after a token history: the longest suffix of the
    /// history that forms a trie path. Only the last `maxDepth` tokens matter.
    public func state(after history: ArraySlice<Int>) -> Int {
        var s = 0
        for token in history.suffix(maxDepth) {
            s = transition(from: s, token: token)
        }
        return s
    }

    /// Dense per-token scores from `state`, before alpha scaling — the scalar
    /// equivalent of NeMo `_advance_pytorch` for one state: walk the backoff
    /// chain, first-seen arc wins, everything else gets the accumulated
    /// backoff plus `unkScore`.
    public func rawScores(state: Int) -> [Float] {
        var out = [Float](repeating: 0, count: vocabSize)
        var filled = [Bool](repeating: false, count: vocabSize)
        var acc: Float = 0
        var cur = state
        while true {
            for (token, weight) in nodes[cur].arcWeight where token >= 0 && token < vocabSize {
                if !filled[token] {
                    out[token] = acc + weight
                    filled[token] = true
                }
            }
            if cur == 0 { break }
            acc += nodes[cur].backoffWeight
            cur = nodes[cur].fail
        }
        let unkValue = acc + config.unkScore
        for t in 0..<vocabSize where !filled[t] {
            out[t] = unkValue
        }
        return out
    }

    /// Whether `state` completes a phrase (final weight applies for AED EOS).
    public func isFinal(state: Int) -> Bool {
        nodes[state].isEnd
    }

    /// Additive boost vector for the next decode step, alpha-scaled.
    ///
    /// - Parameter eosId: AED decoders only — the EOS score is replaced by
    ///   `max(bestBoost, 0)` so boosting never suppresses end-of-sentence,
    ///   plus `finalEosScore` when the state completes a phrase (NeMo
    ///   `GPUBoostingTreeModel.advance(eos_id:)`).
    public func boostLogprobs(previousTokens: ArraySlice<Int>, eosId: Int? = nil) -> [Float] {
        let s = state(after: previousTokens)
        var scores = rawScores(state: s)
        if let eosId, eosId >= 0, eosId < vocabSize {
            let best = scores.max() ?? 0
            scores[eosId] = max(best, 0) + (nodes[s].isEnd ? config.finalEosScore : 0)
        }
        if config.alpha != 1.0 {
            for i in scores.indices { scores[i] *= config.alpha }
        }
        return scores
    }

    /// Per-token boost scores for a whole sentence (unscaled) — mirrors NeMo
    /// `score_sentences` for validation against the golden test vectors.
    public func scoreSentence(_ tokens: [Int]) -> [Float] {
        var s = 0
        var result: [Float] = []
        result.reserveCapacity(tokens.count)
        for token in tokens {
            let scores = rawScores(state: s)
            result.append(token >= 0 && token < vocabSize ? scores[token] : 0)
            s = transition(from: s, token: token)
        }
        return result
    }
}
