import XCTest

@testable import FluidAudio

/// Golden-fixture tests for the Aho-Corasick boosting tree (NeMo `boosting_tree`
/// port). Values mirror `docs/asr-research/nemo-boosting-tree-golden.json` in the
/// Scribion repo (built from v3 SentencePiece ids); see the port spec.
final class TokenBoostTrieTests: XCTestCase {

    // vocabSize 8193 = 8192 BPE + blank(8192); fixture vocab_size is 8192 (blank excluded).
    private let vocabSize = 8193
    private let alpha: Float = 2.0

    // Husten / Diabetes / Atemnot, in fixture order.
    private func tree() -> TokenBoostTrie {
        TokenBoostTrie(
            terms: [
                (tokens: [425, 323, 571], word: "Husten"),
                (tokens: [360, 387, 5634, 283], word: "Diabetes"),
                (tokens: [3078, 311, 7867, 359], word: "Atemnot"),
            ],
            vocabSize: vocabSize
        )
    }

    /// Assert a step's dense vector: `baseline` everywhere (except blank=0),
    /// `exceptions` overriding, and the right number of baseline slots.
    private func assertStep(
        _ boost: [Float], baseline: Float, exceptions: [Int: Float],
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(boost.count, vocabSize, file: file, line: line)
        XCTAssertEqual(boost[vocabSize - 1], 0, "blank slot stays 0", file: file, line: line)
        for (tok, want) in exceptions {
            XCTAssertEqual(boost[tok], want, accuracy: 1e-3, "exc[\(tok)]", file: file, line: line)
        }
        // A few representative non-exception, non-blank tokens should equal baseline.
        for probe in [0, 100, 8000, 8191] where exceptions[probe] == nil {
            XCTAssertEqual(boost[probe], baseline, accuracy: 1e-3, "baseline@\(probe)", file: file, line: line)
        }
    }

    /// Walk into "Husten" (▁H us ten) — the forward-score + backoff-baseline case.
    func testWalkIntoHusten() {
        let t = tree()
        // step 0: root → emit ▁H(425). baseline 0; the 3 term-starts → +2.0.
        assertStep(t.boostLogprobs(previousTokens: [], alpha: alpha),
                   baseline: 0, exceptions: [360: 2.0, 425: 2.0, 3078: 2.0])
        // step 1: after ▁H → emit us(323). baseline −2.0; us → +5.386; other starts → 0.
        assertStep(t.boostLogprobs(previousTokens: [425], alpha: alpha),
                   baseline: -2.0, exceptions: [323: 5.386294, 360: 0, 425: 0, 3078: 0])
        // step 2: after ▁H us → emit ten(571). baseline −7.386; ten → +6.197; starts → −5.386.
        assertStep(t.boostLogprobs(previousTokens: [425, 323], alpha: alpha),
                   baseline: -7.386294,
                   exceptions: [571: 6.197225, 360: -5.386294, 425: -5.386294, 3078: -5.386294])
    }

    /// Cancellation: after ▁H, emit a *different* term's start (▁At=3078).
    /// Its boost nets exactly 0 (−2.0 backoff + 2.0 re-entry) — no false insert.
    func testBreakAfterHCancels() {
        let t = tree()
        let boost = t.boostLogprobs(previousTokens: [425], alpha: alpha)
        XCTAssertEqual(boost[3078], 0, accuracy: 1e-3, "Atemnot start auto-cancels after Husten prefix")
        XCTAssertEqual(boost[323], 5.386294, accuracy: 1e-3, "the Husten continuation is still the only positive")
    }

    /// Node scores follow the locked formula (i0=1.0, i>0=2+ln(i+1)); node_score is the path sum.
    func testDeepDiabetesScores() {
        let t = tree()
        // after ▁D ia bet (Diabetes[0..2], node_score 6.791759) → emit es(283).
        // baseline = −2*6.791759 = −13.583518; es → +2*token_score(es)=2*3.386294=6.772588.
        assertStep(t.boostLogprobs(previousTokens: [360, 387, 5634], alpha: alpha),
                   baseline: -13.583518,
                   exceptions: [283: 6.772589, 360: -11.583519, 425: -11.583519, 3078: -11.583519])
    }
}
