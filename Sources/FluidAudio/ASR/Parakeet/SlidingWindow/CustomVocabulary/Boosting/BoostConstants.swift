import Foundation

/// Constants for the Aho-Corasick boosting tree (NeMo `boosting_tree` port).
///
/// The tree assigns each phrase token a depth-scaled `token_score` and an
/// accumulated `node_score`; at decode time it emits a DENSE per-step boost
/// vector (positive forward score on continuations, negative backoff baseline
/// on everything else) that is added to the token logits before argmax. The
/// negative-on-all-others baseline is the cancellation that lets the tree boost
/// hard without flooding false inserts.
///
/// Locked from the off-device NeMo reference (greedy_batch + boosting_tree),
/// which reproduced lung MedWER 0.4064 → 0.3152 at α=2.0. See
/// `docs/asr-research/nemo-boosting-tree-port-spec.md` and the golden fixture.
public enum BoostConstants {
    /// Base score for a phrase's first token (depth 0).
    public static let contextScore: Double = 1.0

    /// Multiplier on `contextScore` for continuation tokens (depth > 0):
    /// `token_score = contextScore * depthScaling + ln(depth + 1)`.
    public static let depthScaling: Double = 2.0

    /// Global boost strength applied to the whole tree vector (`vector = α · advance`).
    /// NeMo's sweet spot was 2.0 in its log-prob scale; the CoreML `jointSingleStep`
    /// raw-logit scale may differ, so re-sweep on-device.
    public static let defaultAlpha: Float = 2.0
}
