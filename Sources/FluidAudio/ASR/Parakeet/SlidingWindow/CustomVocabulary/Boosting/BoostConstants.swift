import Foundation

/// Defaults for trie-based Parakeet vocabulary boosting.
///
/// Tuned on Mediform/medical_asr_test (German doctor-patient conversations,
/// 14 × ~5 min). Sweep results: `pymlx/run-outputs/SWEEP_SUMMARY.md`.
///
/// Semantics match Canary's deprecated CanaryBoostProcessor so the literature
/// on shallow-fusion context biasing applies directly.
public enum BoostConstants {
    /// Additive log-prob bonus for any token that starts a boosted term.
    /// Lets the decoder 'enter' the trie regardless of context.
    public static let defaultBaseBoost: Float = 0.5

    /// Per-step bonus for tokens that extend a live trie prefix.
    /// Scaled by `(1 + lookback * 0.25)` — longer matched prefixes earn more.
    public static let defaultSequenceBoost: Float = 3.0

    /// How many emitted tokens to consider as a possible prefix when looking
    /// up trie continuations. Longer = more context, more compute per step.
    public static let defaultMaxPrefixLen: Int = 10
}
