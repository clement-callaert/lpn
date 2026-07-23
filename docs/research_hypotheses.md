# Research hypotheses

Long-term hypotheses (from the research proposal):

- H1: multi-start or stochastic latent search improves performance when encoder initialization is in a poor basin.
- H2: gains are larger on fixed out-of-distribution tasks than on in-distribution tasks.
- H3: a proximal penalty centered at the encoder initialization reduces observed-example overfitting and improves query generalization.
- H4: multiple candidates preserve program ambiguity that a single latent cannot express.
- H5: gains remain meaningful after decoder calls, gradient/objective evaluations, wall time, candidate count, and total compute are accounted for.

## Current phase (2026-07-23)

The active goal is to **implement** test-time controllers (compute counters, non-RL baselines, then RL stop/continue), not to confirm H1-H5 on ARC-AGI.

- Control platform: frozen `pattern_2d` / Hugging Face `lpn-2d`.
- Official ARC weights are not available on this account.
- pattern_2d is saturated and in-family; results there support **method correctness**, not H2/ARC claims.
- H5 cannot be tested until explicit compute counters exist.

The official implementation remains the frozen-model control. New methods must not silently change architecture, training objective, greedy decoding, or official metrics.
