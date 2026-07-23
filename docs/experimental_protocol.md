# Experimental protocol

Use one immutable checkpoint per matched comparison. Evaluate every method on identical task arrays and ordering, evaluation keys, decoding, stopping rule, and metric code. Record both training seed and evaluation/search seed. Do not tune on final test tasks.

## Current phase (2026-07-23)

Method implementation uses frozen **pattern_2d** / Hugging Face **lpn-2d** checkpoints. The three local pattern training seeds already satisfy a preliminary multi-seed control for this sandbox. Do not claim ARC-AGI progress from these runs. Official ARC weights are not available on this account; ARC comparisons are optional later under the same protocol with a provenance-clear ARC checkpoint.

The first comparison will contrast `mode=mean` with `mode=gradient_ascent, num_steps=0` to establish whether they are numerically equivalent. Search steps `0, 1, 5, 10, 20` will then be evaluated with the repository's existing objective and optimizer. Any optimizer, initialization, or candidate-count change is a separate axis.

Every run record must include the schema requested in `artifacts/results/`: repository identity, timestamp/host/accelerator, dataset and split, checkpoint hash, method and full hyperparameters, seeds, candidates, steps, compute counters, wall time, metrics, diagnostics, and status/error.

Debugging may use one seed. Preliminary comparisons require at least three independently trained checkpoints; main comparisons require five. Evaluation-key repeats on one checkpoint are reported separately and do not count as independent training seeds.

Fixed-step and compute-matched tables are both required. Until explicit counters exist, decoder-call comparisons must be labeled unavailable rather than inferred loosely.
