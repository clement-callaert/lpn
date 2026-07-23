# Results

## How to read this file (2026-07-23)

The numbers below are the completed **pattern_2d** three-seed original-search baseline. They are a controlled reproduction and implementation sandbox.

They are **not** ARC-AGI results. pattern_2d saturates after five search steps, so it cannot show large gains from stronger search or RL. Use these tables as the matched control when implementing new test-time methods. Do not present them as ARC progress.

Official ARC checkpoints are not available locally (W&B project not visible; no public HF ARC model). See [`README_RESEARCH.md`](../README_RESEARCH.md) and [`docs/negative_results.md`](negative_results.md).

## Baseline reproduction

The exact source/configuration baseline completed in the documented GPU compatibility environment at commit `0adfe56b86d2cba5ae5794edb02da6399a96d98a`. Three independent 200,000-step training seeds are available. This environment uses JAX/JAXlib 0.6.0 and Flax 0.10.2 and is an explicit hardware-compatibility deviation from the repository pins, not an exact dependency reproduction.

| Training seed | Checkpoint SHA-256 | Wall-clock (s) | Final mean exact match | Final 10-step SGD exact match |
| ---: | --- | ---: | ---: | ---: |
| 0 | `3cb6d3e41e698e1e0e62c785ecdf90d1fb4a600384fd6cf21f3460a110d7de13` | 948.08 | 0.73177 | 1.00000 |
| 1 | `bc292f2cd95365f597dec2a3f61c099a7d5a08177dc56050e8823c3a616f72be` | 942.62 | 0.98177 | 0.99740 |
| 2 | `92ecdd494f036d4ac0a865903316575b207444493ee76d6b7c0b0b601e2ee8fc` | 954.32 | 0.93229 | 1.00000 |

Seed-0 scheduled evaluation trajectory (training-run keys, not the frozen-checkpoint ablation key):

| Training step | Mean exact match | 10-step SGD search exact match |
| ---: | ---: | ---: |
| 20,000 | 0.52344 | 0.81510 |
| 40,000 | 0.76302 | 0.87240 |
| 60,000 | 0.66146 | 0.92969 |
| 80,000 | 0.67188 | 0.97656 |
| 100,000 | 0.85677 | 0.99219 |
| 120,000 | 0.88281 | 0.98958 |
| 140,000 | 0.96875 | 1.00000 |
| 160,000 | 0.94531 | 1.00000 |
| 180,000 | 0.94010 | 1.00000 |
| 200,000 | 0.73177 | 1.00000 |

On frozen checkpoints with dataset seed 0 and evaluation seed 0, mean mode and zero-step gradient ascent match exactly for every training seed tested. Zero-step gradient ascent is therefore the accepted search-off control for this execution path when the key is held fixed. Small differences between training-run final mean accuracy and standalone mean accuracy are expected because the standalone evaluator uses an explicitly recorded evaluation key rather than the key reached after 200,000 training-step splits.

No claim about OOD behavior is supported by `pattern_2d`, which evaluates the same procedural family used for training. Timings include JIT compilation and are not compute-matched. Decoder/objective/gradient-call counters remain unimplemented.

## Seed-0 search-step ablation

With the seed-0 checkpoint frozen and dataset/evaluation seeds both fixed at 0:

| Search steps | Exact match | Pixel correctness |
| ---: | ---: | ---: |
| mean | 0.72396 | 0.97900 |
| 0 | 0.72396 | 0.97900 |
| 1 | 0.88542 | 0.99089 |
| 5 | 1.00000 | 1.00000 |
| 10 | 1.00000 | 1.00000 |
| 20 | 1.00000 | 1.00000 |

Source: `artifacts/results/baseline_search_steps_seed_0.json`.

## Seed-1 search-step ablation

With the seed-1 checkpoint frozen and dataset/evaluation seeds both fixed at 0:

| Search steps | Exact match | Pixel correctness |
| ---: | ---: | ---: |
| mean | 0.98698 | 0.99870 |
| 0 | 0.98698 | 0.99870 |
| 1 | 1.00000 | 1.00000 |
| 5 | 1.00000 | 1.00000 |
| 10 | 1.00000 | 1.00000 |
| 20 | 1.00000 | 1.00000 |

Mean matched zero-step search exactly. Source: `artifacts/results/baseline_search_steps_seed_1.json`.

## Seed-2 search-step ablation

With the seed-2 checkpoint frozen and dataset/evaluation seeds both fixed at 0:

| Search steps | Exact match | Pixel correctness |
| ---: | ---: | ---: |
| mean | 0.93490 | 0.99300 |
| 0 | 0.93490 | 0.99300 |
| 1 | 0.99479 | 0.99967 |
| 5 | 1.00000 | 1.00000 |
| 10 | 1.00000 | 1.00000 |
| 20 | 1.00000 | 1.00000 |

Mean matched zero-step search exactly. Source: `artifacts/results/baseline_search_steps_seed_2.json`.

## Three-seed aggregated search-step ablation

Across the three independent **training** seeds (not evaluation-seed repeats), with dataset seed 0, evaluation seed 0, learning rate 0.1, and repository-default SGD:

| Search steps | Mean exact match | Std | Min | Max | Seed 0 | Seed 1 | Seed 2 | Δ vs 0-step mean | Seeds at 1.0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.88194 | 0.13928 | 0.72396 | 0.98698 | 0.72396 | 0.98698 | 0.93490 | 0.00000 | 0 |
| 1 | 0.96007 | 0.06470 | 0.88542 | 1.00000 | 0.88542 | 1.00000 | 0.99479 | +0.07813 | 1 |
| 5 | 1.00000 | 0.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | +0.11806 | 3 |
| 10 | 1.00000 | 0.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | +0.11806 | 3 |
| 20 | 1.00000 | 0.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | +0.11806 | 3 |

- Improvement over zero-step search: mean exact-match rises by about 0.078 at 1 step and by about 0.118 at 5+ steps.
- Saturation point: 5 search steps (smallest evaluated budget at which all three seeds and the three-seed mean reach perfect exact match).
- Seeds reaching perfect accuracy: 0 at 0 steps, 1 at 1 step, 3 at 5/10/20 steps.

Machine-readable aggregate: `artifacts/results/baseline_search_steps_three_seed_aggregate.json`.

### Allowed interpretation

On in-family `pattern_2d`, the original deterministic latent gradient search improves exact-match over the zero-step control for every independently trained checkpoint tested, and saturates at perfect accuracy by 5 steps under this fixed evaluation protocol. This does **not** establish OOD improvement, superiority of stochastic or proximal search, compute efficiency, statistical significance beyond descriptive three-seed dispersion, or general program-synthesis gains.

## Smoke note

The one-step smoke experiment completed the training and evaluation paths with zero exact-match accuracy after one training step. That result is an execution check only; see `artifacts/results/smoke_pattern_2d_seed_0.json` and `artifacts/results/smoke_pattern_2d_gpu_compat_seed_0.json`.
