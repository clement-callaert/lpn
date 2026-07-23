# Stochastic latent-search research branch

This branch investigates whether stochastic, proximal, or population-based test-time latent search improves out-of-distribution program induction over the original deterministic search. The official repository is the control group: dataset generation, encoder/decoder architecture, latent dimensionality, training objective, representation, decoding, and official metrics remain unchanged during the first contribution.

This document is the branch-specific entry point. See the upstream `README.md` for the original project and [postmortem.md](postmortem.md) for the chronological session record.

## Current state

- Branch: `research/stochastic-latent-search`
- Control commit: `0adfe56b86d2cba5ae5794edb02da6399a96d98a`
- Upstream: `https://github.com/clement-bonnet/lpn.git`
- Model/search source changes: none
- Completed full training seeds: 0, 1, and 2
- Completed search-step ablations: seeds 0, 1, and 2 (dataset seed 0, evaluation seed 0)
- Three-seed original-search baseline: complete on in-family `pattern_2d`
- Next required work: review and commit the baseline documentation/evaluator/results, then only after that begin new search-method implementations

The exact pinned CPU environment is `.venv`. The RTX 5090 cannot compile the repository-pinned JAX 0.4.26 CUDA build, so GPU work uses the explicitly documented `.venv-gpu` compatibility environment: JAX/JAXlib 0.6.0, Flax 0.10.2, Optax 0.2.2, and Torch 2.4.1+cpu. Its complete freeze is `artifacts/environment/pip_freeze_gpu_compat.txt`.

## Reproduction status

The unchanged official command is:

```bash
PYTHONPATH="$PWD" python src/train.py --config-name pattern_2d
```

An exact-pin CPU attempt initialized correctly but could not finish its first 1,000-step block after about 18 minutes. In the GPU compatibility environment, the unchanged 200,000-step runs completed as follows:

| Training seed | Checkpoint SHA-256 | Final mean exact match | Final 10-step SGD exact match |
| ---: | --- | ---: | ---: |
| 0 | `3cb6d3e41e698e1e0e62c785ecdf90d1fb4a600384fd6cf21f3460a110d7de13` | 0.73177 | 1.00000 |
| 1 | `bc292f2cd95365f597dec2a3f61c099a7d5a08177dc56050e8823c3a616f72be` | 0.98177 | 0.99740 |
| 2 | `92ecdd494f036d4ac0a865903316575b207444493ee76d6b7c0b0b601e2ee8fc` | 0.93229 | 1.00000 |

Frozen-checkpoint step ablations with common dataset seed 0 and evaluation key 0:

| Search steps | Seed 0 | Seed 1 | Seed 2 | Mean | Std |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.72396 | 0.98698 | 0.93490 | 0.88194 | 0.13928 |
| 1 | 0.88542 | 1.00000 | 0.99479 | 0.96007 | 0.06470 |
| 5 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0.00000 |
| 10 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0.00000 |
| 20 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0.00000 |

Mean and zero-step gradient ascent match exactly under the same evaluation key for every seed. Saturation under this protocol is at 5 steps (all three seeds perfect). First-call timings include separate JIT compilation and are not compute-matched inference measurements. See [docs/results.md](docs/results.md) for the full aggregate table and interpretation limits.

## Resume procedure

Start from the repository root and confirm state before running anything:

```bash
git -c safe.directory="$PWD" rev-parse HEAD
git -c safe.directory="$PWD" status --short
.venv-gpu/bin/python -m pip check
sha256sum artifacts/checkpoints/pattern_2d_seed_{0,1,2}.msgpack
```

If git reports dubious ownership when launching W&B, prefer a one-shot environment override rather than writing git config:

```bash
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="$PWD"
```

The three-seed baseline is complete. Do not retrain seeds 0–2 unless a scientific defect is found. New search methods begin only after the baseline documentation and evaluator are reviewed and committed.

## Evidence map

- Environment: `artifacts/environment/`
- Full logs: `artifacts/logs/`
- Machine-readable records: `artifacts/results/`
- Local ignored checkpoints: `artifacts/checkpoints/`
- Search execution map: `docs/codebase_map.md`
- Reproduction details: `docs/baseline_reproduction.md`
- Experimental controls: `docs/experimental_protocol.md`
- Results: `docs/results.md`
- Failures and negative results: `docs/negative_results.md`
- Chronological handoff: `postmortem.md`

## Scientific guardrails

- Do not compare methods across different checkpoints, task arrays, keys, decoding, stopping rules, or unreported compute budgets.
- Do not count evaluation-key repeats as independent training seeds.
- Do not tune on a final test set.
- `pattern_2d` is in-family and does not answer the OOD hypothesis.
- Decoder/objective/gradient-call counters remain unimplemented, so current wall times are diagnostic only.
- Do not modularize or add Adam, proximal, Langevin, or population methods until the three-seed original-search baseline is reviewed and committed.

## Planned first commits

Keep the eventual history reviewable:

1. `docs: record baseline environment and latent-search path`
2. `exp: add local checkpoint baseline evaluator`
3. `exp: record three-seed latent-search step ablation`

Do not commit `.venv*`, `state.msgpack`, `artifacts/checkpoints/`, W&B caches, secrets, or generated Hydra output directories.
