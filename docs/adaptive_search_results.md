# Adaptive search results (pattern_2d sandbox)

Non-RL adaptive stopping results on frozen `pattern_2d` checkpoints.

- Branch tip during runs: `063d4522be004adeeb619e630bf8b82b181fa467`
- Config: `outputs/2026-07-22/22-55-12/.hydra/config.yaml`
- Evaluation protocol: dataset seed 0, evaluation seed 0, length 96, SGD lr 0.1
- Development protocol: dataset seed 100, length 48 (threshold selection only)
- Fixed-K path: Trainer-matched batched `generate_output` (exact-match control)
- Adaptive path: same batched variational encode, then closed-loop SGD per leave-one-out

This is an implementation sandbox. It is not an ARC-AGI result and not an
out-of-distribution claim.

## 1. Integration regression

Command:

```bash
PYTHONPATH="$PWD" .venv-gpu/bin/python scripts/evaluate_adaptive_search.py \
  --config outputs/2026-07-22/22-55-12/.hydra/config.yaml \
  --checkpoint artifacts/checkpoints/pattern_2d_seed_0.msgpack \
  --output artifacts/results/adaptive_fixed5_seed0_final_20260723.json \
  --dataset-seed 0 --evaluation-seed 0 --length 96 \
  --stopping-rule fixed --max-search-steps 5 --learning-rate 0.1
```

Result: exact match `1.0`, shape `1.0`, pixel `1.0`.
Also rechecked via `scripts/evaluate_pattern_checkpoint.py` with search steps `{5}`: exact match `1.0`.

Early debug runs that used unmatched PRNG or dataset ordering are retained under
unique filenames and must not be used in matched tables.

## 2. Unit tests

```bash
PYTHONPATH="$PWD" .venv-gpu/bin/python -m unittest discover -s tests/search -v
```

Result: 27 tests OK.

## 3. Development split (threshold selection)

Construction:

- Procedural PATTERN tasks, `dataset_seed=100`, `length=48`, `num_pairs=4`
- Same family as evaluation tasks (in-family leakage limitation)
- Used only to choose `min_improvement` and `gradient_norm_threshold`
- Checkpoint: `pattern_2d_seed_0.msgpack`

| Rule | Setting | Exact match | Mean steps | Grad evals |
| --- | --- | ---: | ---: | ---: |
| fixed | 1 | 0.98958 | 1.000 | 192 |
| fixed | 5 | 0.99479 | 5.000 | 960 |
| min_improvement | 0.01 / 0.1 / 1.0 | 0.85417 | ~1.00-1.04 | 192-199 |
| gradient_norm | 0.1 | 0.90104 | 14.719 | 2826 |
| gradient_norm | 0.01 | 0.90104 | 15.422 | 2961 |
| gradient_norm | 0.001 | 0.90104 | 18.958 | 3640 |

Selected for final evaluation:

- `min_improvement = 0.01` (least aggressive among equal-accuracy min-imp settings)
- `gradient_norm_threshold = 0.1` (fewest mean steps among equal-accuracy grad settings)

Neither selected rule reached fixed-5 development accuracy.

## 4. Evaluation results (matched)

Leave-one-out examples per run: 96 tasks * 4 = 384.
Fixed-5 compute reference: 1920 gradient evaluations.

### Seed 0 checkpoint

| Rule | Exact match | Mean steps | Median | Min | Max | Std | Grad evals | vs fixed-5 grads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed 1 | 0.88542 | 1.000 | 1 | 1 | 1 | 0 | 384 | -80% |
| fixed 5 | 1.00000 | 5.000 | 5 | 5 | 5 | 0 | 1920 | 0% |
| fixed 20 | 1.00000 | 20.000 | 20 | 20 | 20 | 0 | 7680 | +300% |
| patience 1 | 0.78906 | 2.169 | 1 | 1 | 18 | - | 833 | -57% |
| patience 2 | 0.81510 | 5.411 | 4 | 2 | 20 | - | 2078 | +8% |
| min_improvement 0.01 | 0.78906 | 1.076 | 1 | 1 | 3 | - | 413 | -78% |
| gradient_norm 0.1 | 0.83854 | 13.852 | 20 | 1 | 20 | - | 5319 | +177% |

### Three-seed comparison (selected rules)

| Rule | Seed 0 EM | Seed 1 EM | Seed 2 EM | Mean EM | Mean steps (0/1/2) |
| --- | ---: | ---: | ---: | ---: | --- |
| fixed 5 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 5.0 / 5.0 / 5.0 |
| patience 1 | 0.78906 | 0.98958 | 0.91667 | 0.89844 | 2.17 / 3.12 / 3.18 |
| min_improvement 0.01 | 0.78906 | 0.98958 | 0.91667 | 0.89844 | 1.08 / 1.00 / 1.01 |
| gradient_norm 0.1 | 0.83854 | 0.99219 | 0.96354 | 0.93142 | 13.85 / 9.58 / 12.99 |

## 5. Interpretation

Central question:

> Can an adaptive controller preserve the accuracy of fixed five-step search
> while using fewer average latent-search steps?

On this pattern_2d sandbox, the tested non-RL rules did **not**.

- Fixed 5 remains the accuracy control (EM 1.0 on all three seeds).
- Patience and min-improvement reduce average steps but lose exact match,
  especially on seed 0.
- Gradient-norm stopping often uses **more** than 5 steps and still fails to
  match fixed-5 accuracy.
- Support-score / gradient signals are imperfect proxies for query exact match.

Per-task step distributions show early stopping on many leave-one-out examples
(median 1 for patience/min-imp) while a minority consume the full budget.

## 6. Failures and root causes

1. Early unmatched fixed-5 harness runs (EM ~0.83-0.99): wrong PRNG key protocol
   and/or missing Trainer dataset permutation / worker settings. Fixed by matching
   Trainer key split `(1, 1)`, `num_workers=8`, and post-generation permutation.
2. First patience run crashed with `UnboundLocalError` on an unused `base_fn`
   delete inside the jitted latent-conditioned search. Fixed and re-run.
3. No experiment file was overwritten; failed outputs were removed and replaced
   under the same unique planned names only after deletion.

## 7. Scientific limitations

- In-family PATTERN only; saturated under fixed 5 steps.
- Development and evaluation splits are the same task family.
- Adaptive closed-loop SGD mirrors the Optax update but is not bit-identical to
  every internal scan detail of upstream GA for partial trajectories.
- Extra post-update objective evaluations used by adaptive stopping are reported
  separately as `extra_objective_evaluations_total`.
- Do not claim ARC-AGI progress, OOD improvement, or SOTA performance.

## 8. Minimal RL interface (next session)

Documented in [`adaptive_search_design.md`](adaptive_search_design.md):

- Actions: stop / continue one SGD step
- Observation: step, scores, deltas, norms, remaining budget (no query labels)
- Preferred reward: terminal EM minus lambda * steps
- Start with logistic or small MLP + REINFORCE after these non-RL tables

## 9. Artifact index

Primary matched outputs:

- `artifacts/results/adaptive_fixed{1,5,20}_seed0_*.json`
- `artifacts/results/adaptive_fixed5_seed{1,2}_20260723.json`
- `artifacts/results/adaptive_patience{1,2}_seed0_20260723.json`
- `artifacts/results/adaptive_patience1_seed{1,2}_20260723.json`
- `artifacts/results/adaptive_minimp0.01_seed{0,1,2}_20260723.json`
- `artifacts/results/adaptive_grad0.1_seed{0,1,2}_20260723.json`
- `artifacts/results/adaptive_dev_*.json`

Logs: `artifacts/logs/adaptive_*.log`
