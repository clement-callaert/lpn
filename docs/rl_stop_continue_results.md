# RL stop/continue results

Matched evaluation of the first stop/continue REINFORCE controller on frozen
`pattern_2d` checkpoints.

- Setting: **B** (train on checkpoints 0 and 1, evaluate on checkpoint 2)
- Final test: checkpoint seed 2, dataset seed 0, evaluation seed 0, length 24
- Horizon: 5
- Result file: `artifacts/rl/results/settingB_ckpt2_ds0_len24_eval_20260723.json`
- Wall time: about 90 minutes

Pattern results are implementation evidence only. Do not claim ARC-AGI progress,
OOD generalization, or SOTA performance.

## Research question

Can a learned stop/continue policy preserve fixed-5 exact match while using
fewer average search steps?

## MDP summary

- Episode: one leave-one-out query prediction
- Actions: `0 = stop`, `1 = one SGD step`
- Observation: 8 features (no query labels)
- Reward: `exact_match - lambda * search_steps`
- Policy: linear Bernoulli stop logit, REINFORCE + moving-average baseline
- Eval action: stop if `p(stop) >= 0.5`

## Train / validation / test split

| Split | Procedural seeds | Checkpoints | Length |
| --- | --- | --- | --- |
| Train | 1000, 1001 | 0 and 1 | 8 |
| Validation | 2000 | 0 and 1 | 8 |
| Final test | 0 | 2 only | 24 |

These are in-family procedural splits, not OOD.

Penalty `lambda` was selected on validation return only. Selected by max
validation return: `lambda = 0.00`.

## Baseline comparison (checkpoint 2, length 24, 96 episodes)

| Controller | Exact match | Mean steps | Median steps |
| --- | ---: | ---: | ---: |
| Fixed 0 | 0.87500 | 0.000 | 0 |
| Fixed 1 | 0.87500 | 1.000 | 1 |
| Fixed 5 | 0.90625 | 5.000 | 5 |
| Patience 1 | 0.87500 | 2.542 | - |
| Min improvement 0.01 | 0.87500 | 1.042 | - |
| Gradient norm 0.1 | 0.90625 | 3.438 | - |
| Random p=0.2 | 0.89583 | 2.448 | - |
| Oracle earliest-correct | 0.90625 | 0.562 | - |
| Policy lambda=0.00 | 0.90625 | 4.281 | - |
| Policy lambda=0.05 | 0.87500 | 0.385 | - |
| Policy lambda=0.10 | 0.87500 | 0.000 | - |

## Main finding

On this matched length-24 test:

1. **Policy lambda=0.00 matches fixed-5 exact match** (0.90625) with fewer mean
   steps (4.281 vs 5.000). That is a small compute saving at matched accuracy.
2. Higher penalties trade accuracy for cost: lambda=0.05 and 0.10 drop to 0.875
   EM (same as fixed-0/1) while using almost no search.
3. **Oracle** shows that matched fixed-5 EM is possible at about 0.56 mean steps
   if query correctness were known. That is an upper bound only, not a method.
4. Gradient-norm heuristic also matched fixed-5 EM (0.90625) at 3.44 mean steps
   on this split. The RL policy with lambda=0 did not beat that cost.

## Accuracy-cost Pareto (selected lambdas)

| Lambda | Val selection role | Test EM | Test mean steps |
| --- | --- | ---: | ---: |
| 0.00 | selected (best val return) | 0.90625 | 4.281 |
| 0.05 | reported for Pareto | 0.87500 | 0.385 |
| 0.10 | collapsed toward always-stop | 0.87500 | 0.000 |

## Failures and limitations

- Training used short length-8 trajectory splits for speed. Final eval used
  length 24, not the full length-96 protocol.
- Fixed-5 on this length-24 / checkpoint-2 slice is 0.90625, not the earlier
  length-96 three-seed saturation of 1.0.
- Policy lambda=0.10 collapsed to always-stop on the test set.
- Cross-checkpoint transfer was tested once. Seed sensitivity of the policy is
  not fully characterized.
- Offline trajectory REINFORCE does not re-query the environment during
  training; it learns a stopping rule on frozen search trajectories.

## Scientific conclusion for Nathanaël's question

A minimal REINFORCE stop/continue controller can be trained and evaluated on
frozen LPN search. On this sandbox split, lambda=0 preserved fixed-5 exact match
with a modest step reduction. Larger compute penalties did not preserve that
accuracy. This is a real but small prototype result on pattern_2d, not an ARC
result.
