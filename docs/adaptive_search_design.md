# Adaptive search design

Design for algorithmic compute accounting, non-RL stopping rules, closed-loop
adaptive SGD search, and a minimal future RL stop/continue interface.

- Branch tip at design time: `063d4522be004adeeb619e630bf8b82b181fa467`
- Upstream control: `0adfe56b86d2cba5ae5794edb02da6399a96d98a`
- Development platform: frozen `pattern_2d` checkpoints
- Related: [`rl_test_time_compute_formulation.md`](rl_test_time_compute_formulation.md)

This document does not claim ARC-AGI improvement. Pattern results are
implementation / sandbox evidence only.

## 1. Upstream fixed-K gradient ascent (control)

Source: [`LPN._get_gradient_ascent_context`](../src/models/lpn.py).

Default pattern settings:

- `include_mean_latent=True`, `include_all_latents=False` => `M = 1` candidate
- `accumulate_gradients_decoder_pairs=False`
- `scan_gradients_latents=False`
- optimizer: `clip_by_global_norm(1.0)` then SGD
- search length: Flax `nn.scan(..., length=num_steps)` (static under JIT)

Per scan step `t = 0 .. K-1`:

1. `value_and_grad` scores current latent `z_t` and returns `grad_t`
2. Optax update produces `z_{t+1}`
3. Emit `(z_{t+1}, score(z_t))`

After the scan:

- Candidate stack: `[z_0, z_1, ..., z_K]`
- Score stack: scores of `z_0 .. z_{K-1}` from the scan, plus a final
  `vmap_log_probs_fn(z_K)` rescoring
- Selection: argmax over the `M * (K + 1)` retained slots

When `K = 0`, there are no gradient steps. Only the init latent is scored once
by the final rescoring path.

The upstream path is the scientific control. Do not change its numerical
behavior. Adaptive search is implemented in [`src/search/`](../src/search/).

## 2. Compute counters

Counters measure logical algorithmic work from array shapes, candidates, steps,
and execution paths. They are not Python call counts. Wall time is measured on
the host only and is never stored inside the JAX counter structure.

### Definitions

| Counter | Meaning |
| --- | --- |
| `search_steps_executed` | Number of Optax latent updates actually run |
| `objective_evaluations` | Full support-set score computations |
| `gradient_evaluations` | Latent gradients from `value_and_grad` |
| `decoder_support_evaluations` | Teacher-forced decoder forwards used to score support pairs |
| `latent_candidates_evaluated` | Candidate starts after `_prepare_latents_before_search` |
| `candidate_step_exposure` | Sum over retained (candidate, iterate) slots |
| `output_decoding_calls` | Calls to greedy `_generate_output_from_context` |
| `autoregressive_token_predictions` | Shape tokens plus grid tokens predicted during greedy decode |

### Default single-candidate SGD formulas

Let `M` be the number of candidates and `K` the number of executed search steps.
Assume no pair-scan and no latent-scan (pattern default).

| Counter | Formula |
| --- | --- |
| `search_steps_executed` | `K` |
| `gradient_evaluations` | `M * K` |
| `objective_evaluations` | `M * (K + 1)` |
| `decoder_support_evaluations` | `M * (K + 1)` |
| `latent_candidates_evaluated` | `M` |
| `candidate_step_exposure` | `M * (K + 1)` |
| `output_decoding_calls` | `1` (or `2` if `return_two_best`) |
| `autoregressive_token_predictions` | `(2 + max_rows * max_cols) * output_decoding_calls` |

Why `objective_evaluations = M * (K + 1)`:

- Each of the `K` updates scores the pre-update latent inside `value_and_grad`
- The final latent `z_K` is scored once more without a gradient
- When `K = 0`, only that final score remains, so the count is `M`

### Pair-scan and multi-candidate notes

If `accumulate_gradients_decoder_pairs=True`, one logical objective evaluation
still counts as one `objective_evaluations` entry, but
`decoder_support_evaluations` becomes `N * objective_evaluations` where `N` is
the number of support pairs.

One vectorized decoder execution over `M` candidates still represents `M`
logical candidate evaluations.

### Double-count risks

- Do not count the final rescoring as a gradient evaluation
- Do not mix teacher-forced support scoring with autoregressive query decoding
- Do not treat a `vmap` over candidates as one logical evaluation
- Do not use wall time as a substitute for these counters

### Implementation location

- Structure and helpers: [`src/search/compute_accounting.py`](../src/search/compute_accounting.py)
- Fill after adaptive or fixed closed-loop search from executed `K` and `M`
- Host serialization via `compute_counters_to_dict`

## 3. Closed-loop adaptive search

Upstream `nn.scan` has a static length, so it cannot early-stop. Adaptive search
uses a closed-loop runner in
[`src/search/adaptive_sgd_search.py`](../src/search/adaptive_sgd_search.py):

1. Encode support pairs once
2. Prepare candidates (`M = 1` by default)
3. Score the init latent
4. While the stopping rule says continue and `step < max_steps`:
   - `value_and_grad` on the support objective
   - one SGD step with the same clip + SGD chain as upstream
   - update rule state and counters
5. Select the best retained iterate
6. Greedy-decode the query once

Fixed-step behavior is the closed-loop runner with a rule that always continues
until `max_steps`. Parity against upstream `generate_output(..., mode="gradient_ascent")`
is required before scientific comparisons.

## 4. Non-RL stopping rules

Pure functions in [`src/search/stopping_rules.py`](../src/search/stopping_rules.py).
A rule never sees query ground truth.

Rules:

1. `fixed`: stop only at `max_steps`
2. `patience`: stop after `P` steps without beating the best support score
3. `min_improvement`: stop when `current_score - previous_score < eps`
   (equality counts as no improvement)
4. `gradient_norm`: stop when gradient global norm is below a threshold
5. `patience_with_max_budget`: patience plus a hard max-step budget

All rules stop at `max_steps`. Non-finite scores or gradient norms trigger a
safe stop.

Tune thresholds on a development split only:

- Development: `dataset_seed=100`, `length=48`
- Final evaluation: `dataset_seed=0`, `length=96`, `evaluation_seed=0`

Both splits are in-family PATTERN tasks. Development tuning is not a claim of
out-of-distribution generalization.

## 5. Evaluation harness

[`scripts/evaluate_adaptive_search.py`](../scripts/evaluate_adaptive_search.py)
reuses model construction, checkpoint loading, PATTERN generation, the support
objective, greedy decoding, and the official leave-one-out metric formulas.

It does not overwrite existing result files. Each run uses a unique output path.
Records include git metadata, environment, checkpoint SHA-256, rule config,
aggregate metrics, per-task step counts, stop reasons, and compute counters.

## 6. Minimal RL interface (not trained in this session)

One task = one episode. First action space:

- `0`: stop and decode
- `1`: continue with one original SGD step

Observation (no query labels):

- normalized current step
- current support score
- score improvement from previous step
- best score improvement so far
- gradient norm
- latent update norm
- latent norm
- remaining budget

Normalize observation statistics on development tasks only.

Suggested training rewards:

- Reward A / C: terminal query exact match minus `lambda * search_steps`
- Reward B (shaped support improvement): risk of overfitting support pairs;
  not the primary training reward

Evaluation metric remains official exact match. Query labels may be used only
as a terminal training reward on train-side / development tasks, never as an
online observation at test time.

Next RL implementation should start with a logistic or small MLP policy and a
simple REINFORCE / contextual baseline after non-RL tables exist. Do not start
with PPO or SAC.

## 7. Scientific question on pattern_2d

Can an adaptive controller preserve the accuracy of fixed five-step search while
using fewer average latent-search steps?

This is an implementation question on an in-family sandbox. It is not an ARC
result.

## 8. Integration regression (expensive)

After the harness exists, verify the seed-0 five-step control still reaches
exact match `1.0` under the same checkpoint, dataset seed, and evaluation seed.
Document the command in [`adaptive_search_results.md`](adaptive_search_results.md).
Do not put the full 96-task GPU run in the default unit suite.
