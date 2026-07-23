# RL stop/continue implementation

Implementation notes for the first reinforcement-learning stop/continue
controller on frozen `pattern_2d` checkpoints.

- Branch tip at implementation time: inspect with `git rev-parse HEAD`
- Platform: frozen pattern_2d / lpn-2d
- Horizon: 5 search steps
- This document does not claim ARC-AGI progress

## 1. Trajectory parity

Closed-loop fixed-K search lives in
[`src/search/adaptive_sgd_search.py`](../src/search/adaptive_sgd_search.py):

- `one_sgd_step`: one clipped SGD update matching the adaptive body
- `run_closed_loop_fixed_k_with_trajectory`: always runs exactly K steps and
  returns latent/score/grad trajectories

Root cause fixed for init-latent mismatch: upstream `generate_output` splits the
PRNG key before variational sampling. The closed-loop path now does the same.

Parity protocol (checkpoint seed 0, synthetic LOO batch, shared eval key, lr
0.1):

| K | output grids | output shapes | max abs context diff | notes |
| --- | --- | --- | --- | --- |
| 0 | bit-identical | bit-identical | about 1e-4 under separate JIT | encode matches bit-exactly outside dual JIT |
| 1 | bit-identical | bit-identical | about 1e-4 | numerically close |
| 5 | bit-identical | bit-identical | about 4e-4 | numerically close |

Documented tolerances used in tests: `atol=1e-3`, `rtol=1e-3`.

Bit-identical fields: decoded grids and shapes.

Numerically close fields: selected context / latents under separate JIT
compilations.

Upstream does not export per-step gradients. Intermediate trajectory fields are
recorded from the closed-loop path that matches upstream finals.

## 2. Offline trajectories

Script: [`scripts/generate_search_trajectories.py`](../scripts/generate_search_trajectories.py)

Each leave-one-out episode stores steps `t = 0 .. max_search_steps` with support
diagnostics plus label-only fields:

- `query_exact_match_if_stopped_now`
- `query_pixel_correctness_if_stopped_now`

Those labels never enter the policy observation.

## 3. Environment

[`src/rl/environment.py`](../src/rl/environment.py)

- Action 0: stop and decode best latent so far
- Action 1: one SGD step; force-stop at `max_steps`
- Terminal reward: `exact_match - compute_penalty * search_steps_executed`
- Query labels stored only for terminal reward

## 4. Observation

Eight features in fixed order ([`src/rl/observations.py`](../src/rl/observations.py)):

1. normalized_current_step
2. normalized_remaining_budget
3. current_support_score
4. score_improvement
5. best_score_improvement
6. gradient_norm
7. latent_norm
8. latent_update_norm

Normalization statistics come only from the policy training split. Zero variance
is replaced by `1.0` and recorded.

## 5. Policy and REINFORCE

- Linear Bernoulli stop logit ([`src/rl/policy.py`](../src/rl/policy.py))
- Init bias toward continue (`continue_bias=1.5`, negative stop logit bias)
- REINFORCE with moving-average return baseline
- Optional entropy bonus
- Adam + global grad clip
- Non-finite loss aborts

Training uses offline trajectories: the frozen search dynamics are precomputed,
and the policy only chooses when to stop.

## 6. Scripts

- `scripts/generate_search_trajectories.py`
- `scripts/train_stop_policy.py`
- `scripts/evaluate_stop_policy.py`

All refuse to overwrite existing outputs and write provenance fields.
