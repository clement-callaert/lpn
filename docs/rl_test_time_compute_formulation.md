# RL formulation for test-time compute in Latent Program Networks

Formal research formulation for controlling test-time latent search compute with reinforcement learning. This document does not implement a policy. It is grounded in the checked-out LPN code and the ARC audit in [`docs/arc_baseline_audit.md`](arc_baseline_audit.md).

- Branch tip: `063d4522be004adeeb619e630bf8b82b181fa467`
- Upstream control: `0adfe56b86d2cba5ae5794edb02da6399a96d98a`
- Reference paper: Searching Latent Program Spaces (arXiv:2411.08706)

## Scope for the current phase (2026-07-23)

**Goal:** implement new test-time controllers (RL and non-RL). Significant ARC-AGI accuracy progress is not required in this phase.

**Development domain:** frozen `pattern_2d` / Hugging Face `lpn-2d` checkpoints. The MDP below is written in ARC language because that is the long-term target, but the same formulation applies to PATTERN tasks (one task = one episode; support pairs; query inputs; no query labels in the observation).

**Do not claim ARC-AGI improvement from pattern results.** Pattern is saturated after a few search steps and is in-family.

Official ARC weights remain unavailable locally (private W&B project; no public HF ARC model). See [`README_RESEARCH.md`](../README_RESEARCH.md).

## 1. Problem statement

At inference time, LPN encodes support pairs into latents, optionally searches in latent space by maximizing a support-set decoder score, then greedily decodes query outputs. Search already exposes step counts, optimizer choice, multi-start candidates, and random perturbations. The open research question is how to allocate that test-time compute per task.

Goal of the controller: improve the tradeoff between task success and measured search compute, without changing the encoder/decoder architecture, training objective, greedy decoder, or official metrics in the first contribution. Retraining the base model is not required.

## 2. Primary formulation: finite-horizon MDP with stop action

**Choice.** Treat one ARC task as one episode of a finite-horizon Markov decision process (MDP) with an explicit stop-and-decode action. Optimal stopping is the subclass of policies that only choose "continue with a default update" versus "stop". Fixed-step budgets are further restricted policies that ignore the state and always act for K steps.

This matches the sequential structure of [`LPN._get_gradient_ascent_context`](../src/models/lpn.py), where each Optax update is a natural transition.

### Episode

- One ARC task from a **train-side** policy-learning split (proposal in the ARC audit).
- Support pairs: `task["train"]` (JSON) or `num_pairs` re-ARC pairs.
- Query inputs: `task["test"]` inputs are visible.
- Query solutions: **not** visible to the policy at decision time.
- Horizon: maximum search budget B (for example max gradient steps or max objective evaluations).

### State and observation

The true latent state includes candidate latents, optimizer state, and decoder parameters (frozen). The policy observation may only use information available without query labels.

Candidate observation variables (several need new instrumentation before use):

| Variable | Available today? | Source |
| --- | --- | --- |
| Current search step t | yes (controller-side) | loop index / `num_steps` |
| Remaining budget B - t | yes (controller-side) | chosen budget |
| Support log-likelihood score | internal in GA/RS | [`_compute_log_probs`](../src/models/lpn.py) |
| Score improvement since last step | not exposed | difference of successive scores |
| Gradient global norm | computed inside clip, not returned | Optax `clip_by_global_norm` in GA |
| Latent update norm | not exposed | `||z_{t+1}-z_t||` |
| Latent / context norm | training metrics only in `__call__` | not in `generate_output` info |
| Top-1 vs top-2 score gap | internal before selection | [`_select_best_and_second_best_latents`](../src/models/lpn.py) |
| Candidate disagreement | not exposed | distance among top candidates |
| Restart / candidate count used | yes from kwargs | `_prepare_latents_before_search` |
| Predicted shape confidence / token entropy | not exposed at search time | would need decoder logits hooks |
| Query grid (pixels) | yes | task test input |
| Query solution | **forbidden in observation** | solutions JSON |

Do not put evaluation-set solutions, or any query ground truth, into the observation or online reward.

### Actions

#### Already expressible with existing repository knobs

| Action | Existing mechanism |
| --- | --- |
| Stop and decode with current best latent | set remaining steps to 0; call greedy decode |
| Take one (or K) SGD steps | `optimizer: sgd`, `num_steps` |
| Take one (or K) Adam steps | `optimizer: adam`, `optimizer_kwargs` |
| Change learning rate / cosine schedule | `lr`, `lr_schedule`, `lr_schedule_exponent` |
| Restart from encoder mean or all pair latents | `include_mean_latent`, `include_all_latents` |
| Sample perturbed starts | `random_perturbation` |
| Random-search candidates | `mode=random_search`, `num_samples`, `scale` |
| Return top-1 only vs keep top-2 attempts | decoding already returns two best; metric can use top-1 or top-2 |

Fixed-step evaluation today is open-loop: the action sequence is chosen before the episode via `inference_kwargs`, not closed-loop per step. Closed-loop control requires a new outer loop later. That is future implementation work, not part of this session.

#### Requires new implementation later

| Action | Why new |
| --- | --- |
| Per-step adaptive stop inside compiled GA scan | current `nn.scan` length is static |
| Learned continuous LR controller | no policy module exists |
| Dynamic prune/keep beyond final argsort | selection is end-of-search only |
| Hierarchical multi-restart with learned restart policy | only static multi-start exists |
| Non-greedy decoding | would change an immutable control for matched tables |

### Transition

For a continue action: update candidate latents with the chosen optimizer step, recompute support scores, advance t. For stop: freeze latents, run [`_generate_output_from_context`](../src/models/lpn.py), end episode.

### Reward (candidates)

Online rewards may use only train-side labels from the policy-learning split. ARC evaluation solutions are reserved for the final frozen report.

#### Reward A: terminal accuracy minus compute

`R = ExactMatch_train_split - lambda * gradient_evaluations`

| Question | Assessment |
| --- | --- |
| Available during policy training? | Yes, on train-side tasks with solutions |
| Leaks evaluation labels? | No if restricted to train-side split |
| Aligns with ARC exact match? | Directly on the proxy split; eval alignment is empirical |
| Encourages premature stopping? | Yes when lambda is large |
| Gameable via support overfitting? | Indirectly: policy may stop when support score is high even if query fails |

#### Reward B: step-wise support improvement minus decoder cost

`r_t = (score_t - score_{t-1}) - lambda * decoder_tf_calls_t` with optional terminal exact-match bonus

| Question | Assessment |
| --- | --- |
| Available during policy training? | Support terms yes without query labels; terminal bonus needs train-side solutions |
| Leaks evaluation labels? | No if train-side only |
| Aligns with ARC exact match? | Weakly; support likelihood is a proxy |
| Encourages premature stopping? | Less than pure cost penalties if improvement remains positive |
| Gameable via support overfitting? | **Yes.** Maximizing support score can overfit pairs and hurt query exact match |

#### Reward C: sparse terminal success with constant step cost

`r_t = -lambda` each continue step; terminal `R = 1` on train-side exact match else `0`

| Question | Assessment |
| --- | --- |
| Available during policy training? | Yes on labeled train-side tasks |
| Leaks evaluation labels? | No if train-side only |
| Aligns with ARC exact match? | Strong on the proxy split |
| Encourages premature stopping? | Yes; explores only if success probability gain exceeds lambda |
| Gameable via support overfitting? | Lower than Reward B because terminal signal is query exact match |

**Final evaluation reward.** Official ARC evaluation `top_1_accuracy` / `top_2_accuracy` from [`Evaluator.evaluate_generations`](../src/evaluator.py) is offline only. It must not be used for online RL updates, early stopping of policy training, or checkpoint picking.

## 3. Non-RL baselines (mandatory before RL)

Tune all adaptive thresholds on a **train-side validation carve-out**. Never tune on the ARC evaluation set.

| Baseline | Observation | Decision rule | Compute budget | Tunable parameters | Validation protocol | Expected failure mode |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed steps | none | always run K steps | K in {0,1,5,10,20,50,200} | K | select K on train-val | wastes compute on easy tasks; under-searches hard tasks |
| Support-score improvement threshold | score delta | stop when delta < eps | max K | eps, K | train-val | stops early on flat but useful landscapes |
| Gradient-norm stop | grad norm | stop when norm < eps | max K | eps, K | train-val | ignores score; sensitive to clipping |
| No-improvement patience | best score so far | stop after P steps without improvement | max K | P, K | train-val | noisy scores cause premature stop |
| Fixed compute budget | counters | stop when objective/grad count hits C | C | C | train-val | needs counters first |
| Random budget allocation | none | sample K from a fixed distribution | random K | distribution | train-val | high variance; weak but important control |
| Oracle budget | full curve including labels | pick best K after seeing outcomes | varies | none usable online | analysis only | **not a deployable method** |
| Existing SGD | open-loop kwargs | fixed SGD GA | K, lr | K, lr | train-val | may be slower/less stable than Adam on ARC configs |
| Existing Adam | open-loop kwargs | fixed Adam GA as in `arc_train.yaml` | K, lr, b2 | K, lr, b2 | train-val | can overfit support with large K |
| Existing random / perturbed multi-start | candidates | score many starts, pick top | samples, scale | samples, scale | train-val | misses local refinement that GA provides |

The oracle is an upper-bound diagnostic only. Do not present it as a usable test-time method.

## 4. Compute accounting design (not implemented in this session)

No compute-efficiency claim is allowed until these counters exist. Count algorithmic work inside JIT/`vmap`/`scan`/`pmap`, not only Python call counts.

| Counter | File | Function | Event | Expected formula (default GA, no pair-scan) | Double-count risk | JAX interaction |
| --- | --- | --- | --- | --- | --- | --- |
| Decoder TF calls | `src/models/lpn.py` | `_get_gradient_ascent_context`, `_get_random_search_context` | teacher-forced decoder for scoring | about `objective_evals` decoder TF forwards (or `N * objective_evals` if accumulating over pairs) | easy to miss final rescoring | inside `vmap`/`scan`; prefer analytic count from shapes |
| Support-objective evaluations | `src/models/lpn.py` | `_compute_log_probs` call sites | each full support score | `M * (num_steps + 1)` for GA including init and final score | final `vmap_log_probs_fn` easy to forget | scanned body runs `num_steps` times |
| Gradient evaluations | `src/models/lpn.py` | `value_and_grad` in GA `update_latents` | each latent grad | `M * num_steps` | do not count objective-only rescoring as grads | compiled with updates |
| Latent candidates evaluated | `src/models/lpn.py` | `_prepare_latents_before_search` (+ RS concat) | candidate construction | M at start; RS adds samples | perturbations vs pair latents | host-side from kwargs |
| Candidate-step exposure | `src/models/lpn.py` | after GA trajectory stack | retained (candidate, step) slots | `M * (num_steps + 1)` | not equal to unique latents if duplicates | post-scan tensor size |
| Autoregressive decoding calls | `src/models/lpn.py` | `_generate_output_from_context` | shape logits + token scan | `(2 + max_rows*max_cols)` per context; times 2 if `return_two_best` | do not mix with TF scoring calls | `nn.scan` over tokens |
| Wall time before JIT | host wrapper | `Evaluator.json_submission` / train eval | first call | measured | includes compile | host clock |
| Wall time after warm-up | same | second timed call | steady state | measured | still device sync dependent | host clock |
| Peak accelerator memory | host | around eval | `device_memory_stats` or nvidia-smi | measured | process sharing | external to JAX tracers |

Until counters ship, result records should keep `decoder_calls: null` (and siblings) rather than inventing values from `num_steps`.

## 5. Files that would eventually require modification

Only after the ARC baseline path and non-RL controls are in place, and only with tests first:

| File | Likely future change |
| --- | --- |
| `src/models/lpn.py` | expose search diagnostics; increment analytic counters; optional outer adaptive loop hooks |
| `src/evaluator.py` or a research wrapper | per-task budget control without changing metrics |
| `scripts/evaluate_*` research scripts | run matched non-RL and later RL controllers with provenance |
| new `research_configs/` (when created) | ARC research eval configs that do not edit upstream YAML |
| tests under a research test path | counter correctness and no-label leakage checks |

## 6. Files that must remain immutable controls

For matched scientific tables in the first ARC RL contribution:

- `src/models/transformer.py`
- Encoder/decoder configs for the chosen ARC model (or freeze by checkpoint hash)
- Training objective in `LPN.__call__`
- `_compute_log_probs` weighting
- Greedy `_generate_output_from_context`
- `Evaluator.evaluate_generations`
- Bundled evaluation JSON identity and ordering for the frozen report
- Checkpoint bytes used in a table (SHA-256)

Do not edit upstream Hydra configs in `src/configs/` for research variants. Add new files under a future `research_configs/` path instead.

## 7. Immediate non-goals

- Architecture changes or new training losses
- Dependency updates
- Claiming ARC-AGI improvement from pattern / `lpn-2d` results
- Claiming compute efficiency without counters
- Waiting for official ARC weights before starting pattern-sandbox implementation

## 8. Next implementation order

1. Implement compute counters behind tests.
2. Run non-RL baselines on frozen pattern / `lpn-2d` checkpoints.
3. Implement a minimal stop/continue policy (Reward C or A on the pattern split).
4. Keep matched tables (same checkpoint, tasks, seeds, metrics).
5. Optional later: repeat on ARC if a provenance-clear checkpoint becomes available.