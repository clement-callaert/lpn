# Research postmortem

This file is the chronological session log for the stochastic latent-search research fork. Append a new dated section after every working session. Record failures and negative results alongside successes. Do not rewrite earlier entries merely because later evidence changes the interpretation; add a correction or follow-up instead.

Each session should capture:

- objective and repository state;
- actions and exact commands/configurations;
- observations and measurements;
- failures and root causes;
- scientific decisions and ambiguities;
- files or artifacts produced;
- unresolved blockers;
- next safe action.

## Session 1 — 2026-07-22 — Repository audit and baseline startup

### Objective

Audit the checked-out repository, reproduce the smallest documented configuration, map latent search, and avoid introducing research methods before the baseline works.

### Repository state

- Commit: `0adfe56b86d2cba5ae5794edb02da6399a96d98a`
- Branch: `research/stochastic-latent-search`
- Worktree at start: clean
- `origin`: `https://github.com/clement-callaert/lpn.git`
- `upstream`: `https://github.com/clement-bonnet/lpn.git`
- Python command in README: `python`; available command on host: `python3` 3.10.12

### Environment findings

- Host environment did not match the repository pins: JAX 0.6.2, jaxlib 0.6.2, Flax 0.10.7, Optax 0.2.8, and W&B 0.27.2 were installed globally.
- Created `.venv` and installed the exact direct requirements: JAX/jaxlib 0.4.26, Flax 0.8.4, Optax 0.2.2, W&B 0.17.3, and the other repository pins.
- Pip selected unpinned transitive dependencies including NumPy 2.2.6 and `orbax-checkpoint==0.6.4`. `pip check` reported no broken requirements and all imports succeeded.
- The unconditional `torch==2.4.1` pin downloaded several gigabytes of CUDA-related wheels even though Torch is only used for data loading and JAX exposed a CPU backend.
- Inside the managed environment, `nvidia-smi` could not initialize NVML. JAX reported one `CpuDevice(id=0)`.
- Official current JAX installation guidance was checked. CPU is the correct installation target given the devices exposed at this point; accelerator status remains to be checked outside the sandbox.
- Pre-install and isolated-environment freezes are stored under `artifacts/environment/`.

### Baseline configuration findings

- The documented command is `python src/train.py --config-name pattern_2d`; the local equivalent is `PYTHONPATH="$PWD" .venv/bin/python src/train.py --config-name pattern_2d`.
- Resolved `pattern_2d` training uses mean latent inference, seed 0, batch size 128, 200,000 steps, logs every 1,000 steps, evaluates every 20,000 steps, and checkpoints at step 200,000.
- Evaluation materializes two fixed 96-task PATTERN datasets using default dataset seed 0: mean decoding and 10-step gradient ascent with learning rate 0.1.
- The gradient-ascent evaluation does not specify `optimizer`, so the repository default is SGD, not Adam. Its name `generator_gradient_ascent_5` is inconsistent with its configured 10 search steps.
- No trained checkpoint is bundled in the checkout.

### Execution attempts

#### W&B startup failure

W&B 0.17.3 starts a local socket service even with offline or disabled mode. The managed sandbox prohibits socket creation, producing `PermissionError: [Errno 1] Operation not permitted` before model initialization. Running outside that socket restriction with `WANDB_MODE=offline` resolves the failure without remote logging. Failure traces are preserved in `artifacts/logs/`.

#### Exact official configuration

The exact configuration successfully:

- initialized JAX on CPU;
- generated both 96-task evaluation datasets;
- initialized 254,664 parameters (126,792 encoder and 127,872 decoder);
- entered the training loop.

It did not finish the first compiled 1,000-step block after approximately 18 minutes and was interrupted. The first official evaluation would occur at step 20,000, so full CPU reproduction is not practical for the initial session. No official metric or checkpoint was produced.

#### One-step smoke configuration

The following Hydra overrides were used only to verify one complete cycle:

```text
training.total_num_steps=1
training.log_every_n_steps=1
training.eval_every_n_logs=1
training.save_checkpoint_every_n_logs=null
eval.test_datasets.0.num_tasks_to_show=0
eval.test_datasets.1.num_tasks_to_show=0
```

The run completed one training step plus both evaluation modes in 51.18 seconds. Training took 12.15 seconds, mean evaluation 4.54 seconds, and 10-step search evaluation 10.90 seconds. Both exact-match accuracies were zero after one training step. These are execution-smoke measurements, not baseline evidence.

Despite disabling checkpointing inside the loop, `run` always writes `state.msgpack` after training. The resulting one-step checkpoint is ignored by Git and must not be treated as a reproduced model.

### Codebase findings

- Latent search is implemented inside `src/models/lpn.py`, primarily in `LPN.generate_output`, `_get_gradient_ascent_context`, `_get_random_search_context`, `_prepare_latents_before_search`, and `_compute_log_probs`.
- Existing gradient search already supports clipped SGD, clipped Adam, a cosine schedule, mean and pair-latent starts, Gaussian perturbed starts, and retention/selection over all iterates.
- The search objective sums observed-pair row-shape log probability, column-shape log probability, and normalized grid-token log probability.
- The returned context is the highest-scoring candidate across initialization and all iterations; it is not necessarily the final iterate.
- The variational encoder samples pair latents during evaluation, including mean mode. Evaluation keys and task ordering must therefore remain fixed.
- Output decoding is greedy and fixed: row argmax, column argmax, then autoregressive grid-token argmax.
- Generated-task exact match is computed in `Trainer`'s generation closure; ARC top-1/top-2 exact match is computed by `Evaluator.evaluate_generations`.
- There are no explicit counters for decoder calls, objective evaluations, gradient evaluations, or candidate-time exposure.
- `pattern_2d` contains only in-family procedural evaluation and cannot answer the ID-versus-OOD hypothesis.

The full symbol, shape, JAX-transformation, PRNG, objective, and call-path map is in `docs/codebase_map.md`.

### Scientific decisions

- Do not run search-method comparisons on the one-step smoke checkpoint.
- Do not interpret evaluation-seed repetitions on one checkpoint as independent training seeds.
- Do not add a search abstraction or new method until the official baseline checkpoint is reproduced or obtained with exact provenance.
- Treat existing Adam and perturbed multi-candidate search as official strong baselines rather than new contributions.
- Test `mode=mean` against `gradient_ascent, num_steps=0` for numerical equivalence before calling either one “search off.”

### Artifacts produced

- `artifacts/environment/system.txt`
- `artifacts/environment/pip_freeze.txt`
- `artifacts/environment/pip_freeze_venv.txt`
- `artifacts/environment/baseline_pattern_2d_resolved.yaml`
- `artifacts/environment/smoke_pattern_2d_resolved.yaml`
- `artifacts/logs/baseline_pattern_2d.log`
- `artifacts/logs/baseline_pattern_2d_wandb_failure.log`
- `artifacts/logs/smoke_pattern_2d.log`
- `artifacts/logs/smoke_pattern_2d_wandb_failure.log`
- `artifacts/results/baseline_pattern_2d_attempt.json`
- `artifacts/results/smoke_pattern_2d_seed_0.json`
- `docs/codebase_map.md`
- `docs/baseline_reproduction.md`
- `docs/experimental_protocol.md`
- `docs/research_hypotheses.md`
- `docs/results.md`
- `docs/negative_results.md`

### Unresolved blockers

1. No official trained checkpoint is present locally.
2. The exact 200,000-step baseline is impractical on the exposed CPU.
3. GPU visibility outside the managed sandbox has not yet been established.
4. The repository has no unambiguous fixed OOD split for the `pattern_2d` baseline.
5. Search compute counters do not exist yet.

### Next safe action

Check host accelerator availability outside the sandbox. If a supported GPU is available, build a separate exact JAX 0.4.26 accelerator environment and rerun the unchanged official configuration. Otherwise, locate an official checkpoint with commit/config provenance before beginning ablations.

## Session 2 — 2026-07-22 — Blackwell GPU compatibility

### Objective

Determine whether the host accelerator can make the unchanged `pattern_2d` baseline practical, while preserving the exact pinned CPU environment as the reproduction reference.

### Accelerator findings

- Outside the managed sandbox, the host exposes an NVIDIA GeForce RTX 5090 with 32,607 MiB, driver 591.86, reported CUDA 13.1, and compute capability 12.0.
- The exact pinned JAX/JAXlib 0.4.26 CUDA environment detected the device but a trivial JIT failed in `ptxas`: `.target 'sm_90a' cannot be compiled to future architecture`. This JAX build predates Blackwell support.
- A separate `.venv-gpu` was retained so the exact `.venv` remained unchanged.
- JAX/JAXlib and the CUDA plugin/PJRT were minimally advanced to 0.6.0. This is the first JAX release documented as built with CUDA 12.8, the CUDA generation introducing Blackwell support. A 1024-by-1024 JIT matrix multiplication then completed on `CudaDevice(id=0)`.
- Flax 0.8.4 failed during model initialization with `AttributeError: 'EvalTrace' object has no attribute 'level'`, caused by incompatibility with JAX's stackless tracing changes.
- Advancing only Flax to 0.10.2 resolved that failure. Optax and all other direct repository packages remain at their pins, except Torch uses the official 2.4.1 CPU wheel because this repository uses it only for data loading.
- `pip check` reports no broken requirements in the resulting compatibility environment.

### GPU smoke execution

The same one-step smoke overrides from Session 1 completed on the GPU with no source-code changes. The model still contains 254,664 parameters. One training step and both 96-task evaluations completed; wall-clock time was 81.87 seconds including startup, dataset generation, and compilation.

All exact-match, correct-shape, and pixel-correctness metrics were zero in this one-step compatibility run. The training loss and other diagnostics also differ materially from the exact-pinned CPU smoke. Because JAX and Flax differ, neither run is a scientific baseline and their numerical difference must not be interpreted as a device comparison.

### Scientific decision

The exact repository dependencies cannot execute on this GPU, while the exact CPU baseline is impractically slow. The GPU environment is therefore a clearly labeled hardware-compatibility baseline: source, data, configuration, seed, architecture, and objectives remain unchanged, but JAX 0.6.0 and Flax 0.10.2 are explicit environmental deviations. All future method comparisons must use one common compatibility environment and checkpoint. No search method will be added before an unchanged full configuration succeeds.

### Artifacts produced

- `artifacts/environment/pip_freeze_gpu_compat.txt`
- `artifacts/environment/smoke_pattern_2d_gpu_compat_resolved.yaml`
- `artifacts/logs/smoke_pattern_2d_gpu_compat.log` (Flax 0.8.4 failure)
- `artifacts/logs/smoke_pattern_2d_gpu_compat_flax_0_10_2.log` (successful cycle)
- `artifacts/results/smoke_pattern_2d_gpu_compat_seed_0.json`

### Next safe action

Launch the unchanged 200,000-step `pattern_2d` configuration in `.venv-gpu`, save the resolved configuration/log/metrics/checkpoint provenance, and stop if runtime or numerical behavior makes completion impractical. Do not begin the search-off/search-on ablation until that run succeeds.

### Session 2 follow-up — Full baseline completed

The unchanged official `pattern_2d` configuration completed successfully in 948.08 seconds. It trained for all 200,000 steps, ran all ten scheduled evaluations, saved `state.msgpack`, and exited with code 0. The checkpoint SHA-256 is `3cb6d3e41e698e1e0e62c785ecdf90d1fb4a600384fd6cf21f3460a110d7de13`.

At step 200,000, mean inference achieved 0.73177 exact-match accuracy and the configured 10-step SGD search achieved 1.0 on the same fixed 96-task generated evaluation set. Search accuracy reached 1.0 at training step 140,000 and remained there at every later scheduled evaluation. The mean result fluctuated and fell from 0.94010 at step 180,000 to 0.73177 at step 200,000, while pixel correctness remained 0.97965. This reinforces the need to use one frozen checkpoint and identical task/PRNG seeds for subsequent comparisons.

The successful result is stored at `artifacts/results/baseline_pattern_2d_gpu_compat_seed_0.json`. It clears the baseline-execution gate but does not clear the three-seed ablation requirement.

### Session 2 follow-up — First search control

A local-checkpoint evaluation wrapper was added at `scripts/evaluate_pattern_checkpoint.py`. It reuses the repository's model construction, PATTERN generator, `Trainer.test_dataset_submission`, greedy decoding, and exact-match implementation. It changes no model or search logic and records the checkpoint hash, task seed, evaluation key, configuration, metrics, and timings.

On the frozen seed-0 checkpoint with dataset seed 0 and evaluation seed 0, mean inference and `gradient_ascent(num_steps=0, lr=0.1)` matched exactly: 0.723958 exact match, 1.0 correct shapes, and 0.979004 pixel correctness. Ten-step SGD search reached 1.0 on all metrics. Zero-step gradient ascent is therefore accepted as the search-off control for subsequent matched runs.

The complete seed-0 step grid then produced exact-match accuracies of 0.723958, 0.885417, 1.0, 1.0, and 1.0 for 0, 1, 5, 10, and 20 steps respectively. Its checkpoint was preserved at `artifacts/checkpoints/pattern_2d_seed_0.msgpack`.

## Session 2 handoff — 2026-07-22

Independent training seed 1 completed all 200,000 unchanged steps in 942.62 seconds. The final scheduled evaluation reported mean exact match 0.98177 and 10-step SGD exact match 0.99740. Its checkpoint was preserved as `artifacts/checkpoints/pattern_2d_seed_1.msgpack` with SHA-256 `bc292f2cd95365f597dec2a3f61c099a7d5a08177dc56050e8823c3a616f72be`.

The seed-1 standalone 0/1/5/10/20 ablation was requested but cancelled before it created a log or result. Process inspection confirmed that no training or evaluation process remained running. Resume instructions are in `README_RESEARCH.md`.

At session end, the repository contains uncommitted documentation, artifacts, `.gitignore` additions, and the local evaluation wrapper. No model/search source was changed and no commit was created. The next scientific action is the seed-1 ablation, followed by unchanged seed-2 training and its matched ablation. Do not add a new search method before those controls are complete.

## Session 3 — 2026-07-23 — Three-seed original-search baseline completed

### Objective

Finish the controlled three-seed baseline for the original deterministic latent search: seed-1 frozen ablation, unchanged seed-2 training, seed-2 matched ablation, and descriptive three-seed aggregation. Do not implement new search methods.

### Repository state

- Commit: `0adfe56b86d2cba5ae5794edb02da6399a96d98a`
- Branch: `research/stochastic-latent-search`
- Environment: `.venv-gpu` (JAX/JAXlib 0.6.0, Flax 0.10.2, Optax 0.2.2, Torch 2.4.1+cpu)
- Accelerator: NVIDIA GeForce RTX 5090, `CudaDevice(id=0)`
- No competing train/eval process at session start

### Actions

1. Verified seed-1 Hydra config at `outputs/2026-07-22/23-24-52/.hydra/config.yaml` (`training.seed: 1`) and checkpoint SHA-256 `bc292f2cd95365f597dec2a3f61c099a7d5a08177dc56050e8823c3a616f72be`.
2. Ran seed-1 step ablation (`0,1,5,10,20`, dataset seed 0, evaluation seed 0, lr 0.1).
3. Trained unchanged `pattern_2d` with only `training.seed=2`, preserved checkpoint immediately, then ran the matched seed-2 ablation.
4. Aggregated descriptive statistics across training seeds 0/1/2 and updated research docs.

### Failures and root causes

1. First seed-1 ablation completed the evaluations but crashed while writing JSON because `git rev-parse` failed with dubious ownership under root execution. Root cause: repository owned by `calla`, process running as root; no `git config` write was used. Fix: `scripts/evaluate_pattern_checkpoint.py` now passes a one-shot `git -c safe.directory=<repo>` flag. The ablation was re-run successfully.
2. First seed-2 training attempt failed during `wandb.init` with `MailboxError: transport failed` after the same dubious-ownership git failure inside W&B. Fix: relaunch with `GIT_CONFIG_COUNT/KEY/VALUE` environment overrides for `safe.directory` (no git config mutation). The successful run overwrote `artifacts/logs/baseline_pattern_2d_gpu_compat_seed_2.log`.

### Observations

#### Seed-1 ablation (`artifacts/results/baseline_search_steps_seed_1.json`)

Exact-match: mean/0-step 0.986979; 1/5/10/20 steps all 1.0. Mean matched zero-step exactly. No NaNs or exceptions in the successful log.

#### Seed-2 training (`artifacts/results/baseline_pattern_2d_gpu_compat_seed_2.json`)

- Wall-clock: 954.32 s (process elapsed); W&B reported runtime 917 s
- Final scheduled mean exact match: 0.93229
- Final scheduled 10-step SGD exact match: 1.0
- Checkpoint: `artifacts/checkpoints/pattern_2d_seed_2.msgpack`
- SHA-256: `92ecdd494f036d4ac0a865903316575b207444493ee76d6b7c0b0b601e2ee8fc`
- Resolved config: `outputs/2026-07-23/11-35-41/.hydra/config.yaml`
- Final `state.msgpack` landed in the repository root rather than the Hydra run directory and was copied immediately before any later overwrite risk

#### Seed-2 ablation (`artifacts/results/baseline_search_steps_seed_2.json`)

Exact-match: mean/0-step 0.934896; 1-step 0.994792; 5/10/20 steps 1.0. Mean matched zero-step exactly.

#### Three-seed aggregate

| Steps | Mean | Std | Min | Max | Seeds at 1.0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.88194 | 0.13928 | 0.72396 | 0.98698 | 0 |
| 1 | 0.96007 | 0.06470 | 0.88542 | 1.00000 | 1 |
| 5+ | 1.00000 | 0.00000 | 1.00000 | 1.00000 | 3 |

Saturation under this protocol: 5 steps. Improvement over zero-step mean: about +0.078 at 1 step and +0.118 at 5+ steps.

### Scientific decisions

- Accept the three-seed original deterministic search ablation as complete for in-family `pattern_2d`.
- Do not claim OOD improvement, stochastic/proximal superiority, compute efficiency, or statistical significance beyond descriptive three-seed dispersion.
- Do not implement new search methods until this baseline package is reviewed and committed.

### Artifacts produced or updated

- `artifacts/results/baseline_search_steps_seed_1.json`
- `artifacts/results/baseline_search_steps_seed_2.json`
- `artifacts/results/baseline_search_steps_three_seed_aggregate.json`
- `artifacts/results/baseline_pattern_2d_gpu_compat_seed_2.json`
- `artifacts/checkpoints/pattern_2d_seed_2.msgpack` (ignored)
- `artifacts/logs/baseline_search_steps_seed_1.log`
- `artifacts/logs/baseline_search_steps_seed_2.log`
- `artifacts/logs/baseline_pattern_2d_gpu_compat_seed_2.log`
- `artifacts/environment/baseline_pattern_2d_gpu_compat_seed_2_resolved.yaml`
- Docs: `docs/results.md`, `docs/baseline_reproduction.md`, `README_RESEARCH.md`, `docs/negative_results.md`, this postmortem
- Minor evaluator fix: one-shot `safe.directory` in `scripts/evaluate_pattern_checkpoint.py`

### Unresolved blockers

1. Explicit decoder/objective/gradient-call counters still absent; wall times remain diagnostic only.
2. No fixed OOD split for `pattern_2d`; OOD hypotheses remain unanswered.
3. Baseline docs/evaluator/results are still uncommitted.

### Next safe action

Review `git status` / diffs, then create the planned documentation and experiment commits without staging checkpoints, venvs, W&B caches, or Hydra outputs. Only after that review, begin implementing the first new search method against the frozen three-seed control.

## Session 4 — 2026-07-23 — ARC-AGI audit and RL formulation

### Objective

Prepare a rigorous ARC-AGI baseline audit and a formal RL formulation for test-time compute allocation. Do not implement an RL policy or new search methods.

### Repository state

- Tip commit: `063d4522be004adeeb619e630bf8b82b181fa467` (pattern_2d research commits already on branch)
- Upstream control: `0adfe56b86d2cba5ae5794edb02da6399a96d98a`
- Branch: `research/stochastic-latent-search`
- Working tree at start: only `.gitignore` modified among tracked files
- Environment: `.venv-gpu` on RTX 5090; `WANDB_API_KEY` and `HF_TOKEN` unset
- No competing train/eval process

### Actions

1. Confirmed git tip/diff and that upstream scientific sources were not unexpectedly modified.
2. Audited ARC train/eval paths, JSON splits, re-ARC ID overlap, checkpoint loading, and leakage risks.
3. Attempted the smallest ARC overfit smoke; recorded OOM then a reduced-batch success.
4. Wrote the RL MDP formulation, non-RL baselines, and compute-counter design.
5. Updated research entrypoint docs.

### Failures and root causes

1. Preferred scientific ARC eval via `evaluate_checkpoint.py` blocked: no W&B API key; script forces `WANDB_MODE=run`.
2. First smoke (`arc_smoke_20260723_122947`) OOM: stock overfit `batch_size=128` compile requested about 18.64 GiB.
3. Second smoke succeeded only after documented Hydra batch/worker reductions. Still not a scientific ARC baseline (tiny overfit model, 1 step, zero metrics).

### Observations

- `ARC_TASK_NAMES` equals the 400 training JSON IDs and has zero overlap with evaluation IDs.
- Bundled `arc-agi_test_challenges.json` is a 100-task subset of training IDs.
- Full `arc_train.yaml` uses latent dim 128, 4/4 layers, 500k steps, and logs evaluation JSON metrics during training.
- Overfit smoke model: 207,264 parameters, latent dim 32, encoder layers 0.
- Successful smoke provenance: `artifacts/results/arc_smoke_20260723_123100.json`.

### Scientific decisions

- Primary RL formulation: finite-horizon MDP with stop-and-decode; fixed budgets are restricted policies.
- Online rewards restricted to train-side labels; ARC evaluation solutions are final-report only.
- Non-RL adaptive baselines are mandatory before RL.
- No compute-efficiency claims until explicit counters exist.
- Do not launch full `arc_train` until cost/storage and a non-leaky logging policy are planned.

### Artifacts produced or updated

- `docs/arc_baseline_audit.md` (new)
- `docs/rl_test_time_compute_formulation.md` (new)
- `docs/negative_results.md`
- `README_RESEARCH.md`
- this postmortem section
- `artifacts/logs/arc_smoke_20260723_122947.log` (failed)
- `artifacts/logs/arc_smoke_20260723_123100.log` (success)
- `artifacts/results/arc_smoke_20260723_123100.json`
- `artifacts/environment/arc_smoke_20260723_123100_resolved.yaml`
- `artifacts/checkpoints/arc_smoke_20260723_123100_state.msgpack` (ignored)

### Unresolved blockers

1. No provenance-clear ARC checkpoint locally.
2. W&B credential/process requirements for official `evaluate_checkpoint.py`.
3. Compute counters unimplemented.
4. Full ARC training cost unmeasured.
5. Stock evaluation-metric logging during training remains a selection-leakage risk if used for picking.

### Next safe action

Obtain a provenance-clear ARC checkpoint (W&B access or a measured training plan), implement compute counters behind tests, define a train-side validation carve-out, and run non-RL search-budget baselines before any RL policy code.

## Session 5 — 2026-07-23 — Goal clarification and documentation update

### Objective

Record the clarified research goal and the checkpoint access situation in the branch markdown files.

### Clarified goal

Implement new test-time methods (RL / non-RL search control) on a frozen LPN. Significant ARC-AGI accuracy progress is not required in this phase.

### Checkpoint findings

1. W&B download attempt with API key failed: `project 'ARC' not found under entity 'TheThinker'`. No official ARC artifact on disk.
2. Hugging Face has no ARC LPN model under `clement-bonnet`; only `clement-bonnet/lpn-2d` (PATTERN). Option B download of that checkpoint succeeded.
3. GitHub and the paper PDF do not ship ARC weights.
4. Full `arc_train` timing: stock batch 128 OOM; batch 8 about 4.1 it/s (~34h for 500k train-only).

### Scientific decision

Use frozen pattern_2d / HF `lpn-2d` as the method-implementation sandbox. Treat pattern metrics as implementation evidence only. Keep ARC audit docs for optional later work. Do not retrain the base model merely to add test-time controllers.

### Docs updated

- `README_RESEARCH.md`
- `docs/arc_baseline_audit.md`
- `docs/rl_test_time_compute_formulation.md`
- `docs/negative_results.md`
- `docs/research_hypotheses.md`
- `docs/results.md`
- this postmortem section

### Next safe action

Implement compute counters, then non-RL baselines, then a minimal RL stop/continue policy on frozen pattern / `lpn-2d` checkpoints.
