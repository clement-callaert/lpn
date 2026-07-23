# Negative and blocked results

## Official CPU reproduction attempt

The exact `pattern_2d` configuration did not complete its first 1,000-step block after approximately 18 minutes on the single exposed CPU device. It was interrupted before any metric was logged. This is an environment/runtime limitation, not a model-quality result.

## W&B under the managed sandbox

W&B 0.17.3 failed before model initialization because its local service could not create a loopback socket. Offline and disabled modes both exhibited this behavior. Running with permission for the local socket resolved it; remote logging remained disabled.

## Exact JAX pin on the RTX 5090

JAX/JAXlib 0.4.26 with its archived CUDA 12 build detected the RTX 5090 but could not compile even a trivial JIT. `ptxas` reported that its `sm_90a` target could not be compiled for the future architecture. Exact dependency reproduction is therefore CPU-only on this host.

## Flax 0.8.4 with JAX 0.6.0

After minimally updating JAX for Blackwell support, the repository-pinned Flax 0.8.4 failed during model initialization with `AttributeError: 'EvalTrace' object has no attribute 'level'`. Flax 0.10.2 fixes this compatibility failure, but the resulting environment is explicitly not an exact pinned reproduction.

## Dubious git ownership under root execution

When the agent process ran as root against a repository owned by `calla`, plain `git` commands failed with dubious ownership. That broke:

1. JSON finalization in `scripts/evaluate_pattern_checkpoint.py` after an otherwise complete seed-1 ablation;
2. `wandb.init` during the first seed-2 training attempt (`MailboxError: transport failed`).

Both were operational failures, not scientific result failures. Remediation used one-shot `safe.directory` overrides (`git -c ...` in the evaluator; `GIT_CONFIG_*` environment variables for training) without writing git config. The ablation and training were then re-run successfully.

## Official ARC checkpoint evaluation without W&B credentials

[`src/evaluate_checkpoint.py`](../src/evaluate_checkpoint.py) loads model weights through W&B artifacts and sets `WANDB_MODE=run`. Early in the project the API key was unset. Later a key was configured, but download still failed with:

```text
project 'ARC' not found under entity 'TheThinker'
```

So authentication works, but the `TheThinker/ARC` project is not visible to this account. No official ARC `state.msgpack` was obtained. This is an access blocker, not a model-quality result.

## No public Hugging Face ARC LPN weights

Under author `clement-bonnet`, Hugging Face lists `clement-bonnet/lpn-2d` only (PATTERN 2D checkpoint `quiet-thunder-789--checkpoint:v0`). There is no public ARC-AGI LPN model repo from this account. Option B download of `lpn-2d` succeeded and is suitable as a pattern sandbox, not as an ARC baseline.

## ARC overfit smoke OOM at stock batch size

The smallest planned ARC smoke used `--config-path configs/arc_train_overfit --config-name 007bbfb7` with one training step. With the stock overfit `training.batch_size: 128`, XLA requested about 18.64 GiB during train-step compile and failed with `RESOURCE_EXHAUSTED` (`artifacts/logs/arc_smoke_20260723_122947.log`).

Root cause: large compiled train graph for batch 128 on `max_rows/max_cols = 30` grids, not a missing dataset. A second smoke with Hydra overrides `training.batch_size=8`, `training.gradient_accumulation_steps=1`, and fewer workers completed successfully (`artifacts/logs/arc_smoke_20260723_123100.log`). Metrics were zero after one step, as expected for an execution smoke. This is not an ARC-AGI accuracy baseline and uses the smaller overfit architecture, not full `arc_train.yaml`.

## Full arc_train timing on RTX 5090

Stock `arc_train` with `batch_size=128` OOM (~206 GiB requested). `batch_size=16` also OOM (~38.5 GiB). `batch_size=8` completed 1000 steps at about 4.1 it/s steady state (6.48M parameters). Extrapolation: about 34 hours for 500k train steps without scheduled evals. Record: `artifacts/results/arc_timing_probe_20260723.json`. Full ARC local train is therefore out of scope for a short method-implementation phase.

## RL stop/continue: penalty collapse and limited scale

On setting B (train ckpt 0+1, test ckpt 2, length 24):

- `lambda=0.10` collapsed to always-stop (mean steps 0.0, EM 0.875).
- `lambda=0.05` also fell to EM 0.875 while cutting steps heavily.
- Only `lambda=0.00` matched fixed-5 EM (0.90625) with a modest step cut (4.28 vs 5.0).
- Training used length-8 trajectory splits for speed; final eval used length 24, not full length 96.
- Gradient-norm heuristic matched fixed-5 EM at lower mean steps than the selected RL policy on this slice.

These are sandbox limitations and tradeoffs, not ARC failures. Details: [`docs/rl_stop_continue_results.md`](rl_stop_continue_results.md).
