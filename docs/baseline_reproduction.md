# Baseline reproduction

## Environment

The audit targets commit `0adfe56b86d2cba5ae5794edb02da6399a96d98a` on branch `research/stochastic-latent-search`. The upstream remote points to `https://github.com/clement-bonnet/lpn.git`.

Only a JAX CPU device is exposed inside the managed sandbox. Outside it, the host exposes an NVIDIA GeForce RTX 5090. The repository's exact direct pins were installed in `.venv`; pip selected unpinned transitive versions including NumPy 2.2.6 and `orbax-checkpoint==0.6.4`. Both the pre-install host freeze and isolated-environment freeze are under `artifacts/environment/`.

W&B 0.17.3 starts a local socket even in offline/disabled mode. Managed-sandbox execution therefore fails before model initialization. Runs must be allowed to create that local socket; `WANDB_MODE=offline` prevents remote tracking and avoids API-key use.

## Official command

```bash
PYTHONPATH="$PWD" .venv/bin/python src/train.py --config-name pattern_2d
```

The local execution also sets `WANDB_MODE=offline`, a workspace-local `WANDB_DIR`, and a writable temporary `MPLCONFIGDIR`. The resolved configuration is `artifacts/environment/baseline_pattern_2d_resolved.yaml`.

The official run initialized one CPU device, generated both 96-task evaluation datasets, initialized 254,664 parameters, and entered training. It was stopped after approximately 18 minutes because the first 1,000-step CPU block had not completed. The first configured evaluation is at step 20,000 and the full run is 200,000 steps, so this hardware cannot produce the official baseline in a practical initial session. No official metric or checkpoint resulted.

## One-step execution smoke test

To verify the complete path without representing it as a baseline, the following overrides were used:

```text
training.total_num_steps=1
training.log_every_n_steps=1
training.eval_every_n_logs=1
training.save_checkpoint_every_n_logs=null
eval.test_datasets.0.num_tasks_to_show=0
eval.test_datasets.1.num_tasks_to_show=0
```

One training step and both official evaluation modes completed in 51.18 seconds. Metrics and the exact override set are stored in `artifacts/results/smoke_pattern_2d_seed_0.json`. Exact-match was zero for both modes after one training step, as expected. These numbers are an execution check only.

Although loop checkpointing was disabled, `run` always writes `state.msgpack` at process exit. That one-step checkpoint is ignored by Git and must not be used as a reproduced model.

## Reproduction status

- Imports: pass.
- Dataset generation: pass.
- One train/evaluation cycle: pass in labeled smoke configuration.
- Exact documented 200,000-step run: incomplete due CPU runtime.
- Official baseline metrics: unavailable.
- Search-off/search-on ablation on a trained checkpoint: blocked pending a reproducible trained checkpoint or usable accelerator.

## GPU compatibility environment

The exact JAX/JAXlib 0.4.26 GPU build cannot compile for the RTX 5090 and fails in `ptxas` with a future-architecture error. A separate `.venv-gpu` advances JAX/JAXlib/CUDA plugin/PJRT to 0.6.0 and Flax to 0.10.2; a trivial GPU JIT and the complete one-step smoke path both pass. The full freeze is `artifacts/environment/pip_freeze_gpu_compat.txt`.

This is not an exact dependency reproduction. It is the minimal tested compatibility path for this hardware, with source code, Hydra configuration, data generation, architecture, objective, and seed unchanged. Its one-step result is stored separately in `artifacts/results/smoke_pattern_2d_gpu_compat_seed_0.json`; it must not be compared scientifically with the exact-pinned CPU smoke.

The unchanged 200,000-step configuration subsequently completed for three independent training seeds in this GPU compatibility environment. Checkpoints remain ignored by Git and are preserved locally under `artifacts/checkpoints/`.

| Training seed | Wall-clock (s) | Final mean exact match | Final 10-step SGD exact match | Checkpoint SHA-256 | Provenance JSON |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 948.08 | 0.73177 | 1.00000 | `3cb6d3e41e698e1e0e62c785ecdf90d1fb4a600384fd6cf21f3460a110d7de13` | `artifacts/results/baseline_pattern_2d_gpu_compat_seed_0.json` |
| 1 | 942.62 | 0.98177 | 0.99740 | `bc292f2cd95365f597dec2a3f61c099a7d5a08177dc56050e8823c3a616f72be` | `artifacts/results/baseline_pattern_2d_gpu_compat_seed_1.json` |
| 2 | 954.32 | 0.93229 | 1.00000 | `92ecdd494f036d4ac0a865903316575b207444493ee76d6b7c0b0b601e2ee8fc` | `artifacts/results/baseline_pattern_2d_gpu_compat_seed_2.json` |

Seed 0 first reached scheduled 10-step search exact match 1.0 at training step 140,000 and retained it through 200,000. Seed 2 used only the override `training.seed=2`, resolved config `outputs/2026-07-23/11-35-41/.hydra/config.yaml` (copy: `artifacts/environment/baseline_pattern_2d_gpu_compat_seed_2_resolved.yaml`), and log `artifacts/logs/baseline_pattern_2d_gpu_compat_seed_2.log`.

Matched frozen-checkpoint search-step ablations for seeds 0–2 (dataset seed 0, evaluation seed 0, lr 0.1, repository-default SGD, steps 0/1/5/10/20) are complete. Aggregate descriptive statistics are in `docs/results.md` and `artifacts/results/baseline_search_steps_three_seed_aggregate.json`. This clears the three-seed original-search baseline gate on in-family `pattern_2d`; it does not answer OOD hypotheses and is not yet a comparison against new search methods.
