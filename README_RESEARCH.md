# Stochastic latent-search research branch

This branch implements new **test-time** methods on Latent Program Networks: reinforcement-learning control of latent search, alternate stopping rules, and related compute-allocation baselines. The official repository remains the control group for data, architecture, training objective, greedy decoding, and official metrics.

**Project goal (current):** implement and validate new methods on a frozen checkpoint.  
**Not required for this phase:** large ARC-AGI accuracy gains or SOTA claims.

Reference paper: Searching Latent Program Spaces (https://arxiv.org/abs/2411.08706). Official code: https://github.com/clement-bonnet/lpn.

This document is the branch-specific entry point. See the upstream `README.md` for the original project and [postmortem.md](postmortem.md) for the chronological session record.

## Current state (2026-07-23)

- Branch: `research/stochastic-latent-search`
- Tip: inspect with `git rev-parse HEAD` (adaptive search landed; RL stop/continue prototype added)
- Upstream scientific control: `0adfe56b86d2cba5ae5794edb02da6399a96d98a`
- **Development platform:** frozen `pattern_2d` / Hugging Face `lpn-2d` checkpoints
- **pattern_2d three-seed baseline:** complete (saturated after 5 search steps)
- **Compute counters + non-RL adaptive search:** complete
- **RL stop/continue prototype:** complete (linear Bernoulli + REINFORCE)
- **Official ARC weights:** not available on this account (see below)
- **Results:** [`docs/rl_stop_continue_results.md`](docs/rl_stop_continue_results.md)

The exact pinned CPU environment is `.venv`. GPU work uses `.venv-gpu` (JAX/JAXlib 0.6.0, Flax 0.10.2): an explicit Blackwell compatibility deviation, not an exact pin reproduction. Freeze: `artifacts/environment/pip_freeze_gpu_compat.txt`.

## Checkpoint situation

| Source | Available? | Role |
| --- | --- | --- |
| Local `pattern_2d` seeds 0/1/2 | yes | Primary sandbox; three-seed control already recorded |
| Hugging Face [`clement-bonnet/lpn-2d`](https://huggingface.co/clement-bonnet/lpn-2d) (`quiet-thunder-789--checkpoint:v0`) | yes (downloaded) | Public pattern-2d weights; same domain family |
| Hugging Face ARC LPN model | **no** | Authors published only `lpn-2d` under this account |
| W&B `TheThinker/ARC/...--checkpoint` | **blocked** | API key works; project not visible (`project 'ARC' not found under entity 'TheThinker'`) |
| Full local `arc_train` (500k steps, batch 128) | impractical here | Stock batch OOM on 32GB; batch 8 fits at ~4.1 it/s (~34h train-only for 500k) |

GitHub code and the paper PDF do **not** include trained ARC weights. Retraining the encoder/decoder is **not** required to implement test-time RL or new search controllers.

## Why pattern / `lpn-2d` is enough for this phase

Meaningful for:

- implementing RL and non-RL test-time controllers on a real frozen LPN
- compute counters, eval harness, matched comparisons
- demonstrating that new methods run correctly

Not meaningful as a final ARC claim:

- in-family PATTERN tasks; saturated near-perfect exact match by 5 search steps
- cannot support "improves ARC-AGI" statements

Report pattern results as **implementation / sandbox evidence**. Optional ARC work later if W&B access appears.

## pattern_2d baseline summary

| Training seed | Checkpoint SHA-256 | Final mean exact match | Final 10-step SGD exact match |
| ---: | --- | ---: | ---: |
| 0 | `3cb6d3e41e698e1e0e62c785ecdf90d1fb4a600384fd6cf21f3460a110d7de13` | 0.73177 | 1.00000 |
| 1 | `bc292f2cd95365f597dec2a3f61c099a7d5a08177dc56050e8823c3a616f72be` | 0.98177 | 0.99740 |
| 2 | `92ecdd494f036d4ac0a865903316575b207444493ee76d6b7c0b0b601e2ee8fc` | 0.93229 | 1.00000 |

Frozen ablation mean exact match across seeds: 0.88194 (0 steps), 0.96007 (1 step), 1.0 (5+ steps). Do not retrain unless a scientific defect is found. Details: [`docs/results.md`](docs/results.md).

## ARC status (optional later)

Audit: [`docs/arc_baseline_audit.md`](docs/arc_baseline_audit.md).  
RL formulation (domain-agnostic MDP; demos on pattern first): [`docs/rl_test_time_compute_formulation.md`](docs/rl_test_time_compute_formulation.md).

Full ARC remains optional. Blockers and timing notes are in [`docs/negative_results.md`](docs/negative_results.md).

## Implementation order (method-first)

1. Compute counters behind tests. **Done.**
2. Non-RL baselines on frozen pattern / `lpn-2d`. **Done.**
3. Minimal RL stop/continue policy. **Done** (prototype).
4. Matched tables on the same checkpoint, tasks, and seeds. **Done** on length-24 setting B.
5. ARC only if a provenance-clear checkpoint becomes available.

See [`docs/rl_stop_continue_implementation.md`](docs/rl_stop_continue_implementation.md) and [`docs/rl_stop_continue_results.md`](docs/rl_stop_continue_results.md).

## Resume procedure

```bash
git -c safe.directory="$PWD" rev-parse HEAD
git -c safe.directory="$PWD" status --short
.venv-gpu/bin/python -m pip check
.venv-gpu/bin/python -c "import jax; print(jax.devices())"
sha256sum artifacts/checkpoints/pattern_2d_seed_{0,1,2}.msgpack
```

HF `lpn-2d` cache path (after Option B download):

```text
~/.cache/huggingface/hub/models--clement-bonnet--lpn-2d/snapshots/.../quiet-thunder-789--checkpoint:v0/
```

## Evidence map

- ARC audit: `docs/arc_baseline_audit.md`
- RL formulation: `docs/rl_test_time_compute_formulation.md`
- Search map: `docs/codebase_map.md`
- pattern results: `docs/results.md`
- Failures / blockers: `docs/negative_results.md`
- Session log: `postmortem.md`
- Environment: `artifacts/environment/`
- Logs / results: `artifacts/logs/`, `artifacts/results/`

## Scientific guardrails

- Do not compare methods across different checkpoints, task arrays, keys, decoding, or unreported compute budgets.
- Do not count evaluation-key repeats as independent training seeds.
- Do not claim ARC-AGI improvement from pattern / `lpn-2d` results.
- Do not claim compute efficiency before explicit counters exist.
- Do not tune on ARC evaluation solutions if ARC is used later.
- Keep encoder/decoder, training objective, greedy decode, and official metrics fixed while adding test-time controllers.
