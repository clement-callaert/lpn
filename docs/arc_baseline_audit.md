# ARC-AGI baseline audit

Audit of the checked-out Latent Program Network repository for ARC-AGI training and evaluation. Every claim below is grounded in this tree. Paper claims that are not visible in code are labeled as such.

- Branch tip at audit time: `063d4522be004adeeb619e630bf8b82b181fa467`
- Upstream scientific control: `0adfe56b86d2cba5ae5794edb02da6399a96d98a`
- Branch: `research/stochastic-latent-search`
- Environment for execution: `.venv-gpu` (JAX/JAXlib 0.6.0, Flax 0.10.2). This is a documented hardware compatibility deviation, not an exact pin reproduction.

This document does not implement RL, new search methods, or architecture changes.

## Status relative to the research goal (2026-07-23)

The active project goal is to **implement new test-time methods**, not to deliver large ARC-AGI accuracy gains in this phase.

| Item | Status |
| --- | --- |
| Official ARC W&B checkpoints (`TheThinker/ARC/...`) | Blocked: project not visible to the local W&B account |
| Public Hugging Face ARC LPN weights | None found; only [`clement-bonnet/lpn-2d`](https://huggingface.co/clement-bonnet/lpn-2d) (PATTERN 2D) |
| Full local `arc_train` on RTX 5090 | Stock `batch_size=128` OOM; `batch_size=8` works at ~4.1 it/s (~34h for 500k train-only, evals extra) |
| Method development platform | Frozen `pattern_2d` / HF `lpn-2d` (see [`README_RESEARCH.md`](../README_RESEARCH.md)) |

This audit remains the map for optional ARC work later. It is **not** a gate that blocks implementing test-time RL or non-RL controllers on pattern checkpoints.

## 1. Training task source

Primary official ARC training path uses an online re-ARC task generator, not the bundled JSON challenges as training batches.

| Item | Location |
| --- | --- |
| Config | [`src/configs/arc_train.yaml`](../src/configs/arc_train.yaml) field `training.task_generator.class: ARC` |
| Generator class | [`ArcTrainTaskGenerator`](../src/datasets/task_gen/task_generator.py) in `src/datasets/task_gen/task_generator.py` |
| Dataloader wiring | [`make_task_gen_dataloader`](../src/datasets/task_gen/dataloader.py) when `class` is `"ARC"` |
| Train loop use | [`Trainer.train_epoch`](../src/train.py) when `self.task_generator` is set |

Relevant `arc_train.yaml` training fields:

- `training.seed: 0`
- `training.total_num_steps: 500000`
- `training.batch_size: 128`
- `training.gradient_accumulation_steps: 2`
- `training.task_generator.num_pairs: 4`
- `training.task_generator.num_workers: 64`
- `training.mixed_precision: True`
- `training.online_data_augmentation: True`
- `training.inference_mode: mean` with nested `inference_kwargs` for optional train-time search

Each training example is `num_pairs` input-output grids sampled from one re-ARC generator function.

## 2. re-ARC generator source

| Item | Location |
| --- | --- |
| Generator source blob | [`GENERATORS_SRC_CODE`](../src/datasets/task_gen/re_arc_generators.py) in `src/datasets/task_gen/re_arc_generators.py` |
| Task ID list | `ARC_TASK_NAMES` in the same file; asserted length 400; shuffled with `random.Random(0)` |
| Runtime binding | `ArcTrainTaskGenerator.__iter__` calls `exec(GENERATORS_SRC_CODE, globals())` then builds `generate_<task_id>` callables |

Verified in this checkout:

- `len(ARC_TASK_NAMES) == 400`
- `set(ARC_TASK_NAMES)` equals the keys of `src/datasets/json/arc-agi_training_challenges.json`
- `set(ARC_TASK_NAMES) ∩ evaluation keys == empty`

Optional filters:

- `overfit_task`: single training task ID
- `only_n_tasks`: first N names after the fixed shuffle

Scaling configs live under [`src/configs/arc_train_scaling/`](../src/configs/arc_train_scaling/). Overfit configs live under [`src/configs/arc_train_overfit/`](../src/configs/arc_train_overfit/).

## 3. ARC challenge JSON source

Bundled under [`src/datasets/json/`](../src/datasets/json/). Base path constant:

- `DATASETS_BASE_PATH = "src/datasets"` in [`src/data_utils.py`](../src/data_utils.py)

| File | Role | Tasks in this checkout |
| --- | --- | ---: |
| `json/arc-agi_training_challenges.json` | Training-set challenges | 400 |
| `json/arc-agi_evaluation_challenges.json` | Evaluation-set challenges | 400 |
| `json/arc-agi_test_challenges.json` | Bundled file named "test" | 100 |

Config fields: `eval.json_datasets[].challenges` (relative to `DATASETS_BASE_PATH`).

Loading path: [`Trainer.test_json_submission`](../src/train.py) opens the challenges file, then [`Evaluator.json_submission`](../src/evaluator.py) builds support pairs from each task's `train` list and predicts each `test` input.

## 4. ARC solution JSON source

| File | Role |
| --- | --- |
| `json/arc-agi_training_solutions.json` | Training-set solutions |
| `json/arc-agi_evaluation_solutions.json` | Evaluation-set solutions |

Config fields: `eval.json_datasets[].solutions`.

Scoring path: [`Evaluator.evaluate_generations`](../src/evaluator.py) compares `attempt_1` / `attempt_2` to solutions.

There is no bundled evaluation-free private contest test solution file in this repository.

## 5. Training and evaluation split definitions

| Split | Definition in this repo | Verified |
| --- | --- | --- |
| Training IDs | Keys of training challenges JSON; identical to `ARC_TASK_NAMES` | yes |
| Evaluation IDs | Keys of evaluation challenges JSON | yes |
| Training ∩ evaluation | empty | yes |
| Bundled `arc-agi_test_challenges.json` | 100 IDs, all ⊆ training | yes |

Important naming trap: the bundled "test" JSON is not a held-out contest private set. Treating it as OOD evaluation would leak training-family tasks.

`Evaluator.json_submission` uses `train=("training" in json_challenges_file)` to choose task ordering when `only_n_tasks` is set: training files follow `ARC_TASK_NAMES`, otherwise the JSON key order.

## 6. Data augmentation

| Field | Config | Behavior |
| --- | --- | --- |
| `training.online_data_augmentation` | `True` in `arc_train.yaml` | Applied in the training path when enabled |
| Overfit configs | often `False` | Smaller debug/overfit setups disable it |

Augmentation does not change the set of task IDs. It transforms grids within the training pipeline.

## 7. Number of input-output support pairs

- Training generator: `training.task_generator.num_pairs: 4` in `arc_train.yaml`
- JSON evaluation: number of support pairs equals `len(task["train"])` for each ARC task (variable; sample task `007bbfb7` has 5 train pairs and 1 test input in the bundled JSON)

## 8. Number of query inputs

- JSON evaluation: one forward per entry in `task["test"]` inside [`Evaluator.json_submission`](../src/evaluator.py)
- Sample task `007bbfb7`: 1 test input
- Some ARC tasks have multiple test inputs; metrics average per test grid within a task, then average over tasks

## 9. Model configuration (full ARC train)

From [`src/configs/arc_train.yaml`](../src/configs/arc_train.yaml):

| Field | Encoder / decoder |
| --- | --- |
| `max_rows` / `max_cols` | 30 / 30 |
| `num_layers` | 4 / 4 |
| `num_heads` | 8 |
| `emb_dim_per_head` | 32 |
| `mlp_dim_factor` | 4.0 |
| `variational` | `True` |
| `latent_projection_bias` | `False` |

Overfit configs intentionally use a **different, smaller** architecture (example: [`007bbfb7.yaml`](../src/configs/arc_train_overfit/007bbfb7.yaml) has encoder `num_layers: 0`, `latent_dim: 32`). They are execution or overfit tools, not the full ARC baseline model.

## 10. Latent dimension

- Full ARC train/eval config: `encoder_transformer.latent_dim: 128`
- pattern_2d research checkpoints: latent dim 2 (incompatible with ARC configs)
- Overfit smoke config `007bbfb7`: latent dim 32

## 11. Search configuration

Full ARC scheduled evaluations in `arc_train.yaml`:

- Generated ARC tests: mean, gradient ascent 1 step, gradient ascent 20 steps
- JSON training and evaluation subsets (`only_n_tasks: 100`): mean and gradient ascent 20 steps
- Default GA kwargs in that file: `lr: 0.1`, `optimizer: adam`, `optimizer_kwargs.b2: 0.9`

Standalone evaluation script examples in [`src/evaluate_checkpoint.py`](../src/evaluate_checkpoint.py) docstring also show random search and longer GA budgets (for example 200 Adam steps).

Search implementation: [`LPN.generate_output`](../src/models/lpn.py), `_get_gradient_ascent_context`, `_get_random_search_context`.

## 12. Candidate initialization

[`LPN._prepare_latents_before_search`](../src/models/lpn.py):

- mean pair latent
- all pair latents
- optional Gaussian perturbations around the mean (`random_perturbation`)
- random search additionally samples candidates around prepared starts

No zero-vector default start path is used for the standard modes.

## 13. Decoder procedure

[`LPN._generate_output_from_context`](../src/models/lpn.py):

1. Greedy argmax row shape
2. Greedy argmax column shape
3. Autoregressive greedy color tokens over `max_rows * max_cols`

Search scoring uses teacher-forced likelihood on support pairs via [`_compute_log_probs`](../src/models/lpn.py). Query decoding is separate and greedy.

## 14. Top-1 and top-2 selection

- Latents: [`_select_best_and_second_best_latents`](../src/models/lpn.py) ranks by support search score
- Outputs: `return_two_best=True` in [`Evaluator.__init__`](../src/evaluator.py) pmap of `generate_output`
- Submission fields: `attempt_1`, `attempt_2` in `json_submission`

Second attempt is the second-best search latent, not a ground-truth-informed choice.

## 15. Exact-match metric

[`Evaluator.evaluate_generations`](../src/evaluator.py) returns:

- `top_1_shape_accuracy`, `top_1_accuracy`, `top_1_pixel_correctness`
- `top_2_shape_accuracy`, `top_2_accuracy`, `top_2_pixel_correctness`

`top_1_accuracy` requires shape match and exact grid equality on attempt 1. `top_2_accuracy` takes the better of attempt 1 and attempt 2. Metrics are averaged per test grid within a task, then over tasks.

Generated (non-JSON) evaluations use leave-one-out exact match in [`Trainer`](../src/train.py), which is a different metric path.

## 16. Checkpoint loading

| Path | Mechanism |
| --- | --- |
| Train save | [`Trainer.save_checkpoint`](../src/train.py) writes `state.msgpack` and logs a W&B artifact |
| Train resume | `training.resume_from_checkpoint` -> [`Trainer.load_checkpoint`](../src/train.py) via `wandb.use_artifact` |
| Offline / standalone eval | [`src/evaluate_checkpoint.py`](../src/evaluate_checkpoint.py) downloads a W&B model artifact; forces `os.environ["WANDB_MODE"] = "run"` |

No official ARC checkpoint is vendored in this repository. Local `artifacts/checkpoints/pattern_2d_seed_*.msgpack` files are pattern models only.

## 17. Expected external downloads

| Resource | Trigger | Notes |
| --- | --- | --- |
| None for bundled JSON | JSON eval | Already under `src/datasets/json/` |
| None for re-ARC code | ARC generator train | In-repo `re_arc_generators.py` |
| Hugging Face `arcenv/arc_datasets` | `use_hf: True` in [`load_datasets`](../src/data_utils.py) | Requires `HF_TOKEN` for gated/rate-limited access in practice |
| W&B artifacts `TheThinker/ARC/...--checkpoint:vN` | `evaluate_checkpoint.py` / resume | Requires `WANDB_API_KEY`; script forces online-capable mode |

At audit time on this host: `WANDB_API_KEY` unset, `HF_TOKEN` unset.

## 18. W&B and Hugging Face dependencies

- W&B init is hard-coded in [`run`](../src/train.py) with `entity="TheThinker"`, `project="ARC"`
- Offline training is possible with `WANDB_MODE=offline` (used for pattern_2d)
- `evaluate_checkpoint.py` currently overwrites mode to `run`, so offline-only hosts cannot use that script as written without credentials or a code change (code change is out of scope for this audit session)
- HF hub is used only for stored `.npy` datasets, not for the primary `arc_train.yaml` generator path

## 19. Possible sources of data leakage

See the dedicated table in section "Data leakage audit" below. Highest practical risk in the stock `arc_train.yaml` path: logging evaluation-set solution metrics to W&B during training, which enables human checkpoint or hyperparameter selection on the evaluation set even though gradients do not use evaluation solutions.

## 20. Expected training compute

Full [`arc_train.yaml`](../src/configs/arc_train.yaml):

- 500,000 gradient steps
- Effective large ARC transformer (latent 128, 30x30, 4 layers)
- Batch 128 with gradient accumulation 2
- 64 dataloader workers
- Frequent eval: every 100 logs, and each log is every 100 steps, so evaluation every 10,000 steps including JSON subsets of 100 tasks with 20-step Adam search

This is far larger than the completed pattern_2d 200,000-step runs. **Do not launch a full ARC train from this audit without a separate cost and storage plan.**

Rough local evidence: pattern_2d (tiny model) took about 940 to 955 seconds per 200k steps on an RTX 5090 in the GPU compatibility environment. ARC is a different model and data path; wall-clock must be measured on a smoke or short scaling run before estimating 500k-step cost.

## Secondary training path: Hugging Face arrays

[`load_datasets`](../src/data_utils.py) can load `grids.npy` / `shapes.npy` / optional `program_ids.npy` from `arcenv/arc_datasets` when `use_hf: True`, or from local `src/datasets/<folder>/`. This path appears in older or debug configs such as [`task_gen.yaml`](../src/configs/task_gen.yaml) and [`debug.yaml`](../src/configs/debug.yaml). It is not the primary `arc_train.yaml` source.

## Evaluation commands expected by the repository

Training (official config name):

```bash
PYTHONPATH="$PWD" python src/train.py --config-name arc_train
```

Standalone checkpoint eval (requires W&B artifact access):

```bash
PYTHONPATH="$PWD" python src/evaluate_checkpoint.py \
  -w TheThinker/ARC/<run>--checkpoint:vN \
  -jc json/arc-agi_evaluation_challenges.json \
  -js json/arc-agi_evaluation_solutions.json \
  -i gradient_ascent --num-steps 20 --lr 0.1
```

Small overfit train (existing config; smaller model):

```bash
PYTHONPATH="$PWD" python src/train.py \
  --config-path configs/arc_train_overfit --config-name 007bbfb7
```

README.md documents pattern training more clearly than ARC. ARC details live in config headers and `evaluate_checkpoint.py` docstrings.

## Data leakage audit

| Source name | Path or download | Task identifiers | Allowed use | Forbidden use | Leakage risk | Verification status |
| --- | --- | --- | --- | --- | --- | --- |
| re-ARC generators | `src/datasets/task_gen/re_arc_generators.py` | `ARC_TASK_NAMES` = training 400 | Model training; train-side validation carve-outs | Final evaluation reporting as if OOD | Low for ID leakage vs eval | Verified: no overlap with eval IDs |
| Training challenges JSON | `src/datasets/json/arc-agi_training_challenges.json` | 400 train IDs | Analysis; JSON eval on train; overfit debug | Claiming private-test performance | Low | Present locally |
| Training solutions JSON | `.../arc-agi_training_solutions.json` | same | Train-set scoring; policy validation if carved out carefully | Using as ARC evaluation score | Medium if used as silent selection set | Present locally |
| Evaluation challenges JSON | `.../arc-agi_evaluation_challenges.json` | 400 eval IDs | Final frozen evaluation only | Policy training features that need labels; unrestricted tuning | High if misused | Present locally; disjoint from train |
| Evaluation solutions JSON | `.../arc-agi_evaluation_solutions.json` | same | Final frozen scoring only | Training loss, reward learning, early stopping, checkpoint picking, policy tuning | High | Present; used today for W&B-logged metrics in `arc_train.yaml` |
| Bundled "test" challenges | `.../arc-agi_test_challenges.json` | 100 ⊆ train | Not a clean holdout | Presenting as unseen contest test | High if mislabeled as OOD | Verified ⊆ train |
| HF `arcenv/arc_datasets` | hub repo via `load_datasets` | depends on chosen folder | Only after folder provenance is documented | Mixing unlabeled eval-derived arrays into train | Unknown until folder inspected | Not downloaded this session |
| W&B eval metrics from `arc_train` | logged by training process | first 100 eval IDs via `only_n_tasks: 100` | Monitoring only if ignored for selection | Model selection, early stop, LR tuning | High for human selection leakage | Config present; behavior documented, not changed |
| Future RL policy data | proposal only | train-ID carve-out | Policy train/val on train-side tasks | Eval solutions in reward or advantage | N/A until designed | Proposal below |

Stock behavior to keep visible: [`arc_train.yaml`](../src/configs/arc_train.yaml) includes evaluation JSON datasets in `eval.json_datasets` and therefore scores evaluation solutions during training logs. Gradients do not consume those solutions. Selection leakage is still possible through humans or any future automated rule that reads those metrics.

### Proposed leakage-free research split (proposal only; not implemented)

1. Model training: re-ARC over training IDs only.
2. Search hyperparameter and future RL validation: held-out subset of **training** task IDs (JSON training challenges/solutions or re-ARC restricted to that subset).
3. Final ARC-AGI report: evaluation JSON once, after freeze of model, search rules, and policy.
4. Forbidden: evaluation solutions in reward learning, early stopping, checkpoint picking, or policy tuning.
5. Do not use bundled `arc-agi_test_challenges.json` as an OOD benchmark.

## Smallest faithful ARC execution (design)

Preferred scientific command (blocked without W&B access):

- Load a provenance-clear ARC checkpoint with [`evaluate_checkpoint.py`](../src/evaluate_checkpoint.py)
- Score a small `only_n_tasks` slice of **training** JSON first
- Keep evaluation JSON for a later frozen report

Smallest local execution smoke (not a scientific ARC baseline):

- Existing overfit config `007bbfb7` with Hydra overrides to one training step and one eval cycle
- Exercises: ARC generator import/exec, ARC-sized padding path, model init, mean generate_output, training-JSON metric path for one task
- Does **not** match full `arc_train` architecture or produce ARC-AGI accuracy claims

### Smoke execution note (2026-07-23)

1. Stock overfit batch size 128 failed with GPU OOM (~18.64 GiB request). Log: `artifacts/logs/arc_smoke_20260723_122947.log`.
2. Reduced-batch override smoke succeeded. Provenance: `artifacts/results/arc_smoke_20260723_123100.json`. All reported accuracies were 0.0 after one step.

## Immutable controls for future ARC work

Do not change for matched comparisons:

- Encoder/decoder architecture and latent dimension of the chosen ARC config
- `_compute_log_probs` objective weighting
- Greedy `_generate_output_from_context`
- `Evaluator.evaluate_generations` metrics
- Evaluation task IDs and ordering for a frozen table
- Checkpoint bytes (record SHA-256)

## Open blockers for a scientific ARC baseline

1. No local ARC checkpoint with clear provenance.
2. W&B: API key can authenticate, but `TheThinker/ARC` is not visible (`project 'ARC' not found under entity 'TheThinker'`). `evaluate_checkpoint.py` also forces `WANDB_MODE=run`.
3. No public Hugging Face ARC LPN model; only `clement-bonnet/lpn-2d` (PATTERN).
4. Full `arc_train` at stock batch 128 does not fit a 32GB RTX 5090; reduced batch is slow for 500k steps.
5. Stock training logs evaluation solutions; research runs need an explicit non-leaky logging policy before ARC method comparisons.

These blockers do not prevent pattern / `lpn-2d` method implementation.
