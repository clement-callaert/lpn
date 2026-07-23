# Codebase map

This map describes commit `0adfe56b86d2cba5ae5794edb02da6399a96d98a`. Shapes use `*B` for zero or more batch dimensions, `N` for observed pairs, `R`/`C` for padded grid dimensions, `H` for latent dimension, and `V` for the color vocabulary.

## End-to-end execution

`src/train.py:run` is the Hydra entry point. It instantiates `EncoderTransformer`, `DecoderTransformer`, `LPN`, W&B, and `Trainer`; creates a Flax `TrainState`; optionally restores a W&B artifact; trains; and writes/uploads `state.msgpack` at exit. `Trainer` builds pmapped train, loss-evaluation, and output-generation functions.

For generated-task prediction the call path is:

1. `Trainer.train_epoch`
2. `Trainer.test_dataset_submission`
3. `build_generate_output_to_be_pmapped` (a closure in `Trainer.__init__`)
4. `LPN.generate_output`
5. `LPN._get_gradient_ascent_context` or `LPN._get_random_search_context`
6. `LPN._select_best_and_second_best_latents`
7. `LPN._generate_output_from_context`

ARC JSON prediction instead passes through `Trainer.test_json_submission`, `Evaluator.json_submission`, and then the same `LPN.generate_output` path.

## Data and splits

### `src/datasets/task_gen/task_generator.py`

- `PatternTaskGenerator`: samples one colored `pattern_size x pattern_size` program and generates `num_pairs` input/output examples at random locations. Python's `random` is seeded per PyTorch worker as `seed + worker_id`.
- `ArcTrainTaskGenerator`: samples programs from re-ARC generator functions. It also seeds Python `random` per worker and records the generator index as `program_id`.

### `src/datasets/task_gen/dataloader.py`

- `make_task_gen_dataloader`: selects the PATTERN or ARC iterable generator and wraps it in `JAXDataLoader`.
- `JAXDataLoader`: uses a PyTorch multiprocessing `DataLoader`, converts NumPy batches to JAX `uint8`, and optionally applies JIT-compiled augmentation on CPU. Its augmentation key is initialized from `seed or 0` and split once per batch.
- `collate_fn`: pads grids and returns grids shaped `(steps, batch, N, R, C, 2)` or `(devices, steps, batch_per_device, N, R, C, 2)`; shapes end in `(N, 2, 2)`.
- `make_dataset`: materializes deterministic evaluation tasks using the dataset entry's seed (default 0).

### `src/data_utils.py`

- `load_datasets`: loads stored arrays either locally or from Hugging Face.
- `shuffle_dataset_into_batches`: permutes a stored dataset with an explicit JAX key.
- `data_augmentation_fn`: splits rotation and color keys and vmaps augmentation across batch dimensions.
- `make_leave_one_out`: constructs the `N` support sets containing the other `N-1` examples.

Configuration, rather than a central split registry, determines train/evaluation/test separation. `training.train_datasets` or `training.task_generator` supplies training data. `eval.eval_datasets` computes teacher-forced loss metrics. `eval.test_datasets` performs leave-one-out exact output generation. `eval.json_datasets` identifies ARC challenge and solution JSON files. In `pattern_2d`, training and test data are independently generated from the same PATTERN family; no OOD split is present.

## Encoding and latent initialization

### `src/models/transformer.py:EncoderTransformer.__call__`

Input pairs have shape `(*B, R, C, 2)` and grid shapes `(*B, 2, 2)`. Grid colors, channel, position, and shape tokens plus a CLS token form a sequence of length `1 + 4 + 2*R*C`. Transformer layers return a CLS representation projected to `latent_mu: (*B, H)` and, when variational, `latent_logvar: (*B, H)`. Encoder weights are ordinary trained Flax parameters; no encoder parameter is optimized at test time.

`LPN.generate_output` invokes the encoder with pair dimension included in `*B`, producing `latents_mu` and `latents_logvar` shaped `(*B, N, H)`. For a variational encoder it splits the supplied key once and samples each pair latent as `mu + exp(0.5*logvar)*normal`. Consequently, evaluation is stochastic even in `mean` mode unless its key and task ordering are fixed.

`LPN._prepare_latents_before_search` performs aggregation:

- default: mean across pair latents, shape `(*B, 1, H)`;
- `include_all_latents=true`: all pair latents, `(*B, N, H)`;
- both flags: mean followed by all pair latents, `(*B, N+1, H)`;
- optional `random_perturbation`: appends `num_samples` Gaussian perturbations centered on the mean.

`remove_encoder_latents=true` replaces pair latents with standard-normal samples. There is no explicit zero-latent initialization in the current repository.

## Training objective and gradient flow

### `src/models/lpn.py:LPN.__call__`

Training is leave-one-out across a task's pairs. In `mean` mode, the context for target pair `i` is the mean of the other `N-1` sampled pair latents. `_loss_from_pair_and_context` teacher-forces the decoder and sums row-shape cross entropy, column-shape cross entropy, and grid-token cross entropy averaged over non-padding tokens. The forward loss then adds the configured prior KL term and optional pairwise KL term.

`Trainer.train_one_step` differentiates this complete loss with respect to `state.params`, averages gradients with `jax.lax.pmean`, and updates all encoder and decoder parameters using global-norm clipping plus AdamW. `Trainer.train_n_steps` splits one key over devices and steps and invokes a pmapped `jax.lax.scan`. Gradient accumulation uses a nested scan.

When training itself uses `gradient_ascent`, `_get_gradient_ascent_context` can differentiate through the search only when `stop_gradient_latent_move=false`. The repository default is `true`, which stops the search-gradient value before the Optax update. Decoder/encoder training still receives gradients through the selected context's subsequent reconstruction loss, subject to the discrete candidate selection.

## Latent-search objective and update

### `src/models/lpn.py:LPN._compute_log_probs`

The observed-example search score is:

`sum_over_pairs(row_shape_log_prob + col_shape_log_prob + normalized_mean_grid_token_log_prob)`.

It is a log-likelihood score, so `_get_gradient_ascent_context` maximizes it. The grid term is averaged over valid cells before pair summation; shape terms are not normalized. This exact weighting must remain fixed in initial search-method comparisons.

### `src/models/lpn.py:LPN._get_gradient_ascent_context`

Inputs are candidates `(*B, M, H)`, observed pairs `(*B, N, R, C, 2)`, grid shapes `(*B, N, 2, 2)`, an optional PRNG key, and search configuration. `jax.value_and_grad` differentiates the score with respect to a candidate latent. Nested `jax.vmap` handles candidates and batch dimensions; optional Flax scans trade memory for time across candidates or observed pairs.

Supported optimizers are:

- `sgd`: `optax.clip_by_global_norm(1.0)` then `optax.sgd`;
- `adam`: the same clipping then `optax.adam(..., eps_root=1e-8)`.

An optional cosine decay schedule replaces the scalar learning rate. A Flax `nn.scan` runs `num_steps`; each step computes score and gradient, passes `-gradient` to Optax, and adds the returned update. The implementation retains the initialization and every updated candidate. It recomputes the last-candidate score, flattens candidate and iteration axes, then chooses the best and second-best score. Thus the returned latent need not be the final iterate.

`num_steps` is static under JIT/Flax scan construction. With default initialization (`M=1`), step zero should select the mean sampled pair latent, but equivalence to `mode=mean` must be regression-tested because both paths still involve variational key flow and different compiled control flow.

### `src/models/lpn.py:LPN._get_random_search_context`

This appends Gaussian candidates around prepared encoder candidates, decodes all observed examples with teacher forcing, scores them with `_compute_log_probs`, and selects the best two. Decoder work may be chunked with `scan_batch_size`.

## Output decoding and official metrics

### `src/models/transformer.py:DecoderTransformer.__call__`

Inputs are flattened input/output sequences `(*B, 2+R*C)` and context `(*B, H)`. It returns row logits `(*B, R)`, column logits `(*B, C)`, and grid logits `(*B, R*C, V)`. All decoder weights remain fixed during test-time latent search.

### `src/models/lpn.py:LPN._generate_output_from_context`

Decoding is greedy and fixed: predict row count by argmax, predict column count by argmax, then autoregressively generate `R*C` color tokens with `nn.scan` and argmax. Search comparisons must use this identical decoding path.

`Trainer`'s generated-dataset closure computes exact shape accuracy, masked pixel correctness, and exact grid accuracy. `Evaluator.evaluate_generations` computes ARC top-1/top-2 shape accuracy, exact-match accuracy, and pixel correctness across tasks.

## JAX transformations and PRNG flow

- `jax.pmap`: training, teacher-forced evaluation, and generated-dataset evaluation across devices.
- `jax.jit`: stored-data preparation, CPU augmentation, leave-one-out construction, plus transformations induced by pmap/Flax apply.
- `jax.vmap`: candidate scoring/gradients, batch dimensions, pair-level decoding, and augmentation.
- `jax.lax.scan`: training steps and gradient accumulation. Flax `nn.scan` implements latent-search iterations, candidate/pair memory-saving variants, random-search chunks, and autoregressive output decoding.
- `jax.lax.map`: evaluation batches and task generation calls inside pmapped functions.

The root training key is `PRNGKey(training.seed)` and splits into initialization and training keys. Epoch, train, and evaluation keys are split explicitly. Generated-dataset evaluation splits one key by device and batch, but the same `test_key` is passed separately to every configured test dataset; comparisons therefore receive identically derived key arrays when shapes match. `LPN.generate_output` splits a key for variational sampling, then passes the remaining key to search. Random perturbation currently consumes one key in a single normal draw rather than a per-step stochastic sequence.

## Configuration fields affecting existing search

`inference_mode` selects `mean`, `all` (training only), `first` (generation only), `random_search`, or `gradient_ascent`. Existing gradient-search fields are `num_steps`, `lr`, `lr_schedule`, `lr_schedule_exponent`, `accumulate_gradients_decoder_pairs`, `scan_gradients_latents`, `optimizer`, `optimizer_kwargs`, `include_mean_latent`, `include_all_latents`, `random_perturbation`, `stop_gradient_latent_move`, and `remove_encoder_latents`. Random search uses `num_samples`, `scale`, `scan_batch_size`, and the include flags.

No current code counts decoder calls, gradient evaluations, objective evaluations, or candidate-time exposure. Those counters must be added before compute-matched claims.
