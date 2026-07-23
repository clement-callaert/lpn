"""Evaluate adaptive and fixed stopping rules on a frozen PATTERN checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import omegaconf

from src.data_utils import make_leave_one_out
from src.datasets.task_gen.dataloader import make_dataset
from src.evaluate_checkpoint import instantiate_model, instantiate_train_state, load_model_weights
from src.search.adaptive_sgd_search import (
    encode_leave_one_out_latents,
    make_adaptive_search_fn,
    make_fixed_search_fn,
    make_latent_conditioned_adaptive_search_fn,
)
from src.search.compute_accounting import (
    add_compute_counters,
    compute_counters_to_dict,
    counters_for_default_sgd_search,
    zero_compute_counters,
)
from src.search.metrics import leave_one_out_metrics
from src.search.stopping_rules import REASON_NAMES, StoppingRuleConfig, validate_stopping_rule_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate adaptive latent-search stopping rules on pattern_2d."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--evaluation-seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument(
        "--stopping-rule",
        type=str,
        required=True,
        choices=[
            "fixed",
            "patience",
            "min_improvement",
            "gradient_norm",
            "patience_with_max_budget",
        ],
    )
    parser.add_argument("--max-search-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min-improvement", type=float, default=None)
    parser.add_argument("--gradient-norm-threshold", type=float, default=None)
    parser.add_argument(
        "--split-name",
        type=str,
        default="evaluation",
        help="Label for the task split (evaluation or development).",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo_root}", *args], text=True
    ).strip()


def package_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "unknown"


def build_rule_config(args: argparse.Namespace) -> StoppingRuleConfig:
    config = StoppingRuleConfig(
        name=args.stopping_rule,
        max_steps=args.max_search_steps,
        patience=args.patience,
        min_improvement=args.min_improvement,
        gradient_norm_threshold=args.gradient_norm_threshold,
    )
    validate_stopping_rule_config(config)
    return config


def _summarize_metrics(
    accuracy_arr: jnp.ndarray,
    correct_shapes_arr: jnp.ndarray,
    pixel_arr: jnp.ndarray,
    per_task_steps: list[list[int]],
    per_task_reasons: list[list[str]],
    total_counters,
    total_extra_objectives: int,
    num_tasks: int,
    num_pairs: int,
) -> dict[str, Any]:
    flat_steps = jnp.asarray([step for task in per_task_steps for step in task])
    return {
        "metrics": {
            "accuracy": float(jnp.mean(accuracy_arr)),
            "correct_shapes": float(jnp.mean(correct_shapes_arr)),
            "pixel_correctness": float(jnp.mean(pixel_arr)),
        },
        "per_example_accuracy": [bool(x) for x in accuracy_arr],
        "per_task_search_steps": per_task_steps,
        "per_task_stop_reasons": per_task_reasons,
        "search_step_stats": {
            "mean": float(jnp.mean(flat_steps)),
            "median": float(jnp.median(flat_steps)),
            "min": int(jnp.min(flat_steps)),
            "max": int(jnp.max(flat_steps)),
            "std": float(jnp.std(flat_steps)),
            "total": int(jnp.sum(flat_steps)),
        },
        "compute_counters": compute_counters_to_dict(total_counters),
        "extra_objective_evaluations_total": total_extra_objectives,
        "number_of_tasks": num_tasks,
        "number_of_leave_one_out_examples": num_tasks * num_pairs,
    }


def evaluate_fixed_matched(
    *,
    model,
    params,
    dataset_grids: jnp.ndarray,
    dataset_shapes: jnp.ndarray,
    evaluation_key: jax.Array,
    num_steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Fixed-K eval matching Trainer leave-one-out + single shared PRNG key."""
    leave_one_out_grids = make_leave_one_out(dataset_grids, axis=-4)
    leave_one_out_shapes = make_leave_one_out(dataset_shapes, axis=-3)
    grids_inputs = dataset_grids[..., 0]
    grids_outputs = dataset_grids[..., 1]
    shapes_inputs = dataset_shapes[..., 0]
    shapes_outputs = dataset_shapes[..., 1]
    num_tasks = int(dataset_grids.shape[0])
    num_pairs = int(dataset_grids.shape[1])

    # Match Trainer.test_dataset_submission key derivation for one device and
    # one full-length batch: split(key, (num_devices, num_batches)).
    num_devices = 1
    num_batches = 1
    batch_keys = jax.random.split(evaluation_key, (num_devices, num_batches))
    batch_key = batch_keys[0, 0]

    output_grids, output_shapes, _info = model.apply(
        {"params": params},
        leave_one_out_grids,
        leave_one_out_shapes,
        grids_inputs,
        shapes_inputs,
        batch_key,
        True,
        "gradient_ascent",
        False,
        method=model.generate_output,
        num_steps=num_steps,
        lr=learning_rate,
    )
    metrics = leave_one_out_metrics(
        output_grids, output_shapes, grids_outputs, shapes_outputs
    )
    per_task_steps = [[num_steps for _ in range(num_pairs)] for _ in range(num_tasks)]
    per_task_reasons = [["fixed" for _ in range(num_pairs)] for _ in range(num_tasks)]
    example_count = num_tasks * num_pairs
    one = counters_for_default_sgd_search(
        candidates=1,
        steps_executed=num_steps,
        max_rows=model.decoder.config.max_rows,
        max_cols=model.decoder.config.max_cols,
        return_two_best=False,
    )
    total_counters = zero_compute_counters()
    for _ in range(example_count):
        total_counters = add_compute_counters(total_counters, one)

    return _summarize_metrics(
        metrics["accuracy"].reshape(-1),
        metrics["correct_shapes"].reshape(-1),
        metrics["pixel_correctness"].reshape(-1),
        per_task_steps,
        per_task_reasons,
        total_counters,
        0,
        num_tasks,
        num_pairs,
    )


def evaluate_adaptive_loop(
    *,
    model,
    params,
    search_fn,
    dataset_grids: jnp.ndarray,
    dataset_shapes: jnp.ndarray,
    evaluation_key: jax.Array,
) -> dict[str, Any]:
    """Adaptive eval with Trainer-matched batched variational encoding."""
    leave_one_out_grids = make_leave_one_out(dataset_grids, axis=-4)
    leave_one_out_shapes = make_leave_one_out(dataset_shapes, axis=-3)
    grids_inputs = dataset_grids[..., 0]
    grids_outputs = dataset_grids[..., 1]
    shapes_inputs = dataset_shapes[..., 0]
    shapes_outputs = dataset_shapes[..., 1]

    num_tasks = dataset_grids.shape[0]
    num_pairs = dataset_grids.shape[1]
    batch_keys = jax.random.split(evaluation_key, (1, 1))
    batch_key = batch_keys[0, 0]
    support_latents = encode_leave_one_out_latents(
        model, params, leave_one_out_grids, leave_one_out_shapes, batch_key
    )

    all_accuracy = []
    all_correct_shapes = []
    all_pixel_correctness = []
    per_task_steps: list[list[int]] = []
    per_task_reasons: list[list[str]] = []
    total_counters = zero_compute_counters()
    total_extra_objectives = 0

    for task_index in range(num_tasks):
        task_steps = []
        task_reasons = []
        for pair_index in range(num_pairs):
            result = search_fn(
                params,
                support_latents[task_index, pair_index],
                leave_one_out_grids[task_index, pair_index],
                leave_one_out_shapes[task_index, pair_index],
                grids_inputs[task_index, pair_index],
                shapes_inputs[task_index, pair_index],
            )
            metrics = leave_one_out_metrics(
                result["output_grids"][None, ...],
                result["output_shapes"][None, ...],
                grids_outputs[task_index, pair_index][None, ...],
                shapes_outputs[task_index, pair_index][None, ...],
            )
            all_accuracy.append(metrics["accuracy"][0])
            all_correct_shapes.append(metrics["correct_shapes"][0])
            all_pixel_correctness.append(metrics["pixel_correctness"][0])
            task_steps.append(int(result["search_steps_executed"]))
            task_reasons.append(
                REASON_NAMES.get(int(result["stop_reason_code"]), "unknown")
            )
            total_counters = add_compute_counters(total_counters, result["counters"])
            total_extra_objectives += int(result["extra_objective_evaluations"])
        per_task_steps.append(task_steps)
        per_task_reasons.append(task_reasons)

    return _summarize_metrics(
        jnp.stack(all_accuracy),
        jnp.stack(all_correct_shapes),
        jnp.stack(all_pixel_correctness),
        per_task_steps,
        per_task_reasons,
        total_counters,
        total_extra_objectives,
        num_tasks,
        num_pairs,
    )


def main() -> None:
    args = parse_args()
    status = "success"
    error_message = None
    result: dict[str, Any]

    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output file: {args.output}. "
            "Choose a unique --output path."
        )

    try:
        rule_config = build_rule_config(args)
        cfg = omegaconf.OmegaConf.load(args.config)
        task_kwargs = {
            "pattern_size": cfg.training.task_generator.pattern_size,
            "num_rows": cfg.training.task_generator.num_rows,
            "num_cols": cfg.training.task_generator.num_cols,
        }
        num_pairs = int(cfg.training.task_generator.num_pairs)

        model = instantiate_model(cfg, mixed_precision=bool(cfg.training.mixed_precision))
        state = instantiate_train_state(model)
        state = load_model_weights(state, str(args.checkpoint.parent), args.checkpoint.name)
        params = state.params

        dataset_grids, dataset_shapes, program_ids = make_dataset(
            args.length,
            num_pairs,
            num_workers=8,
            task_generator_class="PATTERN",
            online_data_augmentation=False,
            seed=args.dataset_seed,
            **task_kwargs,
        )
        # Match Trainer.test_datasets: permute then take a contiguous prefix.
        permute_key = jax.random.PRNGKey(args.dataset_seed)
        indices = jax.random.permutation(permute_key, len(dataset_grids))[: args.length]
        dataset_grids = dataset_grids[indices]
        dataset_shapes = dataset_shapes[indices]
        program_ids = program_ids[indices]

        evaluation_key = jax.random.PRNGKey(args.evaluation_seed)

        warm_start = time.perf_counter()
        if rule_config.name == "fixed":
            search_fn = make_fixed_search_fn(
                model, num_steps=rule_config.max_steps, lr=args.learning_rate
            )
            _ = search_fn(
                params,
                make_leave_one_out(dataset_grids[:1], axis=-4)[0, 0],
                make_leave_one_out(dataset_shapes[:1], axis=-3)[0, 0],
                dataset_grids[0, 0, :, :, 0],
                dataset_shapes[0, 0, :, 0],
                evaluation_key,
            )
        else:
            search_fn = make_latent_conditioned_adaptive_search_fn(
                model, rule_config, lr=args.learning_rate
            )
            warm_latents = encode_leave_one_out_latents(
                model,
                params,
                make_leave_one_out(dataset_grids[:1], axis=-4),
                make_leave_one_out(dataset_shapes[:1], axis=-3),
                evaluation_key,
            )
            _ = search_fn(
                params,
                warm_latents[0, 0],
                make_leave_one_out(dataset_grids[:1], axis=-4)[0, 0],
                make_leave_one_out(dataset_shapes[:1], axis=-3)[0, 0],
                dataset_grids[0, 0, :, :, 0],
                dataset_shapes[0, 0, :, 0],
            )
        warm_up_seconds = time.perf_counter() - warm_start

        start = time.perf_counter()
        if rule_config.name == "fixed":
            evaluation = evaluate_fixed_matched(
                model=model,
                params=params,
                dataset_grids=dataset_grids,
                dataset_shapes=dataset_shapes,
                evaluation_key=evaluation_key,
                num_steps=rule_config.max_steps,
                learning_rate=args.learning_rate,
            )
            key_protocol = "trainer_matched_single_batch_key"
        else:
            evaluation = evaluate_adaptive_loop(
                model=model,
                params=params,
                search_fn=search_fn,
                dataset_grids=dataset_grids,
                dataset_shapes=dataset_shapes,
                evaluation_key=evaluation_key,
            )
            key_protocol = "trainer_matched_batched_encode_then_adaptive_sgd"
        wall_time_seconds = time.perf_counter() - start

        result = {
            "experiment_id": args.output.stem,
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_branch": git_value("branch", "--show-current"),
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
            "hostname": socket.gethostname(),
            "accelerator": str(jax.devices()),
            "environment": {
                "python": platform.python_version(),
                "jax": package_version("jax"),
                "jaxlib": package_version("jaxlib"),
                "flax": package_version("flax"),
                "optax": package_version("optax"),
            },
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": sha256(args.checkpoint),
                "size_bytes": args.checkpoint.stat().st_size,
            },
            "configuration": str(args.config),
            "dataset": {
                "family": "PATTERN",
                "seed": args.dataset_seed,
                "length": args.length,
                "num_pairs": num_pairs,
                "split_name": args.split_name,
                "task_kwargs": task_kwargs,
            },
            "evaluation_seed": args.evaluation_seed,
            "prng_key_protocol": key_protocol,
            "stopping_rule": {
                "name": rule_config.name,
                "max_search_steps": rule_config.max_steps,
                "patience": rule_config.patience,
                "min_improvement": rule_config.min_improvement,
                "gradient_norm_threshold": rule_config.gradient_norm_threshold,
                "learning_rate": args.learning_rate,
                "optimizer": "sgd",
                "search_method": "original_gradient_ascent_sgd",
            },
            "warm_up": {
                "performed": True,
                "wall_clock_seconds": warm_up_seconds,
            },
            "wall_clock_seconds": wall_time_seconds,
            "metrics": evaluation["metrics"],
            "search_step_stats": evaluation["search_step_stats"],
            "compute_counters": evaluation["compute_counters"],
            "extra_objective_evaluations_total": evaluation["extra_objective_evaluations_total"],
            "per_task_search_steps": evaluation["per_task_search_steps"],
            "per_task_stop_reasons": evaluation["per_task_stop_reasons"],
            "number_of_tasks": evaluation["number_of_tasks"],
            "number_of_leave_one_out_examples": evaluation["number_of_leave_one_out_examples"],
            "candidate_count": 1,
            "status": status,
            "error_message": error_message,
            "scientific_use": (
                "Adaptive stopping sandbox on frozen pattern_2d. "
                "Not an ARC-AGI result and not an out-of-distribution claim."
            ),
        }
    except Exception as exc:
        status = "error"
        error_message = f"{type(exc).__name__}: {exc}"
        result = {
            "experiment_id": args.output.stem,
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_branch": git_value("branch", "--show-current"),
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
            "hostname": socket.gethostname(),
            "accelerator": str(jax.devices()),
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": sha256(args.checkpoint) if args.checkpoint.exists() else None,
            },
            "status": status,
            "error_message": error_message,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        raise

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
