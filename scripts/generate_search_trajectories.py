"""Generate offline fixed-horizon search trajectories for stop/continue RL."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import omegaconf

from src.data_utils import make_leave_one_out
from src.datasets.task_gen.dataloader import make_dataset
from src.evaluate_checkpoint import instantiate_model, instantiate_train_state, load_model_weights
from src.rl.trajectory_dataset import (
    compute_norm_stats_from_records,
    generate_fixed_horizon_trajectory,
    records_to_jsonable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed-horizon latent-search trajectories."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-seeds",
        type=int,
        nargs="+",
        required=True,
        help="Procedural dataset seeds for this split.",
    )
    parser.add_argument("--evaluation-seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=48)
    parser.add_argument("--max-search-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--split-name", type=str, required=True)
    parser.add_argument(
        "--write-norm-stats",
        action="store_true",
        help="Write observation normalization stats (train split only).",
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


def main() -> int:
    args = parse_args()
    started = time.time()
    status = "success"
    error_message = None
    if not args.config.exists():
        raise FileNotFoundError(f"config not found: {args.config}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if args.max_search_steps < 0:
        raise ValueError("max-search-steps must be non-negative")
    if args.write_norm_stats and args.split_name != "train":
        raise ValueError("write-norm-stats is only allowed for split-name=train")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    raw_records = []

    try:
        cfg = omegaconf.OmegaConf.load(args.config)
        num_pairs = int(cfg.training.task_generator.num_pairs)
        task_kwargs = {
            "pattern_size": cfg.training.task_generator.pattern_size,
            "num_rows": cfg.training.task_generator.num_rows,
            "num_cols": cfg.training.task_generator.num_cols,
        }
        model = instantiate_model(cfg, mixed_precision=False)
        state = instantiate_train_state(model)
        state = load_model_weights(state, str(args.checkpoint.parent), args.checkpoint.name)
        params = state.params
        checkpoint_hash = sha256(args.checkpoint)

        for dataset_seed in args.dataset_seeds:
            dataset_grids, dataset_shapes, _program_ids = make_dataset(
                args.length,
                num_pairs,
                num_workers=0,
                task_generator_class="PATTERN",
                online_data_augmentation=False,
                seed=dataset_seed,
                **task_kwargs,
            )
            permute_key = jax.random.PRNGKey(dataset_seed)
            indices = jax.random.permutation(permute_key, len(dataset_grids))[: args.length]
            dataset_grids = dataset_grids[indices]
            dataset_shapes = dataset_shapes[indices]
            leave_one_out_grids = make_leave_one_out(dataset_grids, axis=-4)
            leave_one_out_shapes = make_leave_one_out(dataset_shapes, axis=-3)
            num_tasks = int(dataset_grids.shape[0])

            for task_index in range(num_tasks):
                for pair_index in range(num_pairs):
                    # Keep policy/task/latent keys independent by hashing indices.
                    episode_key = jax.random.fold_in(
                        jax.random.PRNGKey(args.evaluation_seed),
                        dataset_seed * 1_000_003 + task_index * 1_001 + pair_index,
                    )
                    records = generate_fixed_horizon_trajectory(
                        model,
                        params,
                        leave_one_out_grids[task_index, pair_index],
                        leave_one_out_shapes[task_index, pair_index],
                        dataset_grids[task_index, pair_index, :, :, 0],
                        dataset_shapes[task_index, pair_index, :, 0],
                        dataset_grids[task_index, pair_index, :, :, 1],
                        dataset_shapes[task_index, pair_index, :, 1],
                        episode_key,
                        task_index=task_index,
                        procedural_task_seed=dataset_seed,
                        leave_one_out_index=pair_index,
                        checkpoint_hash=checkpoint_hash,
                        latent_sampling_seed=int(episode_key[0]),
                        max_search_steps=args.max_search_steps,
                        lr=args.learning_rate,
                    )
                    raw_records.extend(records)
                    all_records.extend(records_to_jsonable(records))

        payload: dict[str, Any] = {
            "status": status,
            "error_message": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_branch": git_value("branch", "--show-current"),
            "python_version": sys.version.split()[0],
            "jax_version": package_version("jax"),
            "flax_version": package_version("flax"),
            "optax_version": package_version("optax"),
            "accelerator": [str(device) for device in jax.devices()],
            "hostname": platform.node(),
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "config_path": str(args.config),
            "dataset_seeds": list(args.dataset_seeds),
            "evaluation_seed": args.evaluation_seed,
            "length": args.length,
            "max_search_steps": args.max_search_steps,
            "learning_rate": args.learning_rate,
            "split_name": args.split_name,
            "episode_count": len(all_records) // (args.max_search_steps + 1),
            "step_record_count": len(all_records),
            "records": all_records,
            "wall_time_seconds": time.time() - started,
        }

        if args.write_norm_stats:
            stats = compute_norm_stats_from_records(raw_records, source_split=args.split_name)
            stats_path = args.output.with_name(args.output.stem + "_norm_stats.json")
            if stats_path.exists():
                raise FileExistsError(f"refusing to overwrite: {stats_path}")
            stats_path.write_text(json.dumps(stats.to_dict(), indent=2) + "\n")
            payload["normalization_stats_path"] = str(stats_path)
            payload["normalization_stats"] = stats.to_dict()

        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"status": status, "output": str(args.output), "episodes": payload["episode_count"]}))
        return 0
    except Exception as exc:
        status = "error"
        error_message = f"{type(exc).__name__}: {exc}"
        failure = {
            "status": status,
            "error_message": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "wall_time_seconds": time.time() - started,
        }
        failure_path = args.output.with_suffix(".error.json")
        if not failure_path.exists():
            failure_path.write_text(json.dumps(failure, indent=2) + "\n")
        print(json.dumps(failure), file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
