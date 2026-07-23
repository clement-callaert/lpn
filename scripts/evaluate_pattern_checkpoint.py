"""Evaluate a local PATTERN checkpoint without changing model or metric code."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import socket
import subprocess
import time
from pathlib import Path

import jax
from jax.tree_util import tree_map
import omegaconf

from src.evaluate_checkpoint import instantiate_model, instantiate_train_state, load_model_weights
from src.train import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-seed", type=int, default=0)
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--search-steps", type=int, nargs="*", default=[0, 10])
    parser.add_argument("--learning-rate", type=float, default=0.1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    # One-shot safe.directory avoids dubious-ownership failures without changing git config.
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo_root}", *args], text=True
    ).strip()


def main() -> None:
    args = parse_args()
    cfg = omegaconf.OmegaConf.load(args.config)

    task_kwargs = {
        "pattern_size": cfg.training.task_generator.pattern_size,
        "num_rows": cfg.training.task_generator.num_rows,
        "num_cols": cfg.training.task_generator.num_cols,
    }
    common_dataset = {
        "generator": "PATTERN",
        "task_generator_kwargs": task_kwargs,
        "num_pairs": cfg.training.task_generator.num_pairs,
        "length": args.length,
        "batch_size": args.batch_size,
        "num_tasks_to_show": 0,
        "seed": args.dataset_seed,
    }
    test_datasets = [{**common_dataset, "name": "mean", "inference_mode": "mean"}]
    test_datasets.extend(
        {
            **common_dataset,
            "name": f"gradient_ascent_{steps}",
            "inference_mode": "gradient_ascent",
            "inference_kwargs": {"num_steps": steps, "lr": args.learning_rate},
        }
        for steps in args.search_steps
    )
    cfg.eval.test_datasets = omegaconf.OmegaConf.create(test_datasets)
    cfg.eval.eval_datasets = None
    cfg.eval.json_datasets = None

    model = instantiate_model(cfg, mixed_precision=cfg.training.mixed_precision)
    state = instantiate_train_state(model)
    state = load_model_weights(state, str(args.checkpoint.parent), args.checkpoint.name)
    trainer = Trainer(cfg, model)
    state = jax.device_put_replicated(state, trainer.devices)

    evaluations = []
    evaluation_key = jax.random.PRNGKey(args.evaluation_seed)
    for dataset in trainer.test_datasets:
        start = time.perf_counter()
        metrics, figure_grids, figure_heatmap, figure_latents = trainer.test_dataset_submission(
            state, key=evaluation_key, **dataset
        )
        elapsed = time.perf_counter() - start
        for figure in (figure_grids, figure_heatmap, figure_latents):
            if figure is not None:
                figure.clear()
        evaluations.append(
            {
                "name": dataset["test_name"],
                "inference_mode": "mean" if dataset["test_name"].endswith("_mean") else "gradient_ascent",
                "number_of_steps": 0
                if dataset["test_name"].endswith("_mean")
                else int(dataset["test_name"].rsplit("_", 1)[-1]),
                "wall_clock_seconds": elapsed,
                "metrics": metrics,
            }
        )

    result = {
        "experiment_id": args.output.stem,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "accelerator": str(jax.devices()),
        "dataset": f"procedural PATTERN, length={args.length}, seed={args.dataset_seed}",
        "split": "fixed generated evaluation tasks from the training family",
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256(args.checkpoint),
            "size_bytes": args.checkpoint.stat().st_size,
        },
        "method": "controlled mean versus original latent-search step ablation",
        "method_hyperparameters": {
            "learning_rate": args.learning_rate,
            "optimizer": "sgd (repository default)",
            "search_steps": args.search_steps,
        },
        "seed": args.evaluation_seed,
        "number_of_candidates": 1,
        "decoder_calls": None,
        "configuration": str(args.config),
        "evaluations": evaluations,
        "status": "success",
        "error_message": None,
        "scientific_use": "Single-checkpoint debugging ablation; not a multi-training-seed comparison.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
