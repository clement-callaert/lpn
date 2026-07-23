"""Evaluate a trained stop/continue policy against fixed and heuristic baselines."""

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
import numpy as np

from src.data_utils import make_leave_one_out
from src.datasets.task_gen.dataloader import make_dataset
from src.evaluate_checkpoint import instantiate_model, instantiate_train_state, load_model_weights
from src.rl.environment import (
    ACTION_CONTINUE,
    ACTION_STOP,
    reset_episode,
    step_episode,
)
from src.rl.observations import (
    FEATURE_NAMES,
    ObservationNormStats,
    observation_dict_to_vector,
    observation_from_episode_state,
)
from src.rl.policy import StopContinuePolicy, evaluate_action, init_policy_params
from src.search.compute_accounting import (
    add_compute_counters,
    compute_counters_to_dict,
    zero_compute_counters,
)
from src.search.metrics import leave_one_out_metrics
from src.search.stopping_rules import StoppingRuleConfig
from src.search.adaptive_sgd_search import decode_query, run_adaptive_sgd_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stop/continue policy.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--policy-meta", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--evaluation-seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument("--max-search-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--compute-penalty", type=float, required=True)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument(
        "--controller",
        type=str,
        default="policy",
        choices=[
            "policy",
            "fixed",
            "patience",
            "min_improvement",
            "gradient_norm",
            "random",
            "oracle",
        ],
    )
    parser.add_argument("--fixed-steps", type=int, default=None)
    parser.add_argument("--random-stop-probability", type=float, default=0.2)
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


def load_policy_params(policy_checkpoint: Path, policy_meta: Path, policy: StopContinuePolicy, seed: int):
    meta = json.loads(policy_meta.read_text())
    blob = np.load(policy_checkpoint, allow_pickle=False)
    leaf_count = len(meta["leaves"])
    leaves = [jnp.asarray(blob[f"arr_{i}"]) for i in range(leaf_count)]
    template = init_policy_params(policy, jax.random.PRNGKey(seed))
    treedef = jax.tree_util.tree_structure(template)
    return jax.tree_util.tree_unflatten(treedef, leaves)


def main() -> int:
    args = parse_args()
    started = time.time()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    if not args.config.exists() or not args.checkpoint.exists():
        raise FileNotFoundError("config or checkpoint missing")

    cfg = omegaconf_load(args.config)
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
    stats = ObservationNormStats.from_dict(json.loads(args.normalization_stats.read_text()))

    policy = StopContinuePolicy(continue_bias=1.5)
    policy_params = None
    if args.controller == "policy":
        policy_params = load_policy_params(
            args.policy_checkpoint, args.policy_meta, policy, args.policy_seed
        )

    dataset_grids, dataset_shapes, _ = make_dataset(
        args.length,
        num_pairs,
        num_workers=0,
        task_generator_class="PATTERN",
        online_data_augmentation=False,
        seed=args.dataset_seed,
        **task_kwargs,
    )
    permute_key = jax.random.PRNGKey(args.dataset_seed)
    indices = jax.random.permutation(permute_key, len(dataset_grids))[: args.length]
    dataset_grids = dataset_grids[indices]
    dataset_shapes = dataset_shapes[indices]
    leave_one_out_grids = make_leave_one_out(dataset_grids, axis=-4)
    leave_one_out_shapes = make_leave_one_out(dataset_shapes, axis=-3)
    num_tasks = int(dataset_grids.shape[0])

    accuracies = []
    pixels = []
    steps_list = []
    returns = []
    stop_hist = {i: 0 for i in range(args.max_search_steps + 1)}
    total_counters = zero_compute_counters()

    for task_index in range(num_tasks):
        for pair_index in range(num_pairs):
            episode_key = jax.random.fold_in(
                jax.random.PRNGKey(args.evaluation_seed),
                args.dataset_seed * 1_000_003 + task_index * 1_001 + pair_index,
            )
            pairs = leave_one_out_grids[task_index, pair_index]
            shapes = leave_one_out_shapes[task_index, pair_index]
            q_in = dataset_grids[task_index, pair_index, :, :, 0]
            q_in_s = dataset_shapes[task_index, pair_index, :, 0]
            q_out = dataset_grids[task_index, pair_index, :, :, 1]
            q_out_s = dataset_shapes[task_index, pair_index, :, 1]

            if args.controller in {"patience", "min_improvement", "gradient_norm", "fixed"}:
                if args.controller == "fixed":
                    if args.fixed_steps is None:
                        raise ValueError("fixed controller requires --fixed-steps")
                    rule = StoppingRuleConfig(name="fixed", max_steps=args.fixed_steps)
                elif args.controller == "patience":
                    rule = StoppingRuleConfig(
                        name="patience", max_steps=args.max_search_steps, patience=1
                    )
                elif args.controller == "min_improvement":
                    rule = StoppingRuleConfig(
                        name="min_improvement",
                        max_steps=args.max_search_steps,
                        min_improvement=0.01,
                    )
                else:
                    rule = StoppingRuleConfig(
                        name="gradient_norm",
                        max_steps=args.max_search_steps,
                        gradient_norm_threshold=0.1,
                    )
                result = run_adaptive_sgd_search(
                    model,
                    params,
                    pairs,
                    shapes,
                    q_in,
                    q_in_s,
                    episode_key,
                    rule_config=rule,
                    lr=args.learning_rate,
                )
                metrics = leave_one_out_metrics(
                    result["output_grids"], result["output_shapes"], q_out, q_out_s
                )
                search_steps = int(result["search_steps_executed"])
                exact = float(metrics["accuracy"])
                pixel = float(metrics["pixel_correctness"])
                reward = exact - args.compute_penalty * search_steps
                total_counters = add_compute_counters(total_counters, result["counters"])
            else:
                env_state = reset_episode(
                    model,
                    params,
                    pairs,
                    shapes,
                    q_in,
                    q_in_s,
                    q_out,
                    q_out_s,
                    episode_key,
                    max_steps=args.max_search_steps,
                    lr=args.learning_rate,
                )
                rng = episode_key
                while not bool(env_state.terminated):
                    if args.controller == "policy":
                        obs = observation_dict_to_vector(
                            observation_from_episode_state(env_state, stats)
                        )
                        action, _ = evaluate_action(policy, policy_params, obs)
                        action = int(action)
                    elif args.controller == "random":
                        rng, sub = jax.random.split(rng)
                        stop = bool(
                            jax.random.bernoulli(sub, args.random_stop_probability)
                        )
                        action = ACTION_STOP if stop else ACTION_CONTINUE
                    elif args.controller == "oracle":
                        grids, shapes_out = decode_query(
                            model,
                            params,
                            env_state.best_latent,
                            env_state.query_input,
                            env_state.query_input_shape,
                        )
                        probe_metrics = leave_one_out_metrics(
                            grids, shapes_out, q_out, q_out_s
                        )
                        if float(probe_metrics["accuracy"]) >= 1.0:
                            action = ACTION_STOP
                        elif int(env_state.current_step) >= args.max_search_steps:
                            action = ACTION_STOP
                        else:
                            action = ACTION_CONTINUE
                    else:
                        raise ValueError(f"unsupported controller: {args.controller}")

                    env_state, _, _ = step_episode(
                        model,
                        params,
                        env_state,
                        action,
                        lr=args.learning_rate,
                        compute_penalty=args.compute_penalty,
                    )

                exact = float(
                    leave_one_out_metrics(
                        env_state.output_grids,
                        env_state.output_shapes,
                        q_out,
                        q_out_s,
                    )["accuracy"]
                )
                pixel = float(
                    leave_one_out_metrics(
                        env_state.output_grids,
                        env_state.output_shapes,
                        q_out,
                        q_out_s,
                    )["pixel_correctness"]
                )
                search_steps = int(env_state.search_steps_executed)
                reward = float(env_state.last_reward)
                total_counters = add_compute_counters(total_counters, env_state.counters)

            accuracies.append(exact)
            pixels.append(pixel)
            steps_list.append(search_steps)
            returns.append(reward)
            stop_hist[search_steps] = stop_hist.get(search_steps, 0) + 1

    payload = {
        "status": "success",
        "error_message": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "python_version": sys.version.split()[0],
        "jax_version": package_version("jax"),
        "flax_version": package_version("flax"),
        "optax_version": package_version("optax"),
        "accelerator": [str(d) for d in jax.devices()],
        "hostname": platform.node(),
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "policy_checkpoint": str(args.policy_checkpoint),
        "normalization_stats": str(args.normalization_stats),
        "dataset_seed": args.dataset_seed,
        "evaluation_seed": args.evaluation_seed,
        "length": args.length,
        "max_search_steps": args.max_search_steps,
        "compute_penalty": args.compute_penalty,
        "controller": args.controller,
        "fixed_steps": args.fixed_steps,
        "reward_definition": "exact_match - compute_penalty * search_steps",
        "observation_features": list(FEATURE_NAMES),
        "metrics": {
            "accuracy": float(np.mean(accuracies)),
            "pixel_correctness": float(np.mean(pixels)),
            "mean_return": float(np.mean(returns)),
            "mean_search_steps": float(np.mean(steps_list)),
            "median_search_steps": float(np.median(steps_list)),
            "std_search_steps": float(np.std(steps_list)),
        },
        "stop_histogram": stop_hist,
        "compute_counters": compute_counters_to_dict(total_counters),
        "episode_count": len(accuracies),
        "wall_time_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "success", "metrics": payload["metrics"]}))
    return 0


def omegaconf_load(path: Path):
    import omegaconf

    return omegaconf.OmegaConf.load(path)


if __name__ == "__main__":
    raise SystemExit(main())
