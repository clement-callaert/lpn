"""Train a Bernoulli stop/continue policy with REINFORCE on offline trajectories."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
import optax

from src.rl.observations import FEATURE_NAMES, OBSERVATION_DIM, ObservationNormStats
from src.rl.policy import StopContinuePolicy, init_policy_params
from src.rl.reinforce import MovingAverageBaseline, compute_episode_return, make_adam_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stop/continue REINFORCE policy.")
    parser.add_argument("--train-trajectories", type=Path, required=True)
    parser.add_argument("--validation-trajectories", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-type", type=str, default="linear", choices=["linear"])
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--compute-penalty", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-updates", type=int, default=200)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--baseline-momentum", type=float, default=0.9)
    return parser.parse_args()


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


def load_trajectory_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"trajectory file not found: {path}")
    return json.loads(path.read_text())


def group_episodes(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    episodes: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            record["procedural_task_seed"],
            record["task_index"],
            record["leave_one_out_index"],
            record["checkpoint_hash"],
        )
        episodes.setdefault(key, []).append(record)
    ordered = []
    for key in sorted(episodes):
        steps = sorted(episodes[key], key=lambda item: item["current_step"])
        ordered.append(steps)
    return ordered


def raw_observation_vector(step: dict[str, Any], stats: ObservationNormStats) -> np.ndarray:
    max_steps = max(float(step["maximum_horizon"]), 1.0)
    values = {
        "normalized_current_step": float(step["current_step"]) / max_steps,
        "normalized_remaining_budget": float(step["remaining_budget"]) / max_steps,
        "current_support_score": (float(step["current_support_score"]) - stats.score_mean)
        / stats.score_std,
        "score_improvement": float(step["score_improvement"]),
        "best_score_improvement": float(step["best_score_improvement"]),
        "gradient_norm": (float(step["gradient_norm"]) - stats.grad_norm_mean)
        / stats.grad_norm_std,
        "latent_norm": (float(step["latent_norm"]) - stats.latent_norm_mean)
        / stats.latent_norm_std,
        "latent_update_norm": (
            float(step["latent_update_norm"]) - stats.latent_update_norm_mean
        )
        / stats.latent_update_norm_std,
    }
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)


def precompute_episode_arrays(
    episodes: list[list[dict[str, Any]]], stats: ObservationNormStats
) -> list[dict[str, np.ndarray]]:
    prepared = []
    for episode in episodes:
        obs = np.stack([raw_observation_vector(step, stats) for step in episode], axis=0)
        exact = np.asarray(
            [float(step["query_exact_match_if_stopped_now"]) for step in episode],
            dtype=np.float32,
        )
        prepared.append(
            {
                "observations": obs,
                "exact_match_if_stopped": exact,
                "horizon": len(episode) - 1,
            }
        )
    return prepared


def rollout_prepared(
    policy: StopContinuePolicy,
    params: Any,
    episode: dict[str, np.ndarray],
    rng: np.random.Generator,
    *,
    compute_penalty: float,
    deterministic: bool,
) -> dict[str, Any]:
    observations = episode["observations"]
    exact = episode["exact_match_if_stopped"]
    actions = []
    rewards = []
    stop_probs = []
    stopped_step = None
    exact_match = 0.0
    for t in range(observations.shape[0]):
        obs = jnp.asarray(observations[t])
        stop_logit = policy.apply(params, obs)
        p_stop = float(jax.nn.sigmoid(stop_logit))
        p_stop = min(max(p_stop, 1e-6), 1.0 - 1e-6)
        stop_probs.append(p_stop)
        if deterministic:
            stop = p_stop >= 0.5
        else:
            stop = bool(rng.random() < p_stop)
        # env: 0=stop, 1=continue
        action = 0 if stop else 1
        actions.append(action)
        if stop:
            stopped_step = t
            exact_match = float(exact[t])
            rewards.append(exact_match - compute_penalty * t)
            break
        rewards.append(0.0)
    else:
        stopped_step = observations.shape[0] - 1
        exact_match = float(exact[stopped_step])
        rewards[-1] = exact_match - compute_penalty * stopped_step

    return {
        "observations": jnp.asarray(observations[: len(actions)], dtype=jnp.float32),
        "actions": jnp.asarray(actions, dtype=jnp.int32),
        "stop_probs": stop_probs,
        "rewards": rewards,
        "return": float(compute_episode_return(rewards)),
        "exact_match": exact_match,
        "search_steps": int(stopped_step),
    }


def evaluate_prepared(
    policy: StopContinuePolicy,
    params: Any,
    episodes: list[dict[str, np.ndarray]],
    *,
    compute_penalty: float,
    seed: int,
    deterministic: bool,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    returns = []
    matches = []
    steps = []
    stop_hist: dict[int, int] = {}
    for episode in episodes:
        result = rollout_prepared(
            policy,
            params,
            episode,
            rng,
            compute_penalty=compute_penalty,
            deterministic=deterministic,
        )
        returns.append(result["return"])
        matches.append(result["exact_match"])
        steps.append(result["search_steps"])
        stop_hist[result["search_steps"]] = stop_hist.get(result["search_steps"], 0) + 1
    return {
        "mean_return": float(np.mean(returns)),
        "mean_exact_match": float(np.mean(matches)),
        "mean_search_steps": float(np.mean(steps)),
        "median_search_steps": float(np.median(steps)),
        "stop_histogram": stop_hist,
    }


def main() -> int:
    args = parse_args()
    started = time.time()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to use non-empty output dir: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_payload = load_trajectory_payload(args.train_trajectories)
    val_payload = load_trajectory_payload(args.validation_trajectories)
    stats = ObservationNormStats.from_dict(json.loads(args.normalization_stats.read_text()))
    if stats.source_split != "train":
        raise ValueError(
            f"normalization stats must come from train split, got {stats.source_split}"
        )

    train_episodes = precompute_episode_arrays(
        group_episodes(train_payload["records"]), stats
    )
    val_episodes = precompute_episode_arrays(group_episodes(val_payload["records"]), stats)
    if not train_episodes or not val_episodes:
        raise ValueError("empty train or validation episodes")

    policy = StopContinuePolicy(continue_bias=1.5)
    params = init_policy_params(policy, jax.random.PRNGKey(args.policy_seed))
    optimizer = make_adam_optimizer(args.learning_rate, max_grad_norm=args.max_grad_norm)
    opt_state = optimizer.init(params)
    baseline = MovingAverageBaseline(momentum=args.baseline_momentum)
    rng = np.random.default_rng(args.policy_seed + 1000)
    max_len = max(ep["observations"].shape[0] for ep in train_episodes)

    def pad_rollout(observations: jnp.ndarray, actions: jnp.ndarray):
        t = int(observations.shape[0])
        obs_pad = jnp.zeros((max_len, OBSERVATION_DIM), dtype=jnp.float32)
        act_pad = jnp.zeros((max_len,), dtype=jnp.int32)
        mask = jnp.zeros((max_len,), dtype=jnp.float32)
        obs_pad = obs_pad.at[:t].set(observations)
        act_pad = act_pad.at[:t].set(actions)
        mask = mask.at[:t].set(1.0)
        return obs_pad, act_pad, mask

    @jax.jit
    def batch_update(params, opt_state, obs_batch, act_batch, mask_batch, advantages):
        # obs_batch: (B, T, D), act_batch: (B, T), mask_batch: (B, T), advantages: (B,)
        def loss_fn(current_params):
            def one_episode(obs, acts, mask, advantage):
                from src.rl.policy import log_probability_of_action

                log_probs = jax.vmap(
                    lambda o, a: log_probability_of_action(policy, current_params, o, a)
                )(obs, acts)
                weighted = jnp.sum(log_probs * mask)
                p_stop = jax.nn.sigmoid(policy.apply(current_params, obs))
                p_stop = jnp.clip(p_stop, 1e-6, 1.0 - 1e-6)
                entropy = -jnp.sum(
                    mask
                    * (p_stop * jnp.log(p_stop) + (1.0 - p_stop) * jnp.log(1.0 - p_stop))
                ) / jnp.maximum(jnp.sum(mask), 1.0)
                return -jax.lax.stop_gradient(advantage) * weighted - args.entropy_coef * entropy

            losses = jax.vmap(one_episode)(obs_batch, act_batch, mask_batch, advantages)
            return jnp.mean(losses)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    history = []
    best_val_return = -1e9
    best_params = params

    for update_index in range(args.num_updates):
        batch_returns = []
        batch_matches = []
        batch_steps = []
        obs_list = []
        act_list = []
        mask_list = []
        adv_list = []
        for _ in range(args.batch_size):
            episode = train_episodes[int(rng.integers(0, len(train_episodes)))]
            rollout = rollout_prepared(
                policy,
                params,
                episode,
                rng,
                compute_penalty=args.compute_penalty,
                deterministic=False,
            )
            episode_return = rollout["return"]
            baseline_value = baseline.update(episode_return)
            advantage = episode_return - baseline_value
            obs_pad, act_pad, mask = pad_rollout(rollout["observations"], rollout["actions"])
            obs_list.append(obs_pad)
            act_list.append(act_pad)
            mask_list.append(mask)
            adv_list.append(advantage)
            batch_returns.append(episode_return)
            batch_matches.append(rollout["exact_match"])
            batch_steps.append(rollout["search_steps"])

        params, opt_state, loss = batch_update(
            params,
            opt_state,
            jnp.stack(obs_list),
            jnp.stack(act_list),
            jnp.stack(mask_list),
            jnp.asarray(adv_list, dtype=jnp.float32),
        )
        if not bool(jnp.isfinite(loss)):
            raise ValueError(f"non-finite loss at update {update_index}: {loss}")

        val_metrics = evaluate_prepared(
            policy,
            params,
            val_episodes,
            compute_penalty=args.compute_penalty,
            seed=args.policy_seed + 10_000 + update_index,
            deterministic=True,
        )
        row = {
            "update": update_index,
            "train_mean_return": float(np.mean(batch_returns)),
            "train_mean_exact_match": float(np.mean(batch_matches)),
            "train_mean_steps": float(np.mean(batch_steps)),
            "val_mean_return": val_metrics["mean_return"],
            "val_mean_exact_match": val_metrics["mean_exact_match"],
            "val_mean_steps": val_metrics["mean_search_steps"],
            "baseline": baseline.value,
            "loss": float(loss),
        }
        history.append(row)
        if val_metrics["mean_return"] > best_val_return:
            best_val_return = val_metrics["mean_return"]
            best_params = params
        if update_index % 20 == 0 or update_index == args.num_updates - 1:
            print(json.dumps(row), flush=True)

    flat_leaves, treedef = jax.tree_util.tree_flatten(best_params)
    policy_path = args.output_dir / "best_policy.npz"
    np.savez(policy_path, *[np.asarray(leaf) for leaf in flat_leaves])
    meta_path = args.output_dir / "best_policy_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "treedef": str(treedef),
                "leaves": [
                    {
                        "index": i,
                        "shape": list(np.asarray(leaf).shape),
                        "dtype": str(np.asarray(leaf).dtype),
                    }
                    for i, leaf in enumerate(flat_leaves)
                ],
            },
            indent=2,
        )
        + "\n"
    )

    summary = {
        "status": "success",
        "error_message": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "python_version": sys.version.split()[0],
        "jax_version": package_version("jax"),
        "flax_version": package_version("flax"),
        "optax_version": package_version("optax"),
        "accelerator": [str(device) for device in jax.devices()],
        "hostname": platform.node(),
        "train_trajectories": str(args.train_trajectories),
        "validation_trajectories": str(args.validation_trajectories),
        "normalization_stats": str(args.normalization_stats),
        "policy_type": args.policy_type,
        "hidden_dim": args.hidden_dim,
        "compute_penalty": args.compute_penalty,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_updates": args.num_updates,
        "policy_seed": args.policy_seed,
        "entropy_coef": args.entropy_coef,
        "max_grad_norm": args.max_grad_norm,
        "reward_definition": "terminal exact_match - compute_penalty * search_steps",
        "observation_features": list(FEATURE_NAMES),
        "observation_dim": OBSERVATION_DIM,
        "best_validation_return": best_val_return,
        "history": history,
        "policy_checkpoint": str(policy_path),
        "wall_time_seconds": time.time() - started,
    }
    (args.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "success", "best_validation_return": best_val_return}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
