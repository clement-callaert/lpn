"""Minimal REINFORCE trainer for the stop/continue policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax

from src.rl.policy import StopContinuePolicy, log_probability_of_action


@dataclass
class MovingAverageBaseline:
    """Scalar moving-average return baseline."""

    value: float = 0.0
    count: int = 0
    momentum: float = 0.9

    def update(self, episode_return: float) -> float:
        if self.count == 0:
            self.value = float(episode_return)
        else:
            self.value = self.momentum * self.value + (1.0 - self.momentum) * float(episode_return)
        self.count += 1
        return self.value


def compute_episode_return(rewards: list[float] | jnp.ndarray) -> jnp.ndarray:
    """Sum undiscounted rewards for one episode."""
    arr = jnp.asarray(rewards, dtype=jnp.float32)
    if arr.size == 0:
        raise ValueError("rewards must be non-empty")
    if not bool(jnp.all(jnp.isfinite(arr))):
        raise ValueError(f"non-finite rewards: {arr}")
    return jnp.sum(arr)


def reinforce_loss(
    policy: StopContinuePolicy,
    params: Any,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    advantage: jnp.ndarray,
    *,
    entropy_coef: float = 0.0,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """REINFORCE loss with optional entropy bonus.

    observations: (T, obs_dim)
    actions: (T,) env actions
    advantage: scalar
    """
    if observations.ndim != 2:
        raise ValueError(f"observations must be rank 2, got shape {observations.shape}")
    if actions.ndim != 1:
        raise ValueError(f"actions must be rank 1, got shape {actions.shape}")
    if observations.shape[0] != actions.shape[0]:
        raise ValueError("observations and actions length mismatch")

    log_probs = jax.vmap(lambda obs, act: log_probability_of_action(policy, params, obs, act))(
        observations, actions
    )
    advantage = jax.lax.stop_gradient(jnp.asarray(advantage, dtype=jnp.float32))
    policy_loss = -advantage * jnp.sum(log_probs)

    p_stop = jax.nn.sigmoid(policy.apply(params, observations))
    p_stop = jnp.clip(p_stop, 1e-6, 1.0 - 1e-6)
    entropy = -jnp.mean(p_stop * jnp.log(p_stop) + (1.0 - p_stop) * jnp.log(1.0 - p_stop))
    loss = policy_loss - jnp.asarray(entropy_coef, dtype=jnp.float32) * entropy
    if not jnp.isfinite(loss):
        raise ValueError(f"non-finite REINFORCE loss: {loss}")
    metrics = {
        "policy_loss": policy_loss.astype(jnp.float32),
        "entropy": entropy.astype(jnp.float32),
        "mean_log_prob": jnp.mean(log_probs).astype(jnp.float32),
        "advantage": advantage.astype(jnp.float32),
    }
    return loss.astype(jnp.float32), metrics


def make_adam_optimizer(learning_rate: float, max_grad_norm: float = 1.0) -> optax.GradientTransformation:
    """Adam with global gradient clipping."""
    if learning_rate < 0:
        raise ValueError(f"learning_rate must be non-negative, got {learning_rate}")
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate),
    )


def apply_policy_update(
    policy: StopContinuePolicy,
    params: Any,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    advantage: float,
    *,
    entropy_coef: float = 0.01,
) -> tuple[Any, Any, dict[str, float]]:
    """One REINFORCE parameter update."""

    def loss_fn(current_params):
        loss, metrics = reinforce_loss(
            policy,
            current_params,
            observations,
            actions,
            jnp.asarray(advantage, dtype=jnp.float32),
            entropy_coef=entropy_coef,
        )
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    if not jnp.isfinite(loss):
        raise ValueError(f"non-finite loss before update: {loss}")
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    grad_leaves = jax.tree_util.tree_leaves(grads)
    grad_norm = float(
        jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in grad_leaves))
    )
    host_metrics = {name: float(value) for name, value in metrics.items()}
    host_metrics["loss"] = float(loss)
    host_metrics["grad_norm"] = grad_norm
    return new_params, new_opt_state, host_metrics
