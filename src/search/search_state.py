"""Search-state containers for adaptive and future RL controllers."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax.struct import dataclass

from src.search.compute_accounting import ComputeCounters, zero_compute_counters


@dataclass
class SearchState:
    """Mutable-looking but immutable carry for closed-loop latent search.

    Query labels are intentionally absent so this state can feed a future RL
    observation without leakage.
    """

    current_step: jnp.ndarray
    max_steps: jnp.ndarray
    current_score: jnp.ndarray
    previous_score: jnp.ndarray
    best_score: jnp.ndarray
    steps_since_improvement: jnp.ndarray
    gradient_norm: jnp.ndarray
    latent_norm: jnp.ndarray
    latent_update_norm: jnp.ndarray
    remaining_budget: jnp.ndarray
    stop_reason_code: jnp.ndarray
    counters: ComputeCounters


def initial_search_state(
    *,
    max_steps: int,
    init_score: jnp.ndarray,
    init_latent_norm: jnp.ndarray,
    dtype_int=jnp.int32,
    dtype_float=jnp.float32,
) -> SearchState:
    """Build the search state before any SGD update."""
    if max_steps < 0:
        raise ValueError(f"max_steps must be non-negative, got {max_steps}")
    max_steps_arr = jnp.asarray(max_steps, dtype=dtype_int)
    zero_i = jnp.asarray(0, dtype=dtype_int)
    zero_f = jnp.asarray(0.0, dtype=dtype_float)
    return SearchState(
        current_step=zero_i,
        max_steps=max_steps_arr,
        current_score=jnp.asarray(init_score, dtype=dtype_float),
        previous_score=jnp.asarray(init_score, dtype=dtype_float),
        best_score=jnp.asarray(init_score, dtype=dtype_float),
        steps_since_improvement=zero_i,
        gradient_norm=zero_f,
        latent_norm=jnp.asarray(init_latent_norm, dtype=dtype_float),
        latent_update_norm=zero_f,
        remaining_budget=max_steps_arr,
        stop_reason_code=zero_i,
        counters=zero_compute_counters(dtype=dtype_int),
    )


def rl_observation_from_search_state(
    state: SearchState,
    *,
    score_mean: float = 0.0,
    score_std: float = 1.0,
    grad_norm_mean: float = 0.0,
    grad_norm_std: float = 1.0,
    latent_norm_mean: float = 0.0,
    latent_norm_std: float = 1.0,
    latent_update_norm_mean: float = 0.0,
    latent_update_norm_std: float = 1.0,
) -> dict[str, jnp.ndarray]:
    """Build the minimal RL observation vector components.

    Normalization statistics must come from development tasks, never from the
    final evaluation split. Query ground truth is not included.
    """
    if score_std <= 0:
        raise ValueError(f"score_std must be positive, got {score_std}")
    if grad_norm_std <= 0:
        raise ValueError(f"grad_norm_std must be positive, got {grad_norm_std}")
    if latent_norm_std <= 0:
        raise ValueError(f"latent_norm_std must be positive, got {latent_norm_std}")
    if latent_update_norm_std <= 0:
        raise ValueError(
            f"latent_update_norm_std must be positive, got {latent_update_norm_std}"
        )

    max_steps = jnp.maximum(state.max_steps.astype(jnp.float32), 1.0)
    normalized_step = state.current_step.astype(jnp.float32) / max_steps
    score_improvement = state.current_score - state.previous_score
    best_score_improvement = state.best_score - state.previous_score
    remaining_budget = state.remaining_budget.astype(jnp.float32) / max_steps

    def normalize(value: jnp.ndarray, mean: float, std: float) -> jnp.ndarray:
        return (value - mean) / std

    return {
        "normalized_step": normalized_step,
        "current_support_score": normalize(state.current_score, score_mean, score_std),
        "score_improvement": score_improvement,
        "best_score_improvement": best_score_improvement,
        "gradient_norm": normalize(state.gradient_norm, grad_norm_mean, grad_norm_std),
        "latent_update_norm": normalize(
            state.latent_update_norm, latent_update_norm_mean, latent_update_norm_std
        ),
        "latent_norm": normalize(state.latent_norm, latent_norm_mean, latent_norm_std),
        "remaining_budget": remaining_budget,
    }


def search_state_to_host_dict(state: SearchState) -> dict[str, Any]:
    """Convert a search state to plain Python values for logging."""
    from src.search.compute_accounting import compute_counters_to_dict

    return {
        "current_step": int(state.current_step),
        "max_steps": int(state.max_steps),
        "current_score": float(state.current_score),
        "previous_score": float(state.previous_score),
        "best_score": float(state.best_score),
        "steps_since_improvement": int(state.steps_since_improvement),
        "gradient_norm": float(state.gradient_norm),
        "latent_norm": float(state.latent_norm),
        "latent_update_norm": float(state.latent_update_norm),
        "remaining_budget": int(state.remaining_budget),
        "stop_reason_code": int(state.stop_reason_code),
        "counters": compute_counters_to_dict(state.counters),
    }
