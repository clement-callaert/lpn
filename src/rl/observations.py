"""Observation builders for the stop/continue policy.

Query labels must never enter the observation vector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax.numpy as jnp

from src.rl.environment import EpisodeState

FEATURE_NAMES: tuple[str, ...] = (
    "normalized_current_step",
    "normalized_remaining_budget",
    "current_support_score",
    "score_improvement",
    "best_score_improvement",
    "gradient_norm",
    "latent_norm",
    "latent_update_norm",
)

OBSERVATION_DIM: int = len(FEATURE_NAMES)


@dataclass(frozen=True)
class ObservationNormStats:
    """Train-split normalization statistics with provenance fields."""

    score_mean: float
    score_std: float
    grad_norm_mean: float
    grad_norm_std: float
    latent_norm_mean: float
    latent_norm_std: float
    latent_update_norm_mean: float
    latent_update_norm_std: float
    source_split: str
    zero_variance_replaced: tuple[str, ...]
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_names"] = list(self.feature_names)
        payload["zero_variance_replaced"] = list(self.zero_variance_replaced)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObservationNormStats":
        return cls(
            score_mean=float(payload["score_mean"]),
            score_std=float(payload["score_std"]),
            grad_norm_mean=float(payload["grad_norm_mean"]),
            grad_norm_std=float(payload["grad_norm_std"]),
            latent_norm_mean=float(payload["latent_norm_mean"]),
            latent_norm_std=float(payload["latent_norm_std"]),
            latent_update_norm_mean=float(payload["latent_update_norm_mean"]),
            latent_update_norm_std=float(payload["latent_update_norm_std"]),
            source_split=str(payload["source_split"]),
            zero_variance_replaced=tuple(payload.get("zero_variance_replaced", ())),
            feature_names=tuple(payload.get("feature_names", FEATURE_NAMES)),
        )


def _safe_std(values: list[float], *, name: str, replaced: list[str]) -> float:
    if not values:
        raise ValueError(f"cannot compute std for empty feature list: {name}")
    arr = jnp.asarray(values, dtype=jnp.float32)
    std = float(jnp.std(arr))
    if std <= 0.0:
        replaced.append(name)
        return 1.0
    return std


def build_norm_stats_from_raw_features(
    *,
    support_scores: list[float],
    gradient_norms: list[float],
    latent_norms: list[float],
    latent_update_norms: list[float],
    source_split: str,
) -> ObservationNormStats:
    """Compute normalization stats from training-split raw features only."""
    replaced: list[str] = []
    score_mean = float(jnp.mean(jnp.asarray(support_scores, dtype=jnp.float32)))
    grad_mean = float(jnp.mean(jnp.asarray(gradient_norms, dtype=jnp.float32)))
    latent_mean = float(jnp.mean(jnp.asarray(latent_norms, dtype=jnp.float32)))
    update_mean = float(jnp.mean(jnp.asarray(latent_update_norms, dtype=jnp.float32)))
    return ObservationNormStats(
        score_mean=score_mean,
        score_std=_safe_std(support_scores, name="current_support_score", replaced=replaced),
        grad_norm_mean=grad_mean,
        grad_norm_std=_safe_std(gradient_norms, name="gradient_norm", replaced=replaced),
        latent_norm_mean=latent_mean,
        latent_norm_std=_safe_std(latent_norms, name="latent_norm", replaced=replaced),
        latent_update_norm_mean=update_mean,
        latent_update_norm_std=_safe_std(
            latent_update_norms, name="latent_update_norm", replaced=replaced
        ),
        source_split=source_split,
        zero_variance_replaced=tuple(replaced),
    )


def observation_from_episode_state(
    state: EpisodeState,
    norm_stats: ObservationNormStats,
) -> dict[str, jnp.ndarray]:
    """Build the eight-feature observation dict. No query labels."""
    max_steps = jnp.maximum(state.max_steps.astype(jnp.float32), 1.0)
    normalized_current_step = state.current_step.astype(jnp.float32) / max_steps
    normalized_remaining_budget = state.remaining_budget.astype(jnp.float32) / max_steps
    score_improvement = state.current_score - state.previous_score
    best_score_improvement = state.best_score - state.previous_score

    def normalize(value: jnp.ndarray, mean: float, std: float) -> jnp.ndarray:
        return (value - jnp.asarray(mean, dtype=jnp.float32)) / jnp.asarray(std, dtype=jnp.float32)

    return {
        "normalized_current_step": normalized_current_step,
        "normalized_remaining_budget": normalized_remaining_budget,
        "current_support_score": normalize(
            state.current_score, norm_stats.score_mean, norm_stats.score_std
        ),
        "score_improvement": score_improvement.astype(jnp.float32),
        "best_score_improvement": best_score_improvement.astype(jnp.float32),
        "gradient_norm": normalize(
            state.gradient_norm, norm_stats.grad_norm_mean, norm_stats.grad_norm_std
        ),
        "latent_norm": normalize(
            state.latent_norm, norm_stats.latent_norm_mean, norm_stats.latent_norm_std
        ),
        "latent_update_norm": normalize(
            state.latent_update_norm,
            norm_stats.latent_update_norm_mean,
            norm_stats.latent_update_norm_std,
        ),
    }


def observation_dict_to_vector(observation: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Pack observation fields into a fixed-order float32 vector."""
    missing = [name for name in FEATURE_NAMES if name not in observation]
    if missing:
        raise ValueError(f"observation missing features: {missing}")
    extra = [name for name in observation if name not in FEATURE_NAMES]
    if extra:
        raise ValueError(f"observation has unexpected features: {extra}")
    return jnp.stack([observation[name].astype(jnp.float32) for name in FEATURE_NAMES], axis=0)


def assert_no_query_labels(observation: dict[str, jnp.ndarray]) -> None:
    """Fail if forbidden query-label keys appear in an observation."""
    forbidden = {
        "query_exact_match",
        "query_exact_match_if_stopped_now",
        "query_pixel_correctness",
        "query_target",
        "true_output",
        "accuracy",
    }
    overlap = forbidden.intersection(observation)
    if overlap:
        raise ValueError(f"query labels leaked into observation: {sorted(overlap)}")
