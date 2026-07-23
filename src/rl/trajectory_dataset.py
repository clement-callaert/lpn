"""Offline trajectory generation for stop/continue policy learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp

from src.models.lpn import LPN
from src.rl.observations import ObservationNormStats, build_norm_stats_from_raw_features
from src.search.adaptive_sgd_search import (
    decode_query,
    run_closed_loop_fixed_k_with_trajectory,
)
from src.search.metrics import leave_one_out_metrics


@dataclass
class TrajectoryStepRecord:
    """One decision point along a forced fixed-horizon trajectory."""

    task_index: int
    procedural_task_seed: int
    leave_one_out_index: int
    checkpoint_hash: str
    latent_sampling_seed: int
    current_step: int
    maximum_horizon: int
    current_support_score: float
    previous_support_score: float
    score_improvement: float
    best_support_score: float
    best_score_improvement: float
    gradient_norm: float
    latent_norm: float
    latent_update_norm: float
    remaining_budget: int
    query_exact_match_if_stopped_now: float
    query_pixel_correctness_if_stopped_now: float
    selected_best_iterate_index: int
    counters: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_fixed_horizon_trajectory(
    model: LPN,
    params: Any,
    pairs: jnp.ndarray,
    grid_shapes: jnp.ndarray,
    query_input: jnp.ndarray,
    query_input_shape: jnp.ndarray,
    query_output: jnp.ndarray,
    query_output_shape: jnp.ndarray,
    key: jnp.ndarray,
    *,
    task_index: int,
    procedural_task_seed: int,
    leave_one_out_index: int,
    checkpoint_hash: str,
    latent_sampling_seed: int,
    max_search_steps: int,
    lr: float = 0.1,
) -> list[TrajectoryStepRecord]:
    """Force continue to the horizon and record a labeled step at each t.

    Uses one closed-loop fixed-K run plus a decode at each iterate. Query labels
    are stored only as training diagnostics and must not enter observations.
    """
    if max_search_steps < 0:
        raise ValueError(f"max_search_steps must be non-negative, got {max_search_steps}")

    traj = run_closed_loop_fixed_k_with_trajectory(
        model,
        params,
        pairs,
        grid_shapes,
        query_input,
        query_input_shape,
        key,
        num_steps=max_search_steps,
        lr=lr,
    )
    latent_traj = traj["latent_trajectory"]
    score_traj = traj["score_trajectory"]
    grad_norm_traj = traj["gradient_norm_trajectory"]
    update_norm_traj = traj["latent_update_norm_trajectory"]

    # Decode every iterate once, then select the best-so-far for each stop time.
    def decode_one(latent_vector):
        return decode_query(model, params, latent_vector, query_input, query_input_shape)

    all_grids, all_shapes = jax.vmap(decode_one)(latent_traj)

    records: list[TrajectoryStepRecord] = []
    for step in range(max_search_steps + 1):
        scores_prefix = score_traj[: step + 1]
        best_index = int(jnp.argmax(scores_prefix))
        metrics = leave_one_out_metrics(
            all_grids[best_index],
            all_shapes[best_index],
            query_output,
            query_output_shape,
        )
        current_score = float(score_traj[step])
        previous_score = float(score_traj[step - 1]) if step > 0 else current_score
        best_score = float(jnp.max(scores_prefix))
        gradient_norm = float(grad_norm_traj[step - 1]) if step > 0 else 0.0
        update_norm = float(update_norm_traj[step - 1]) if step > 0 else 0.0
        latent_norm = float(jnp.sqrt(jnp.sum(jnp.square(latent_traj[step]))))
        records.append(
            TrajectoryStepRecord(
                task_index=task_index,
                procedural_task_seed=procedural_task_seed,
                leave_one_out_index=leave_one_out_index,
                checkpoint_hash=checkpoint_hash,
                latent_sampling_seed=latent_sampling_seed,
                current_step=step,
                maximum_horizon=max_search_steps,
                current_support_score=current_score,
                previous_support_score=previous_score,
                score_improvement=current_score - previous_score,
                best_support_score=best_score,
                best_score_improvement=best_score - previous_score,
                gradient_norm=gradient_norm,
                latent_norm=latent_norm,
                latent_update_norm=update_norm,
                remaining_budget=max_search_steps - step,
                query_exact_match_if_stopped_now=float(metrics["accuracy"]),
                query_pixel_correctness_if_stopped_now=float(metrics["pixel_correctness"]),
                selected_best_iterate_index=best_index,
                counters={
                    "search_steps_executed": step,
                    "objective_evaluations": step + 1,
                    "gradient_evaluations": step,
                    "decoder_support_evaluations": step + 1,
                    "output_decoding_calls": 0,
                },
            )
        )
    return records


def compute_norm_stats_from_records(
    records: list[TrajectoryStepRecord],
    *,
    source_split: str,
) -> ObservationNormStats:
    """Build observation normalization statistics from training records only."""
    if not records:
        raise ValueError("cannot compute normalization stats from empty records")
    return build_norm_stats_from_raw_features(
        support_scores=[r.current_support_score for r in records],
        gradient_norms=[r.gradient_norm for r in records],
        latent_norms=[r.latent_norm for r in records],
        latent_update_norms=[r.latent_update_norm for r in records],
        source_split=source_split,
    )


def earliest_correct_step(records: list[TrajectoryStepRecord]) -> int | None:
    """Oracle: earliest step with correct query output, or None if never."""
    for record in records:
        if record.query_exact_match_if_stopped_now >= 1.0:
            return record.current_step
    return None


def records_to_jsonable(records: list[TrajectoryStepRecord]) -> list[dict[str, Any]]:
    return [record.to_dict() for record in records]
