"""Logical compute counters for latent search.

These counters measure algorithmic work from candidates, steps, and execution
paths. They are not Python call counts. Wall time stays on the host.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class ComputeCounters:
    """Immutable JAX-compatible counters for one search episode or aggregate."""

    search_steps_executed: jnp.ndarray
    objective_evaluations: jnp.ndarray
    gradient_evaluations: jnp.ndarray
    decoder_support_evaluations: jnp.ndarray
    latent_candidates_evaluated: jnp.ndarray
    candidate_step_exposure: jnp.ndarray
    output_decoding_calls: jnp.ndarray
    autoregressive_token_predictions: jnp.ndarray


COUNTER_FIELD_NAMES: tuple[str, ...] = (
    "search_steps_executed",
    "objective_evaluations",
    "gradient_evaluations",
    "decoder_support_evaluations",
    "latent_candidates_evaluated",
    "candidate_step_exposure",
    "output_decoding_calls",
    "autoregressive_token_predictions",
)


def zero_compute_counters(dtype=jnp.int32) -> ComputeCounters:
    """Return a counter structure filled with zeros."""
    zero = jnp.asarray(0, dtype=dtype)
    return ComputeCounters(
        search_steps_executed=zero,
        objective_evaluations=zero,
        gradient_evaluations=zero,
        decoder_support_evaluations=zero,
        latent_candidates_evaluated=zero,
        candidate_step_exposure=zero,
        output_decoding_calls=zero,
        autoregressive_token_predictions=zero,
    )


def add_compute_counters(left: ComputeCounters, right: ComputeCounters) -> ComputeCounters:
    """Add two counter structures field by field."""
    return ComputeCounters(
        search_steps_executed=left.search_steps_executed + right.search_steps_executed,
        objective_evaluations=left.objective_evaluations + right.objective_evaluations,
        gradient_evaluations=left.gradient_evaluations + right.gradient_evaluations,
        decoder_support_evaluations=(
            left.decoder_support_evaluations + right.decoder_support_evaluations
        ),
        latent_candidates_evaluated=(
            left.latent_candidates_evaluated + right.latent_candidates_evaluated
        ),
        candidate_step_exposure=left.candidate_step_exposure + right.candidate_step_exposure,
        output_decoding_calls=left.output_decoding_calls + right.output_decoding_calls,
        autoregressive_token_predictions=(
            left.autoregressive_token_predictions + right.autoregressive_token_predictions
        ),
    )


def compute_counters_to_dict(counters: ComputeCounters) -> dict[str, int]:
    """Convert counters to a Python dict with stable field names."""
    return {name: int(getattr(counters, name)) for name in COUNTER_FIELD_NAMES}


def validate_compute_counters_host(counters: ComputeCounters) -> None:
    """Reject negative counters on the host after device-to-host conversion."""
    values = compute_counters_to_dict(counters)
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")


def counters_for_default_sgd_search(
    *,
    candidates: int,
    steps_executed: int,
    max_rows: int,
    max_cols: int,
    return_two_best: bool = False,
    accumulate_over_pairs: bool = False,
    num_support_pairs: int = 1,
    dtype=jnp.int32,
) -> ComputeCounters:
    """Fill analytic counters for default teacher-forced SGD latent search.

    For the default path (no pair accumulation):
      objective_evaluations = candidates * (steps_executed + 1)
      gradient_evaluations = candidates * steps_executed
      candidate_step_exposure = candidates * (steps_executed + 1)

    The +1 accounts for scoring the final iterate (and the only score when
    steps_executed is 0).
    """
    if candidates < 0:
        raise ValueError(f"candidates must be non-negative, got {candidates}")
    if steps_executed < 0:
        raise ValueError(f"steps_executed must be non-negative, got {steps_executed}")
    if max_rows < 1:
        raise ValueError(f"max_rows must be positive, got {max_rows}")
    if max_cols < 1:
        raise ValueError(f"max_cols must be positive, got {max_cols}")
    if num_support_pairs < 1:
        raise ValueError(f"num_support_pairs must be positive, got {num_support_pairs}")

    objective_evaluations = candidates * (steps_executed + 1)
    gradient_evaluations = candidates * steps_executed
    if accumulate_over_pairs:
        decoder_support_evaluations = objective_evaluations * num_support_pairs
    else:
        decoder_support_evaluations = objective_evaluations

    output_decoding_calls = 2 if return_two_best else 1
    autoregressive_token_predictions = (2 + max_rows * max_cols) * output_decoding_calls

    def as_array(value: int) -> Any:
        return jnp.asarray(value, dtype=dtype)

    return ComputeCounters(
        search_steps_executed=as_array(steps_executed),
        objective_evaluations=as_array(objective_evaluations),
        gradient_evaluations=as_array(gradient_evaluations),
        decoder_support_evaluations=as_array(decoder_support_evaluations),
        latent_candidates_evaluated=as_array(candidates),
        candidate_step_exposure=as_array(candidates * (steps_executed + 1)),
        output_decoding_calls=as_array(output_decoding_calls),
        autoregressive_token_predictions=as_array(autoregressive_token_predictions),
    )
