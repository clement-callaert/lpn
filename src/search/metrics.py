"""Official leave-one-out metric formulas copied for the adaptive harness.

Keep these formulas aligned with src/train.py. Do not edit train.py.
"""

from __future__ import annotations

import jax.numpy as jnp


def leave_one_out_metrics(
    generated_grids: jnp.ndarray,
    generated_shapes: jnp.ndarray,
    true_grids: jnp.ndarray,
    true_shapes: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Compute shape / pixel / exact-match metrics.

    Args:
        generated_grids: predicted grids. Shape (*B, R, C).
        generated_shapes: predicted shapes. Shape (*B, 2).
        true_grids: ground-truth output grids. Shape (*B, R, C).
        true_shapes: ground-truth shapes. Shape (*B, 2).
    """
    correct_shapes = jnp.all(generated_shapes == true_shapes, axis=-1)
    batch_ndims = len(true_grids.shape[:-2])

    row_arange = jnp.arange(true_grids.shape[-2]).reshape((*batch_ndims * (1,), true_grids.shape[-2]))
    col_arange = jnp.arange(true_grids.shape[-1]).reshape((*batch_ndims * (1,), true_grids.shape[-1]))
    row_mask = row_arange < true_shapes[..., :1]
    col_mask = col_arange < true_shapes[..., 1:]
    valid_mask = row_mask[..., None] & col_mask[..., None, :]

    pixels_equal = jnp.where(
        valid_mask & correct_shapes[..., None, None],
        generated_grids == true_grids,
        False,
    )
    pixel_correctness = pixels_equal.sum(axis=(-1, -2)) / true_shapes.prod(axis=-1)
    accuracy = pixels_equal.sum(axis=(-1, -2)) == true_shapes.prod(axis=-1)
    return {
        "correct_shapes": correct_shapes,
        "pixel_correctness": pixel_correctness,
        "accuracy": accuracy,
    }


def aggregate_metric_means(metrics: dict[str, jnp.ndarray]) -> dict[str, float]:
    """Mean-aggregate boolean or float per-example metrics."""
    return {name: float(jnp.mean(values)) for name, values in metrics.items()}
