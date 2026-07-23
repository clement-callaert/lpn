"""Unit tests for observation helpers."""

from __future__ import annotations

import unittest

import jax.numpy as jnp

from src.rl.observations import (
    FEATURE_NAMES,
    OBSERVATION_DIM,
    ObservationNormStats,
    assert_no_query_labels,
    build_norm_stats_from_raw_features,
    observation_dict_to_vector,
)
from src.rl.trajectory_dataset import TrajectoryStepRecord, compute_norm_stats_from_records


class TestObservations(unittest.TestCase):
    def test_feature_order_and_dimension_stable(self):
        self.assertEqual(OBSERVATION_DIM, 8)
        self.assertEqual(len(FEATURE_NAMES), 8)
        self.assertEqual(FEATURE_NAMES[0], "normalized_current_step")
        self.assertEqual(FEATURE_NAMES[-1], "latent_update_norm")

    def test_query_labels_rejected(self):
        with self.assertRaises(ValueError):
            assert_no_query_labels({"query_exact_match_if_stopped_now": jnp.asarray(1.0)})

    def test_normalization_uses_training_statistics_only(self):
        train_records = [
            TrajectoryStepRecord(
                task_index=0,
                procedural_task_seed=1000,
                leave_one_out_index=0,
                checkpoint_hash="abc",
                latent_sampling_seed=0,
                current_step=0,
                maximum_horizon=5,
                current_support_score=2.0,
                previous_support_score=2.0,
                score_improvement=0.0,
                best_support_score=2.0,
                best_score_improvement=0.0,
                gradient_norm=1.0,
                latent_norm=0.5,
                latent_update_norm=0.1,
                remaining_budget=5,
                query_exact_match_if_stopped_now=0.0,
                query_pixel_correctness_if_stopped_now=0.0,
                selected_best_iterate_index=0,
                counters={},
            ),
            TrajectoryStepRecord(
                task_index=1,
                procedural_task_seed=1000,
                leave_one_out_index=1,
                checkpoint_hash="abc",
                latent_sampling_seed=0,
                current_step=1,
                maximum_horizon=5,
                current_support_score=4.0,
                previous_support_score=2.0,
                score_improvement=2.0,
                best_support_score=4.0,
                best_score_improvement=2.0,
                gradient_norm=3.0,
                latent_norm=1.5,
                latent_update_norm=0.3,
                remaining_budget=4,
                query_exact_match_if_stopped_now=1.0,
                query_pixel_correctness_if_stopped_now=1.0,
                selected_best_iterate_index=1,
                counters={},
            ),
        ]
        stats = compute_norm_stats_from_records(train_records, source_split="train")
        self.assertEqual(stats.source_split, "train")
        self.assertAlmostEqual(stats.score_mean, 3.0)
        # Validation values must not mutate stats.
        frozen = ObservationNormStats.from_dict(stats.to_dict())
        self.assertEqual(frozen.score_mean, stats.score_mean)

    def test_zero_variance_replaced_safely(self):
        stats = build_norm_stats_from_raw_features(
            support_scores=[1.0, 1.0],
            gradient_norms=[0.0, 0.0],
            latent_norms=[2.0, 2.0],
            latent_update_norms=[0.5, 0.5],
            source_split="train",
        )
        self.assertIn("current_support_score", stats.zero_variance_replaced)
        self.assertEqual(stats.score_std, 1.0)

    def test_vector_packing_rejects_extra_keys(self):
        obs = {name: jnp.asarray(0.0, dtype=jnp.float32) for name in FEATURE_NAMES}
        obs["extra"] = jnp.asarray(1.0)
        with self.assertRaises(ValueError):
            observation_dict_to_vector(obs)


if __name__ == "__main__":
    unittest.main()
