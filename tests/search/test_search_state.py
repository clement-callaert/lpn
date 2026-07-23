"""Unit tests for search-state helpers."""

from __future__ import annotations

import unittest

import jax.numpy as jnp

from src.search.search_state import (
    initial_search_state,
    rl_observation_from_search_state,
    search_state_to_host_dict,
)


class TestSearchState(unittest.TestCase):
    def test_initial_state(self):
        state = initial_search_state(
            max_steps=20,
            init_score=jnp.asarray(1.25, dtype=jnp.float32),
            init_latent_norm=jnp.asarray(0.5, dtype=jnp.float32),
        )
        self.assertEqual(int(state.current_step), 0)
        self.assertEqual(int(state.max_steps), 20)
        self.assertEqual(float(state.current_score), 1.25)
        self.assertEqual(float(state.best_score), 1.25)
        self.assertEqual(int(state.remaining_budget), 20)
        self.assertEqual(int(state.counters.search_steps_executed), 0)

    def test_rl_observation_fields(self):
        state = initial_search_state(
            max_steps=10,
            init_score=jnp.asarray(2.0, dtype=jnp.float32),
            init_latent_norm=jnp.asarray(1.0, dtype=jnp.float32),
        )
        observation = rl_observation_from_search_state(state)
        expected = {
            "normalized_current_step",
            "normalized_remaining_budget",
            "current_support_score",
            "score_improvement",
            "best_score_improvement",
            "gradient_norm",
            "latent_norm",
            "latent_update_norm",
        }
        self.assertEqual(set(observation), expected)
        self.assertEqual(float(observation["normalized_current_step"]), 0.0)
        self.assertEqual(float(observation["normalized_remaining_budget"]), 1.0)

    def test_host_dict(self):
        state = initial_search_state(
            max_steps=5,
            init_score=jnp.asarray(0.0, dtype=jnp.float32),
            init_latent_norm=jnp.asarray(0.0, dtype=jnp.float32),
        )
        payload = search_state_to_host_dict(state)
        self.assertIn("counters", payload)
        self.assertEqual(payload["max_steps"], 5)

    def test_rejects_negative_max_steps(self):
        with self.assertRaises(ValueError):
            initial_search_state(
                max_steps=-1,
                init_score=jnp.asarray(0.0),
                init_latent_norm=jnp.asarray(0.0),
            )


if __name__ == "__main__":
    unittest.main()
