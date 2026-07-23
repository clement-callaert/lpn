"""Unit tests for the Bernoulli stop/continue policy."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp

from src.rl.observations import OBSERVATION_DIM
from src.rl.policy import (
    StopContinuePolicy,
    evaluate_action,
    init_policy_params,
    log_probability_of_action,
    sample_action,
    stop_probability,
)


class TestPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = StopContinuePolicy(continue_bias=1.5)
        self.params = init_policy_params(self.policy, jax.random.PRNGKey(0))
        self.observation = jnp.zeros((OBSERVATION_DIM,), dtype=jnp.float32)

    def test_probabilities_are_finite(self):
        p_stop = stop_probability(self.policy, self.params, self.observation)
        self.assertTrue(bool(jnp.isfinite(p_stop)))
        self.assertGreater(float(p_stop), 0.0)
        self.assertLess(float(p_stop), 1.0)

    def test_init_does_not_always_stop(self):
        p_stop = float(stop_probability(self.policy, self.params, self.observation))
        self.assertLess(p_stop, 0.5)

    def test_sampled_actions_are_valid(self):
        key = jax.random.PRNGKey(1)
        for i in range(20):
            key, sub = jax.random.split(key)
            action, log_prob, p_stop = sample_action(
                self.policy, self.params, self.observation, sub
            )
            self.assertIn(int(action), (0, 1))
            self.assertTrue(bool(jnp.isfinite(log_prob)))
            self.assertTrue(bool(jnp.isfinite(p_stop)))

    def test_log_probabilities_match_sampled_actions(self):
        key = jax.random.PRNGKey(2)
        action, log_prob, _ = sample_action(
            self.policy, self.params, self.observation, key
        )
        recomputed = log_probability_of_action(
            self.policy, self.params, self.observation, action
        )
        self.assertAlmostEqual(float(log_prob), float(recomputed), places=5)

    def test_different_policy_seeds_can_differ(self):
        params_a = init_policy_params(self.policy, jax.random.PRNGKey(10))
        params_b = init_policy_params(self.policy, jax.random.PRNGKey(11))
        # Kernel weights should differ for different seeds.
        w_a = params_a["params"]["Dense_0"]["kernel"]
        w_b = params_b["params"]["Dense_0"]["kernel"]
        self.assertFalse(bool(jnp.array_equal(w_a, w_b)))

    def test_deterministic_threshold_eval(self):
        action, p_stop = evaluate_action(
            self.policy, self.params, self.observation, threshold=0.5
        )
        expected = 0 if float(p_stop) >= 0.5 else 1
        self.assertEqual(int(action), expected)


if __name__ == "__main__":
    unittest.main()
