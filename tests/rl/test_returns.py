"""Unit tests for REINFORCE helpers."""

from __future__ import annotations

import copy
import unittest

import jax
import jax.numpy as jnp
import optax

from src.rl.observations import OBSERVATION_DIM
from src.rl.policy import StopContinuePolicy, init_policy_params
from src.rl.reinforce import (
    MovingAverageBaseline,
    apply_policy_update,
    compute_episode_return,
    make_adam_optimizer,
    reinforce_loss,
)


class TestReturns(unittest.TestCase):
    def test_return_calculation(self):
        episode_return = compute_episode_return([0.0, 0.0, 0.9])
        self.assertAlmostEqual(float(episode_return), 0.9)

    def test_non_finite_rewards_fail(self):
        with self.assertRaises(ValueError):
            compute_episode_return([0.0, float("nan")])

    def test_baseline_subtraction_preserves_shapes(self):
        baseline = MovingAverageBaseline()
        value = baseline.update(1.0)
        advantage = 1.0 - value
        self.assertIsInstance(advantage, float)

    def test_policy_update_changes_parameters(self):
        policy = StopContinuePolicy()
        params = init_policy_params(policy, jax.random.PRNGKey(0))
        optimizer = make_adam_optimizer(1e-2)
        opt_state = optimizer.init(params)
        # Non-zero observations so the Dense kernel receives a gradient.
        observations = jnp.ones((3, OBSERVATION_DIM), dtype=jnp.float32)
        actions = jnp.asarray([1, 1, 0], dtype=jnp.int32)
        new_params, _, metrics = apply_policy_update(
            policy,
            params,
            opt_state,
            optimizer,
            observations,
            actions,
            advantage=1.0,
            entropy_coef=0.01,
        )
        self.assertTrue(jnp.isfinite(metrics["loss"]))
        old_w = params["params"]["Dense_0"]["kernel"]
        new_w = new_params["params"]["Dense_0"]["kernel"]
        self.assertFalse(bool(jnp.array_equal(old_w, new_w)))

    def test_zero_learning_rate_preserves_parameters(self):
        policy = StopContinuePolicy()
        params = init_policy_params(policy, jax.random.PRNGKey(1))
        optimizer = make_adam_optimizer(0.0)
        opt_state = optimizer.init(params)
        observations = jnp.ones((2, OBSERVATION_DIM), dtype=jnp.float32)
        actions = jnp.asarray([0, 1], dtype=jnp.int32)
        new_params, _, _ = apply_policy_update(
            policy,
            params,
            opt_state,
            optimizer,
            observations,
            actions,
            advantage=0.5,
        )
        old_w = params["params"]["Dense_0"]["kernel"]
        new_w = new_params["params"]["Dense_0"]["kernel"]
        self.assertTrue(bool(jnp.allclose(old_w, new_w)))

    def test_reinforce_loss_shapes(self):
        policy = StopContinuePolicy()
        params = init_policy_params(policy, jax.random.PRNGKey(2))
        observations = jnp.zeros((4, OBSERVATION_DIM), dtype=jnp.float32)
        actions = jnp.asarray([1, 1, 1, 0], dtype=jnp.int32)
        loss, metrics = reinforce_loss(
            policy, params, observations, actions, advantage=jnp.asarray(0.25)
        )
        self.assertEqual(loss.shape, ())
        self.assertEqual(metrics["advantage"].shape, ())


if __name__ == "__main__":
    unittest.main()
