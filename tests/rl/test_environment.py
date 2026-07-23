"""Unit tests for the stop/continue RL environment."""

from __future__ import annotations

import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import omegaconf

from src.evaluate_checkpoint import instantiate_model, instantiate_train_state, load_model_weights
from src.rl.environment import (
    ACTION_CONTINUE,
    ACTION_STOP,
    compute_terminal_reward,
    reset_episode,
    run_fixed_continue_policy,
    step_episode,
)
from src.rl.observations import (
    FEATURE_NAMES,
    ObservationNormStats,
    assert_no_query_labels,
    observation_dict_to_vector,
    observation_from_episode_state,
)
from src.search.adaptive_sgd_search import run_fixed_sgd_search_like_upstream
from src.search.compute_accounting import counters_for_default_sgd_search
from src.search.metrics import leave_one_out_metrics


def _find_config() -> Path:
    for path in (
        Path("outputs/2026-07-22/22-55-12/.hydra/config.yaml"),
        Path("src/configs/pattern_2d.yaml"),
    ):
        if path.exists():
            return path
    raise FileNotFoundError("No pattern_2d config found")


def _find_checkpoint() -> Path | None:
    path = Path("artifacts/checkpoints/pattern_2d_seed_0.msgpack")
    return path if path.exists() else None


def _identity_norm_stats() -> ObservationNormStats:
    return ObservationNormStats(
        score_mean=0.0,
        score_std=1.0,
        grad_norm_mean=0.0,
        grad_norm_std=1.0,
        latent_norm_mean=0.0,
        latent_norm_std=1.0,
        latent_update_norm_mean=0.0,
        latent_update_norm_std=1.0,
        source_split="unit_test",
        zero_variance_replaced=(),
    )


class TestEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = _find_checkpoint()
        if checkpoint is None:
            raise unittest.SkipTest("pattern_2d_seed_0 checkpoint not available")
        cfg = omegaconf.OmegaConf.load(_find_config())
        if "training" not in cfg or cfg.training is None:
            raise unittest.SkipTest("config missing training section")
        cls.model = instantiate_model(cfg, mixed_precision=False)
        state = instantiate_train_state(cls.model)
        cls.state = load_model_weights(state, str(checkpoint.parent), checkpoint.name)
        cls.params = cls.state.params

    def _make_task(self, seed: int = 0):
        key = jax.random.PRNGKey(seed)
        pairs = jax.random.randint(key, (3, 4, 4, 2), 0, 10)
        shapes = jnp.ones((3, 2, 2), dtype=jnp.int32) * 2
        query_input = pairs[0, :, :, 0]
        query_input_shape = shapes[0, :, 0]
        query_output = pairs[0, :, :, 1]
        query_output_shape = shapes[0, :, 1]
        return pairs, shapes, query_input, query_input_shape, query_output, query_output_shape

    def test_deterministic_reset(self):
        task = self._make_task(1)
        key = jax.random.PRNGKey(7)
        a = reset_episode(self.model, self.params, *task, key, max_steps=5)
        b = reset_episode(self.model, self.params, *task, key, max_steps=5)
        self.assertTrue(jnp.array_equal(a.current_latent, b.current_latent))
        self.assertEqual(float(a.current_score), float(b.current_score))

    def test_stop_at_step_zero(self):
        task = self._make_task(2)
        state = reset_episode(
            self.model, self.params, *task, jax.random.PRNGKey(3), max_steps=5
        )
        state, reward, done = step_episode(
            self.model, self.params, state, ACTION_STOP, compute_penalty=0.01
        )
        self.assertTrue(bool(done))
        self.assertEqual(int(state.search_steps_executed), 0)
        self.assertTrue(jnp.isfinite(reward))
        self.assertEqual(int(state.counters.output_decoding_calls), 1)
        # With zero steps, reward equals exact match.
        metrics = leave_one_out_metrics(
            state.output_grids,
            state.output_shapes,
            state.query_output,
            state.query_output_shape,
        )
        self.assertAlmostEqual(float(reward), float(metrics["accuracy"]) - 0.0)

    def test_one_continue_transition(self):
        task = self._make_task(3)
        state = reset_episode(
            self.model, self.params, *task, jax.random.PRNGKey(4), max_steps=5
        )
        z0 = state.current_latent
        state, reward, done = step_episode(
            self.model, self.params, state, ACTION_CONTINUE, compute_penalty=0.0
        )
        self.assertFalse(bool(done))
        self.assertEqual(float(reward), 0.0)
        self.assertEqual(int(state.current_step), 1)
        self.assertFalse(bool(jnp.array_equal(z0, state.current_latent)))

    def test_forced_termination_at_max_steps(self):
        task = self._make_task(4)
        state = reset_episode(
            self.model, self.params, *task, jax.random.PRNGKey(5), max_steps=2
        )
        state, _, done = step_episode(self.model, self.params, state, ACTION_CONTINUE)
        self.assertFalse(bool(done))
        state, _, done = step_episode(self.model, self.params, state, ACTION_CONTINUE)
        self.assertTrue(bool(done))
        self.assertEqual(int(state.search_steps_executed), 2)

    def test_query_labels_absent_from_observations(self):
        task = self._make_task(5)
        state = reset_episode(
            self.model, self.params, *task, jax.random.PRNGKey(6), max_steps=5
        )
        obs = observation_from_episode_state(state, _identity_norm_stats())
        assert_no_query_labels(obs)
        self.assertEqual(tuple(obs.keys()), FEATURE_NAMES)
        vector = observation_dict_to_vector(obs)
        self.assertEqual(vector.shape, (len(FEATURE_NAMES),))
        self.assertTrue(bool(jnp.all(jnp.isfinite(vector))))

    def test_no_transition_after_termination(self):
        task = self._make_task(6)
        state = reset_episode(
            self.model, self.params, *task, jax.random.PRNGKey(8), max_steps=5
        )
        state, _, _ = step_episode(self.model, self.params, state, ACTION_STOP)
        with self.assertRaises(ValueError):
            step_episode(self.model, self.params, state, ACTION_CONTINUE)

    def test_reward_equals_exact_match_minus_step_cost(self):
        reward = compute_terminal_reward(1.0, 5, 0.02)
        self.assertAlmostEqual(float(reward), 1.0 - 0.1)
        with self.assertRaises(ValueError):
            compute_terminal_reward(float("nan"), 1, 0.01)

    def test_fixed_continue_matches_fixed_k_output(self):
        task = self._make_task(9)
        pairs, shapes, q_in, q_in_s, q_out, q_out_s = task
        key = jax.random.PRNGKey(11)
        for k in (0, 1, 5):
            with self.subTest(k=k):
                env_state = run_fixed_continue_policy(
                    self.model,
                    self.params,
                    pairs,
                    shapes,
                    q_in,
                    q_in_s,
                    q_out,
                    q_out_s,
                    key,
                    num_steps=k,
                )
                upstream = run_fixed_sgd_search_like_upstream(
                    self.model,
                    self.params,
                    pairs,
                    shapes,
                    q_in,
                    q_in_s,
                    key,
                    num_steps=k,
                )
                self.assertTrue(
                    jnp.array_equal(env_state.output_grids, upstream["output_grids"])
                )
                self.assertTrue(
                    jnp.array_equal(env_state.output_shapes, upstream["output_shapes"])
                )

    def test_compute_counters_match_formula(self):
        task = self._make_task(10)
        state = run_fixed_continue_policy(
            self.model, self.params, *task, jax.random.PRNGKey(12), num_steps=3
        )
        expected = counters_for_default_sgd_search(
            candidates=1,
            steps_executed=3,
            max_rows=self.model.decoder.config.max_rows,
            max_cols=self.model.decoder.config.max_cols,
        )
        self.assertEqual(int(state.counters.search_steps_executed), 3)
        self.assertEqual(int(state.counters.gradient_evaluations), 3)
        self.assertEqual(int(state.counters.objective_evaluations), 4)
        self.assertEqual(
            int(state.counters.output_decoding_calls),
            int(expected.output_decoding_calls),
        )


if __name__ == "__main__":
    unittest.main()
