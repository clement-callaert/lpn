"""Parity tests between official fixed-K search and the adaptive harness path."""

from __future__ import annotations

import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import omegaconf

from src.evaluate_checkpoint import instantiate_model, instantiate_train_state, load_model_weights
from src.search.adaptive_sgd_search import (
    run_adaptive_sgd_search,
    run_fixed_sgd_search_like_upstream,
)
from src.search.stopping_rules import StoppingRuleConfig


def _find_config() -> Path:
    candidates = [
        Path("outputs/2026-07-22/22-55-12/.hydra/config.yaml"),
        Path("src/configs/pattern_2d.yaml"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No pattern_2d config found for parity tests")


def _find_checkpoint() -> Path | None:
    path = Path("artifacts/checkpoints/pattern_2d_seed_0.msgpack")
    return path if path.exists() else None


class TestAdaptiveSgdParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = _find_checkpoint()
        if checkpoint is None:
            raise unittest.SkipTest("pattern_2d_seed_0 checkpoint not available")
        cfg = omegaconf.OmegaConf.load(_find_config())
        # Hydra interpolations may remain in the checked-in yaml; prefer resolved
        # training outputs when present.
        if "training" not in cfg or cfg.training is None:
            raise unittest.SkipTest("config missing training section")
        cls.model = instantiate_model(cfg, mixed_precision=False)
        state = instantiate_train_state(cls.model)
        cls.state = load_model_weights(state, str(checkpoint.parent), checkpoint.name)
        cls.params = cls.state.params

    def test_fixed_rule_delegates_to_upstream(self):
        key = jax.random.PRNGKey(0)
        pairs = jax.random.randint(key, (3, 4, 4, 2), 0, 10)
        shapes = jnp.ones((3, 2, 2), dtype=jnp.int32) * 2
        query = pairs[0, :, :, 0]
        query_shape = shapes[0, :, 0]
        key_eval = jax.random.PRNGKey(1)

        upstream = run_fixed_sgd_search_like_upstream(
            self.model,
            self.params,
            pairs,
            shapes,
            query,
            query_shape,
            key_eval,
            num_steps=2,
            lr=0.1,
        )
        adaptive = run_adaptive_sgd_search(
            self.model,
            self.params,
            pairs,
            shapes,
            query,
            query_shape,
            key_eval,
            rule_config=StoppingRuleConfig(name="fixed", max_steps=2),
            lr=0.1,
        )
        self.assertTrue(jnp.array_equal(upstream["output_grids"], adaptive["output_grids"]))
        self.assertTrue(jnp.array_equal(upstream["output_shapes"], adaptive["output_shapes"]))
        self.assertEqual(adaptive["search_steps_executed"], 2)

    def test_zero_step_matches_mean_context_decode_path(self):
        # GA with 0 steps still uses generate_output gradient_ascent mode.
        key = jax.random.PRNGKey(2)
        pairs = jax.random.randint(key, (3, 4, 4, 2), 0, 10)
        shapes = jnp.ones((3, 2, 2), dtype=jnp.int32) * 2
        query = pairs[0, :, :, 0]
        query_shape = shapes[0, :, 0]
        key_eval = jax.random.PRNGKey(3)
        result = run_adaptive_sgd_search(
            self.model,
            self.params,
            pairs,
            shapes,
            query,
            query_shape,
            key_eval,
            rule_config=StoppingRuleConfig(name="fixed", max_steps=0),
            lr=0.1,
        )
        self.assertEqual(result["search_steps_executed"], 0)
        self.assertEqual(int(result["counters"].objective_evaluations), 1)
        self.assertEqual(int(result["counters"].gradient_evaluations), 0)


if __name__ == "__main__":
    unittest.main()
