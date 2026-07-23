"""Parity tests for closed-loop fixed-K trajectories vs upstream search."""

from __future__ import annotations

import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import omegaconf

from src.evaluate_checkpoint import instantiate_model, instantiate_train_state, load_model_weights
from src.search.adaptive_sgd_search import (
    run_closed_loop_fixed_k_with_trajectory,
    run_fixed_sgd_search_like_upstream,
)


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


class TestClosedLoopTrajectoryParity(unittest.TestCase):
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

    def test_closed_loop_matches_upstream_outputs_for_k_0_1_5(self):
        key = jax.random.PRNGKey(0)
        pairs = jax.random.randint(key, (3, 4, 4, 2), 0, 10)
        shapes = jnp.ones((3, 2, 2), dtype=jnp.int32) * 2
        query = pairs[0, :, :, 0]
        query_shape = shapes[0, :, 0]
        key_eval = jax.random.PRNGKey(1)
        atol = 1e-3
        rtol = 1e-3

        for num_steps in (0, 1, 5):
            with self.subTest(num_steps=num_steps):
                upstream = run_fixed_sgd_search_like_upstream(
                    self.model,
                    self.params,
                    pairs,
                    shapes,
                    query,
                    query_shape,
                    key_eval,
                    num_steps=num_steps,
                    lr=0.1,
                )
                closed = run_closed_loop_fixed_k_with_trajectory(
                    self.model,
                    self.params,
                    pairs,
                    shapes,
                    query,
                    query_shape,
                    key_eval,
                    num_steps=num_steps,
                    lr=0.1,
                )
                self.assertTrue(
                    jnp.array_equal(upstream["output_grids"], closed["output_grids"])
                )
                self.assertTrue(
                    jnp.array_equal(upstream["output_shapes"], closed["output_shapes"])
                )
                ctx_diff = float(jnp.max(jnp.abs(upstream["context"] - closed["context"])))
                self.assertLessEqual(ctx_diff, atol)
                self.assertTrue(
                    jnp.allclose(
                        upstream["context"], closed["context"], atol=atol, rtol=rtol
                    )
                )
                self.assertEqual(int(closed["search_steps_executed"]), num_steps)
                self.assertEqual(closed["latent_trajectory"].shape[0], num_steps + 1)
                self.assertEqual(closed["score_trajectory"].shape[0], num_steps + 1)


if __name__ == "__main__":
    unittest.main()
