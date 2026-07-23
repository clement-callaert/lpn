"""Unit tests for offline trajectory records."""

from __future__ import annotations

import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import omegaconf

from src.evaluate_checkpoint import instantiate_model, instantiate_train_state, load_model_weights
from src.rl.trajectory_dataset import (
    earliest_correct_step,
    generate_fixed_horizon_trajectory,
)


def _find_config() -> Path:
    for path in (
        Path("outputs/2026-07-22/22-55-12/.hydra/config.yaml"),
        Path("src/configs/pattern_2d.yaml"),
    ):
        if path.exists():
            return path
    raise FileNotFoundError("No pattern_2d config found")


class TestTrajectoryDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = Path("artifacts/checkpoints/pattern_2d_seed_0.msgpack")
        if not checkpoint.exists():
            raise unittest.SkipTest("pattern_2d_seed_0 checkpoint not available")
        cfg = omegaconf.OmegaConf.load(_find_config())
        cls.model = instantiate_model(cfg, mixed_precision=False)
        state = instantiate_train_state(cls.model)
        cls.state = load_model_weights(state, str(checkpoint.parent), checkpoint.name)
        cls.params = cls.state.params

    def test_generate_trajectory_length_and_labels(self):
        key = jax.random.PRNGKey(0)
        pairs = jax.random.randint(key, (3, 4, 4, 2), 0, 10)
        shapes = jnp.ones((3, 2, 2), dtype=jnp.int32) * 2
        records = generate_fixed_horizon_trajectory(
            self.model,
            self.params,
            pairs,
            shapes,
            pairs[0, :, :, 0],
            shapes[0, :, 0],
            pairs[0, :, :, 1],
            shapes[0, :, 1],
            jax.random.PRNGKey(1),
            task_index=0,
            procedural_task_seed=1000,
            leave_one_out_index=0,
            checkpoint_hash="testhash",
            latent_sampling_seed=1,
            max_search_steps=5,
        )
        self.assertEqual(len(records), 6)
        for index, record in enumerate(records):
            self.assertEqual(record.current_step, index)
            self.assertIn(record.query_exact_match_if_stopped_now, (0.0, 1.0))
        oracle = earliest_correct_step(records)
        if oracle is not None:
            self.assertGreaterEqual(oracle, 0)
            self.assertLessEqual(oracle, 5)

    def test_same_seeds_reproduce_same_rollout(self):
        key = jax.random.PRNGKey(2)
        pairs = jax.random.randint(key, (3, 4, 4, 2), 0, 10)
        shapes = jnp.ones((3, 2, 2), dtype=jnp.int32) * 2
        kwargs = dict(
            model=self.model,
            params=self.params,
            pairs=pairs,
            grid_shapes=shapes,
            query_input=pairs[0, :, :, 0],
            query_input_shape=shapes[0, :, 0],
            query_output=pairs[0, :, :, 1],
            query_output_shape=shapes[0, :, 1],
            key=jax.random.PRNGKey(3),
            task_index=0,
            procedural_task_seed=1000,
            leave_one_out_index=0,
            checkpoint_hash="testhash",
            latent_sampling_seed=3,
            max_search_steps=2,
        )
        a = generate_fixed_horizon_trajectory(**kwargs)
        b = generate_fixed_horizon_trajectory(**kwargs)
        self.assertEqual(
            [r.current_support_score for r in a],
            [r.current_support_score for r in b],
        )


if __name__ == "__main__":
    unittest.main()
