"""Unit tests for pure stopping rules."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp

from src.search.stopping_rules import (
    REASON_CONTINUE,
    REASON_FIXED,
    REASON_GRADIENT_NORM,
    REASON_MAX_STEPS,
    REASON_MIN_IMPROVEMENT,
    REASON_NON_FINITE,
    REASON_PATIENCE,
    StoppingObservation,
    StoppingRuleConfig,
    apply_fixed_rule,
    apply_gradient_norm_rule,
    apply_min_improvement_rule,
    apply_patience_rule,
    apply_stopping_rule,
    update_improvement_tracker,
    validate_stopping_rule_config,
)


def _obs(
    *,
    step: int,
    max_steps: int,
    current_score: float,
    best_score: float,
    previous_score: float,
    steps_since_improvement: int,
    gradient_norm: float,
) -> StoppingObservation:
    return StoppingObservation(
        current_step=jnp.asarray(step, dtype=jnp.int32),
        max_steps=jnp.asarray(max_steps, dtype=jnp.int32),
        current_score=jnp.asarray(current_score, dtype=jnp.float32),
        best_score=jnp.asarray(best_score, dtype=jnp.float32),
        previous_score=jnp.asarray(previous_score, dtype=jnp.float32),
        steps_since_improvement=jnp.asarray(steps_since_improvement, dtype=jnp.int32),
        gradient_norm=jnp.asarray(gradient_norm, dtype=jnp.float32),
        remaining_budget=jnp.asarray(max_steps - step, dtype=jnp.int32),
    )


class TestStoppingRules(unittest.TestCase):
    def test_fixed_stops_at_exact_step(self):
        decision = apply_fixed_rule(
            _obs(
                step=5,
                max_steps=5,
                current_score=1.0,
                best_score=1.0,
                previous_score=0.9,
                steps_since_improvement=0,
                gradient_norm=1.0,
            )
        )
        self.assertTrue(bool(decision.should_stop))
        self.assertEqual(int(decision.stop_reason_code), REASON_FIXED)

    def test_fixed_continues_before_budget(self):
        decision = apply_fixed_rule(
            _obs(
                step=4,
                max_steps=5,
                current_score=1.0,
                best_score=1.0,
                previous_score=0.9,
                steps_since_improvement=0,
                gradient_norm=1.0,
            )
        )
        self.assertFalse(bool(decision.should_stop))
        self.assertEqual(int(decision.stop_reason_code), REASON_CONTINUE)

    def test_all_rules_stop_at_max_steps(self):
        configs = [
            StoppingRuleConfig(name="fixed", max_steps=3),
            StoppingRuleConfig(name="patience", max_steps=3, patience=10),
            StoppingRuleConfig(name="min_improvement", max_steps=3, min_improvement=0.01),
            StoppingRuleConfig(name="gradient_norm", max_steps=3, gradient_norm_threshold=1e-6),
            StoppingRuleConfig(name="patience_with_max_budget", max_steps=3, patience=10),
        ]
        observation = _obs(
            step=3,
            max_steps=3,
            current_score=2.0,
            best_score=1.0,
            previous_score=1.5,
            steps_since_improvement=0,
            gradient_norm=10.0,
        )
        for config in configs:
            decision = apply_stopping_rule(config, observation)
            self.assertTrue(bool(decision.should_stop), msg=config.name)

    def test_patience_resets_after_improvement(self):
        best, steps = update_improvement_tracker(
            current_score=jnp.asarray(1.5),
            best_score=jnp.asarray(1.0),
            steps_since_improvement=jnp.asarray(4),
        )
        self.assertEqual(float(best), 1.5)
        self.assertEqual(int(steps), 0)

    def test_patience_stops_after_configured_non_improving_steps(self):
        decision = apply_patience_rule(
            _obs(
                step=2,
                max_steps=20,
                current_score=1.0,
                best_score=1.2,
                previous_score=1.1,
                steps_since_improvement=2,
                gradient_norm=1.0,
            ),
            patience=2,
        )
        self.assertTrue(bool(decision.should_stop))
        self.assertEqual(int(decision.stop_reason_code), REASON_PATIENCE)

    def test_patience_zero_stops_immediately(self):
        decision = apply_patience_rule(
            _obs(
                step=0,
                max_steps=20,
                current_score=1.0,
                best_score=1.0,
                previous_score=1.0,
                steps_since_improvement=0,
                gradient_norm=jnp.inf,
            ),
            patience=0,
        )
        self.assertTrue(bool(decision.should_stop))
        self.assertEqual(int(decision.stop_reason_code), REASON_PATIENCE)

    def test_improvement_threshold_equality_is_no_improvement(self):
        decision = apply_min_improvement_rule(
            _obs(
                step=1,
                max_steps=20,
                current_score=1.0,
                best_score=1.0,
                previous_score=1.0,
                steps_since_improvement=0,
                gradient_norm=1.0,
            ),
            min_improvement=0.0,
        )
        # gain == 0 and threshold == 0 => gain < threshold is False; continue.
        self.assertFalse(bool(decision.should_stop))

        decision_strict = apply_min_improvement_rule(
            _obs(
                step=1,
                max_steps=20,
                current_score=1.0,
                best_score=1.0,
                previous_score=1.0,
                steps_since_improvement=0,
                gradient_norm=1.0,
            ),
            min_improvement=0.1,
        )
        self.assertTrue(bool(decision_strict.should_stop))
        self.assertEqual(int(decision_strict.stop_reason_code), REASON_MIN_IMPROVEMENT)

    def test_gradient_rule_stops_below_threshold(self):
        decision = apply_gradient_norm_rule(
            _obs(
                step=1,
                max_steps=20,
                current_score=1.0,
                best_score=1.0,
                previous_score=0.9,
                steps_since_improvement=0,
                gradient_norm=1e-5,
            ),
            gradient_norm_threshold=1e-4,
        )
        self.assertTrue(bool(decision.should_stop))
        self.assertEqual(int(decision.stop_reason_code), REASON_GRADIENT_NORM)

    def test_gradient_rule_stops_at_exact_threshold(self):
        decision = apply_gradient_norm_rule(
            _obs(
                step=1,
                max_steps=20,
                current_score=1.0,
                best_score=1.0,
                previous_score=0.9,
                steps_since_improvement=0,
                gradient_norm=1e-4,
            ),
            gradient_norm_threshold=1e-4,
        )
        self.assertTrue(bool(decision.should_stop))

    def test_gradient_rule_does_not_stop_above_threshold(self):
        decision = apply_gradient_norm_rule(
            _obs(
                step=1,
                max_steps=20,
                current_score=1.0,
                best_score=1.0,
                previous_score=0.9,
                steps_since_improvement=0,
                gradient_norm=1e-3,
            ),
            gradient_norm_threshold=1e-4,
        )
        self.assertFalse(bool(decision.should_stop))

    def test_nan_score_safe_stop(self):
        decision = apply_fixed_rule(
            _obs(
                step=1,
                max_steps=20,
                current_score=float("nan"),
                best_score=1.0,
                previous_score=1.0,
                steps_since_improvement=0,
                gradient_norm=1.0,
            )
        )
        self.assertTrue(bool(decision.should_stop))
        self.assertEqual(int(decision.stop_reason_code), REASON_NON_FINITE)

    def test_zero_step_budget_stops_immediately(self):
        decision = apply_fixed_rule(
            _obs(
                step=0,
                max_steps=0,
                current_score=1.0,
                best_score=1.0,
                previous_score=1.0,
                steps_since_improvement=0,
                gradient_norm=jnp.inf,
            )
        )
        self.assertTrue(bool(decision.should_stop))
        self.assertEqual(int(decision.stop_reason_code), REASON_FIXED)

    def test_results_are_deterministic(self):
        observation = _obs(
            step=2,
            max_steps=5,
            current_score=1.0,
            best_score=1.2,
            previous_score=1.1,
            steps_since_improvement=2,
            gradient_norm=0.5,
        )
        config = StoppingRuleConfig(name="patience", max_steps=5, patience=2)

        @jax.jit
        def once(obs):
            return apply_stopping_rule(config, obs)

        first = once(observation)
        second = once(observation)
        self.assertEqual(bool(first.should_stop), bool(second.should_stop))
        self.assertEqual(int(first.stop_reason_code), int(second.stop_reason_code))

    def test_validate_rejects_incompatible_args(self):
        with self.assertRaises(ValueError):
            validate_stopping_rule_config(
                StoppingRuleConfig(name="fixed", max_steps=5, patience=1)
            )
        with self.assertRaises(ValueError):
            validate_stopping_rule_config(
                StoppingRuleConfig(name="patience", max_steps=-1, patience=1)
            )


if __name__ == "__main__":
    unittest.main()
