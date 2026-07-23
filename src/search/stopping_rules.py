"""Pure non-RL stopping rules for closed-loop latent search."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class StoppingObservation:
    """Information available to a stopping rule at test time.

    Query ground truth is intentionally absent. The caller owns updates to
    best_score and steps_since_improvement before invoking a rule.
    """

    current_step: jnp.ndarray
    max_steps: jnp.ndarray
    current_score: jnp.ndarray
    best_score: jnp.ndarray
    previous_score: jnp.ndarray
    steps_since_improvement: jnp.ndarray
    gradient_norm: jnp.ndarray
    remaining_budget: jnp.ndarray


@dataclass
class StoppingDecision:
    """Output of one stopping-rule evaluation."""

    should_stop: jnp.ndarray
    stop_reason_code: jnp.ndarray


# Stable integer codes for host-side logging.
REASON_CONTINUE = 0
REASON_MAX_STEPS = 1
REASON_FIXED = 2
REASON_PATIENCE = 3
REASON_MIN_IMPROVEMENT = 4
REASON_GRADIENT_NORM = 5
REASON_NON_FINITE = 6

REASON_NAMES: dict[int, str] = {
    REASON_CONTINUE: "continue",
    REASON_MAX_STEPS: "max_steps",
    REASON_FIXED: "fixed",
    REASON_PATIENCE: "patience",
    REASON_MIN_IMPROVEMENT: "min_improvement",
    REASON_GRADIENT_NORM: "gradient_norm",
    REASON_NON_FINITE: "non_finite",
}


class StoppingRuleConfig(NamedTuple):
    """Host-side configuration for a named stopping rule."""

    name: str
    max_steps: int
    patience: int | None = None
    min_improvement: float | None = None
    gradient_norm_threshold: float | None = None


def validate_stopping_rule_config(config: StoppingRuleConfig) -> None:
    """Validate host-side stopping-rule arguments."""
    if config.max_steps < 0:
        raise ValueError(f"max_steps must be non-negative, got {config.max_steps}")
    if config.name not in {
        "fixed",
        "patience",
        "min_improvement",
        "gradient_norm",
        "patience_with_max_budget",
    }:
        raise ValueError(f"unsupported stopping rule: {config.name}")

    if config.name in {"patience", "patience_with_max_budget"}:
        if config.patience is None:
            raise ValueError(f"patience is required for rule {config.name}")
        if config.patience < 0:
            raise ValueError(f"patience must be non-negative, got {config.patience}")
    elif config.patience is not None:
        raise ValueError(f"patience is only valid for patience-based rules, got rule={config.name}")

    if config.name == "min_improvement":
        if config.min_improvement is None:
            raise ValueError("min_improvement is required for rule min_improvement")
        if not jnp.isfinite(config.min_improvement):
            raise ValueError(f"min_improvement must be finite, got {config.min_improvement}")
    elif config.min_improvement is not None:
        raise ValueError(
            f"min_improvement is only valid for min_improvement rule, got rule={config.name}"
        )

    if config.name == "gradient_norm":
        if config.gradient_norm_threshold is None:
            raise ValueError("gradient_norm_threshold is required for rule gradient_norm")
        if config.gradient_norm_threshold < 0:
            raise ValueError(
                "gradient_norm_threshold must be non-negative, "
                f"got {config.gradient_norm_threshold}"
            )
        if not jnp.isfinite(config.gradient_norm_threshold):
            raise ValueError(
                f"gradient_norm_threshold must be finite, got {config.gradient_norm_threshold}"
            )
    elif config.gradient_norm_threshold is not None:
        raise ValueError(
            "gradient_norm_threshold is only valid for gradient_norm rule, "
            f"got rule={config.name}"
        )


def _non_finite_score_or_grad(observation: StoppingObservation) -> jnp.ndarray:
    # +inf gradient_norm is allowed as a "not computed yet" sentinel.
    score_bad = jnp.logical_not(jnp.isfinite(observation.current_score))
    grad = observation.gradient_norm
    grad_bad = jnp.logical_or(jnp.isnan(grad), jnp.isneginf(grad))
    return jnp.logical_or(score_bad, grad_bad)


def _at_max_steps(observation: StoppingObservation) -> jnp.ndarray:
    return observation.current_step >= observation.max_steps


def _finish_decision(
    observation: StoppingObservation,
    rule_stop: jnp.ndarray,
    rule_reason: int,
) -> StoppingDecision:
    non_finite = _non_finite_score_or_grad(observation)
    at_max = _at_max_steps(observation)
    should_stop = jnp.logical_or(non_finite, jnp.logical_or(at_max, rule_stop))
    reason = jnp.where(
        non_finite,
        jnp.asarray(REASON_NON_FINITE, dtype=jnp.int32),
        jnp.where(
            at_max,
            jnp.asarray(REASON_MAX_STEPS if rule_reason != REASON_FIXED else REASON_FIXED, dtype=jnp.int32),
            jnp.where(
                rule_stop,
                jnp.asarray(rule_reason, dtype=jnp.int32),
                jnp.asarray(REASON_CONTINUE, dtype=jnp.int32),
            ),
        ),
    )
    return StoppingDecision(should_stop=should_stop, stop_reason_code=reason)


def apply_fixed_rule(observation: StoppingObservation) -> StoppingDecision:
    """Stop only when the fixed step budget is reached."""
    return _finish_decision(
        observation,
        rule_stop=jnp.asarray(False),
        rule_reason=REASON_FIXED,
    )


def apply_patience_rule(observation: StoppingObservation, patience: int) -> StoppingDecision:
    """Stop after patience non-improving steps, or at max_steps.

    The caller must set steps_since_improvement. Setting the initial best score
    does not count as a non-improving step.
    """
    if patience < 0:
        raise ValueError(f"patience must be non-negative, got {patience}")
    patience_hit = observation.steps_since_improvement >= patience
    # Patience 0 stops before any search step once the init score is set.
    return _finish_decision(observation, rule_stop=patience_hit, rule_reason=REASON_PATIENCE)


def apply_min_improvement_rule(
    observation: StoppingObservation, min_improvement: float
) -> StoppingDecision:
    """Stop when the score gain is strictly below the threshold.

    Equality counts as no improvement. The rule is inactive at step 0 so the
    initial score alone does not trigger an immediate stop.
    """
    if not jnp.isfinite(min_improvement):
        raise ValueError(f"min_improvement must be finite, got {min_improvement}")
    gain = observation.current_score - observation.previous_score
    below_threshold = jnp.logical_and(observation.current_step > 0, gain < min_improvement)
    return _finish_decision(
        observation, rule_stop=below_threshold, rule_reason=REASON_MIN_IMPROVEMENT
    )


def apply_gradient_norm_rule(
    observation: StoppingObservation, gradient_norm_threshold: float
) -> StoppingDecision:
    """Stop when the gradient global norm is at or below the threshold.

    Exact equality at the threshold stops. The rule is inactive at step 0
    before any gradient has been computed for an update decision.
    """
    if gradient_norm_threshold < 0:
        raise ValueError(
            f"gradient_norm_threshold must be non-negative, got {gradient_norm_threshold}"
        )
    if not jnp.isfinite(gradient_norm_threshold):
        raise ValueError(
            f"gradient_norm_threshold must be finite, got {gradient_norm_threshold}"
        )
    below_or_equal = jnp.logical_and(
        observation.current_step > 0,
        observation.gradient_norm <= gradient_norm_threshold,
    )
    return _finish_decision(
        observation, rule_stop=below_or_equal, rule_reason=REASON_GRADIENT_NORM
    )


def apply_patience_with_max_budget_rule(
    observation: StoppingObservation, patience: int
) -> StoppingDecision:
    """Patience rule that always respects max_steps as a hard budget."""
    return apply_patience_rule(observation, patience=patience)


def apply_stopping_rule(
    config: StoppingRuleConfig, observation: StoppingObservation
) -> StoppingDecision:
    """Dispatch to a named pure stopping rule."""
    validate_stopping_rule_config(config)
    if config.name == "fixed":
        return apply_fixed_rule(observation)
    if config.name == "patience":
        assert config.patience is not None
        return apply_patience_rule(observation, patience=config.patience)
    if config.name == "min_improvement":
        assert config.min_improvement is not None
        return apply_min_improvement_rule(observation, min_improvement=config.min_improvement)
    if config.name == "gradient_norm":
        assert config.gradient_norm_threshold is not None
        return apply_gradient_norm_rule(
            observation, gradient_norm_threshold=config.gradient_norm_threshold
        )
    if config.name == "patience_with_max_budget":
        assert config.patience is not None
        return apply_patience_with_max_budget_rule(observation, patience=config.patience)
    raise ValueError(f"unsupported stopping rule: {config.name}")


def update_improvement_tracker(
    current_score: jnp.ndarray,
    best_score: jnp.ndarray,
    steps_since_improvement: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Update best score and patience counter after observing a new score."""
    improved = current_score > best_score
    new_best = jnp.where(improved, current_score, best_score)
    new_steps = jnp.where(
        improved,
        jnp.zeros_like(steps_since_improvement),
        steps_since_improvement + 1,
    )
    return new_best, new_steps
