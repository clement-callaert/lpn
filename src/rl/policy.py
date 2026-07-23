"""Bernoulli stop/continue policy.

Action 0 = stop, action 1 = continue. The network outputs one stop logit.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from src.rl.observations import OBSERVATION_DIM


class StopContinuePolicy(nn.Module):
    """Linear policy mapping observation -> stop logit.

    Initialization uses a negative bias so the initial stop probability is below
    0.5 and the policy does not collapse to always-stop.
    """

    continue_bias: float = 1.5

    @nn.compact
    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        single = observation.ndim == 1
        if single:
            observation = observation[None, :]
        # Dense default init is lecun_normal for kernel; set bias toward continue.
        stop_logit = nn.Dense(
            features=1,
            bias_init=nn.initializers.constant(-self.continue_bias),
        )(observation)
        stop_logit = stop_logit.squeeze(-1)
        if single:
            stop_logit = stop_logit[0]
        return stop_logit


def init_policy_params(
    policy: StopContinuePolicy,
    rng_key: jnp.ndarray,
    observation_dim: int = OBSERVATION_DIM,
) -> Any:
    """Initialize policy parameters from a policy PRNG key."""
    dummy = jnp.zeros((observation_dim,), dtype=jnp.float32)
    return policy.init(rng_key, dummy)


def stop_probability(policy: StopContinuePolicy, params: Any, observation: jnp.ndarray) -> jnp.ndarray:
    """Return p(stop | observation)."""
    stop_logit = policy.apply(params, observation)
    return jax.nn.sigmoid(stop_logit)


def sample_action(
    policy: StopContinuePolicy,
    params: Any,
    observation: jnp.ndarray,
    rng_key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample a Bernoulli stop action and return (action, log_prob, p_stop)."""
    p_stop = stop_probability(policy, params, observation)
    # Clip for numerical stability of log.
    p_stop = jnp.clip(p_stop, 1e-6, 1.0 - 1e-6)
    action = jax.random.bernoulli(rng_key, p_stop).astype(jnp.int32)
    # action 1 means stop in Bernoulli(p_stop); map to env codes:
    # env ACTION_STOP=0, ACTION_CONTINUE=1.
    # Bernoulli True/1 => stop => env action 0.
    env_action = jnp.where(action == 1, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(1, dtype=jnp.int32))
    log_prob = jnp.where(action == 1, jnp.log(p_stop), jnp.log(1.0 - p_stop))
    return env_action, log_prob.astype(jnp.float32), p_stop.astype(jnp.float32)


def evaluate_action(
    policy: StopContinuePolicy,
    params: Any,
    observation: jnp.ndarray,
    threshold: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Deterministic eval: stop if p(stop) >= threshold."""
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    p_stop = stop_probability(policy, params, observation)
    env_action = jnp.where(
        p_stop >= threshold,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
    )
    return env_action, p_stop.astype(jnp.float32)


def log_probability_of_action(
    policy: StopContinuePolicy,
    params: Any,
    observation: jnp.ndarray,
    env_action: jnp.ndarray,
) -> jnp.ndarray:
    """Log probability of an environment action under the current policy."""
    p_stop = stop_probability(policy, params, observation)
    p_stop = jnp.clip(p_stop, 1e-6, 1.0 - 1e-6)
    # env 0 = stop, env 1 = continue
    return jnp.where(
        env_action == 0,
        jnp.log(p_stop),
        jnp.log(1.0 - p_stop),
    ).astype(jnp.float32)
