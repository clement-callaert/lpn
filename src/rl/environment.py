"""Finite-horizon stop/continue environment for latent SGD search.

One episode is one leave-one-out query prediction. Query labels are stored for
terminal reward only and are never part of the policy observation.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from src.models.lpn import LPN
from src.search.adaptive_sgd_search import (
    decode_query,
    make_clip_sgd_optimizer,
    one_sgd_step,
    score_latent,
)
from src.search.compute_accounting import (
    ComputeCounters,
    counters_for_default_sgd_search,
    zero_compute_counters,
)
from src.search.metrics import leave_one_out_metrics

ACTION_STOP: int = 0
ACTION_CONTINUE: int = 1


@dataclass
class EpisodeState:
    """Immutable env carry for one leave-one-out search episode."""

    current_latent: jnp.ndarray
    best_latent: jnp.ndarray
    opt_state: Any
    current_score: jnp.ndarray
    previous_score: jnp.ndarray
    best_score: jnp.ndarray
    gradient_norm: jnp.ndarray
    latent_norm: jnp.ndarray
    latent_update_norm: jnp.ndarray
    current_step: jnp.ndarray
    max_steps: jnp.ndarray
    remaining_budget: jnp.ndarray
    terminated: jnp.ndarray
    counters: ComputeCounters
    # Support sequences needed for continue transitions.
    input_seq: jnp.ndarray
    output_seq: jnp.ndarray
    query_input: jnp.ndarray
    query_input_shape: jnp.ndarray
    # Query labels for terminal reward only. Never expose in observations.
    query_output: jnp.ndarray
    query_output_shape: jnp.ndarray
    best_index: jnp.ndarray
    output_grids: jnp.ndarray
    output_shapes: jnp.ndarray
    last_reward: jnp.ndarray
    search_steps_executed: jnp.ndarray


def _latent_norm(latent: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(jnp.square(latent))).astype(jnp.float32)


def compute_terminal_reward(
    query_exact_match: jnp.ndarray | float,
    search_steps_executed: jnp.ndarray | int,
    compute_penalty: float,
) -> jnp.ndarray:
    """Terminal reward = exact match minus compute penalty times steps."""
    if compute_penalty < 0:
        raise ValueError(f"compute_penalty must be non-negative, got {compute_penalty}")
    match_value = jnp.asarray(query_exact_match, dtype=jnp.float32)
    steps_value = jnp.asarray(search_steps_executed, dtype=jnp.float32)
    reward = match_value - jnp.asarray(compute_penalty, dtype=jnp.float32) * steps_value
    if not jnp.isfinite(reward):
        raise ValueError(f"non-finite terminal reward: {reward}")
    return reward.astype(jnp.float32)


def _empty_output_placeholders(query_input: jnp.ndarray, query_input_shape: jnp.ndarray):
    return (
        jnp.zeros_like(query_input),
        jnp.zeros_like(query_input_shape),
    )


def reset_episode(
    model: LPN,
    params: Any,
    pairs: jnp.ndarray,
    grid_shapes: jnp.ndarray,
    query_input: jnp.ndarray,
    query_input_shape: jnp.ndarray,
    query_output: jnp.ndarray,
    query_output_shape: jnp.ndarray,
    key: jnp.ndarray,
    *,
    max_steps: int,
    lr: float = 0.1,
) -> EpisodeState:
    """Reset one episode from a leave-one-out task and a PRNG key."""
    if max_steps < 0:
        raise ValueError(f"max_steps must be non-negative, got {max_steps}")
    if key is None:
        raise ValueError("key is required for variational pattern checkpoints")

    # Match generate_output: split before variational sampling.
    unused_key, key_latents = jax.random.split(key)

    def encode_body(module: LPN):
        latents_mu, latents_logvar = module.encoder(pairs, grid_shapes, dropout_eval=True)
        if latents_logvar is not None:
            sampled, *_ = module._sample_latents(latents_mu, latents_logvar, key_latents)
            return sampled
        return latents_mu

    latents = model.apply({"params": params}, method=encode_body)
    candidate = LPN._prepare_latents_before_search(True, False, latents, None, unused_key)
    input_seq, output_seq = LPN._flatten_input_output_for_decoding(pairs, grid_shapes)
    z0 = candidate[0]
    optimizer = make_clip_sgd_optimizer(lr)
    opt_state0 = optimizer.init(z0)
    init_score = score_latent(model, params, z0, input_seq, output_seq).astype(jnp.float32)
    empty_grids, empty_shapes = _empty_output_placeholders(query_input, query_input_shape)
    max_steps_arr = jnp.asarray(max_steps, dtype=jnp.int32)
    return EpisodeState(
        current_latent=z0,
        best_latent=z0,
        opt_state=opt_state0,
        current_score=init_score,
        previous_score=init_score,
        best_score=init_score,
        gradient_norm=jnp.asarray(0.0, dtype=jnp.float32),
        latent_norm=_latent_norm(z0),
        latent_update_norm=jnp.asarray(0.0, dtype=jnp.float32),
        current_step=jnp.asarray(0, dtype=jnp.int32),
        max_steps=max_steps_arr,
        remaining_budget=max_steps_arr,
        terminated=jnp.asarray(False),
        counters=zero_compute_counters(),
        input_seq=input_seq,
        output_seq=output_seq,
        query_input=query_input,
        query_input_shape=query_input_shape,
        query_output=query_output,
        query_output_shape=query_output_shape,
        best_index=jnp.asarray(0, dtype=jnp.int32),
        output_grids=empty_grids,
        output_shapes=empty_shapes,
        last_reward=jnp.asarray(0.0, dtype=jnp.float32),
        search_steps_executed=jnp.asarray(0, dtype=jnp.int32),
    )


def _decode_and_reward(
    model: LPN,
    params: Any,
    state: EpisodeState,
    *,
    compute_penalty: float,
    max_rows: int,
    max_cols: int,
) -> EpisodeState:
    output_grids, output_shapes = decode_query(
        model,
        params,
        state.best_latent,
        state.query_input,
        state.query_input_shape,
    )
    metrics = leave_one_out_metrics(
        output_grids,
        output_shapes,
        state.query_output,
        state.query_output_shape,
    )
    reward = compute_terminal_reward(
        metrics["accuracy"],
        state.search_steps_executed,
        compute_penalty,
    )
    counters = counters_for_default_sgd_search(
        candidates=1,
        steps_executed=int(state.search_steps_executed),
        max_rows=max_rows,
        max_cols=max_cols,
        return_two_best=False,
    )
    # Prefer live executed counters for adaptive early stop.
    counters = counters.replace(
        search_steps_executed=state.search_steps_executed.astype(jnp.int32),
        objective_evaluations=(state.search_steps_executed + 1).astype(jnp.int32),
        gradient_evaluations=state.search_steps_executed.astype(jnp.int32),
        decoder_support_evaluations=(state.search_steps_executed + 1).astype(jnp.int32),
        candidate_step_exposure=(state.search_steps_executed + 1).astype(jnp.int32),
        output_decoding_calls=jnp.asarray(1, dtype=jnp.int32),
        autoregressive_token_predictions=jnp.asarray(
            2 + max_rows * max_cols, dtype=jnp.int32
        ),
        latent_candidates_evaluated=jnp.asarray(1, dtype=jnp.int32),
    )
    return state.replace(
        terminated=jnp.asarray(True),
        output_grids=output_grids,
        output_shapes=output_shapes,
        last_reward=reward,
        counters=counters,
        remaining_budget=jnp.asarray(0, dtype=jnp.int32),
    )


def step_episode(
    model: LPN,
    params: Any,
    state: EpisodeState,
    action: int | jnp.ndarray,
    *,
    lr: float = 0.1,
    compute_penalty: float = 0.0,
) -> tuple[EpisodeState, jnp.ndarray, jnp.ndarray]:
    """Apply one stop/continue action.

    Returns:
        new_state, reward, terminated
    """
    if bool(state.terminated):
        raise ValueError("cannot transition after termination")
    action_int = int(action)
    if action_int not in (ACTION_STOP, ACTION_CONTINUE):
        raise ValueError(f"action must be 0 (stop) or 1 (continue), got {action_int}")

    max_rows = model.decoder.config.max_rows
    max_cols = model.decoder.config.max_cols
    optimizer = make_clip_sgd_optimizer(lr)

    if action_int == ACTION_STOP:
        new_state = _decode_and_reward(
            model,
            params,
            state,
            compute_penalty=compute_penalty,
            max_rows=max_rows,
            max_cols=max_cols,
        )
        return new_state, new_state.last_reward, new_state.terminated

    # Continue: one SGD step. Intermediate reward is zero.
    # The compute penalty is applied once in the terminal reward.
    step_result = one_sgd_step(
        model,
        params,
        state.current_latent,
        state.opt_state,
        state.input_seq,
        state.output_seq,
        optimizer,
    )
    new_latent = step_result["latent"]
    post_score = step_result["score_post"]
    improved = post_score > state.best_score
    best_latent = jnp.where(improved, new_latent, state.best_latent)
    best_score = jnp.where(improved, post_score, state.best_score)
    new_step = state.current_step + 1
    new_steps_executed = state.search_steps_executed + 1
    best_index = jnp.where(improved, new_step, state.best_index)
    new_state = state.replace(
        current_latent=new_latent,
        best_latent=best_latent,
        opt_state=step_result["opt_state"],
        previous_score=state.current_score,
        current_score=post_score,
        best_score=best_score,
        gradient_norm=step_result["gradient_norm"],
        latent_norm=_latent_norm(new_latent),
        latent_update_norm=step_result["latent_update_norm"],
        current_step=new_step.astype(jnp.int32),
        remaining_budget=(state.max_steps - new_step).astype(jnp.int32),
        search_steps_executed=new_steps_executed.astype(jnp.int32),
        best_index=best_index.astype(jnp.int32),
        last_reward=jnp.asarray(0.0, dtype=jnp.float32),
    )

    # Force terminate at the horizon.
    if int(new_state.current_step) >= int(new_state.max_steps):
        new_state = _decode_and_reward(
            model,
            params,
            new_state,
            compute_penalty=compute_penalty,
            max_rows=max_rows,
            max_cols=max_cols,
        )
        return new_state, new_state.last_reward, new_state.terminated

    return new_state, new_state.last_reward, new_state.terminated


def run_fixed_continue_policy(
    model: LPN,
    params: Any,
    pairs: jnp.ndarray,
    grid_shapes: jnp.ndarray,
    query_input: jnp.ndarray,
    query_input_shape: jnp.ndarray,
    query_output: jnp.ndarray,
    query_output_shape: jnp.ndarray,
    key: jnp.ndarray,
    *,
    num_steps: int,
    lr: float = 0.1,
    compute_penalty: float = 0.0,
) -> EpisodeState:
    """Always continue for num_steps, then stop. Used for fixed-K parity."""
    state = reset_episode(
        model,
        params,
        pairs,
        grid_shapes,
        query_input,
        query_input_shape,
        query_output,
        query_output_shape,
        key,
        max_steps=num_steps,
        lr=lr,
    )
    for _ in range(num_steps):
        state, _, terminated = step_episode(
            model, params, state, ACTION_CONTINUE, lr=lr, compute_penalty=compute_penalty
        )
        if bool(terminated):
            return state
    state, _, _ = step_episode(
        model, params, state, ACTION_STOP, lr=lr, compute_penalty=compute_penalty
    )
    return state
