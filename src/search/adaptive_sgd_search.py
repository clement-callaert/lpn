"""Closed-loop adaptive SGD latent search.

Mirrors the default Optax path in LPN._get_gradient_ascent_context without
editing that function. Fixed-step numerical control uses official
generate_output. Adaptive rules use a compiled while-loop with the same
optimizer and support objective.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax

from src.models.lpn import LPN
from src.search.compute_accounting import counters_for_default_sgd_search
from src.search.stopping_rules import (
    REASON_FIXED,
    REASON_GRADIENT_NORM,
    REASON_MAX_STEPS,
    REASON_MIN_IMPROVEMENT,
    REASON_NON_FINITE,
    REASON_PATIENCE,
    StoppingRuleConfig,
    validate_stopping_rule_config,
)


def _global_norm(tree: Any) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def make_clip_sgd_optimizer(lr: float) -> optax.GradientTransformation:
    """Build the default clip-then-SGD chain used by upstream GA."""
    return optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(learning_rate=lr))


def score_latent(
    model: LPN,
    params: Any,
    latent_vector: jnp.ndarray,
    input_seq: jnp.ndarray,
    output_seq: jnp.ndarray,
) -> jnp.ndarray:
    """Score one latent against support pairs with the frozen decoder."""
    repeated = jnp.broadcast_to(
        latent_vector[None, :],
        (output_seq.shape[-2], latent_vector.shape[-1]),
    )

    def body(module: LPN):
        row_logits, col_logits, grid_logits = module.decoder(
            input_seq, output_seq, repeated, dropout_eval=True
        )
        return module._compute_log_probs(row_logits, col_logits, grid_logits, output_seq)

    return model.apply({"params": params}, method=body)


def decode_query(
    model: LPN,
    params: Any,
    context: jnp.ndarray,
    query_input: jnp.ndarray,
    query_input_shape: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Greedy-decode one query from a latent context."""

    def body(module: LPN):
        grids, shapes = module._generate_output_from_context(
            context[None, ...],
            query_input[None, ...],
            query_input_shape[None, ...],
            dropout_eval=True,
        )
        return grids[0], shapes[0]

    return model.apply({"params": params}, method=body)


def one_sgd_step(
    model: LPN,
    params: Any,
    latent_vector: jnp.ndarray,
    opt_state: Any,
    input_seq: jnp.ndarray,
    output_seq: jnp.ndarray,
    optimizer: optax.GradientTransformation,
) -> dict[str, Any]:
    """Apply one clipped SGD latent update matching the adaptive search body.

    Returns the post-update latent, optimizer state, pre/post scores, gradient
    diagnostics, and the Optax update vector.
    """

    def objective(candidate: jnp.ndarray) -> jnp.ndarray:
        return score_latent(model, params, candidate, input_seq, output_seq)

    score_pre, grads = jax.value_and_grad(objective)(latent_vector)
    gradient_norm = _global_norm(grads).astype(jnp.float32)
    updates, new_opt_state = optimizer.update(-grads, opt_state, latent_vector)
    latent_update_norm = _global_norm(updates).astype(jnp.float32)
    new_latent = optax.apply_updates(latent_vector, updates)
    score_post = objective(new_latent).astype(jnp.float32)
    return {
        "latent": new_latent,
        "opt_state": new_opt_state,
        "score_pre": score_pre.astype(jnp.float32),
        "score_post": score_post,
        "grads": grads,
        "gradient_norm": gradient_norm,
        "latent_update_norm": latent_update_norm,
        "updates": updates,
    }


def make_closed_loop_fixed_k_trajectory_fn(
    model: LPN, *, num_steps: int, lr: float
) -> Callable[..., dict[str, Any]]:
    """Return a JIT fn that runs exactly K SGD steps and records a trajectory.

    This path does not short-circuit to generate_output. Final selected latent
    and decoded grids must match the upstream fixed-K control under matched
    keys.
    """
    if num_steps < 0:
        raise ValueError(f"num_steps must be non-negative, got {num_steps}")
    max_rows = model.decoder.config.max_rows
    max_cols = model.decoder.config.max_cols
    optimizer = make_clip_sgd_optimizer(lr)
    latent_dim = model.encoder.config.latent_dim

    @jax.jit
    def closed_loop_fixed_k(
        params: Any,
        pairs: jnp.ndarray,
        grid_shapes: jnp.ndarray,
        query_input: jnp.ndarray,
        query_input_shape: jnp.ndarray,
        key: jnp.ndarray,
    ) -> dict[str, Any]:
        # Match generate_output: split key before variational sampling.
        unused_key, key_latents = jax.random.split(key)

        def encode_body(module: LPN):
            latents_mu, latents_logvar = module.encoder(pairs, grid_shapes, dropout_eval=True)
            if latents_logvar is not None:
                sampled, *_ = module._sample_latents(latents_mu, latents_logvar, key_latents)
                return sampled
            return latents_mu

        latents = model.apply({"params": params}, method=encode_body)
        # Upstream also passes the leftover key into prepare; unused without perturbation.
        candidate = LPN._prepare_latents_before_search(True, False, latents, None, unused_key)
        input_seq, output_seq = LPN._flatten_input_output_for_decoding(pairs, grid_shapes)
        z0 = candidate[0]
        opt_state0 = optimizer.init(z0)
        init_score = score_latent(model, params, z0, input_seq, output_seq).astype(jnp.float32)

        latent_traj = jnp.zeros((num_steps + 1, z0.shape[-1]), dtype=z0.dtype)
        score_traj = jnp.zeros((num_steps + 1,), dtype=jnp.float32)
        grad_norm_traj = jnp.zeros((num_steps,), dtype=jnp.float32)
        update_norm_traj = jnp.zeros((num_steps,), dtype=jnp.float32)
        grad_traj = jnp.zeros((num_steps, z0.shape[-1]), dtype=z0.dtype)
        update_traj = jnp.zeros((num_steps, z0.shape[-1]), dtype=z0.dtype)
        latent_traj = latent_traj.at[0].set(z0)
        score_traj = score_traj.at[0].set(init_score)

        def scan_body(carry, _unused):
            step, z, opt_state, score_traj_local = carry
            step_result = one_sgd_step(
                model, params, z, opt_state, input_seq, output_seq, optimizer
            )
            new_z = step_result["latent"]
            new_opt = step_result["opt_state"]
            # Match adaptive scoring: store pre-update score at the current index.
            score_traj_local = score_traj_local.at[step].set(step_result["score_pre"])
            new_step = step + 1
            return (
                new_step,
                new_z,
                new_opt,
                score_traj_local,
            ), {
                "latent": new_z,
                "score_pre": step_result["score_pre"],
                "score_post": step_result["score_post"],
                "gradient_norm": step_result["gradient_norm"],
                "latent_update_norm": step_result["latent_update_norm"],
                "grads": step_result["grads"],
                "updates": step_result["updates"],
            }

        if num_steps == 0:
            steps_executed = jnp.asarray(0, dtype=jnp.int32)
            final_z = z0
            final_score = init_score
            best_index = jnp.asarray(0, dtype=jnp.int32)
            best_context = z0
            best_score = init_score
        else:
            (_final_step, final_z, _opt_state, score_traj), step_outputs = jax.lax.scan(
                scan_body,
                (jnp.asarray(0, dtype=jnp.int32), z0, opt_state0, score_traj),
                xs=None,
                length=num_steps,
            )
            steps_executed = jnp.asarray(num_steps, dtype=jnp.int32)
            latent_traj = latent_traj.at[1:].set(step_outputs["latent"])
            score_traj = score_traj.at[num_steps].set(step_outputs["score_post"][-1])
            grad_norm_traj = step_outputs["gradient_norm"]
            update_norm_traj = step_outputs["latent_update_norm"]
            grad_traj = step_outputs["grads"]
            update_traj = step_outputs["updates"]
            final_score = step_outputs["score_post"][-1]
            best_index = jnp.argmax(score_traj)
            best_context = latent_traj[best_index]
            best_score = score_traj[best_index]

        output_grids, output_shapes = decode_query(
            model, params, best_context, query_input, query_input_shape
        )
        counters = counters_for_default_sgd_search(
            candidates=1,
            steps_executed=num_steps,
            max_rows=max_rows,
            max_cols=max_cols,
            return_two_best=False,
        )
        return {
            "output_grids": output_grids,
            "output_shapes": output_shapes,
            "context": best_context,
            "search_steps_executed": steps_executed,
            "stop_reason_code": jnp.asarray(REASON_FIXED, dtype=jnp.int32),
            "counters": counters,
            "extra_objective_evaluations": jnp.asarray(num_steps, dtype=jnp.int32),
            "best_support_score": best_score.astype(jnp.float32),
            "final_support_score": final_score.astype(jnp.float32),
            "gradient_norm": (
                grad_norm_traj[-1]
                if num_steps > 0
                else jnp.asarray(0.0, dtype=jnp.float32)
            ),
            "initial_latent": z0,
            "initial_support_score": init_score,
            "latent_trajectory": latent_traj,
            "score_trajectory": score_traj,
            "gradient_trajectory": grad_traj,
            "gradient_norm_trajectory": grad_norm_traj,
            "update_trajectory": update_traj,
            "latent_update_norm_trajectory": update_norm_traj,
            "best_latent_index": best_index.astype(jnp.int32),
            "final_latent": final_z,
            "latent_dim": jnp.asarray(latent_dim, dtype=jnp.int32),
        }

    return closed_loop_fixed_k


def run_closed_loop_fixed_k_with_trajectory(
    model: LPN,
    params: Any,
    pairs: jnp.ndarray,
    grid_shapes: jnp.ndarray,
    query_input: jnp.ndarray,
    query_input_shape: jnp.ndarray,
    key: jnp.ndarray | None,
    num_steps: int,
    lr: float = 0.1,
) -> dict[str, Any]:
    """Run closed-loop fixed-K search and return per-step trajectory fields."""
    if key is None:
        raise ValueError("key is required for variational pattern checkpoints")
    search_fn = make_closed_loop_fixed_k_trajectory_fn(model, num_steps=num_steps, lr=lr)
    return search_fn(params, pairs, grid_shapes, query_input, query_input_shape, key)


def make_fixed_search_fn(
    model: LPN, *, num_steps: int, lr: float
) -> Callable[..., dict[str, Any]]:
    """Return a JIT-compiled fixed-K search using official generate_output."""
    if num_steps < 0:
        raise ValueError(f"num_steps must be non-negative, got {num_steps}")
    max_rows = model.decoder.config.max_rows
    max_cols = model.decoder.config.max_cols

    @jax.jit
    def fixed_search(
        params: Any,
        pairs: jnp.ndarray,
        grid_shapes: jnp.ndarray,
        query_input: jnp.ndarray,
        query_input_shape: jnp.ndarray,
        key: jnp.ndarray,
    ) -> dict[str, Any]:
        output_grids, output_shapes, info = model.apply(
            {"params": params},
            pairs[None, ...],
            grid_shapes[None, ...],
            query_input[None, ...],
            query_input_shape[None, ...],
            key,
            True,
            "gradient_ascent",
            False,
            method=model.generate_output,
            num_steps=num_steps,
            lr=lr,
        )
        counters = counters_for_default_sgd_search(
            candidates=1,
            steps_executed=num_steps,
            max_rows=max_rows,
            max_cols=max_cols,
            return_two_best=False,
        )
        return {
            "output_grids": output_grids[0],
            "output_shapes": output_shapes[0],
            "context": info["context"][0],
            "search_steps_executed": jnp.asarray(num_steps, dtype=jnp.int32),
            "stop_reason_code": jnp.asarray(REASON_FIXED, dtype=jnp.int32),
            "counters": counters,
            "extra_objective_evaluations": jnp.asarray(0, dtype=jnp.int32),
            "best_support_score": jnp.asarray(0.0, dtype=jnp.float32),
            "final_support_score": jnp.asarray(0.0, dtype=jnp.float32),
            "gradient_norm": jnp.asarray(0.0, dtype=jnp.float32),
        }

    return fixed_search


def make_adaptive_search_fn(
    model: LPN, rule_config: StoppingRuleConfig, *, lr: float
) -> Callable[..., dict[str, Any]]:
    """Return a JIT-compiled closed-loop search for one leave-one-out example."""
    validate_stopping_rule_config(rule_config)
    if rule_config.name == "fixed":
        return make_fixed_search_fn(model, num_steps=rule_config.max_steps, lr=lr)

    max_steps = int(rule_config.max_steps)
    patience = 0 if rule_config.patience is None else int(rule_config.patience)
    min_improvement = (
        0.0 if rule_config.min_improvement is None else float(rule_config.min_improvement)
    )
    grad_threshold = (
        0.0
        if rule_config.gradient_norm_threshold is None
        else float(rule_config.gradient_norm_threshold)
    )
    rule_name = rule_config.name
    max_rows = model.decoder.config.max_rows
    max_cols = model.decoder.config.max_cols
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(learning_rate=lr))

    def score_latent(params, latent_vector, input_seq, output_seq):
        repeated = jnp.broadcast_to(
            latent_vector[None, :],
            (output_seq.shape[-2], latent_vector.shape[-1]),
        )

        def body(module: LPN):
            row_logits, col_logits, grid_logits = module.decoder(
                input_seq, output_seq, repeated, dropout_eval=True
            )
            return module._compute_log_probs(row_logits, col_logits, grid_logits, output_seq)

        return model.apply({"params": params}, method=body)

    def encode(params, pairs, grid_shapes, key):
        def body(module: LPN):
            latents_mu, latents_logvar = module.encoder(pairs, grid_shapes, dropout_eval=True)
            if latents_logvar is not None:
                sampled, *_ = module._sample_latents(latents_mu, latents_logvar, key)
                return sampled
            return latents_mu

        return model.apply({"params": params}, method=body)

    def decode(params, context, query_input, query_input_shape):
        def body(module: LPN):
            grids, shapes = module._generate_output_from_context(
                context[None, ...],
                query_input[None, ...],
                query_input_shape[None, ...],
                dropout_eval=True,
            )
            return grids[0], shapes[0]

        return model.apply({"params": params}, method=body)

    def should_stop(
        step, current_score, previous_score, steps_since_improvement, gradient_norm
    ):
        score_bad = jnp.logical_not(jnp.isfinite(current_score))
        grad_bad = jnp.logical_or(jnp.isnan(gradient_norm), jnp.isneginf(gradient_norm))
        non_finite = jnp.logical_or(score_bad, grad_bad)
        at_max = step >= max_steps

        if rule_name in {"patience", "patience_with_max_budget"}:
            rule_hit = steps_since_improvement >= patience
            rule_reason = REASON_PATIENCE
        elif rule_name == "min_improvement":
            gain = current_score - previous_score
            rule_hit = jnp.logical_and(step > 0, gain < min_improvement)
            rule_reason = REASON_MIN_IMPROVEMENT
        elif rule_name == "gradient_norm":
            rule_hit = jnp.logical_and(step > 0, gradient_norm <= grad_threshold)
            rule_reason = REASON_GRADIENT_NORM
        else:
            raise ValueError(f"unsupported adaptive rule: {rule_name}")

        should = jnp.logical_or(non_finite, jnp.logical_or(at_max, rule_hit))
        reason_code = jnp.where(
            non_finite,
            jnp.asarray(REASON_NON_FINITE, dtype=jnp.int32),
            jnp.where(
                at_max,
                jnp.asarray(REASON_MAX_STEPS, dtype=jnp.int32),
                jnp.where(
                    rule_hit,
                    jnp.asarray(rule_reason, dtype=jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                ),
            ),
        )
        return should, reason_code

    @jax.jit
    def adaptive_search(
        params: Any,
        pairs: jnp.ndarray,
        grid_shapes: jnp.ndarray,
        query_input: jnp.ndarray,
        query_input_shape: jnp.ndarray,
        key: jnp.ndarray,
    ) -> dict[str, Any]:
        latents = encode(params, pairs, grid_shapes, key)
        candidate = LPN._prepare_latents_before_search(True, False, latents, None, None)
        input_seq, output_seq = LPN._flatten_input_output_for_decoding(pairs, grid_shapes)
        z0 = candidate[0]
        opt_state0 = optimizer.init(z0)
        init_score = score_latent(params, z0, input_seq, output_seq).astype(jnp.float32)

        latent_traj = jnp.zeros((max_steps + 1, z0.shape[-1]), dtype=z0.dtype)
        score_traj = jnp.zeros((max_steps + 1,), dtype=jnp.float32)
        latent_traj = latent_traj.at[0].set(z0)
        score_traj = score_traj.at[0].set(init_score)

        def cond_fn(carry):
            step = carry[0]
            current_score = carry[3]
            previous_score = carry[5]
            steps_since = carry[6]
            grad_norm = carry[7]
            should, _ = should_stop(
                step, current_score, previous_score, steps_since, grad_norm
            )
            return jnp.logical_and(step < max_steps, jnp.logical_not(should))

        def body_fn(carry):
            (
                step,
                z,
                opt_state,
                current_score,
                best_score,
                previous_score,
                steps_since_improvement,
                _gradient_norm,
                extra_obj,
                latent_traj,
                score_traj,
            ) = carry

            def objective(latent_vector):
                return score_latent(params, latent_vector, input_seq, output_seq)

            score_pre, grads = jax.value_and_grad(objective)(z)
            new_grad_norm = _global_norm(grads).astype(jnp.float32)
            updates, new_opt_state = optimizer.update(-grads, opt_state, z)
            new_z = optax.apply_updates(z, updates)
            post_score = objective(new_z).astype(jnp.float32)
            new_step = step + 1

            improved = post_score > best_score
            new_best = jnp.where(improved, post_score, best_score)
            new_steps_since = jnp.where(
                improved,
                jnp.asarray(0, dtype=jnp.int32),
                steps_since_improvement + 1,
            )
            new_latent_traj = latent_traj.at[new_step].set(new_z)
            new_score_traj = score_traj.at[step].set(score_pre.astype(jnp.float32))

            return (
                new_step,
                new_z,
                new_opt_state,
                post_score,
                new_best,
                current_score,
                new_steps_since,
                new_grad_norm,
                extra_obj + 1,
                new_latent_traj,
                new_score_traj,
            )

        init_carry = (
            jnp.asarray(0, dtype=jnp.int32),
            z0,
            opt_state0,
            init_score,
            init_score,
            init_score,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(jnp.inf, dtype=jnp.float32),
            jnp.asarray(1, dtype=jnp.int32),
            latent_traj,
            score_traj,
        )
        final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        (
            steps_executed,
            _final_z,
            _opt_state,
            current_score,
            _best_score,
            previous_score,
            steps_since,
            gradient_norm,
            extra_obj,
            latent_traj,
            score_traj,
        ) = final_carry

        _should, reason_code = should_stop(
            steps_executed, current_score, previous_score, steps_since, gradient_norm
        )
        score_traj = score_traj.at[steps_executed].set(current_score)

        indices = jnp.arange(max_steps + 1)
        valid = indices <= steps_executed
        masked_scores = jnp.where(valid, score_traj, jnp.asarray(-1e30, dtype=jnp.float32))
        best_index = jnp.argmax(masked_scores)
        best_context = latent_traj[best_index]
        output_grids, output_shapes = decode(
            params, best_context, query_input, query_input_shape
        )

        counters = counters_for_default_sgd_search(
            candidates=1,
            steps_executed=0,
            max_rows=max_rows,
            max_cols=max_cols,
            return_two_best=False,
        ).replace(
            search_steps_executed=steps_executed.astype(jnp.int32),
            objective_evaluations=(steps_executed + 1).astype(jnp.int32),
            gradient_evaluations=steps_executed.astype(jnp.int32),
            decoder_support_evaluations=(steps_executed + 1).astype(jnp.int32),
            candidate_step_exposure=(steps_executed + 1).astype(jnp.int32),
        )

        return {
            "output_grids": output_grids,
            "output_shapes": output_shapes,
            "context": best_context,
            "search_steps_executed": steps_executed.astype(jnp.int32),
            "stop_reason_code": reason_code,
            "counters": counters,
            "extra_objective_evaluations": extra_obj.astype(jnp.int32),
            "best_support_score": jnp.max(masked_scores),
            "final_support_score": current_score,
            "gradient_norm": gradient_norm,
        }

    return adaptive_search


def make_latent_conditioned_adaptive_search_fn(
    model: LPN, rule_config: StoppingRuleConfig, *, lr: float
) -> Callable[..., dict[str, Any]]:
    """JIT search that starts from a provided support latent tensor.

    Used after a Trainer-matched batched encode so variational noise matches
    the fixed-K control path.
    """
    validate_stopping_rule_config(rule_config)
    if rule_config.name == "fixed":
        raise ValueError("latent-conditioned search is for adaptive rules only")

    max_steps = int(rule_config.max_steps)
    patience = 0 if rule_config.patience is None else int(rule_config.patience)
    min_improvement = (
        0.0 if rule_config.min_improvement is None else float(rule_config.min_improvement)
    )
    grad_threshold = (
        0.0
        if rule_config.gradient_norm_threshold is None
        else float(rule_config.gradient_norm_threshold)
    )
    rule_name = rule_config.name
    max_rows = model.decoder.config.max_rows
    max_cols = model.decoder.config.max_cols
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(learning_rate=lr))

    def score_latent(params, latent_vector, input_seq, output_seq):
        repeated = jnp.broadcast_to(
            latent_vector[None, :],
            (output_seq.shape[-2], latent_vector.shape[-1]),
        )

        def body(module: LPN):
            row_logits, col_logits, grid_logits = module.decoder(
                input_seq, output_seq, repeated, dropout_eval=True
            )
            return module._compute_log_probs(row_logits, col_logits, grid_logits, output_seq)

        return model.apply({"params": params}, method=body)

    def decode(params, context, query_input, query_input_shape):
        def body(module: LPN):
            grids, shapes = module._generate_output_from_context(
                context[None, ...],
                query_input[None, ...],
                query_input_shape[None, ...],
                dropout_eval=True,
            )
            return grids[0], shapes[0]

        return model.apply({"params": params}, method=body)

    def should_stop(step, current_score, previous_score, steps_since_improvement, gradient_norm):
        score_bad = jnp.logical_not(jnp.isfinite(current_score))
        grad_bad = jnp.logical_or(jnp.isnan(gradient_norm), jnp.isneginf(gradient_norm))
        non_finite = jnp.logical_or(score_bad, grad_bad)
        at_max = step >= max_steps
        if rule_name in {"patience", "patience_with_max_budget"}:
            rule_hit = steps_since_improvement >= patience
            rule_reason = REASON_PATIENCE
        elif rule_name == "min_improvement":
            gain = current_score - previous_score
            rule_hit = jnp.logical_and(step > 0, gain < min_improvement)
            rule_reason = REASON_MIN_IMPROVEMENT
        elif rule_name == "gradient_norm":
            rule_hit = jnp.logical_and(step > 0, gradient_norm <= grad_threshold)
            rule_reason = REASON_GRADIENT_NORM
        else:
            raise ValueError(f"unsupported adaptive rule: {rule_name}")
        should = jnp.logical_or(non_finite, jnp.logical_or(at_max, rule_hit))
        reason_code = jnp.where(
            non_finite,
            jnp.asarray(REASON_NON_FINITE, dtype=jnp.int32),
            jnp.where(
                at_max,
                jnp.asarray(REASON_MAX_STEPS, dtype=jnp.int32),
                jnp.where(
                    rule_hit,
                    jnp.asarray(rule_reason, dtype=jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                ),
            ),
        )
        return should, reason_code

    @jax.jit
    def adaptive_from_latents(
        params: Any,
        support_latents: jnp.ndarray,
        pairs: jnp.ndarray,
        grid_shapes: jnp.ndarray,
        query_input: jnp.ndarray,
        query_input_shape: jnp.ndarray,
    ) -> dict[str, Any]:
        candidate = LPN._prepare_latents_before_search(True, False, support_latents, None, None)
        input_seq, output_seq = LPN._flatten_input_output_for_decoding(pairs, grid_shapes)
        z0 = candidate[0]
        opt_state0 = optimizer.init(z0)
        init_score = score_latent(params, z0, input_seq, output_seq).astype(jnp.float32)

        latent_traj = jnp.zeros((max_steps + 1, z0.shape[-1]), dtype=z0.dtype)
        score_traj = jnp.zeros((max_steps + 1,), dtype=jnp.float32)
        latent_traj = latent_traj.at[0].set(z0)
        score_traj = score_traj.at[0].set(init_score)

        def cond_fn(carry):
            step = carry[0]
            current_score = carry[3]
            previous_score = carry[5]
            steps_since = carry[6]
            grad_norm = carry[7]
            should, _ = should_stop(
                step, current_score, previous_score, steps_since, grad_norm
            )
            return jnp.logical_and(step < max_steps, jnp.logical_not(should))

        def body_fn(carry):
            (
                step,
                z,
                opt_state,
                current_score,
                best_score,
                previous_score,
                steps_since_improvement,
                _gradient_norm,
                extra_obj,
                latent_traj,
                score_traj,
            ) = carry

            def objective(latent_vector):
                return score_latent(params, latent_vector, input_seq, output_seq)

            score_pre, grads = jax.value_and_grad(objective)(z)
            new_grad_norm = _global_norm(grads).astype(jnp.float32)
            updates, new_opt_state = optimizer.update(-grads, opt_state, z)
            new_z = optax.apply_updates(z, updates)
            post_score = objective(new_z).astype(jnp.float32)
            new_step = step + 1
            improved = post_score > best_score
            new_best = jnp.where(improved, post_score, best_score)
            new_steps_since = jnp.where(
                improved,
                jnp.asarray(0, dtype=jnp.int32),
                steps_since_improvement + 1,
            )
            return (
                new_step,
                new_z,
                new_opt_state,
                post_score,
                new_best,
                current_score,
                new_steps_since,
                new_grad_norm,
                extra_obj + 1,
                latent_traj.at[new_step].set(new_z),
                score_traj.at[step].set(score_pre.astype(jnp.float32)),
            )

        init_carry = (
            jnp.asarray(0, dtype=jnp.int32),
            z0,
            opt_state0,
            init_score,
            init_score,
            init_score,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(jnp.inf, dtype=jnp.float32),
            jnp.asarray(1, dtype=jnp.int32),
            latent_traj,
            score_traj,
        )
        final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        (
            steps_executed,
            _final_z,
            _opt_state,
            current_score,
            _best_score,
            previous_score,
            steps_since,
            gradient_norm,
            extra_obj,
            latent_traj,
            score_traj,
        ) = final_carry
        _should, reason_code = should_stop(
            steps_executed, current_score, previous_score, steps_since, gradient_norm
        )
        score_traj = score_traj.at[steps_executed].set(current_score)
        indices = jnp.arange(max_steps + 1)
        valid = indices <= steps_executed
        masked_scores = jnp.where(valid, score_traj, jnp.asarray(-1e30, dtype=jnp.float32))
        best_context = latent_traj[jnp.argmax(masked_scores)]
        output_grids, output_shapes = decode(
            params, best_context, query_input, query_input_shape
        )
        counters = counters_for_default_sgd_search(
            candidates=1,
            steps_executed=0,
            max_rows=max_rows,
            max_cols=max_cols,
            return_two_best=False,
        ).replace(
            search_steps_executed=steps_executed.astype(jnp.int32),
            objective_evaluations=(steps_executed + 1).astype(jnp.int32),
            gradient_evaluations=steps_executed.astype(jnp.int32),
            decoder_support_evaluations=(steps_executed + 1).astype(jnp.int32),
            candidate_step_exposure=(steps_executed + 1).astype(jnp.int32),
        )
        return {
            "output_grids": output_grids,
            "output_shapes": output_shapes,
            "context": best_context,
            "search_steps_executed": steps_executed.astype(jnp.int32),
            "stop_reason_code": reason_code,
            "counters": counters,
            "extra_objective_evaluations": extra_obj.astype(jnp.int32),
            "best_support_score": jnp.max(masked_scores),
            "final_support_score": current_score,
            "gradient_norm": gradient_norm,
        }

    return adaptive_from_latents


def encode_leave_one_out_latents(
    model: LPN,
    params: Any,
    leave_one_out_pairs: jnp.ndarray,
    leave_one_out_shapes: jnp.ndarray,
    key: jnp.ndarray,
) -> jnp.ndarray:
    """Encode a full leave-one-out batch with one shared PRNG key."""

    def body(module: LPN):
        latents_mu, latents_logvar = module.encoder(
            leave_one_out_pairs, leave_one_out_shapes, dropout_eval=True
        )
        if latents_logvar is not None:
            sampled, *_ = module._sample_latents(latents_mu, latents_logvar, key)
            return sampled
        return latents_mu

    return model.apply({"params": params}, method=body)


def run_fixed_sgd_search_like_upstream(
    model: LPN,
    params: Any,
    pairs: jnp.ndarray,
    grid_shapes: jnp.ndarray,
    query_input: jnp.ndarray,
    query_input_shape: jnp.ndarray,
    key: jnp.ndarray | None,
    num_steps: int,
    lr: float = 0.1,
) -> dict[str, Any]:
    """Fixed-K search via official generate_output for numerical control."""
    if key is None:
        raise ValueError("key is required for variational pattern checkpoints")
    search_fn = make_fixed_search_fn(model, num_steps=num_steps, lr=lr)
    return search_fn(params, pairs, grid_shapes, query_input, query_input_shape, key)


def run_adaptive_sgd_search(
    model: LPN,
    params: Any,
    pairs: jnp.ndarray,
    grid_shapes: jnp.ndarray,
    query_input: jnp.ndarray,
    query_input_shape: jnp.ndarray,
    key: jnp.ndarray | None,
    rule_config: StoppingRuleConfig,
    lr: float = 0.1,
) -> dict[str, Any]:
    """Run search for one leave-one-out problem."""
    if key is None:
        raise ValueError("key is required for variational pattern checkpoints")
    search_fn = make_adaptive_search_fn(model, rule_config, lr=lr)
    return search_fn(params, pairs, grid_shapes, query_input, query_input_shape, key)
