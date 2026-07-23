"""Unit tests for logical compute counters."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp

from src.search.compute_accounting import (
    COUNTER_FIELD_NAMES,
    add_compute_counters,
    compute_counters_to_dict,
    counters_for_default_sgd_search,
    validate_compute_counters_host,
    zero_compute_counters,
)


class TestComputeAccounting(unittest.TestCase):
    def test_zero_counters(self):
        counters = zero_compute_counters()
        values = compute_counters_to_dict(counters)
        self.assertEqual(set(values), set(COUNTER_FIELD_NAMES))
        for name, value in values.items():
            self.assertEqual(value, 0, msg=name)

    def test_counter_addition(self):
        left = counters_for_default_sgd_search(
            candidates=1, steps_executed=5, max_rows=4, max_cols=4
        )
        right = counters_for_default_sgd_search(
            candidates=1, steps_executed=2, max_rows=4, max_cols=4
        )
        total = add_compute_counters(left, right)
        values = compute_counters_to_dict(total)
        self.assertEqual(values["search_steps_executed"], 7)
        self.assertEqual(values["gradient_evaluations"], 7)
        self.assertEqual(values["objective_evaluations"], 9)
        self.assertEqual(values["candidate_step_exposure"], 9)

    def test_negative_counters_rejected(self):
        counters = zero_compute_counters()
        counters = counters.replace(search_steps_executed=jnp.asarray(-1, dtype=jnp.int32))
        with self.assertRaises(ValueError):
            validate_compute_counters_host(counters)

    def test_dictionary_serialization_stable_names(self):
        counters = counters_for_default_sgd_search(
            candidates=2, steps_executed=3, max_rows=4, max_cols=4, return_two_best=True
        )
        values = compute_counters_to_dict(counters)
        self.assertEqual(list(values.keys()), list(COUNTER_FIELD_NAMES))
        self.assertEqual(values["latent_candidates_evaluated"], 2)
        self.assertEqual(values["gradient_evaluations"], 6)
        self.assertEqual(values["objective_evaluations"], 8)
        self.assertEqual(values["output_decoding_calls"], 2)
        self.assertEqual(values["autoregressive_token_predictions"], 2 * (2 + 16))

    def test_zero_steps_formula(self):
        counters = counters_for_default_sgd_search(
            candidates=1, steps_executed=0, max_rows=4, max_cols=4
        )
        values = compute_counters_to_dict(counters)
        self.assertEqual(values["gradient_evaluations"], 0)
        self.assertEqual(values["objective_evaluations"], 1)
        self.assertEqual(values["candidate_step_exposure"], 1)

    def test_works_inside_jit(self):
        @jax.jit
        def add_zeros():
            return add_compute_counters(zero_compute_counters(), zero_compute_counters())

        values = compute_counters_to_dict(add_zeros())
        self.assertEqual(values["search_steps_executed"], 0)

    def test_works_inside_scan(self):
        def body(carry, _):
            one = counters_for_default_sgd_search(
                candidates=1, steps_executed=1, max_rows=4, max_cols=4
            )
            return add_compute_counters(carry, one), None

        final, _ = jax.lax.scan(body, zero_compute_counters(), xs=None, length=3)
        values = compute_counters_to_dict(final)
        self.assertEqual(values["search_steps_executed"], 3)
        self.assertEqual(values["gradient_evaluations"], 3)
        self.assertEqual(values["objective_evaluations"], 6)


if __name__ == "__main__":
    unittest.main()
