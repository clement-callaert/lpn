"""Public exports for adaptive latent-search helpers."""

from src.search.adaptive_sgd_search import (
    encode_leave_one_out_latents,
    make_adaptive_search_fn,
    make_fixed_search_fn,
    make_latent_conditioned_adaptive_search_fn,
    run_adaptive_sgd_search,
    run_fixed_sgd_search_like_upstream,
)
from src.search.compute_accounting import (
    COUNTER_FIELD_NAMES,
    ComputeCounters,
    add_compute_counters,
    compute_counters_to_dict,
    counters_for_default_sgd_search,
    validate_compute_counters_host,
    zero_compute_counters,
)
from src.search.search_state import (
    SearchState,
    initial_search_state,
    rl_observation_from_search_state,
    search_state_to_host_dict,
)
from src.search.stopping_rules import (
    REASON_NAMES,
    StoppingDecision,
    StoppingObservation,
    StoppingRuleConfig,
    apply_stopping_rule,
    validate_stopping_rule_config,
)

__all__ = [
    "COUNTER_FIELD_NAMES",
    "ComputeCounters",
    "REASON_NAMES",
    "SearchState",
    "StoppingDecision",
    "StoppingObservation",
    "StoppingRuleConfig",
    "add_compute_counters",
    "apply_stopping_rule",
    "compute_counters_to_dict",
    "counters_for_default_sgd_search",
    "initial_search_state",
    "encode_leave_one_out_latents",
    "make_adaptive_search_fn",
    "make_fixed_search_fn",
    "make_latent_conditioned_adaptive_search_fn",
    "rl_observation_from_search_state",
    "run_adaptive_sgd_search",
    "run_fixed_sgd_search_like_upstream",
    "search_state_to_host_dict",
    "validate_compute_counters_host",
    "validate_stopping_rule_config",
    "zero_compute_counters",
]
