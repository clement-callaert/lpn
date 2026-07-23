"""Reinforcement learning helpers for stop/continue latent search control."""

from src.rl.environment import (
    ACTION_CONTINUE,
    ACTION_STOP,
    EpisodeState,
    compute_terminal_reward,
    reset_episode,
    step_episode,
)
from src.rl.observations import (
    FEATURE_NAMES,
    ObservationNormStats,
    observation_dict_to_vector,
    observation_from_episode_state,
)
from src.rl.policy import StopContinuePolicy, evaluate_action, sample_action
from src.rl.reinforce import (
    MovingAverageBaseline,
    compute_episode_return,
    reinforce_loss,
)
from src.rl.trajectory_dataset import (
    TrajectoryStepRecord,
    compute_norm_stats_from_records,
    generate_fixed_horizon_trajectory,
)

__all__ = [
    "ACTION_CONTINUE",
    "ACTION_STOP",
    "EpisodeState",
    "FEATURE_NAMES",
    "MovingAverageBaseline",
    "ObservationNormStats",
    "StopContinuePolicy",
    "TrajectoryStepRecord",
    "compute_episode_return",
    "compute_norm_stats_from_records",
    "compute_terminal_reward",
    "evaluate_action",
    "generate_fixed_horizon_trajectory",
    "observation_dict_to_vector",
    "observation_from_episode_state",
    "reinforce_loss",
    "reset_episode",
    "sample_action",
    "step_episode",
]
