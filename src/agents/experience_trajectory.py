"""Experience Trajectory public API for RAP."""

from src.agents.decision_memory import (
    descriptor as experience_descriptor,
    enrich_episode as enrich_experience_trajectory,
    retrieve_decision_memory as retrieve_experience_trajectories,
)
from src.agents.episode_memory import (
    append_episode as append_experience_trajectory,
    build_episode as build_experience_trajectory,
    build_episode_reflection_messages,
    clear_episode_memory,
    load_episodes,
    parse_episode_reflection_response,
)


def retrieve_experience_trajectory(*args, **kwargs):
    """Singular compatibility entry point for Experience Trajectory retrieval."""

    return retrieve_experience_trajectories(*args, **kwargs)


__all__ = [
    "append_experience_trajectory",
    "build_experience_trajectory",
    "build_episode_reflection_messages",
    "clear_episode_memory",
    "enrich_experience_trajectory",
    "experience_descriptor",
    "load_episodes",
    "parse_episode_reflection_response",
    "retrieve_experience_trajectories",
    "retrieve_experience_trajectory",
]
