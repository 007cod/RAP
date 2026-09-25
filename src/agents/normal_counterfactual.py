"""Normal Counterfactual Generation public API.

The implementation remains in ``non_incident_retrieval`` so cached indexes
and existing imports stay valid.  This module provides the terminology used by
the RAP paper without introducing a second implementation.
"""

from src.agents.non_incident_retrieval import (
    NoIncidentCandidatesError,
    NoIncidentHistoryIndex,
    NormalCounterfactualCandidatesError,
    NormalCounterfactualIndex,
    get_no_incident_history_index,
)
from src.agents.tools import (
    build_normal_counterfactual_sequence,
    build_no_incident_normal_reference,
)


def generate_normal_counterfactual_sequence(*args, **kwargs):
    return build_normal_counterfactual_sequence(*args, **kwargs)


def get_normal_counterfactual_index(*args, **kwargs):
    return get_no_incident_history_index(*args, **kwargs)


__all__ = [
    "NoIncidentCandidatesError",
    "NormalCounterfactualCandidatesError",
    "NormalCounterfactualIndex",
    "generate_normal_counterfactual_sequence",
    "build_normal_counterfactual_sequence",
    "build_no_incident_normal_reference",
    "get_normal_counterfactual_index",
]
