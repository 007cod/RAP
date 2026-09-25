"""Outcome-blind, non-compensating ranking within the existing recall pool."""

import numpy as np


# Paper-facing retrieval-group names.
PAPER_FOUNDATION_NORMAL_STATE_GROUP = "foundation_normal_state_similarity"
PAPER_INCIDENT_SIMILARITY_GROUP = "incident_similarity"
PAPER_SEQUENCE_SIMILARITY_GROUP = "sequence_similarity"

# Persisted experiment names retained for compatibility with existing outputs,
# reports, and callers.  The model-facing projection maps these to the paper
# names in ``terminology.canonicalize_model_context``.
FOUNDATION_NORMAL_STATE_GROUP = "base_normal_similarity"
BASE_NORMAL_GROUP = FOUNDATION_NORMAL_STATE_GROUP
INCIDENT_SIMILARITY_GROUP = "incident_similarity"
SEQUENCE_SIMILARITY_GROUP = "sequence_similarity"
FOUNDATION_NORMAL_STATE_NAME = PAPER_FOUNDATION_NORMAL_STATE_GROUP
LEGACY_INCIDENT_SIMILARITY_GROUP = "incident_network_time_similarity"
LEGACY_SEQUENCE_SIMILARITY_GROUP = "history_similarity"
STATE_AXES = tuple(
    f"{trajectory}.{metric}"
    for trajectory in ("base", "normal", "signed_gap")
    for metric in ("rmse", "step_change_rmse")
)


def rank_foundation_normal_candidates(rows):
    """Minimize the worst normalized mismatch, then the next worst, and so on.

    A common query-history scale preserves units across every candidate/axis.
    No observed future, historical winner, or old group threshold is consulted.
    Lexicographic minimax is Pareto-monotone without an O(n²) frontier search.
    It ranks relative matches; even rank one may be a poor absolute match.
    """
    eligible = []
    for row in rows:
        paired = row.get("paired_state_diagnostics", {})
        if paired.get("status") != "available":
            continue
        comparisons = paired.get("trajectory_comparisons", {})
        try:
            scale = float(paired["query_history_scale"])
            values = np.asarray([
                comparisons[trajectory][metric]
                for trajectory, metric in (axis.split(".") for axis in STATE_AXES)
            ], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(values).all() or (values < 0).any():
            continue
        normalized = values / scale
        row["base_normal_match"] = {
            "normalized_distances": dict(zip(STATE_AXES, normalized.tolist())),
            "worst_dimension": STATE_AXES[int(np.argmax(normalized))],
            "worst_normalized_distance": float(normalized.max()),
            "query_history_scale": scale,
            "ranking_key": sorted(normalized.tolist(), reverse=True),
        }
        eligible.append(row)
    return sorted(eligible, key=lambda row: (
        *row["base_normal_match"]["ranking_key"],
        -row["retrieval_score"], row["_timestamp"], row["incident_id"],
    ))


# Compatibility alias for experiment scripts and persisted-evaluation tooling.
rank_base_normal_candidates = rank_foundation_normal_candidates
