"""Descriptive analog evidence; never selects or modifies a forecast."""

import numpy as np


def trajectory_comparison(query, candidate):
    """Separate signed level offset from shape; all errors use flow units."""
    query, candidate = (np.asarray(x, dtype=np.float64) for x in (query, candidate))
    if any(x.shape != (12,) or not np.isfinite(x).all() for x in (query, candidate)):
        raise ValueError("Trajectory comparison requires 12 finite values")
    difference = candidate - query
    offset = float(difference.mean())
    return {
        "candidate_minus_query_by_step": difference.tolist(),
        "mean_signed_offset": offset,
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "centered_shape_rmse": float(np.sqrt(np.mean((difference - offset) ** 2))),
        "step_change_rmse": float(np.sqrt(np.mean(np.diff(difference) ** 2))),
        "query_end_minus_start": float(query[-1] - query[0]),
        "candidate_end_minus_start": float(candidate[-1] - candidate[0]),
    }


def compare_paired_state(
    query_base,
    query_normal,
    base,
    normal,
    observed,
    scale,
                         *, query_history=None, candidate_history=None):
    arrays = [np.asarray(x, dtype=np.float64) for x in (
        query_base,
        query_normal,
        base,
        normal,
        observed,
    )]
    if any(x.shape != (12,) or not np.isfinite(x).all() for x in arrays):
        raise ValueError("Paired-state trajectories must contain 12 finite values")
    query_foundation_model_prediction, query_normal_counterfactual_sequence, foundation_model_prediction, normal_counterfactual_sequence, observed = arrays
    scale = max(float(scale), 1.0)
    if not np.isfinite(scale):
        raise ValueError("Paired-state scale must be finite")
    gap = foundation_model_prediction - normal_counterfactual_sequence
    query_gap = query_foundation_model_prediction - query_normal_counterfactual_sequence
    errors = {
        name: float(np.sqrt(np.mean((candidate - query) ** 2)))
        for name, candidate, query in (
            ("base", foundation_model_prediction, query_foundation_model_prediction),
            ("normal", normal_counterfactual_sequence, query_normal_counterfactual_sequence),
            ("signed_gap", gap, query_gap),
        )
    }
    valid = observed != 0
    comparisons = {
        name: trajectory_comparison(query, candidate)
        for name, query, candidate in (
            ("base", query_foundation_model_prediction, foundation_model_prediction),
            ("normal", query_normal_counterfactual_sequence, normal_counterfactual_sequence),
            ("signed_gap", query_gap, gap),
        )
    }
    # Canonical names are added alongside the persisted aliases.  The values
    # are identical and no ranking/evaluation branch depends on either key.
    comparisons["foundation_model_prediction"] = comparisons["base"]
    if query_history is not None and candidate_history is not None:
        comparisons["history"] = trajectory_comparison(query_history, candidate_history)
    result = {
        "status": "available",
        "candidate_base_forecast": foundation_model_prediction.tolist(),
        "candidate_normal_flow": normal_counterfactual_sequence.tolist(),
        "candidate_foundation_model_prediction": foundation_model_prediction.tolist(),
        "candidate_normal_counterfactual_sequence": normal_counterfactual_sequence.tolist(),
        "candidate_base_normal_gap": gap.tolist(),
        "query_base_normal_gap": query_gap.tolist(),
        "candidate_foundation_normal_gap": gap.tolist(),
        "query_foundation_normal_gap": query_gap.tolist(),
        "gap_difference_by_step": (gap - query_gap).tolist(),
        "rmse_to_query": errors,
        "trajectory_comparisons": comparisons,
        "historical_outcome": {
            "observed_future_flow": observed.tolist(),
            "valid_observation_mask": valid.tolist(),
            "observed_minus_base": [float(v) if ok else None for v, ok in zip(observed - foundation_model_prediction, valid)],
            "observed_minus_foundation_model": [float(v) if ok else None for v, ok in zip(observed - foundation_model_prediction, valid)],
            "observed_minus_normal": [float(v) if ok else None for v, ok in zip(observed - normal_counterfactual_sequence, valid)],
            "definition": "Positive residual means underprediction; null denotes missing zero observations.",
        },
        "query_history_scale": scale,
        "similarity_to_query": {k: float(np.exp(-v / scale)) for k, v in errors.items()},
        "similarity_definition": "exp(-RMSE/query_history_scale); descriptive, not a probability",
        "gap_sign_agreement_fraction": float(np.mean(np.sign(gap) == np.sign(query_gap))),
        "historical_valid_steps": int(valid.sum()),
        "historical_base_mae": float(np.mean(np.abs(foundation_model_prediction[valid] - observed[valid]))) if valid.any() else None,
        "historical_foundation_model_mae": float(np.mean(np.abs(foundation_model_prediction[valid] - observed[valid]))) if valid.any() else None,
        "historical_normal_mae": float(np.mean(np.abs(normal_counterfactual_sequence[valid] - observed[valid]))) if valid.any() else None,
    }
    return result


def compare_foundation_normal_state(
    query_foundation_model_prediction,
    query_normal_counterfactual_sequence,
    foundation_model_prediction,
    normal_counterfactual_sequence,
    observed_future_flow,
    query_history_scale,
    **kwargs,
):
    """Compare the Foundation--Normal state while preserving the old API."""

    return compare_paired_state(
        query_foundation_model_prediction,
        query_normal_counterfactual_sequence,
        foundation_model_prediction,
        normal_counterfactual_sequence,
        observed_future_flow,
        query_history_scale,
        **kwargs,
    )


__all__ = ["compare_paired_state", "compare_foundation_normal_state", "trajectory_comparison"]
