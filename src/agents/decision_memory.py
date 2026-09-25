"""Independent, outcome-blind retrieval of forecasting decision experiences.

Queries use only observations and candidate forecasts. Outcomes describe evidence
after selection, never the relevance score. Full provenance stays in raw context.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import RLock

import numpy as np
import pandas as pd

from src.agents.horizon_evidence import HORIZON_BANDS, local_outcomes
from src.config import DEFAULT_MEMORY_MIN_SIMILARITY

SCHEMA = "decision-local-evidence-v3"
_LOCK = RLock()
_INDEX = {}  # One immutable snapshot per experiment file, invalidated on append.


def descriptor(context):
    h = np.asarray(context.get("traffic_history", context.get("target_history_flow", [])), dtype=float)
    foundation_model_prediction = np.asarray(
        context.get("foundation_model_prediction", context.get("base_forecast", [])),
        dtype=float,
    )
    normal_context = (
        context.get("normal_counterfactual_sequence")
        or context.get("non_incident_normal_reference")
        or {}
    )
    n = np.asarray(
        normal_context.get("sequence", normal_context.get("weighted_normal_flow", [])),
        dtype=float,
    )
    normal_available = n.shape == (12,)
    if h.shape != (12,) or foundation_model_prediction.shape != (12,) or (n.size and not normal_available):
        return None
    arrays = (h, foundation_model_prediction, n) if normal_available else (h, foundation_model_prediction)
    if not all(np.isfinite(x).all() for x in arrays) or not np.any(h):
        return None
    scale = max(float(h.std()), abs(float(h.mean())) * .1, 1.)
    state = np.r_[(h - h[-1]) / scale, np.diff(h[-5:]) / scale]
    normal_decision = (n - h[-1]) / scale if normal_available else np.zeros(12)
    decision = np.r_[(foundation_model_prediction - h[-1]) / scale, normal_decision]
    normal_ref = context.get("non_incident_normal_reference") or {}
    profile = normal_ref.get(
        "historical_foundation_model_error",
        normal_ref.get("historical_base_error"),
    ) or {}
    bands = profile.get("bands") or []
    reliability = []
    for band in bands[:3]:
        # Normalize errors by the current history scale so nodes with different
        # flow magnitudes remain comparable during vector reranking.
        reliability.extend([float(band.get("mae", 0.0) or 0.0) / scale,
                             float(band.get("bias", 0.0) or 0.0) / scale])
    if len(reliability) < 6:
        reliability.extend([0.0] * (6 - len(reliability)))
    # Explicit Base-vs-Normal disagreement shape, independent of absolute flow.
    gap = (foundation_model_prediction - n) / scale if normal_available else np.zeros(12)
    return {
        "state": state.tolist(), "decision": decision.tolist(),
        "foundation_model_reliability": reliability[:6],
        "foundation_normal_gap": gap.tolist(),
        "traffic_history": h.tolist(),
        "foundation_model_prediction": foundation_model_prediction.tolist(),
        "normal_counterfactual_sequence": n.tolist() if normal_available else None,
        # Legacy descriptor aliases keep old memory files readable.
        "base_reliability": reliability[:6], "base_normal_gap": gap.tolist(),
        "history": h.tolist(), "base": foundation_model_prediction.tolist(),
        "normal": n.tolist() if normal_available else None,
        "normal_available": normal_available,
        "scale": scale,
        "window_signature": hashlib.sha256(np.round(h, 4).tobytes()).hexdigest(),
    }


def experience_trajectory_descriptor(context):
    """Build the retrieval descriptor for an Experience Trajectory."""

    return descriptor(context)


def enrich_episode(episode, context, prediction, truth):
    episode = dict(episode)
    episode["schema_version"] = SCHEMA
    episode["decision_descriptor"] = descriptor(context)
    t = pd.Timestamp(context["incident"]["time"]).floor("5min")
    episode["forecast_origin"] = t.isoformat()
    episode["outcome_available_at"] = (t + pd.Timedelta(minutes=5 * len(truth))).isoformat()
    normal = (
        context.get("normal_counterfactual_sequence")
        or context.get("non_incident_normal_reference")
        or {}
    ).get("sequence", (
        context.get("normal_counterfactual_sequence")
        or context.get("non_incident_normal_reference")
        or {}
    ).get("weighted_normal_flow"))
    episode["trajectories"] = {
        "history": context.get("traffic_history") or context.get("target_history_flow", []),
        "foundation_model_prediction": context.get("foundation_model_prediction") or context.get("base_forecast", []),
        "base": context.get("foundation_model_prediction") or context.get("base_forecast", []),
        **({"normal": normal} if normal is not None else {}),
        "prediction": prediction,
        "truth": truth,
    }
    foundation_model_prediction = context.get("foundation_model_prediction") or context.get("base_forecast", [])
    episode["horizon_outcomes"] = local_outcomes(
        foundation_model_prediction, normal, prediction, truth
    )
    intervals = [(p["start_step"], p["end_step"]) for p in episode.get("phase_analysis", [])
                 if isinstance(p, dict) and isinstance(p.get("start_step"), int) and isinstance(p.get("end_step"), int)]
    episode["phase_outcomes"] = (
        local_outcomes(foundation_model_prediction, normal, prediction, truth, intervals)
        if intervals else []
    )
    return episode


def _snapshot(path):
    path = Path(path).resolve()
    with _LOCK:
        if not path.exists():
            return [], np.empty((0, 16)), np.empty((0, 24)), np.empty((0, 6)), np.empty((0, 12))
        stat = path.stat(); version = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        cached = _INDEX.get(str(path))
        if cached and cached[0] == version:
            return cached[1]
        entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        entries = [e for e in entries if e.get("schema_version") == SCHEMA and e.get("decision_descriptor")
                   and e.get("horizon_outcomes")]
        states = np.asarray([e["decision_descriptor"]["state"] for e in entries], dtype=float).reshape(-1, 16)
        decisions = np.asarray([e["decision_descriptor"]["decision"] for e in entries], dtype=float).reshape(-1, 24)
        reliability = np.asarray([e["decision_descriptor"].get("base_reliability", [0.0]*6) for e in entries], dtype=float).reshape(-1, 6)
        gaps = np.asarray([e["decision_descriptor"].get("base_normal_gap", [0.0]*12) for e in entries], dtype=float).reshape(-1, 12)
        snapshot = (entries, states, decisions, reliability, gaps)
        if len(_INDEX) >= 8:
            _INDEX.pop(next(iter(_INDEX)))
        _INDEX[str(path)] = (version, snapshot)
        return snapshot


def _relation(context):
    return context.get("incident", {}).get("target_road_relation", {})


def _physical_similarity(episode, context):
    a, b = _relation(episode), _relation(context)
    keys = ("sensor_freeway", "sensor_direction", "same_direction", "same_road")
    observed = [float(str(a[k]) == str(b[k])) for k in keys if a.get(k) is not None and b.get(k) is not None]
    return float(np.mean(observed)) if observed else .5


def _differences(episode, context, query):
    d = episode["decision_descriptor"]
    differences = []
    if int(episode["target_node_id"]) != int(context["target_node_id"]):
        differences.append("different sensor; absolute flow levels may not transfer")
    layer = context.get("layer_index", context["incident"].get("distance_layer", 0))
    if episode["layer_index"] != layer:
        differences.append("different distance layer")
    a, b = _relation(episode), _relation(context)
    if a.get("sensor_direction") != b.get("sensor_direction"):
        differences.append("different sensor direction")
    source_hour = pd.Timestamp(episode["incident_time"]).hour
    current_hour = pd.Timestamp(context["incident"]["time"]).hour
    if min(abs(source_hour-current_hour), 24-abs(source_hour-current_hour)) >= 3:
        differences.append("different time of day")
    if np.sign(d["history"][-1]-d["history"][-2]) != np.sign(query["history"][-1]-query["history"][-2]):
        differences.append("latest observed change has a different direction")
    return differences


def local_evidence(episode, context, query):
    d = episode["decision_descriptor"]
    # All three bands remain visible: an overall success can fail locally.
    outcomes = []
    for outcome in episode["horizon_outcomes"]:
        metrics = {k: v["mae"] for k, v in outcome["metrics"].items()}
        if "base" in metrics:
            metrics["foundation_model_prediction"] = metrics["base"]
        if "prediction" in metrics:
            metrics["llm_prediction"] = metrics["prediction"]
        outcomes.append({
            "steps": [outcome["start_step"], outcome["end_step"]],
            "mae": metrics,
            "outcome_vs_base": outcome["outcome_vs_base"],
            "outcome_vs_foundation_model": outcome["outcome_vs_base"],
        })
    def change(values):
        return {"first": values[0], "last": values[-1]}
    foundation_model_candidate = change(
        d.get("foundation_model_prediction", d.get("base", []))
    )
    past_candidates = {
        # Legacy key remains in the raw memory record for compatibility.
        "base": foundation_model_candidate,
        "foundation_model_prediction": foundation_model_candidate,
    }
    if query.get("normal_available") and d.get("normal") is not None:
        normal_candidate = change(d["normal"])
        past_candidates["normal"] = normal_candidate
        past_candidates["normal_counterfactual_sequence"] = normal_candidate
    llm_prediction = change(episode["trajectories"]["prediction"])
    observed_future_flow = change(episode["trajectories"]["truth"])
    return {
        "memory_id": episode["memory_id"],
        "past_interpretation": str(episode.get("forecast_explanation", ""))[:180],
        "source_state": {"latest_change": d["history"][-1]-d["history"][-2],
                         "recent_changes": np.diff(d["history"][-4:]).tolist(),
                         "earlier_largest_rise": float(np.max(np.diff(d["history"][:-1]))),
                         "latest_observed": d["history"][-1]},
        "past_candidates": past_candidates,
        "past_prediction": change(episode["trajectories"]["prediction"]),
        "llm_prediction": llm_prediction,
        "observed_future": change(episode["trajectories"]["truth"]),
        "observed_future_flow": observed_future_flow,
        "observed_outcomes": outcomes,
        "transfer_differences": _differences(episode, context, query),
    }


def _learning_relevance(episode, *, min_base_mae=0.0, min_failure_delta=10.0):
    """Keep experiences where the decision had material room to learn.

    A low-error, successful correction is usually not useful as a transfer
    example. Large Base error remains useful; a large Tool LLM regression is
    retained as a counterexample so the forecast model can reject that move.
    """
    outcomes = episode.get("horizon_outcomes") or []
    valid = 0
    base_error = 0.0
    prediction_error = 0.0
    for outcome in outcomes:
        count = int(outcome.get("valid_points", 0) or 0)
        base = outcome.get("metrics", {}).get("base", {}).get("mae")
        prediction = outcome.get("metrics", {}).get("prediction", {}).get("mae")
        if count > 0 and base is not None and prediction is not None:
            valid += count
            base_error += float(base) * count
            prediction_error += float(prediction) * count
    if valid <= 0:
        return {"eligible": False, "reason": "no_valid_outcome", "base_mae": None, "delta_mae": None}
    base_mae = base_error / valid
    delta_mae = (prediction_error - base_error) / valid
    failure = delta_mae >= float(min_failure_delta)
    large_base_error = base_mae >= float(min_base_mae)
    return {
        "eligible": bool(large_base_error or failure),
        "reason": "large_base_error" if large_base_error else "large_tool_regression" if failure else "low_value_success",
        "base_mae": base_mae,
        "delta_mae": delta_mae,
    }


def retrieve_decision_memory(path, context, top_k=3, *, search_window_days=120,
                             recency_half_life_days=30, min_similarity=DEFAULT_MEMORY_MIN_SIMILARITY, recall_k=24,
                             min_base_mae=0.0, min_failure_delta=10.0):
    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("memory.min_similarity must be between 0 and 1")
    for name, value in (("min_base_mae", min_base_mae), ("min_failure_delta", min_failure_delta)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"memory.{name} must be finite and nonnegative")
    if top_k <= 0 or recency_half_life_days <= 0 or search_window_days < 0:
        raise ValueError("Invalid decision memory search configuration")
    query = descriptor(context)
    episodes, states, decisions, reliabilities, gaps = _snapshot(path)
    result = {"source": str(path), "retrieval_method": "independent_state_decision_vector_recall_v3",
              "available_episode_count": 0, "compatible_episode_count": 0,
              "selected_episode_count": 0, "selected_memory_ids": [], "local_evidence": [],
              "retrieval_diagnostics": {"min_similarity": min_similarity, "recall_k": recall_k,
                                        "min_base_mae": min_base_mae, "min_failure_delta": min_failure_delta,
                                        "excluded_duplicates": [], "ranked_candidates": []}}
    if query is None or not episodes:
        return result
    origin = pd.Timestamp(context["incident"]["time"]).floor("5min")
    eligible = [i for i, e in enumerate(episodes)
                if str(e["incident_id"]) != str(context["sample_id"])
                and pd.Timestamp(e["outcome_available_at"]) <= origin
                and 0 <= (origin-pd.Timestamp(e["forecast_origin"])).total_seconds()/86400 <= search_window_days
                and bool(e["decision_descriptor"].get("normal_available", True)) == bool(query["normal_available"])]
    result["available_episode_count"] = len(eligible)
    if not eligible:
        return result
    indices = np.asarray(eligible)
    state = np.exp(-np.sqrt(np.mean((states[indices]-query["state"])**2, axis=1)))
    decision_dimensions = 24 if query["normal_available"] else 12
    decision = np.exp(-np.sqrt(np.mean(
        (decisions[indices, :decision_dimensions] - np.asarray(query["decision"])[:decision_dimensions]) ** 2,
        axis=1,
    )))
    if query["normal_available"]:
        reliability = np.exp(-np.sqrt(np.mean((reliabilities[indices]-np.asarray(query["base_reliability"]))**2, axis=1)))
        gap_similarity = np.exp(-np.sqrt(np.mean((gaps[indices]-np.asarray(query["base_normal_gap"]))**2, axis=1)))
    else:
        reliability = np.ones(len(indices), dtype=float)
        gap_similarity = np.ones(len(indices), dtype=float)
    # Two independent exact vector recalls. Incident candidate rank is irrelevant.
    pool = sorted(set(np.argsort(-state, kind="stable")[:recall_k]) | set(np.argsort(-decision, kind="stable")[:recall_k]))
    ranked = []
    for j in pool:
        e = episodes[indices[j]]
        physical = _physical_similarity(e, context)
        age = (origin-pd.Timestamp(e["forecast_origin"])).total_seconds()/86400
        recency = float(np.exp(-np.log(2)*age/recency_half_life_days))
        base_score = float((.45*state[j] + .45*decision[j] + .1*physical) * (.9+.1*recency))
        # Ablation: keep the old qualification threshold unchanged; only rerank.
        if query["normal_available"]:
            score_components = .35*state[j] + .35*decision[j] + .12*reliability[j] + .08*gap_similarity[j] + .1*physical
        else:
            score_components = .45*state[j] + .45*decision[j] + .1*physical
        score = float(score_components * (.9+.1*recency))
        accepted = bool(base_score >= min_similarity and state[j] >= .5 and decision[j] >= .5)
        info = {"memory_id": e["memory_id"], "state_similarity": float(state[j]),
                "decision_similarity": float(decision[j]), "physical_similarity": physical,
                **({
                    "base_reliability_similarity": float(reliability[j]),
                    "base_normal_gap_similarity": float(gap_similarity[j]),
                } if query["normal_available"] else {}),
                "qualification_score": base_score,
                "recency": recency, "score": score, "qualified": accepted,
                "source_paths": e.get("paths", {})}
        ranked.append((score, e, info))
    ranked.sort(key=lambda x: (-x[0], x[1]["memory_id"]))
    result["retrieval_diagnostics"]["ranked_candidates"] = [x[2] for x in ranked]
    qualified = []
    for item in ranked:
        learning = _learning_relevance(item[1], min_base_mae=min_base_mae, min_failure_delta=min_failure_delta)
        item[2]["learning_relevance"] = learning
        if item[2]["qualified"] and learning["eligible"]:
            qualified.append(item)
    result["compatible_episode_count"] = len(qualified)
    selected = []
    for score, e, info in qualified:
        duplicate = any(int(e["target_node_id"]) == int(s["target_node_id"]) and (
            e["decision_descriptor"]["window_signature"] == s["decision_descriptor"]["window_signature"]
            or abs((pd.Timestamp(e["forecast_origin"])-pd.Timestamp(s["forecast_origin"])).total_seconds()) <= 15*60)
            for s in selected)
        if duplicate:
            result["retrieval_diagnostics"]["excluded_duplicates"].append(e["memory_id"])
            continue
        if len(selected) >= top_k:
            continue
        selected.append(e)
    result["selected_memory_ids"] = [e["memory_id"] for e in selected]
    result["selected_episode_count"] = len(selected)
    result["local_evidence"] = [local_evidence(e, context, query) for e in selected]
    return result


def retrieve_experience_trajectories(*args, **kwargs):
    """Retrieve Experience Trajectories with the paper-facing name."""

    return retrieve_decision_memory(*args, **kwargs)
