from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock

import numpy as np
import torch
import pandas as pd

from src.evaluation.metrics import compute_metric_dict
from src.agents.decision_memory import SCHEMA, enrich_episode, retrieve_decision_memory
from src.agents.horizon_evidence import local_outcomes
from src.config import DEFAULT_MEMORY_MIN_SIMILARITY
from src.agents.terminology import (
    FOUNDATION_MODEL_PREDICTION,
    NORMAL_COUNTERFACTUAL_SEQUENCE,
    TRAFFIC_HISTORY,
)


DEFAULT_EPISODES_PATH = Path("artifacts_sacramento/memory/episodes_reflection.jsonl")
DEFAULT_EPISODE_RETENTION_DAYS = 90
DEFAULT_EPISODE_RECENCY_HALF_LIFE_DAYS = 30
DEFAULT_EPISODE_SEARCH_WINDOW_DAYS = 120
DEFAULT_EPISODE_MAX_ENTRIES = 2000
DEFAULT_EPISODE_LOOKBACK_MINUTES = 60
MEMORY_SCHEMA_VERSION = SCHEMA
_WRITE_LOCK = RLock()
REFLECTION_VERDICTS = {"improved", "degraded", "equal"}
REFLECTION_FAILURE_MODES = {
    "none",
    "sign_error",
    "over_correction",
    "under_correction",
    "too_low",
    "too_high",
    "phase_mismatch",
}
EPISODE_REFLECTION_SYSTEM_PROMPT = """
You are an Experience Trajectory reviewer. Review one completed incident-aware
traffic prediction after its outcome is known. The program computes the verdict;
Write one concise paragraph stating the main agreement or error between the
prediction and the observed outcome, one observable condition under which the
experience may transfer, and its limitation. Foundation Model Prediction
residuals are forecast errors, not measured causal incident effects. Use only
the supplied evidence, record observations rather than commands, and do not
repeat full trajectories or invent causes.
""".strip()


def append_episode(
    path: str | Path,
    episode: dict,
    *,
    retention_days: int = DEFAULT_EPISODE_RETENTION_DAYS,
    max_entries: int = DEFAULT_EPISODE_MAX_ENTRIES,
) -> None:
    path = Path(path)
    with _WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        episodes = load_episodes(path)
        episodes.append(_compact_stored_episode(episode))
        episodes = _dedupe_episodes(episodes)
        reference_time = max(_parse_episode_time(item) for item in episodes)
        episodes = _prune_episodes(episodes, reference_time, retention_days=int(retention_days), max_entries=int(max_entries))
        _write_episodes(path, episodes)


def append_experience_trajectory(*args, **kwargs) -> None:
    """Persist one Experience Trajectory using the existing storage schema."""

    return append_episode(*args, **kwargs)


def clear_episode_memory(path: str | Path) -> int:
    """Clear the model-specific episode memory before a new Tool LLM task."""
    path = Path(path)
    with _WRITE_LOCK:
        existing_count = len(load_episodes(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    return existing_count


def load_episodes(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    episodes = []
    with _WRITE_LOCK:
        lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        episode = json.loads(line)
        if not isinstance(episode, dict):
            raise ValueError(f"Episode memory line must be a JSON object: path={path}, line={line_number}")
        episodes.append(episode)
    return episodes


def retrieve_episode_memory(
    path: str | Path,
    context: dict,
    top_k: int = 3,
    *,
    search_window_days: int = DEFAULT_EPISODE_SEARCH_WINDOW_DAYS,
    recency_half_life_days: int = DEFAULT_EPISODE_RECENCY_HALF_LIFE_DAYS,
    min_similarity: float = DEFAULT_MEMORY_MIN_SIMILARITY,
    min_base_mae: float = 0.0,
    min_failure_delta: float = 10.0,
) -> dict:
    return retrieve_decision_memory(
        path, context, top_k=top_k, search_window_days=search_window_days,
        recency_half_life_days=recency_half_life_days,
        min_similarity=min_similarity,
        min_base_mae=min_base_mae,
        min_failure_delta=min_failure_delta,
    )


def retrieve_experience_trajectories(*args, **kwargs) -> dict:
    """Retrieve the paper-named Experience Trajectory evidence."""

    return retrieve_episode_memory(*args, **kwargs)


def _foundation_model_prediction(evaluation: dict) -> list[float]:
    """Read the canonical Foundation Model Prediction with legacy fallback."""

    values = evaluation.get(FOUNDATION_MODEL_PREDICTION, evaluation.get("base_forecast"))
    if values is None:
        raise KeyError("evaluation is missing foundation_model_prediction")
    return [float(value) for value in values]


def _normal_counterfactual_sequence(context: dict):
    reference = (
        context.get(NORMAL_COUNTERFACTUAL_SEQUENCE)
        or context.get("non_incident_normal_reference")
        or {}
    )
    values = reference.get("sequence", reference.get("weighted_normal_flow"))
    return None if values is None else [float(value) for value in values]


def build_episode(
    context: dict,
    evaluation: dict,
    parsed: dict,
    reflection: dict,
    result_record_path: str | Path,
    raw_response_path: str | Path,
    reflection_raw_response_path: str | Path,
) -> dict:
    y_true = [float(value) for value in evaluation["y_true"]]
    foundation_model_prediction = _foundation_model_prediction(evaluation)
    llm_predicted_flow = [float(value) for value in parsed["predicted_flow"]]
    foundation_model_metrics = compute_metric_dict(
        torch.tensor(foundation_model_prediction, dtype=torch.float32),
        torch.tensor(y_true, dtype=torch.float32),
        null_val=0.0,
    )
    llm_metrics = compute_metric_dict(
        torch.tensor(llm_predicted_flow, dtype=torch.float32),
        torch.tensor(y_true, dtype=torch.float32),
        null_val=0.0,
    )
    delta_mae = float(llm_metrics["mae"] - foundation_model_metrics["mae"])
    assessment = _correction_assessment(
        foundation_model_prediction, llm_predicted_flow, y_true, delta_mae
    )
    if reflection["verdict"] != assessment["verdict"]:
        raise ValueError(
            "Episode reflection verdict disagrees with objective assessment: "
            f"expected={assessment['verdict']}, actual={reflection['verdict']}"
        )
    episode = {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "memory_id": _memory_id(context),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "incident_time": context["incident"]["time"],
        "incident_id": str(context["sample_id"]),
        "target_node_id": int(context["target_node_id"]),
        "target_node_index": int(context["target_node_index"]),
        "layer_index": int(context["layer_index"]),
        "incident": {
            "time": context["incident"]["time"],
            "description": context["incident"]["description"],
            "type": context["incident"]["type"],
            "area": context["incident"]["area"],
            "location": context["incident"]["location"],
            "freeway": context["incident"]["freeway"],
            "direction": context["incident"]["direction"],
            "nearest_sensor_node_id": int(context["incident"]["nearest_sensor_node_id"]),
            "target_road_relation": context["incident"]["target_road_relation"],
            "topology_path": context["incident"]["topology_path"],
        },
        "target_history_summary": _series_summary(
            context.get(TRAFFIC_HISTORY, context.get("target_history_flow", []))
        ),
        "base_forecast_summary": _series_summary(foundation_model_prediction),
        "llm_predicted_flow_summary": _series_summary(llm_predicted_flow),
        "phase_analysis": parsed["phase_analysis"],
        "forecast_explanation": parsed.get("forecast_explanation", ""),
        "phase_outcomes": [],
        "metrics": {
            "base": foundation_model_metrics,
            "tool_llm": llm_metrics,
            "delta_mae_tool_minus_base": delta_mae,
            "verdict": _verdict(delta_mae),
        },
        "assessment": assessment,
        "reflection": reflection,
        "paths": {
            "context": context["_paths"]["context"],
            "messages": context["_paths"]["messages"],
            "result_record": str(result_record_path),
            "raw_response": str(raw_response_path),
            "reflection_raw_response": str(reflection_raw_response_path),
        },
    }

    return enrich_episode(episode, context, llm_predicted_flow, y_true)


def build_experience_trajectory(*args, **kwargs) -> dict:
    """Construct one persisted Experience Trajectory."""

    return build_episode(*args, **kwargs)


def build_episode_reflection_messages(
    context: dict,
    evaluation: dict,
    parsed: dict,
) -> tuple[list[dict], dict]:
    y_true = [float(value) for value in evaluation["y_true"]]
    foundation_model_prediction = _foundation_model_prediction(evaluation)
    predicted_flow = [float(value) for value in parsed["predicted_flow"]]
    foundation_model_metrics = compute_metric_dict(
        torch.tensor(foundation_model_prediction, dtype=torch.float32),
        torch.tensor(y_true, dtype=torch.float32),
        null_val=0.0,
    )
    llm_metrics = compute_metric_dict(
        torch.tensor(predicted_flow, dtype=torch.float32),
        torch.tensor(y_true, dtype=torch.float32),
        null_val=0.0,
    )
    assessment = _correction_assessment(
        foundation_model_prediction,
        predicted_flow,
        y_true,
        float(llm_metrics["mae"] - foundation_model_metrics["mae"]),
    )
    normal_counterfactual_sequence = _normal_counterfactual_sequence(context)
    canonical_runtime_context = (
        FOUNDATION_MODEL_PREDICTION in context or TRAFFIC_HISTORY in context
    )
    payload = {
        "objective_assessment": assessment,
        "horizon_outcomes": local_outcomes(
            foundation_model_prediction, normal_counterfactual_sequence,
            predicted_flow, y_true,
        ),
        "incident": context["incident"],
        "target_node_id": int(context["target_node_id"]),
        "layer_index": int(context["layer_index"]),
        (
            "traffic_history" if canonical_runtime_context else "target_history_flow"
        ): context.get(TRAFFIC_HISTORY, context.get("target_history_flow", [])),
        (
            "foundation_model_prediction" if canonical_runtime_context else "base_forecast"
        ): foundation_model_prediction,
    }
    if normal_counterfactual_sequence is not None:
        payload[
            "normal_counterfactual_sequence" if canonical_runtime_context else "normal_forecast"
        ] = normal_counterfactual_sequence
    payload.update({
        "llm_prediction" if canonical_runtime_context else "predicted_flow": predicted_flow,
        "observed_future_flow" if canonical_runtime_context else "realized_future_flow": y_true,
        "phase_analysis": parsed["phase_analysis"],
        "forecast_explanation": parsed.get("forecast_explanation", ""),
    })
    return [
        {"role": "system", "content": EPISODE_REFLECTION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ], assessment


def parse_episode_reflection_response(raw_response: str, expected_verdict: str) -> dict:
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("Episode reflection must be a non-empty paragraph")
    if expected_verdict not in REFLECTION_VERDICTS:
        raise ValueError(f"Invalid program-computed episode verdict: {expected_verdict!r}")
    return {"verdict": expected_verdict, "reflection": raw_response.strip()}


def _compact_stored_episode(episode: dict) -> dict:
    # Storage is the audit record; only the separate model-facing projection is
    # compressed. Preserve all metrics, interpretations and source paths here.
    return dict(episode)


def _memory_id(context: dict) -> str:
    return (
        f"incident_{context['sample_id']}"
        f"_node_{context['target_node_id']}"
        f"_layer_{context['layer_index']}"
    )


def _series_summary(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "first": float(array[0]),
        "last": float(array[-1]),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "delta_last_first": float(array[-1] - array[0]),
    }


def _verdict(delta_mae: float) -> str:
    if delta_mae < 0:
        return "improved"
    if delta_mae > 0:
        return "degraded"
    return "equal"


def _correction_assessment(
    base_forecast: list[float],
    llm_predicted_flow: list[float],
    y_true: list[float],
    delta_mae: float,
) -> dict:
    base = np.asarray(base_forecast, dtype=np.float64)
    llm = np.asarray(llm_predicted_flow, dtype=np.float64)
    truth = np.asarray(y_true, dtype=np.float64)
    valid = np.isfinite(truth) & (truth != 0) & np.isfinite(base) & np.isfinite(llm)
    if not valid.any():
        return {"verdict": "equal", "failure_mode": "none", "reason": "No valid future observations."}
    base, llm, truth = base[valid], llm[valid], truth[valid]
    predicted_effect = llm - base
    realized_effect = truth - base
    predicted_effect_mean = float(np.mean(predicted_effect))
    realized_effect_mean = float(np.mean(realized_effect))
    predicted_direction = _effect_direction(predicted_effect_mean)
    realized_direction = _effect_direction(realized_effect_mean)
    magnitude_ratio = _safe_ratio(abs(predicted_effect_mean), abs(realized_effect_mean))
    if delta_mae < 0:
        reason = "Prediction reduced overall masked MAE versus Base; local outcomes can still include failures."
        return {
            "verdict": "improved",
            "failure_mode": "none",
            "reason": reason,
            "predicted_effect_direction": predicted_direction,
            "realized_effect_direction": realized_direction,
            "predicted_effect_mean": predicted_effect_mean,
            "realized_effect_mean": realized_effect_mean,
            "magnitude_ratio": magnitude_ratio,
        }
    if predicted_direction != realized_direction and "flat" not in {predicted_direction, realized_direction}:
        failure_mode = "sign_error"
        reason = (
            f"LLM changed the baseline in the wrong direction: predicted {predicted_direction} "
            f"but the realized effect was {realized_direction}."
        )
    elif magnitude_ratio > 1.5:
        failure_mode = "over_correction"
        reason = (
            "LLM over-corrected the baseline and pushed the forecast farther than the observed Base prediction residual."
        )
    elif magnitude_ratio < 0.5:
        failure_mode = "under_correction"
        reason = (
            "LLM stayed too close to the history-conditioned Base forecast and missed most of the observed Base prediction residual."
        )
    elif float(np.mean(llm)) < float(np.mean(base)) < float(np.mean(truth)):
        failure_mode = "too_low"
        reason = "LLM lowered a baseline that was already below the realized flow."
    elif float(np.mean(llm)) > float(np.mean(base)) > float(np.mean(truth)):
        failure_mode = "too_high"
        reason = "LLM raised a baseline that was already above the realized flow."
    else:
        failure_mode = "phase_mismatch"
        reason = "LLM captured the incident direction imperfectly but the phase shape did not match the realized trajectory."
    return {
        "verdict": "degraded" if delta_mae > 0 else "equal",
        "failure_mode": failure_mode,
        "reason": reason,
        "predicted_effect_direction": predicted_direction,
        "realized_effect_direction": realized_direction,
        "predicted_effect_mean": predicted_effect_mean,
        "realized_effect_mean": realized_effect_mean,
        "magnitude_ratio": magnitude_ratio,
    }


def _effect_direction(value: float) -> str:
    if value > 0.5:
        return "increase"
    if value < -0.5:
        return "decrease"
    return "flat"


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-6:
        return 999.0 if numerator > 1e-6 else 1.0
    return float(numerator / denominator)


def _parse_episode_time(episode: dict) -> datetime:
    if "incident_time" in episode:
        return _parse_time_value(episode["incident_time"])
    incident = episode.get("incident", {})
    if isinstance(incident, dict) and "time" in incident:
        return _parse_time_value(incident["time"])
    created_at = episode.get("created_at")
    if created_at:
        return _parse_time_value(created_at)
    raise ValueError("Episode is missing incident_time/incident.time/created_at")


def _parse_time_value(value) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.to_pydatetime()


def _prune_episodes(
    episodes: list[dict],
    reference_time: datetime,
    *,
    retention_days: int,
    max_entries: int | None,
) -> list[dict]:
    if retention_days < 0:
        raise ValueError(f"retention_days must be >= 0, got {retention_days}")
    if max_entries is not None and max_entries <= 0:
        raise ValueError(f"max_entries must be positive, got {max_entries}")
    reference = pd.Timestamp(reference_time).to_pydatetime()
    retained = []
    for episode in episodes:
        episode_time = _parse_episode_time(episode)
        age_days = (reference - episode_time).total_seconds() / 86400.0
        if age_days < 0:
            continue
        if age_days > float(retention_days):
            continue
        retained.append(episode)
    retained = _dedupe_episodes(retained)
    retained.sort(key=_episode_sort_key, reverse=True)
    if max_entries is not None:
        retained = retained[: int(max_entries)]
    return retained


def _dedupe_episodes(episodes: list[dict]) -> list[dict]:
    deduped = {}
    for episode in episodes:
        deduped[_episode_dedupe_key(episode)] = episode
    return list(deduped.values())


def _episode_dedupe_key(episode: dict) -> str:
    if "memory_id" in episode:
        return str(episode["memory_id"])
    incident_id = str(episode.get("incident_id", ""))
    target_node_id = str(episode.get("target_node_id", ""))
    layer_index = str(episode.get("layer_index", ""))
    return "|".join([incident_id, target_node_id, layer_index])


def _episode_sort_key(episode: dict):
    return _parse_episode_time(episode), str(episode.get("created_at", ""))


def _write_episodes(path: Path, episodes: list[dict]) -> None:
    lines = [json.dumps(episode, ensure_ascii=False, separators=(",", ":")) for episode in episodes]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp_path.replace(path)
