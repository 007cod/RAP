"""Canonical terminology used by the RAP implementation.

The experiment artifacts written by earlier runs intentionally keep their
original field names.  Runtime contexts expose the paper terminology as the
primary view and retain legacy aliases at serialization boundaries so old
reports and resume jobs remain readable.
"""

from __future__ import annotations

from typing import Any


# Paper-facing module names and context fields.  These constants are the
# canonical vocabulary used at runtime; legacy names below are serialization
# and API aliases only.
CONTEXT_AWARE_PHASED_REASONING = "context_aware_phased_reasoning"
HISTORICAL_KNOWLEDGE_DISCOVERY = "historical_knowledge_discovery"
NORMAL_COUNTERFACTUAL_GENERATION = "normal_counterfactual_generation"
HISTORICAL_INCIDENT_RETRIEVAL = "historical_incident_retrieval"
EXPERIENCE_TRAJECTORY = "experience_trajectory"
CONTEXT_ANALYSIS = "context_analysis"
PHASED_PREDICTION = "phased_prediction"

FOUNDATION_MODEL_PREDICTION = "foundation_model_prediction"
NORMAL_COUNTERFACTUAL_SEQUENCE = "normal_counterfactual_sequence"
INCIDENT_PATTERNS = "incident_patterns"
EXPERIENCE_TRAJECTORIES = "experience_trajectories"
PREVIOUS_LAYER_PREDICTIONS = "previous_layer_predictions"
TRAFFIC_HISTORY = "traffic_history"

# The three evidence groups described in Historical Incident Retrieval.
INCIDENT_SIMILARITY = "incident_similarity"
SEQUENCE_SIMILARITY = "sequence_similarity"
FOUNDATION_NORMAL_STATE_SIMILARITY = "foundation_normal_state_similarity"

# Persisted group names from earlier experiments.  They remain readable and
# writable so existing outputs and evaluation scripts are not invalidated.
LEGACY_EVIDENCE_GROUP_ALIASES = {
    INCIDENT_SIMILARITY: "incident_network_time_similarity",
    SEQUENCE_SIMILARITY: "history_similarity",
    FOUNDATION_NORMAL_STATE_SIMILARITY: "base_normal_similarity",
}
CANONICAL_EVIDENCE_GROUP_ALIASES = {
    legacy: canonical
    for canonical, legacy in LEGACY_EVIDENCE_GROUP_ALIASES.items()
}


def canonical_evidence_group_name(name: str) -> str:
    """Return the paper-facing name for one retrieval evidence group."""

    return CANONICAL_EVIDENCE_GROUP_ALIASES.get(str(name), str(name))


def legacy_evidence_group_name(name: str) -> str:
    """Return the persisted name for one retrieval evidence group."""

    name = canonical_evidence_group_name(name)
    return LEGACY_EVIDENCE_GROUP_ALIASES.get(name, name)

# Human-facing module/option names from the paper.  These are accepted in
# configuration and normalized to the canonical context fields above.
CONTEXT_OPTION_ALIASES = {
    "similar_incident_cases": INCIDENT_PATTERNS,
    "normal_reference": NORMAL_COUNTERFACTUAL_SEQUENCE,
    "episode_memory": EXPERIENCE_TRAJECTORIES,
    "previous_layer_nodes": PREVIOUS_LAYER_PREDICTIONS,
    "historical_incident_retrieval": INCIDENT_PATTERNS,
    "normal_counterfactual_generation": NORMAL_COUNTERFACTUAL_SEQUENCE,
    "experience_trajectory": EXPERIENCE_TRAJECTORIES,
    "experience_trajectories": EXPERIENCE_TRAJECTORIES,
    "previous_layer_predictions": PREVIOUS_LAYER_PREDICTIONS,
    "foundation_model_prediction": FOUNDATION_MODEL_PREDICTION,
}

# Historical artifact/config names retained for compatibility.
LEGACY_CONTEXT_ALIASES = {
    FOUNDATION_MODEL_PREDICTION: "base_forecast",
    NORMAL_COUNTERFACTUAL_SEQUENCE: "non_incident_normal_reference",
    INCIDENT_PATTERNS: "retrieve_similar_cases_top5",
    EXPERIENCE_TRAJECTORIES: "episode_memory",
    PREVIOUS_LAYER_PREDICTIONS: "previous_layer_nodes",
}

LEGACY_DATA_ALIASES = {
    TRAFFIC_HISTORY: "target_history_flow",
}

# Nested model-facing fields.  Raw experiment records may still contain the
# legacy spelling; canonicalize_model_context projects them for the LLM.
LEGACY_MODEL_FIELD_ALIASES = {
    "sequence": "weighted_normal_flow",
    "foundation_normal_gap": "base_forecast_normal_gap",
    "historical_foundation_model_error": "historical_base_error",
    "weighted_average_foundation_model_error": "weighted_average_base_error",
    "candidate_foundation_model_prediction": "candidate_base_forecast",
    "normal_counterfactual_sequence": "normal_counterfactual_flow",
    "foundation_model_residual_distribution": "base_residual_distribution",
    "query_foundation_normal_gap": "query_base_normal_gap",
    "candidate_foundation_normal_gap": "candidate_base_normal_gap",
    "historical_foundation_model_mae": "historical_base_mae",
    "foundation_normal_state_match": "base_normal_match",
}

CANONICAL_CONTEXT_ALIASES = {
    **{value: key for key, value in LEGACY_CONTEXT_ALIASES.items()},
}


def canonical_context_key(key: str) -> str:
    """Return the paper-facing name for a context field or option."""

    key = str(key)
    normalized = CANONICAL_CONTEXT_ALIASES.get(key, key)
    return CONTEXT_OPTION_ALIASES.get(normalized, normalized)


def legacy_context_key(key: str) -> str:
    """Return the persisted legacy name for a canonical context field."""

    key = canonical_context_key(key)
    return LEGACY_CONTEXT_ALIASES.get(key, key)


def add_legacy_context_aliases(context: dict[str, Any]) -> dict[str, Any]:
    """Expose old field names without duplicating or changing their values."""

    for canonical, legacy in LEGACY_CONTEXT_ALIASES.items():
        if canonical in context and legacy not in context:
            context[legacy] = context[canonical]
        elif legacy in context and canonical not in context:
            context[canonical] = context[legacy]
    return context


def canonical_context_view(context: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow context view keyed by the paper terminology."""

    result = dict(context)
    for canonical, legacy in LEGACY_CONTEXT_ALIASES.items():
        if canonical not in result and legacy in result:
            result[canonical] = result[legacy]
        result.pop(legacy, None)
    for canonical, legacy in LEGACY_DATA_ALIASES.items():
        if canonical not in result and legacy in result:
            result[canonical] = result[legacy]
        result.pop(legacy, None)
    return result


def canonicalize_model_context(context: dict[str, Any]) -> dict[str, Any]:
    """Rename model-facing nested evidence fields without changing values."""

    result = canonical_context_view(context)
    normal = result.get(NORMAL_COUNTERFACTUAL_SEQUENCE)
    if isinstance(normal, dict):
        for canonical, legacy in LEGACY_MODEL_FIELD_ALIASES.items():
            if canonical not in normal and legacy in normal:
                normal[canonical] = normal[legacy]
            normal.pop(legacy, None)

    def canonicalize_experience_payload(payload: dict[str, Any]) -> None:
        """Project compact Experience Trajectory evidence for the LLM only."""

        local_evidence = payload.get("local_evidence")
        if not isinstance(local_evidence, list):
            return
        for item in local_evidence:
            if not isinstance(item, dict):
                continue
            candidates = item.get("past_candidates")
            if isinstance(candidates, dict):
                if "foundation_model_prediction" not in candidates and "base" in candidates:
                    candidates["foundation_model_prediction"] = candidates["base"]
                if "normal_counterfactual_sequence" not in candidates and "normal" in candidates:
                    candidates["normal_counterfactual_sequence"] = candidates["normal"]
                candidates.pop("base", None)
                candidates.pop("normal", None)
            if "llm_prediction" not in item and "past_prediction" in item:
                item["llm_prediction"] = item["past_prediction"]
            item.pop("past_prediction", None)
            if "observed_future_flow" not in item and "observed_future" in item:
                item["observed_future_flow"] = item["observed_future"]
            item.pop("observed_future", None)
            outcomes = item.get("observed_outcomes")
            if isinstance(outcomes, list):
                for outcome in outcomes:
                    if not isinstance(outcome, dict):
                        continue
                    metrics = outcome.get("mae")
                    if isinstance(metrics, dict):
                        if "foundation_model_prediction" not in metrics and "base" in metrics:
                            metrics["foundation_model_prediction"] = metrics["base"]
                        if "llm_prediction" not in metrics and "prediction" in metrics:
                            metrics["llm_prediction"] = metrics["prediction"]
                        metrics.pop("base", None)
                        metrics.pop("prediction", None)
                    if "outcome_vs_foundation_model" not in outcome and "outcome_vs_base" in outcome:
                        outcome["outcome_vs_foundation_model"] = outcome["outcome_vs_base"]
                    outcome.pop("outcome_vs_base", None)

    experiences = result.get(EXPERIENCE_TRAJECTORIES)
    if isinstance(experiences, dict):
        canonicalize_experience_payload(experiences)

    patterns = result.get(INCIDENT_PATTERNS)
    if isinstance(patterns, dict):
        for case in patterns.get("selection", {}).get("cases", []):
            if not isinstance(case, dict):
                continue
            paired = case.get("paired_state")
            if isinstance(paired, dict):
                for canonical, legacy in LEGACY_MODEL_FIELD_ALIASES.items():
                    if canonical not in paired and legacy in paired:
                        paired[canonical] = paired[legacy]
                    paired.pop(legacy, None)

    # Retrieval group names are implementation details in stored records but
    # are evidence semantics in the LLM context.  Rewrite only the model view.
    patterns = result.get(INCIDENT_PATTERNS)
    if isinstance(patterns, dict):
        selection = patterns.get("selection")
        if isinstance(selection, dict):
            for container_key in ("groups", "group_scores", "group_ranks"):
                container = selection.get(container_key)
                if isinstance(container, dict):
                    renamed = {}
                    for key, value in container.items():
                        renamed[canonical_evidence_group_name(key)] = value
                    selection[container_key] = renamed
            for case in selection.get("cases", []):
                if not isinstance(case, dict):
                    continue
                for canonical, legacy in LEGACY_MODEL_FIELD_ALIASES.items():
                    if canonical not in case and legacy in case:
                        case[canonical] = case[legacy]
                    case.pop(legacy, None)
                historical_effect = case.get("historical_effect")
                if isinstance(historical_effect, dict):
                    for canonical, legacy in LEGACY_MODEL_FIELD_ALIASES.items():
                        if canonical not in historical_effect and legacy in historical_effect:
                            historical_effect[canonical] = historical_effect[legacy]
                        historical_effect.pop(legacy, None)
                if isinstance(case.get("source_groups"), list):
                    case["source_groups"] = [
                        canonical_evidence_group_name(group)
                        for group in case["source_groups"]
                    ]
                for container_key in ("group_scores", "group_ranks"):
                    container = case.get(container_key)
                    if isinstance(container, dict):
                        case[container_key] = {
                            canonical_evidence_group_name(key): value
                            for key, value in container.items()
                        }
        if isinstance(patterns.get("evidence_group_names"), dict):
            patterns["evidence_group_names"] = {
                canonical_evidence_group_name(key): value
                for key, value in patterns["evidence_group_names"].items()
            }
        semantics = patterns.get("evidence_semantics")
        if isinstance(semantics, dict):
            for container_key in ("group_scores", "evidence_group_roles"):
                container = semantics.get(container_key)
                if isinstance(container, dict):
                    semantics[container_key] = {
                        canonical_evidence_group_name(key): value
                        for key, value in container.items()
                    }
    return result


__all__ = [
    "CONTEXT_AWARE_PHASED_REASONING",
    "HISTORICAL_KNOWLEDGE_DISCOVERY",
    "NORMAL_COUNTERFACTUAL_GENERATION",
    "HISTORICAL_INCIDENT_RETRIEVAL",
    "EXPERIENCE_TRAJECTORY",
    "CONTEXT_ANALYSIS",
    "PHASED_PREDICTION",
    "FOUNDATION_MODEL_PREDICTION",
    "NORMAL_COUNTERFACTUAL_SEQUENCE",
    "INCIDENT_PATTERNS",
    "EXPERIENCE_TRAJECTORIES",
    "PREVIOUS_LAYER_PREDICTIONS",
    "TRAFFIC_HISTORY",
    "INCIDENT_SIMILARITY",
    "SEQUENCE_SIMILARITY",
    "FOUNDATION_NORMAL_STATE_SIMILARITY",
    "LEGACY_EVIDENCE_GROUP_ALIASES",
    "CANONICAL_EVIDENCE_GROUP_ALIASES",
    "LEGACY_MODEL_FIELD_ALIASES",
    "CONTEXT_OPTION_ALIASES",
    "LEGACY_CONTEXT_ALIASES",
    "LEGACY_DATA_ALIASES",
    "canonical_context_key",
    "legacy_context_key",
    "add_legacy_context_aliases",
    "canonical_context_view",
    "canonicalize_model_context",
    "canonical_evidence_group_name",
    "legacy_evidence_group_name",
]
