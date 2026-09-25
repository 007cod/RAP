import numpy as np

from src.agents.episode_memory import DEFAULT_EPISODES_PATH, retrieve_episode_memory
from src.agents.incident_vector_index import IncidentVectorIndex
from src.agents.non_incident_retrieval import NoIncidentHistoryIndex, get_no_incident_history_index
from src.agents.time_context import time_context
from src.agents.tools import (
    build_normal_counterfactual_sequence as _build_normal_counterfactual_sequence,
    incident_payload,
    retrieve_historical_incident_patterns as _retrieve_historical_incident_patterns,
)

# Legacy patch/import points remain available for existing tests and callers;
# both names resolve to the same implementation.
build_no_incident_normal_reference = _build_normal_counterfactual_sequence
retrieve_similar_cases = _retrieve_historical_incident_patterns
from src.agents.terminology import (
    EXPERIENCE_TRAJECTORIES,
    FOUNDATION_MODEL_PREDICTION,
    INCIDENT_PATTERNS,
    NORMAL_COUNTERFACTUAL_SEQUENCE,
    PREVIOUS_LAYER_PREDICTIONS,
    add_legacy_context_aliases,
    canonical_context_key,
)
from src.data.traffic import TrafficData
from src.config import (
    DEFAULT_MEMORY_MIN_SIMILARITY,
    LLM_CONTEXT_KEYS,
    ImpactScopeConfig,
    LLMContextConfig,
)
from src.tool_llm.timing import timed_section


def generate_normal_counterfactual_sequence(*args, **kwargs):
    """Paper-named entry point for Normal Counterfactual Generation."""

    aliases = {
        "foundation_model_provider": "base_forecast_provider",
        "current_foundation_model_prediction": "current_base_forecast",
    }
    for source, target in aliases.items():
        if source in kwargs and target not in kwargs:
            kwargs[target] = kwargs.pop(source)
    return build_no_incident_normal_reference(*args, **kwargs)


def retrieve_historical_incident_patterns(*args, **kwargs):
    """Paper-named entry point for Historical Incident Retrieval."""

    aliases = {
        "query_foundation_model_prediction": "query_base_forecast",
        "query_normal_counterfactual_sequence": "query_normal_flow",
        "foundation_model_provider": "base_forecast_provider",
        "include_normal_counterfactual": "include_normal_reference",
    }
    for source, target in aliases.items():
        if source in kwargs and target not in kwargs:
            kwargs[target] = kwargs.pop(source)
    return retrieve_similar_cases(*args, **kwargs)


def retrieve_experience_trajectories(*args, **kwargs):
    """Paper-named entry point for Experience Trajectory retrieval."""

    return retrieve_episode_memory(*args, **kwargs)


def _topology_payload(data: TrafficData, incident_node_index: int, target_node_index: int) -> dict:
    try:
        path_indices = data.shortest_path(incident_node_index, target_node_index)
    except ValueError as exc:
        if "No graph path between nodes" not in str(exc):
            raise
        return {
            "connected": False,
            "incident_node_index": int(incident_node_index),
            "incident_node_id": int(data.node_order[incident_node_index]),
            "target_node_index": int(target_node_index),
            "target_node_id": int(data.node_order[target_node_index]),
            "path_node_indices": [],
            "path_node_ids": [],
            "hop_count": None,
            "path_distance_km": None,
        }
    path_node_ids = [int(data.node_order[index]) for index in path_indices]
    edge_distances = [
        data.node_distance_km(path_indices[index], path_indices[index + 1])
        for index in range(len(path_indices) - 1)
    ]
    return {
        "connected": True,
        "incident_node_index": int(incident_node_index),
        "incident_node_id": int(data.node_order[incident_node_index]),
        "target_node_index": int(target_node_index),
        "target_node_id": int(data.node_order[target_node_index]),
        "path_node_indices": [int(index) for index in path_indices],
        "path_node_ids": path_node_ids,
        "hop_count": int(len(path_indices) - 1),
        "path_distance_km": float(sum(edge_distances)),
    }


def _compact_topology_path(topology: dict) -> dict:
    return {
        "connected": bool(topology["connected"]),
        "incident_node_id": topology["incident_node_id"],
        "target_node_id": topology["target_node_id"],
        "path_node_ids": topology["path_node_ids"],
        "hop_count": topology["hop_count"],
        "path_distance_km": topology["path_distance_km"],
    }


def build_traffic_context(
    data: TrafficData,
    incident_id: str,
    base_forecast,
    base_forecast_source: str,
    impact_scope: ImpactScopeConfig,
    llm_context: LLMContextConfig,
    node_index: int | None = None,
    layer_index: int = 0,
    previous_layer: list[int] | None = None,
    llm_predicted_flow: dict[int, list[float]] | None = None,
    incident_retrieval_min_score: float = 0.8,
    similar_incident_vector_candidates: int = 50,
    normal_reference_top_k: int = 10,
    normal_reference_min_score: float = 0.6,
    episodes_path: str | None = None,
    episode_memory_top_k: int = 3,
    episode_search_window_days: int = 120,
    episode_recency_half_life_days: int = 30,
    episode_min_similarity: float = DEFAULT_MEMORY_MIN_SIMILARITY,
    episode_min_base_mae: float = 0.0,
    episode_min_failure_delta: float = 10.0,
    normal_reference_base_forecast_provider=None,
    no_incident_history_index: NoIncidentHistoryIndex | None = None,
    incident_vector_index: IncidentVectorIndex | None = None,
    no_incident_local_radius_km: float = 5.0,
    incident_report_delay_buffer_minutes: int = 30,
    unknown_incident_duration_minutes: int = 60,
    build_all_evidence: bool = False,
    timings=None,
    timing_metadata: dict | None = None,
    incident_network_time_top_k: int = 3,
    incident_history_top_k: int = 3,
    incident_base_normal_top_k: int = 3,
    **canonical_kwargs,
) -> dict:
    """Build the node context consumed by Context-Aware Phased Reasoning.

    Canonical paper-facing keywords are accepted alongside the historical
    API names.  They are normalized before retrieval and do not alter any
    computation or persisted artifact.
    """
    foundation_model_prediction_input = canonical_kwargs.pop(
        "foundation_model_prediction", base_forecast
    )
    foundation_model_source = canonical_kwargs.pop(
        "foundation_model_source", base_forecast_source
    )
    normal_counterfactual_provider = canonical_kwargs.pop(
        "normal_counterfactual_provider", normal_reference_base_forecast_provider
    )
    previous_layer_predictions = canonical_kwargs.pop(
        "previous_layer_predictions", previous_layer
    )
    previous_layer_prediction_values = canonical_kwargs.pop(
        "previous_layer_prediction_values", llm_predicted_flow
    )
    no_incident_history_index = canonical_kwargs.pop(
        "normal_counterfactual_index", no_incident_history_index
    )
    incident_vector_index = canonical_kwargs.pop(
        "historical_incident_retrieval_index", incident_vector_index
    )
    episode_memory_top_k = canonical_kwargs.pop(
        "experience_trajectory_top_k", episode_memory_top_k
    )
    normal_reference_top_k = canonical_kwargs.pop(
        "normal_counterfactual_top_k", normal_reference_top_k
    )
    normal_reference_min_score = canonical_kwargs.pop(
        "normal_counterfactual_min_score", normal_reference_min_score
    )
    if canonical_kwargs:
        raise TypeError(f"Unexpected context arguments: {sorted(canonical_kwargs)}")
    timing_metadata = dict(timing_metadata or {})
    matches = data.incidents.loc[data.incidents["incident_id"].astype(str) == str(incident_id)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one incident: incident_id={incident_id}, matches={len(matches)}")
    incident = matches.iloc[0]
    osrm_scope = data.osrm_impact_scope(incident, impact_scope)
    incident_node_index = osrm_scope["anchor_node_index"]
    if incident_node_index is None:
        raise ValueError(f"No cached OSRM route is inside the impact distance bands: incident_id={incident_id}")
    if node_index is None:
        node_index = incident_node_index
    node_index = int(node_index)
    window = data.traffic_window(incident["dt_parsed"], node_index)
    target_node_id = int(data.node_order[node_index])
    osrm_route = data.osrm_incident_node_route(incident, node_index)
    foundation_model_prediction_enabled = bool(llm_context.foundation_model_prediction)
    normal_enabled = llm_context.is_enabled(NORMAL_COUNTERFACTUAL_SEQUENCE)
    enabled_context_key_set = set(llm_context.enabled_keys)
    if not foundation_model_prediction_enabled:
        enabled_context_key_set.discard("episode_memory")
    enabled_context_keys = sorted(enabled_context_key_set)
    # Other context ablations remain model-facing projections. Base and Normal
    # are strict computation ablations: disabled signals must not be built or
    # consumed by retrieval, memory, or diagnostics.
    available_context_keys = sorted(
        LLM_CONTEXT_KEYS if build_all_evidence else llm_context.enabled_keys
    )
    available_canonical_context_keys = {
        canonical_context_key(key) for key in available_context_keys
    }
    previous_layer_predictions = (
        [] if previous_layer_predictions is None else previous_layer_predictions
    )
    previous_layer_prediction_values = (
        {} if previous_layer_prediction_values is None else previous_layer_prediction_values
    )
    if not foundation_model_prediction_enabled or foundation_model_prediction_input is None:
        target_base_forecast = []
    else:
        foundation_array = np.asarray(foundation_model_prediction_input)
        target_base_forecast = (
            foundation_array[node_index]
            if foundation_array.ndim >= 2
            else foundation_array
        )
    target_history_flow = [float(value) for value in window.history_flow]
    target_history_flow_missing = bool(target_history_flow) and all(
        value == 0.0 for value in target_history_flow
    )
    context = {
        "sample_id": int(incident["incident_id"]),
        "target_node_index": int(node_index),
        "target_node_id": target_node_id,
        "target_history_flow": target_history_flow,
        "traffic_history": target_history_flow,
        "target_history_flow_status": (
            "missing_all_zero" if target_history_flow_missing else "observed"
        ),
        "target_history_flow_status_explanation": (
            "sensor_failure_missing_records"
            if target_history_flow_missing
            else "sensor_records_available"
        ),
        "incident": {
            **incident_payload(incident, include_duration=False),
            "time_context": time_context(incident["dt_parsed"]),
            "nearest_sensor_node_index": int(incident_node_index),
            "nearest_sensor_node_id": int(data.node_order[incident_node_index]),
            "distance_layer": int(layer_index),
            "osrm_driving_route": {
                "distance_km": osrm_route["distance_km"],
                "duration_s": osrm_route["duration_s"],
                "bearing_deg": osrm_route["bearing_deg"],
            },
            "target_road_relation": {
                "driving_distance_km": osrm_route["distance_km"],
                "driving_duration_s": osrm_route["duration_s"],
                "bearing_deg": osrm_route["bearing_deg"],
                **osrm_route["road_metadata_relation"],
            },
            "topology_path": _compact_topology_path(_topology_payload(data, incident_node_index, node_index)),
        },
        "_llm_context_keys": enabled_context_keys,
        "_evaluation": {
            "y_true": window.future_flow,
        },
    }
    if foundation_model_prediction_enabled:
        if len(target_base_forecast) != 12:
            raise ValueError(
                "foundation_model_prediction is enabled but no 12-step Foundation Model Prediction was provided"
            )
        foundation_model_prediction = [float(value) for value in target_base_forecast]
        context[FOUNDATION_MODEL_PREDICTION] = foundation_model_prediction
        # Keep the serialized field used by existing evaluation artifacts.
        context["base_forecast"] = foundation_model_prediction
        context["incident"]["base_forecast_source"] = foundation_model_source
        context["_evaluation"]["base_forecast"] = foundation_model_prediction
    if target_history_flow_missing:
        context["_llm_context_keys"] = []
        add_legacy_context_aliases(context)
        return context
    if normal_enabled and NORMAL_COUNTERFACTUAL_SEQUENCE in available_canonical_context_keys:
        no_incident_history_index = no_incident_history_index or get_no_incident_history_index(data)
        with timed_section(timings, "context.no_incident_retrieval", **timing_metadata):
            normal_counterfactual_sequence = generate_normal_counterfactual_sequence(
                data,
                incident,
                node_index,
                top_k=int(normal_reference_top_k),
                min_score=float(normal_reference_min_score),
                base_forecast_provider=(
                    normal_counterfactual_provider
                    if foundation_model_prediction_enabled and normal_enabled else None
                ),
                index=no_incident_history_index,
                local_radius_km=float(no_incident_local_radius_km),
                report_delay_buffer_minutes=int(incident_report_delay_buffer_minutes),
                unknown_duration_minutes=int(unknown_incident_duration_minutes),
                current_base_forecast=(
                    [float(value) for value in target_base_forecast]
                    if foundation_model_prediction_enabled else None
                ),
                timings=timings,
                timing_metadata=timing_metadata,
            )
            context[NORMAL_COUNTERFACTUAL_SEQUENCE] = normal_counterfactual_sequence
            context["non_incident_normal_reference"] = normal_counterfactual_sequence
    if INCIDENT_PATTERNS in available_canonical_context_keys:
        with timed_section(timings, "context.incident_retrieval", **timing_metadata):
            incident_patterns = retrieve_historical_incident_patterns(
                data,
                incident,
                node_index,
                min_score=float(incident_retrieval_min_score),
                vector_index=incident_vector_index,
                vector_candidates=int(similar_incident_vector_candidates),
                incident_network_time_top_k=incident_network_time_top_k,
                incident_history_top_k=incident_history_top_k,
                incident_base_normal_top_k=incident_base_normal_top_k,
                normal_reference_top_k=int(normal_reference_top_k),
                normal_reference_min_score=float(normal_reference_min_score),
                no_incident_history_index=no_incident_history_index,
                no_incident_local_radius_km=float(no_incident_local_radius_km),
                report_delay_buffer_minutes=int(incident_report_delay_buffer_minutes),
                unknown_duration_minutes=int(unknown_incident_duration_minutes),
                query_base_forecast=(
                    [float(value) for value in target_base_forecast]
                    if foundation_model_prediction_enabled else None
                ),
                query_normal_flow=(
                    context.get(NORMAL_COUNTERFACTUAL_SEQUENCE, {}).get(
                        "sequence",
                        context.get("non_incident_normal_reference", {}).get("weighted_normal_flow"),
                    )
                    if normal_enabled else None
                ),
                base_forecast_provider=(
                    normal_counterfactual_provider
                    if foundation_model_prediction_enabled and normal_enabled else None
                ),
                include_normal_reference=normal_enabled,
                timings=timings,
                timing_metadata=timing_metadata,
            )
            context[INCIDENT_PATTERNS] = incident_patterns
            context["retrieve_similar_cases_top5"] = incident_patterns
    if PREVIOUS_LAYER_PREDICTIONS in available_canonical_context_keys:
        missing_predictions = [
            int(previous_node_index)
            for previous_node_index in previous_layer_predictions
            if int(previous_node_index) not in previous_layer_prediction_values
        ]
        if missing_predictions:
            raise ValueError(
                f"Previous layer predictions are missing: layer_index={layer_index}, "
                f"missing_node_indices={missing_predictions}"
            )
        connected_previous_layer_nodes = []
        for previous_node_index in previous_layer_predictions:
            previous_node_index = int(previous_node_index)
            has_predecessor, path_distance_km = data.has_network_predecessor(
                incident_node_index,
                previous_node_index,
                node_index,
            )
            if not has_predecessor:
                continue
            connected_previous_layer_nodes.append(
                {
                    "node_index": previous_node_index,
                    "node_id": int(data.node_order[previous_node_index]),
                    "edge_to_target": {
                        "direction": "previous_to_target",
                        "path_distance_km": path_distance_km,
                    },
                    "llm_predicted_flow": [float(value) for value in previous_layer_prediction_values[previous_node_index]],
                }
            )
        previous_layer_nodes = {
            "layer_index": int(layer_index),
            "previous_layer_node_ids": [int(data.node_order[int(index)]) for index in previous_layer_predictions],
            "connected_previous_layer_node_count": len(connected_previous_layer_nodes),
            "connected_previous_layer_nodes": [
                {
                    "node_id": node["node_id"],
                    "edge_to_target": node["edge_to_target"],
                    "llm_predicted_flow": node["llm_predicted_flow"],
                }
                for node in connected_previous_layer_nodes
            ],
        }
        context[PREVIOUS_LAYER_PREDICTIONS] = previous_layer_nodes
        context["previous_layer_nodes"] = previous_layer_nodes
    # The current episode-memory schema evaluates corrections against Base.
    # Suppress the component when Base is ablated rather than leaking stored
    # Base trajectories through retrieval or reflection.
    if foundation_model_prediction_enabled and EXPERIENCE_TRAJECTORIES in available_canonical_context_keys:
        episodes_path = str(DEFAULT_EPISODES_PATH) if episodes_path is None else episodes_path
        memory_context = {**context, "layer_index": int(layer_index)}
        experience_trajectories = retrieve_experience_trajectories(
            episodes_path,
            memory_context,
            top_k=int(episode_memory_top_k),
            search_window_days=int(episode_search_window_days),
            recency_half_life_days=int(episode_recency_half_life_days),
            min_similarity=float(episode_min_similarity),
            min_base_mae=float(episode_min_base_mae),
            min_failure_delta=float(episode_min_failure_delta),
        )
        context[EXPERIENCE_TRAJECTORIES] = experience_trajectories
        context["episode_memory"] = experience_trajectories
    add_legacy_context_aliases(context)
    return context
