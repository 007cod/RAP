from __future__ import annotations

import copy
import json
from difflib import SequenceMatcher
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.agents.non_incident_retrieval import (
    NoIncidentCandidatesError,
    NoIncidentHistoryIndex,
    get_no_incident_history_index,
)
from src.agents.incident_vector_index import (
    IncidentVectorIndex,
    get_incident_vector_index,
)
from src.agents.incident_history_index import get_incident_history_vector_index
from src.agents.history_similarity import (
    HISTORY_SIMILARITY_WEIGHTS,
    history_similarity_components,
)
from src.agents.paired_state import compare_foundation_normal_state
from src.agents.base_normal_selection import (
    BASE_NORMAL_GROUP,
    rank_base_normal_candidates,
)
from src.agents.terminology import (
    EXPERIENCE_TRAJECTORIES,
    FOUNDATION_MODEL_PREDICTION,
    INCIDENT_PATTERNS,
    NORMAL_COUNTERFACTUAL_SEQUENCE,
    PREVIOUS_LAYER_PREDICTIONS,
    TRAFFIC_HISTORY,
    canonical_context_key,
    canonicalize_model_context,
    FOUNDATION_NORMAL_STATE_SIMILARITY,
    INCIDENT_SIMILARITY,
    SEQUENCE_SIMILARITY,
)
from src.agents.time_context import time_context
from src.agents.json_utils import parse_json_object_response
from src.tool_llm.timing import timed_section

if TYPE_CHECKING:
    from src.data.traffic import TrafficData


MAIN_SYSTEM_PROMPT = """
You are an incident-aware traffic-flow forecasting agent.
Use only the supplied context and do not invent unavailable evidence.
Follow the task and the output requirements below.
""".strip()


# Keep the full prediction request synchronized with overleaf/prompts.tex.
MAIN_USER_TASK = """
Task
Forecast all 12 future five-minute steps for the target node from the supplied
context. Compare the evidence and competing future hypotheses, then construct
the flow sequence that best fits this sample. The resulting shape can be
non-uniform or non-linear when supported by the evidence.
""".strip()


MAIN_USER_EVIDENCE_GUIDANCE = """
Evidence principles
1. Current observations. The query is the current incident and target node;
   `traffic_history` and `candidate_history_flow` from a case are observed
   pre-forecast states, never future observations for the query.

2. Hypotheses. Foundation Model Prediction (`foundation_model_prediction`) is a
   history-conditioned baseline learned without an incident indicator; it is
   neither a strict no-incident counterfactual nor an incident-conditioned
   forecast and may already contain effects visible in the history. Normal
   counterfactual sequence (`normal_counterfactual_sequence`) is the conditional future
   of matched incident-free windows. Both are candidate descriptions of the
   future, not observations or truth. The signed Foundation--Normal gap is the
   Foundation Model Prediction minus the Normal counterfactual sequence at each
   step; `foundation_normal_gap.G` is its mean absolute separation, not its
   direction. The gap is derived, not independent evidence. Historical
   Foundation Model errors are errors on matched incident-free windows: horizon
   bands are step-wise error statistics, not phases, and positive bias means the
   Foundation Model Prediction overpredicted. Normal spread is historical
   variation, not calibrated confidence.

3. Historical outcomes. `historical_effect.observed_future_flow` and, when
   supplied, `paired_state.historical_outcome.observed_future_flow` are observed
   futures after a historical case pre-forecast state. `effect_residual` is that
   future minus the local no-incident reference for the case; paired outcomes also contain
   observed-minus-Foundation-Model and observed-minus-Normal residuals. Null observations do
   not establish accuracy. These residuals and before/after observations are
   uncertain analog evidence, never causal estimates. `trajectory_comparisons`
   separates signed level offsets, centered shape changes, and step changes;
   differences are in flow units, and gap differences (Foundation Model
   differences minus Normal differences) are not independent votes.

4. Historical Incident Retrieval evidence. Incident Patterns contain three
   evidence groups: `sequence_similarity`,
   `incident_similarity`, and `foundation_normal_state_similarity`,
   corresponding to Sequence Similarity, Incident Similarity, and
   Foundation--Normal State Similarity. They have different evidence roles. A case may inform only level,
   direction, amplitude, persistence, a turning point, or one interval; rank and
   similarity do not establish transferability. The normalized `retrieval_score`
   uses time (0.2), cached OSRM route (1.0), target history (1.0), and incident
   attributes (0.5). Target-history similarity uses wavelet-scattering shape (0.45), robust
   level (0.20), latest three steps (0.20), and changepoints (0.15). The
   Foundation--Normal state similarity compares the Foundation Model
   Prediction, Normal counterfactual sequence, and signed gap jointly and minimizes
   the worst normalized mismatch before the next worst, so one strong component
   cannot hide a severe mismatch. Group, rank, score, counts, and programmatic
   direction/shape aggregates are descriptive metadata, not votes or reliability
   estimates.

5. Physical and memory evidence. Cached OSRM driving route, road relation,
   topology, target history, and time context inform whether and when an incident
   could reach the target. Reachability is only a possible exposure and timing
   signal: it does not establish the sign, magnitude, or persistence of a flow
   effect. An incident adjustment needs compatible target-state outcomes or an
   already observed target change; route closeness alone is insufficient.
   Previous-layer predictions are weak cross-node evidence only: node-scale values
   differ, so an upstream level or trajectory is not the target forecast and does
   not prove propagation direction or size. Experience Trajectories are
   conditional historical forecasting evidence, not a prescription or causal
   estimate.

6. Rules shared by all sources. Evaluate every enabled source against the
   complete sample; preserve conflict, weak coverage, and competing explanations. Do not majority-vote,
   mechanically average cases, or treat a historical winner as decisive. When
   conflicting outcomes cannot be distinguished by observable pre-forecast
   features, keep the result unresolved rather than inventing a discriminator or
   selecting the most extreme case. Judge transferability from observable
   query-case features and its limitations. Preserve competing explanations
   when the evidence does not distinguish them. Infer freely: do not copy an
   existing trajectory, force the result between the Foundation Model Prediction
   and Normal, average them,
   or prefer either when analogs are weak. Context Analysis is the evidence
   consolidation step; Phased Prediction is the resulting horizon-structured
   forecast in the same LLM response.
""".strip()


MAIN_USER_REASONING = """
Reasoning process
1. Establish the current state from the complete target history: level, overall
   and recent trend, volatility, turning points, and likely persistence. Treat the
   latest direction as a hypothesis, then test whether it is likely to persist,
   reverse, or develop further against the full history, the Foundation Model
   Prediction, the Normal counterfactual sequence, and
   matched-case continuations to form the current-state hypothesis. The latest
   one-to-three observations alone must not determine future amplitude or persistence. Use time of day, weekday,
   comparable windows, and ordinary traffic evolution; the incident contribution
   may be small or unresolved.

2. Develop competing futures from the Foundation Model Prediction, Normal
   counterfactual sequence, signed Foundation--Normal gap, and structural case
   summaries. Compare them with the current-state hypothesis, use `G` and raw
   trajectories to locate differences, and use historical Foundation Model error
   summaries to judge matched-window reliability. Check the premise and
   history-match quality of the Normal counterfactual. Separate ordinary
   demand evolution from an incident effect: the
   cached OSRM driving route (`incident.osrm_driving_route.distance_km`,
   `duration_s`), road relation, and topology constrain possible reachability and
   timing, but do not determine effect sign or magnitude without compatible
   outcome evidence. Include target history and connected previous-layer predictions
   only with their transfer limits; never use straight-line distance. Report time
   may lag onset: do not assume a fixed delay or invent a time, and allow effects
   already visible in history.

3. Evaluate only retained Incident Patterns and Experience Trajectories. Inspect group scores, ranks,
   retrieval and OSRM components, `effect_shape`, `effect_residual`, direction,
   spread, per-step distributions, and `trajectory_comparisons`; inspect raw
   paths when summaries hide a turning point or local mismatch. For the Sequence
   Similarity group, compare `candidate_history_flow` with the query `traffic_history`, then its future continuation:
   turning points, short deviations, recovery, direction, ordering, persistence,
   relative magnitude, and timing. Adapt it to target level, recent dynamics,
   calendar, and physical evidence; road relation does not determine
   transferability alone. For every case, separate pre-forecast state similarity
   from outcome transfer: a mismatch in flow scale, timing, recent shape, or
   incident duration limits the transferred amplitude even when the route is
   close. For Incident Similarity cases, use `effect_residual` to assess a
   possible incident mechanism and interval transfer from incident/route context,
   reachability, node scale, and residuals. For Foundation--Normal State Similarity cases, compare
   the Foundation Model Prediction, Normal counterfactual sequence, and signed
   gap together before using the outcome. Judge each case
   independently by interval, allowing support or conflict; use memory with the
   same transfer limitations.

4. Synthesize plausible continuations from all enabled sources. For each
   hypothesis and case, identify supporting and contradicting conditions,
   historical performance, transferable features, and limits. Treat evidence as
   support, counterexample, or unresolved rather than a vote. Explain which
   observable query-case features transfer level, direction, persistence,
   magnitude, or turning time. When cases conflict, compare the strongest
   supporting case and counterexample, but use a distinguishing feature only when
   it is observable in the supplied pre-forecast state. If no such feature
   separates the outcomes, explicitly preserve the interval as unresolved. A
   single extreme historical outcome, a close route, or a high rank does not set
   the forecast amplitude or persistence. Select the explanation that best
   accounts for each interval from convergent evidence, while allowing competing
   explanations to remain unresolved.

5. Construct phases and flow values. Before assigning values, state internally
   whether an incident effect is established, directionally supported, or
   unresolved for each interval; do not invent an incident sign or magnitude
   when the evidence only establishes reachability. Divide steps 1--12 only where supported by
   the future evolution; the reported incident time is not a phase boundary.
   Relate evidence to each phase and connect direction and magnitude to the
   explanation. Before values, give a concise explanation naming material cases
   by incident ID and citing query/case features, timing or road context,
   Foundation--Normal/gap differences, historical performance, transfer value, limits,
   counterevidence, and uncertainty. Summarize decisions without repeating the
   full evidence review, then generate the flow sequence.
""".strip()


ABLATION_USER_EVIDENCE_GUIDANCE = """
Evidence principles
Historical outcomes are uncertain analog evidence, not causality or future truth.
""".strip()


DIRECT_PREDICTION_REASONING = """
5. Generate flow values. From the supported future evolution, connect predicted
   direction and magnitude to the selected evidence over the forecast horizon.
   Complete the evidence assessment internally and output only `predicted_flow`
   as required by the system message.
""".strip()


MAX_PHASE_COUNT = 4
DEFAULT_CONTEXT_TOOL_KEYS = {
    INCIDENT_PATTERNS,
    NORMAL_COUNTERFACTUAL_SEQUENCE,
    PREVIOUS_LAYER_PREDICTIONS,
    EXPERIENCE_TRAJECTORIES,
}
CANONICAL_CONTEXT_TOOL_KEYS = {
    INCIDENT_PATTERNS,
    NORMAL_COUNTERFACTUAL_SEQUENCE,
    PREVIOUS_LAYER_PREDICTIONS,
    EXPERIENCE_TRAJECTORIES,
}
DEFAULT_SIMILAR_CASE_LOOKBACK_MINUTES = 60
EFFECT_SHAPE_PHASE_COUNT = 4
EFFECT_SHAPE_DEADBAND_FRACTION = 0.02
RETRIEVAL_SIMILARITY_WEIGHTS = {
    "time": 0.2,
    "network": 1.0,
    "history": 1.0,
    "incident": 0.5,
}


RESPONSE_TEMPLATE = {
    "target_node_id": 0,
    "forecast_explanation": "...",
    # This is a shape example only. The model must replace the placeholder
    # phase with the phases supported by the evidence.
    "phase_analysis": [{
        "start_step": "<start_step>",
        "end_step": "<end_step>",
        "evidence_references": [],
        "explanation": "<phase explanation>",
        "predicted_flow": ["<one value per step in this phase>"],
    }],
}


def _context_for_llm(
    context: dict,
    historical_error_mode: str = "horizon",
    *,
    enabled_context_keys=None,
    include_base_prediction: bool = True,
    include_foundation_model_prediction: bool | None = None,
) -> dict:
    """Return the model-facing context without retrieval implementation metadata."""
    if include_foundation_model_prediction is not None:
        include_base_prediction = bool(include_foundation_model_prediction)
    compact = copy.deepcopy({k: v for k, v in context.items() if not k.startswith("_")})
    # The LLM boundary always uses the paper terminology.  Persisted records
    # may still be read elsewhere, but no legacy field names are exposed in
    # the model-facing context.
    compact = canonicalize_model_context(compact)
    normal_key = NORMAL_COUNTERFACTUAL_SEQUENCE
    patterns_key = INCIDENT_PATTERNS
    memory_key = EXPERIENCE_TRAJECTORIES
    history_key = TRAFFIC_HISTORY
    foundation_key = FOUNDATION_MODEL_PREDICTION
    if historical_error_mode not in {"global", "horizon"}:
        raise ValueError("historical_error_mode must be global or horizon")
    if enabled_context_keys is not None:
        enabled_context_keys = {canonical_context_key(key) for key in enabled_context_keys}
        for key in DEFAULT_CONTEXT_TOOL_KEYS - enabled_context_keys:
            compact.pop(key, None)
    normal_enabled = (
        True
        if enabled_context_keys is None
        else NORMAL_COUNTERFACTUAL_SEQUENCE in enabled_context_keys
    )
    if not include_base_prediction:
        compact.pop(FOUNDATION_MODEL_PREDICTION, None)
        compact.pop(EXPERIENCE_TRAJECTORIES, None)
        incident = compact.get("incident")
        if isinstance(incident, dict):
            incident.pop("base_forecast_source", None)
        normal = compact.get(normal_key)
        if isinstance(normal, dict):
            for key in (
                "foundation_normal_gap",
                "historical_foundation_model_error",
                "weighted_average_foundation_model_error",
                "foundation_model_residual_distribution",
            ):
                normal.pop(key, None)
            for case in normal.get("selection", {}).get("selected_cases", []):
                if isinstance(case, dict):
                    case.pop("mean_foundation_model_residual", None)
                    paired = case.get("paired_state")
                    if isinstance(paired, dict):
                        paired.pop("historical_foundation_model_mae", None)
    similar = compact.get(patterns_key)
    if isinstance(similar, dict) and (not include_base_prediction or not normal_enabled):
        selection = similar.get("selection", {})
        selection.get("groups", {}).pop(FOUNDATION_NORMAL_STATE_SIMILARITY, None)
        selection.pop("paired_state_candidate_audit", None)
        for case in selection.get("cases", []):
            if not isinstance(case, dict):
                continue
            case.pop("paired_state", None)
            case.pop("base_normal_match", None)
            if isinstance(case.get("source_groups"), list):
                case["source_groups"] = [
                    group for group in case["source_groups"] if group != FOUNDATION_NORMAL_STATE_SIMILARITY
                ]
            for key in ("group_scores", "group_ranks"):
                if isinstance(case.get(key), dict):
                    case[key].pop(FOUNDATION_NORMAL_STATE_SIMILARITY, None)
            if not normal_enabled:
                historical_effect = case.pop("historical_effect", None)
                if isinstance(historical_effect, dict) and historical_effect.get("observed_future_flow"):
                    case["historical_continuation"] = {
                        "definition": "observed future following the candidate history",
                        "observed_future_flow": historical_effect["observed_future_flow"],
                    }
        similar.pop("incident_correction_evidence", None)
        if not normal_enabled:
            similar.pop("effect_definition", None)
    if memory_key in compact:
        # Only local, verifiable evidence reaches the model. Diagnostics, paths,
        # rankings and full episode records remain in the saved raw context.
        memory = compact.get(memory_key) or {}
        compact[memory_key] = {"local_evidence": memory.get("local_evidence", [])}

    # Put the few signals needed for the first reasoning pass in one compact
    # block. Detailed retrieval metadata remains available for audit, but the
    # model can begin from this stable representation.
    high_value_signals = {TRAFFIC_HISTORY: compact.get(history_key, [])}
    if foundation_key in compact:
        high_value_signals[FOUNDATION_MODEL_PREDICTION] = compact[foundation_key]
    if normal_key in compact:
        normal_payload = compact[normal_key]
        if isinstance(normal_payload, dict):
            high_value_signals[NORMAL_COUNTERFACTUAL_SEQUENCE] = normal_payload.get("sequence", [])
    if patterns_key in compact:
        available_groups = set(
            compact[patterns_key].get("selection", {}).get("groups", {})
        )
        canonical_groups = available_groups
        if FOUNDATION_NORMAL_STATE_SIMILARITY in canonical_groups:
            high_value_signals["retrieval_structure"] = (
                "sequence_similarity cases describe pre-incident traffic-state continuation; "
                "incident_similarity cases describe incident-related deviation; "
                "foundation_normal_state_similarity cases compare jointly similar Foundation--Normal states. "
                "Group membership is not evidence strength or a vote for either forecast."
            )
        else:
            group_descriptions = []
            if SEQUENCE_SIMILARITY in canonical_groups:
                group_descriptions.append("sequence_similarity cases describe state continuation")
            if INCIDENT_SIMILARITY in canonical_groups:
                group_descriptions.append(
                    "incident_similarity cases describe incident-related evidence"
                )
            high_value_signals["retrieval_structure"] = (
                "; ".join(group_descriptions)
                + ". Group membership is not evidence strength or a forecast vote."
            )
    compact["high_value_signals"] = high_value_signals

    similar = compact.get(patterns_key)
    if isinstance(similar, dict):
        # The current incident already carries this relation under
        # ``incident``; avoid sending a second copy of the query geometry.
        similar.pop("query_spatial_relation", None)
        similar.pop("method", None)
        similar.pop("effect_definition", None)
        selection = similar.get("selection")
        if isinstance(selection, dict):
            for key in (
                "skipped_no_normal_reference_count",
                "eligible_historical_candidate_count",
                "scored_candidate_count",
                "vector_recall_candidate_count",
                "metadata_vector_recall_candidate_count",
                "history_vector_recall_candidate_count",
                "missing_osrm_route_count",
                "below_score_threshold_count",
                "selection_policy",
                "paired_state_candidate_audit",
            ):
                selection.pop(key, None)
            for case in selection.get("cases", []):
                if not isinstance(case, dict):
                    continue
                # The query/target route is already present in ``incident``;
                # repeating the same target and matched route for every case
                # adds tokens without changing the decision evidence.
                case.pop("target_node", None)
                case.pop("matched_spatial_relation", None)
                paired = case.get("paired_state")
                if isinstance(paired, dict):
                    # Keep exponential similarity scores for audit only.
                    paired.pop("similarity_to_query", None)
                    paired.pop("similarity_definition", None)
                diagnostics = case.get("retrieval_diagnostics")
                if isinstance(diagnostics, dict):
                    for key in (
                        "vector_recall_score", "vector_recall_rank",
                        "metadata_vector_score", "metadata_vector_rank",
                        "history_vector_score", "history_vector_rank",
                        "vector_recall_pool_size", "eligible_historical_candidate_count",
                        "scored_candidate_count", "skipped_incomplete_window_count",
                        "similarity_weights", "rerank_score",
                        "history_similarity_definition", "network_similarity_components",
                    ):
                        diagnostics.pop(key, None)
                effect = case.get("historical_effect")
                if isinstance(effect, dict):
                    effect.pop("definition", None)
        # Programmatic direction/shape aggregates are retained in raw context
        # for auditing, but deliberately hidden from the model.  Classification
        # retrieval creates heterogeneous evidence; only the LLM should decide
        # which individual cases transfer to each forecast phase.
        similar.pop("incident_correction_evidence", None)
        available_groups = set(similar.get("selection", {}).get("groups", {}))
        full_paired_evidence = FOUNDATION_NORMAL_STATE_SIMILARITY in available_groups
        if full_paired_evidence:
            semantics = {
                "effect_residual": "observed future flow minus local no-incident reference",
                "retrieval_score": "weighted time, OSRM route, history, and incident similarity",
                "group_scores": {
                    INCIDENT_SIMILARITY: "incident and route-context similarity used for that retrieval group",
                    SEQUENCE_SIMILARITY: "observed traffic-state and recent-dynamics similarity used for that retrieval group",
                    FOUNDATION_NORMAL_STATE_SIMILARITY: "1/(1+worst normalized mismatch); descriptive match score, not probability",
                },
                "group_ranks": "within-group relative match rank, not reliability or forecast preference",
                "foundation_normal_state_match": "Compare the Foundation Model Prediction, Normal counterfactual sequence, and signed gap levels/trajectories and step changes jointly; "
                "ranking minimizes the worst mismatch first. Rank one may still be dissimilar. "
                "The gap is derived from the Foundation Model Prediction and Normal, not an independent vote.",
                "evidence_group_roles": {
                    FOUNDATION_NORMAL_STATE_SIMILARITY: {
                        "role": "comparable model/reference states and their historical outcomes",
                        "primary_fields": ["paired_state.trajectory_comparisons", "paired_state.historical_outcome"],
                    },
                    SEQUENCE_SIMILARITY: {
                        "role": "traffic-state continuation evidence",
                        "primary_fields": [
                            "candidate_history_flow",
                            "historical_effect.observed_future_flow",
                        ],
                    },
                    INCIDENT_SIMILARITY: {
                        "role": "incident-related deviation evidence",
                        "primary_fields": ["historical_effect.effect_residual"],
                    },
                },
            }
        else:
            semantics = {
                "retrieval_score": "weighted time, OSRM route, history, and incident similarity",
                "group_scores": {
                    group: description
                    for group, description in (
                        (INCIDENT_SIMILARITY, "incident and route-context similarity used for that retrieval group"),
                        (SEQUENCE_SIMILARITY, "observed traffic-state and recent-dynamics similarity used for that retrieval group"),
                        (FOUNDATION_NORMAL_STATE_SIMILARITY, "1/(1+worst normalized mismatch); descriptive match score, not probability"),
                    )
                    if group in available_groups
                },
                "group_ranks": "within-group relative match rank, not reliability or forecast preference",
                "evidence_group_roles": {
                    **({FOUNDATION_NORMAL_STATE_SIMILARITY: {
                        "role": "comparable model/reference states and their historical outcomes",
                        "primary_fields": ["paired_state.trajectory_comparisons", "paired_state.historical_outcome"],
                    }} if FOUNDATION_NORMAL_STATE_SIMILARITY in available_groups else {}),
                    **({SEQUENCE_SIMILARITY: {
                        "role": "traffic-state continuation evidence",
                        "primary_fields": [
                            "candidate_history_flow",
                            (
                                "historical_effect.observed_future_flow"
                                if "historical_effect" in next(iter(similar.get("selection", {}).get("cases", [])), {})
                                else "historical_continuation.observed_future_flow"
                            ),
                        ],
                    }} if SEQUENCE_SIMILARITY in available_groups else {}),
                    **({INCIDENT_SIMILARITY: {
                        "role": "incident-related deviation evidence",
                        "primary_fields": [
                            "historical_effect.effect_residual"
                            if "historical_effect" in next(iter(similar.get("selection", {}).get("cases", [])), {})
                            else "historical_continuation.observed_future_flow"
                        ],
                    }} if INCIDENT_SIMILARITY in available_groups else {}),
                },
            }
            if any("historical_effect" in case for case in similar.get("selection", {}).get("cases", [])):
                semantics["effect_residual"] = "observed future flow minus local no-incident reference"
        similar["evidence_semantics"] = semantics
        available_group_names = set(
            similar.get("selection", {}).get("groups", {})
        )
        similar["evidence_group_names"] = sorted(available_group_names)

    normal = compact.get(normal_key)
    if isinstance(normal, dict):
        profile = normal.pop("historical_foundation_model_error", None)
        normal.pop("foundation_model_residual_distribution", None)
        if historical_error_mode == "horizon":
            normal.pop("weighted_average_foundation_model_error", None)
            if profile:
                normal["historical_foundation_model_error"] = {
                    "bands": profile["bands"], "bias_definition": profile["bias_definition"],
                }
        for key in (
            "method",
            "local_incident_filter",
            "eligible_candidate_count",
            "exact_rerank_candidate_count",
            "coarse_candidate_filter",
            "weighted_normal_flow_definition",
            "weighted_average_foundation_model_error_definition",
        ):
            normal.pop(key, None)
        gap = normal.get("foundation_normal_gap")
        if isinstance(gap, dict):
            gap.pop("definition", None)

        # Candidate-level score components are retrieval diagnostics.  Keep a
        # small audit summary, while exposing the aggregate trajectory and
        # uncertainty below as the model-facing evidence.
        selection = normal.get("selection")
        if isinstance(selection, dict):
            selected_cases = selection.get("selected_cases")
            if isinstance(selected_cases, list):
                compact_cases = []
                for case in selected_cases:
                    if not isinstance(case, dict):
                        continue
                    compact_cases.append(
                        {
                            key: case[key]
                            for key in ("rank", "sample_time", "retrieval_score", "mean_future_flow")
                            if key in case
                        }
                    )
                selection["selected_cases"] = compact_cases

        # Per-candidate base residuals and per-step standard deviations repeat
        # the same uncertainty already represented by the aggregate gap and
        # quantiles.  Retain the directional distribution only.
        residuals = normal.get("foundation_model_residual_distribution")
        if isinstance(residuals, dict):
            for key in ("weighted_std",):
                residuals.pop(key, None)
        # These values describe how closely the historical input windows match;
        # naming them as generic evidence quality makes the aggregate future
        # appear more authoritative than Base or the incident cases.
        history_match_quality = normal.pop("evidence_quality", None)
        if isinstance(history_match_quality, dict):
            # Do not allow obsolete aliases to conflict with retained-weight metrics.
            history_match_quality.pop("mean_retrieval_score", None)
            history_match_quality.pop("reference_confidence", None)
            normal["history_match_quality"] = history_match_quality
        # This pre-interprets differences from the latest observation as
        # recovery evidence.  The raw history and trajectories already let the
        # model infer the state without favoring a recovery narrative.
        normal.pop("recovery_evidence", None)
        reference_semantics = {
            "source": "same target node; matched incident-free windows",
            "sequence": "conditional future if the target continues like the matched incident-free windows",
            "history_match_quality": "similarity of the matched historical inputs, not correctness of their future for this incident",
        }
        if "foundation_normal_gap" in normal:
            reference_semantics["G"] = (
                "mean absolute per-step separation between the Foundation Model Prediction and the Normal counterfactual sequence"
            )
        if "historical_foundation_model_error" in normal:
            reference_semantics["historical_foundation_model_error"] = (
                "Foundation Model Prediction errors on matched normal windows; horizon bands are statistical "
                "intervals, not incident phases or current-future accuracy"
            )
        normal["reference_semantics"] = reference_semantics

    # The model-facing payload uses the paper terminology.  Legacy names are
    # retained only in the raw audit context and in persisted historical
    # artifacts; this conversion does not alter any evidence values.
    rounded = _round_float_values(compact)
    return rounded


def _round_float_values(value):
    """Keep numeric values in the LLM context and response at two decimals."""
    if isinstance(value, dict):
        return {key: _round_float_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_float_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_round_float_values(item) for item in value)
    if isinstance(value, Real) and not isinstance(value, (bool, int)):
        return round(float(value), 2)
    return value


def build_chat_messages(
    context: dict,
    tool_keys: list[str],
    *,
    historical_error_mode: str = "horizon",
    include_base_prediction: bool = True,
    phased_prediction: bool = True,
    include_foundation_model_prediction: bool | None = None,
) -> list[dict]:
    """Build one request with shared section order and configuration-specific evidence."""
    if include_foundation_model_prediction is not None:
        include_base_prediction = bool(include_foundation_model_prediction)
    if len(set(tool_keys)) != len(tool_keys):
        raise ValueError(f"tool_keys must not contain duplicates, got {tool_keys}")
    tool_keys = [canonical_context_key(key) for key in tool_keys]
    normal_available = NORMAL_COUNTERFACTUAL_SEQUENCE in tool_keys

    if phased_prediction:
        template = {**RESPONSE_TEMPLATE, "target_node_id": context["target_node_id"]}
        response_contract = f"""
- Return one JSON object with exactly `target_node_id`, `forecast_explanation`,
  and `phase_analysis`, in that order.
- `forecast_explanation` is a brief natural-language string describing the key
  evidence and important uncertainty. Mention cases when they materially
  affect the decision; no separate analytical fields or candidate tables are
  required.
- `phase_analysis` is the final field and has 1--{MAX_PHASE_COUNT} contiguous phases covering
  steps 1--12 without gaps or overlap. Every phase contains
  `start_step`, `end_step`, `evidence_references`, `explanation`, and
  `predicted_flow`.
  These are the following four semantic parts, in this order: `Range times`
  (`start_step` and `end_step`, inclusive); `Evidence refs`
  (`evidence_references`, a JSON list identifying supplied evidence that materially
  supports the phase); `Explanation` (`explanation`, a concise account of how
  that evidence supports or limits the phase); and `Prediction`
  (`predicted_flow`, the phase's flow values). Include all four parts in every
  phase; use an empty list only when no specific source applies for
  `evidence_references`. Predicted values are finite, non-negative, exactly the
  phase length, and rounded to at most two decimals.
- Do not rename keys, add wrappers or extra top-level keys, return a second JSON
  object, or append another explanation after the phase predictions. The phase
  object in the skeleton is illustrative only: replace its placeholders with
  one or more evidence-supported phases covering the full horizon.
""".strip()
    else:
        template = {"predicted_flow": []}
        response_contract = """
- Return one JSON object containing exactly one field: `predicted_flow`.
- `predicted_flow` must be a JSON list of exactly 12 finite, non-negative flow
  values in chronological order, rounded to at most two decimals.
- Do not output a target ID, explanation, phases, evidence references, analysis,
  wrappers, markdown, commentary, or any other field or text.
""".strip()

    if include_base_prediction and normal_available:
        evidence_guidance = MAIN_USER_EVIDENCE_GUIDANCE
        reasoning = MAIN_USER_REASONING
        if not phased_prediction:
            # Keep the same evidence and reasoning for the output-format ablation.
            # Only omit phase construction and the request for a written explanation.
            reasoning = reasoning.split("\n5. Construct phases and flow values.", 1)[0].rstrip()
            reasoning += "\n\n" + DIRECT_PREDICTION_REASONING
    else:
        # Base/Normal ablations must not describe or request disabled evidence.
        evidence_rules = [
            "`traffic_history` is the complete observed traffic history, not future truth.",
            "Incident, OSRM route, time, and road relation describe physical context for possible transfer.",
        ]
        if include_base_prediction:
            evidence_rules.append(
                "Foundation Model Prediction denotes `foundation_model_prediction`, a learned history-conditioned continuation, not future truth."
            )
        if normal_available:
            evidence_rules.append(
                "Normal counterfactual sequence denotes `normal_counterfactual_sequence`, the conditional continuation of matched incident-free windows, not future truth."
            )
        if INCIDENT_PATTERNS in tool_keys:
            if normal_available:
                evidence_rules.append(
                    "Retrieved incidents provide observable pre-forecast states and physical context; "
                    "`historical_effect` describes historical outcomes and residuals whose transfer depends on those features."
                )
            else:
                evidence_rules.append(
                    "Retrieved incidents provide observable pre-forecast states and physical context; "
                    "`historical_continuation.observed_future_flow` is a historical continuation, not a vote or direct target forecast."
                )
        if PREVIOUS_LAYER_PREDICTIONS in tool_keys:
            evidence_rules.append(
                "Connected previous-layer predictions are weak propagation evidence; nodes can have different flow scales."
            )
        if include_base_prediction and EXPERIENCE_TRAJECTORIES in tool_keys:
            evidence_rules.append(
                "Experience Trajectories are conditional historical evidence with observable-state requirements and transfer limitations."
            )
        evidence_guidance = ABLATION_USER_EVIDENCE_GUIDANCE + "\n" + "\n".join(
            f"- {rule}" for rule in evidence_rules
        )
        reasoning = """
Reasoning process
1. Establish the current state. Use the complete observed target history to
   infer level, recent dynamics, turning points, and persistence.
2. Develop competing future hypotheses from the supplied evidence. Use incident,
   OSRM route, time, road relation, and any supplied connected-node evidence
   only when its values support transfer; account for node-scale differences.
3. Evaluate supplied historical evidence. Compare observable pre-forecast state
   and physical context; use historical continuations or residuals only where
   those features support transfer to the current decision interval. Consider
   any supplied episode memory's observable-state match and transfer limitations.
4. Synthesize support, conflict, and uncertainty. Explain which currently
   observable features make an analog transferable, including its transfer
   value and limitations. Preserve competing explanations when evidence
   conflicts. Infer the trajectory freely; do not majority-vote, mechanically
   average cases, or copy a historical continuation.
""".strip()
        if phased_prediction:
            reasoning += """

5. Construct phases and flow values. Select the trajectory that best accounts
   for each forecast phase. Record the key evidence decisions and material
   uncertainty in the brief explanation before generating values; describe an
   analog's transfer value and limitations when it materially affects a phase.
""".rstrip()
        else:
            reasoning += "\n\n" + DIRECT_PREDICTION_REASONING

    llm_context = _context_for_llm(
        context,
        historical_error_mode,
        enabled_context_keys=tool_keys,
        include_base_prediction=include_base_prediction,
        include_foundation_model_prediction=include_base_prediction,
    )
    user_content = f"""{MAIN_USER_TASK}

{evidence_guidance}

{reasoning}

Input context
{json.dumps(llm_context, ensure_ascii=False, indent=2)}"""
    system_prompt = f"""{MAIN_SYSTEM_PROMPT}

Output requirements
{response_contract}

Required JSON top-level skeleton
{json.dumps(template, ensure_ascii=False, indent=2)}"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def has_full_window(data: TrafficData, timestamp) -> bool:
    month, step = data.month_step(timestamp)
    array = data.month_arrays[month]
    return step - 11 >= 0 and step + 13 <= array.shape[1]


def incident_payload(
    incident,
    include_duration: bool,
) -> dict:
    payload = {
        "incident_id": str(incident.incident_id),
        "time": str(incident.dt_parsed),
        "description": str(incident.DESCRIPTION),
        "type": str(incident.Type),
        "area": str(incident.AREA),
        "location": str(incident.LOCATION),
        "freeway": str(incident.Fwy),
        "direction": str(incident.Freeway_direction),
        "latitude": float(incident.Latitude_num),
        "longitude": float(incident.Longitude_num),
    }
    if include_duration:
        payload["duration_minutes"] = float(incident.duration_minutes)
    else:
        payload["duration_status"] = "unknown_at_prediction_time"
    return payload


def incident_to_node_relation(data: TrafficData, incident, node_index: int) -> dict:
    node = data.sensors.iloc[int(node_index)]
    distance_km = data.incident_node_distance_km(
        float(incident.Latitude_num),
        float(incident.Longitude_num),
        int(node_index),
    )
    bearing_deg = _bearing_deg(
        float(incident.Latitude_num),
        float(incident.Longitude_num),
        float(node["Lat"]),
        float(node["Lng"]),
    )
    return {
        "distance_km": float(distance_km),
        "bearing_deg": float(bearing_deg),
        "target_freeway": str(node["Fwy"]),
        "target_road_direction": str(node["Direction"]),
    }


def retrieve_similar_cases(
    data: TrafficData,
    query_incident,
    node_index: int,
    min_score: float = 0.8,
    vector_index: IncidentVectorIndex | None = None,
    history_vector_index=None,
    vector_candidates: int = 50,
    normal_reference_top_k: int = 10,
    no_incident_history_index: NoIncidentHistoryIndex | None = None,
    no_incident_local_radius_km: float = 5.0,
    report_delay_buffer_minutes: int = 30,
    unknown_duration_minutes: int = 60,
    normal_reference_min_score: float = 0.6,
    query_base_forecast: list[float] | None = None,
    query_normal_flow: list[float] | None = None,
    base_forecast_provider=None,
    include_normal_reference: bool = True,
    timings=None,
    timing_metadata: dict | None = None,
    incident_network_time_top_k: int = 3,
    incident_history_top_k: int = 3,
    incident_base_normal_top_k: int = 3,
) -> dict:
    """Recall candidates and select each evidence group's configured top-k."""
    timing_metadata = dict(timing_metadata or {})
    include_normal_reference = bool(include_normal_reference)
    for name, value in (
        ("incident_network_time_top_k", incident_network_time_top_k),
        ("incident_history_top_k", incident_history_top_k),
        ("incident_base_normal_top_k", incident_base_normal_top_k),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    minimum_score = float(min_score)
    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError(f"min_score must be between 0 and 1, got {minimum_score}")
    query_time = query_incident.dt_parsed
    cutoff_time = query_time - pd.Timedelta(minutes=DEFAULT_SIMILAR_CASE_LOOKBACK_MINUTES)
    query_route = data.osrm_incident_node_route(query_incident, int(node_index))
    reference_time_context = time_context(query_time)
    query_window = data.traffic_window(query_time, node_index)
    query_base = np.asarray(query_base_forecast, dtype=np.float64) if query_base_forecast is not None else None
    query_normal = np.asarray(query_normal_flow, dtype=np.float64) if query_normal_flow is not None else None
    if query_base is not None and (query_base.shape != (12,) or not np.isfinite(query_base).all()):
        raise ValueError("query_base_forecast must contain 12 values")
    if query_normal is not None and (query_normal.shape != (12,) or not np.isfinite(query_normal).all()):
        raise ValueError("query_normal_flow must contain 12 values")
    if not include_normal_reference and query_normal is not None:
        raise ValueError("query_normal_flow must be omitted when Normal is disabled")
    paired_state_enabled = (
        include_normal_reference
        and query_base is not None
        and query_normal is not None
        and base_forecast_provider is not None
    )
    vector_index = vector_index or get_incident_vector_index(data)
    with timed_section(timings, "incident.vector_recall", **timing_metadata):
        metadata_recalled_cases = vector_index.recall(
            query_incident,
            cutoff_time,
            top_k=int(vector_candidates),
        )
    # History dynamics have an independent ANN index.  Fall back only for
    # lightweight fixtures that cannot materialize the index.
    history_recalled_cases = []
    with timed_section(timings, "incident.history_vector_recall", **timing_metadata):
        try:
            history_vector_index = history_vector_index or get_incident_history_vector_index(data, vector_index)
            history_recalled_cases = history_vector_index.recall(
                query_window.history_flow, cutoff_time, int(node_index), top_k=int(vector_candidates)
            )
        except Exception:
            history_vector_index = None
    recalled_by_position = {}
    for rank, (position, score) in enumerate(metadata_recalled_cases, start=1):
        recalled_by_position.setdefault(int(position), {}).update(
            metadata_vector_score=float(score), metadata_vector_rank=int(rank)
        )
    for rank, (position, score) in enumerate(history_recalled_cases, start=1):
        recalled_by_position.setdefault(int(position), {}).update(
            history_vector_score=float(score), history_vector_rank=int(rank)
        )
    recalled_cases = [
        (position, values.get("metadata_vector_score", values.get("history_vector_score", 0.0)))
        for position, values in recalled_by_position.items()
    ]
    rows = []
    skipped_incomplete_window_count = 0
    missing_osrm_route_count = 0
    historical_candidates = data.incidents.loc[data.incidents["dt_parsed"] <= cutoff_time]
    if include_normal_reference:
        no_incident_history_index = no_incident_history_index or get_no_incident_history_index(data)

    with timed_section(timings, "incident.candidate_window_scoring", **timing_metadata):
        for row_position, vector_recall_score in recalled_cases:
            vector_meta = recalled_by_position[int(row_position)]
            incident = data.incidents.iloc[int(row_position)]
            if not has_full_window(data, incident.dt_parsed):
                skipped_incomplete_window_count += 1
                continue
            candidate_node_index = int(node_index)
            candidate_window = data.traffic_window(incident.dt_parsed, candidate_node_index)
            incident_similarity = _incident_similarity(query_incident, incident)
            candidate_time_context = time_context(incident.dt_parsed)
            time_similarity = _time_similarity(reference_time_context, candidate_time_context)
            history_components = history_similarity_components(
                query_window.history_flow,
                candidate_window.history_flow,
            )
            history_similarity = history_components["combined"]
            try:
                candidate_route = data.osrm_incident_node_route(incident, candidate_node_index)
            except KeyError:
                candidate_route = None
                missing_osrm_route_count += 1
            network_similarity, network_components = _network_similarity(query_route, candidate_route)
            total_similarity = _combine_retrieval_similarity(
                incident_similarity=incident_similarity,
                time_similarity=time_similarity,
                history_similarity=history_similarity,
                network_similarity=network_similarity,
            )
            rows.append(
                {
                    "incident_id": str(incident.incident_id),
                    "_timestamp": pd.Timestamp(incident.dt_parsed),
                    "_incident_record": incident,
                    "_target_node_index": candidate_node_index,
                    "_history_flow": candidate_window.history_flow,
                    "_future_flow": candidate_window.future_flow,
                    "retrieval_score": total_similarity,
                    "retrieval_diagnostics": {
                        "incident_similarity": float(incident_similarity),
                        "time_similarity": float(time_similarity),
                        "history_similarity": float(history_similarity),
                        "history_similarity_components": history_components,
                        "history_similarity_definition": {
                            "method": "wavelet_scattering_level_recent_changepoint",
                            "weights": dict(HISTORY_SIMILARITY_WEIGHTS),
                            "wavelet": "Kymatio 1D scattering J=3 Q=(2,1) max_order=2",
                            "recent_steps": 3,
                        },
                        "network_similarity": float(network_similarity),
                        "metadata_vector_score": vector_meta.get("metadata_vector_score"),
                        "metadata_vector_rank": vector_meta.get("metadata_vector_rank"),
                        "history_vector_score": vector_meta.get("history_vector_score"),
                        "history_vector_rank": vector_meta.get("history_vector_rank"),
                        "network_similarity_components": network_components,
                    },
                    "incident": _compact_historical_incident(
                        incident_payload(incident, include_duration=True)
                    ),
                    "time_context": candidate_time_context,
                    "target_node": {
                        "node_id": int(data.node_order[candidate_node_index]),
                        "freeway": str(data.sensors.iloc[candidate_node_index]["Fwy"]),
                        "road_direction": str(data.sensors.iloc[candidate_node_index]["Direction"]),
                    },
                    "matched_spatial_relation": _route_relation_payload(
                        data, candidate_node_index, query_route, candidate_route, network_components
                    ),
                }
            )

    # Reuse bounded parallel Normal/disk caches and historical Base caches.
    # The new group consumes these existing annotations; no additional retrieval.
    if paired_state_enabled:
        with timed_section(timings, "incident.paired_state_evidence", **timing_metadata):
            query_scale = max(float(np.mean(np.abs(query_window.history_flow))), 1.0)
            def annotate(row):
                try:
                    candidate_normal = no_incident_history_index.retrieve(
                        data=data,
                        query_incident=row["_incident_record"],
                        node_index=row["_target_node_index"],
                        cutoff_time=row["_timestamp"] - pd.Timedelta(minutes=DEFAULT_SIMILAR_CASE_LOOKBACK_MINUTES),
                        top_k=int(normal_reference_top_k),
                        base_forecast_provider=None,
                        local_radius_km=float(no_incident_local_radius_km),
                        report_delay_buffer_minutes=int(report_delay_buffer_minutes),
                        unknown_duration_minutes=int(unknown_duration_minutes),
                        min_score=float(normal_reference_min_score),
                        current_base_forecast=None,
                        timings=timings,
                        timing_metadata={**timing_metadata, "historical_incident_id": row["incident_id"]},
                    )
                    candidate_normal_flow = np.asarray(candidate_normal["weighted_normal_flow"], dtype=np.float64)
                    candidate_base_flow = np.asarray(
                        base_forecast_provider(row["_timestamp"], int(row["_target_node_index"])), dtype=np.float64
                    )
                    row["_normal_reference"] = candidate_normal
                    row["paired_state_diagnostics"] = compare_foundation_normal_state(
                        query_base, query_normal, candidate_base_flow,
                        candidate_normal_flow, row["_future_flow"], query_scale,
                        query_history=query_window.history_flow,
                        candidate_history=row["_history_flow"],
                    )
                except NoIncidentCandidatesError:
                    row["paired_state_diagnostics"] = {
                        "status": "unavailable", "reason": "no_historical_normal_reference"
                    }

            runtime = getattr(no_incident_history_index, 'runtime', None)
            if runtime is not None:
                runtime.map(annotate, rows)
            else:
                for row in rows:
                    annotate(row)

    # Independent retrieval groups: network/time, history, and Base/Normal. Keep the
    # component-specific scores and the within-group ranks so the model can
    # distinguish traffic-state similarity from incident/network similarity.
    with timed_section(timings, "incident.rerank", **timing_metadata):
        for row in rows:
            d = row["retrieval_diagnostics"]
            row["network_time_score"] = float(
                0.55 * d["network_similarity"] + 0.25 * d["time_similarity"]
                + 0.20 * d["incident_similarity"]
            )
            row["history_score"] = float(d["history_similarity"])
        network_rows = sorted(rows, key=lambda x: (-x["network_time_score"], x["_timestamp"], x["incident_id"]))
        history_rows = sorted(rows, key=lambda x: (-x["history_score"], x["_timestamp"], x["incident_id"]))
        network_eligible = [r for r in network_rows if r["network_time_score"] >= minimum_score]
        history_eligible = [r for r in history_rows if r["history_score"] >= max(0.75, minimum_score)]
        if paired_state_enabled:
            with timed_section(timings, "incident.base_normal_rerank", **timing_metadata):
                state_rows = rank_base_normal_candidates(rows)
        else:
            state_rows = []
        state_rank_by_id = {row["incident_id"]: rank for rank, row in enumerate(state_rows, 1)}
        state_top_ids = {r["incident_id"] for r in state_rows[:incident_base_normal_top_k]}
        network_top_ids = {r["incident_id"] for r in network_eligible[:incident_network_time_top_k]}
        history_top_ids = {r["incident_id"] for r in history_eligible[:incident_history_top_k]}
        network_rank_by_id = {
            row["incident_id"]: int(rank)
            for rank, row in enumerate(network_eligible, start=1)
        }
        history_rank_by_id = {
            row["incident_id"]: int(rank)
            for rank, row in enumerate(history_eligible, start=1)
        }
        for row in rows:
            groups = []
            if row["incident_id"] in network_top_ids:
                groups.append("incident_network_time_similarity")
            if row["incident_id"] in history_top_ids:
                groups.append("history_similarity")
            # Preserve old group membership and ordering, append only new cases.
            row["_legacy_group_count"] = len(groups)
            if row["incident_id"] in state_top_ids:
                groups.append(BASE_NORMAL_GROUP)
            row["source_groups"] = groups
            row["group_scores"] = {
                "incident_network_time_similarity": float(row["network_time_score"]),
                "history_similarity": float(row["history_score"]),
            }
            if "base_normal_match" in row:
                row["group_scores"][BASE_NORMAL_GROUP] = 1.0 / (1.0 + row["base_normal_match"]["worst_normalized_distance"])
            row["group_ranks"] = {
                group: rank
                for group, rank in (
                    ("incident_network_time_similarity", network_rank_by_id.get(row["incident_id"])),
                    ("history_similarity", history_rank_by_id.get(row["incident_id"])),
                    (BASE_NORMAL_GROUP, state_rank_by_id.get(row["incident_id"])),
                )
                if rank is not None
            }
        rows.sort(key=lambda x: (
            -x["_legacy_group_count"],
            -x["retrieval_score"] if x["_legacy_group_count"] else state_rank_by_id.get(x["incident_id"], len(rows) + 1),
            x["_timestamp"], x["incident_id"],
        ))
    threshold_rows = [row for row in rows if row.get("source_groups")]
    # Legacy aggregate-score count is audit-only; group selection above is the
    # authoritative candidate policy.
    cases = []
    skipped_no_normal_reference_count = 0
    with timed_section(timings, "incident.case_outcomes", **timing_metadata):
        for row in threshold_rows:
            rank = len(cases) + 1
            historical_time = row["_timestamp"]
            historical_timing_metadata = {
                **timing_metadata,
                "historical_incident_id": str(row["incident_id"]),
                "historical_node_index": int(row["_target_node_index"]),
                "historical_rank": int(rank),
            }
            observed = np.asarray(row["_future_flow"], dtype=np.float64)
            if not include_normal_reference:
                cases.append(
                    {
                        "rank": int(rank),
                        "incident_id": row["incident_id"],
                        "retrieval_score": float(row["retrieval_score"]),
                        "source_groups": list(row.get("source_groups", [])),
                        "group_scores": dict(row.get("group_scores", {})),
                        "group_ranks": dict(row.get("group_ranks", {})),
                        "candidate_history_flow": [round(float(v), 2) for v in row["_history_flow"]],
                        "retrieval_diagnostics": dict(row["retrieval_diagnostics"]),
                        "incident": row["incident"],
                        "time_context": row["time_context"],
                        "target_node": row["target_node"],
                        "matched_spatial_relation": row["matched_spatial_relation"],
                        "historical_continuation": {
                            "definition": "observed future following the candidate history",
                            "observed_future_flow": [float(value) for value in observed],
                        },
                    }
                )
                continue
            normal_reference = row.get("_normal_reference")
            if normal_reference is None:
                try:
                    normal_reference = no_incident_history_index.retrieve(
                        data=data,
                        query_incident=row["_incident_record"],
                        node_index=row["_target_node_index"],
                        cutoff_time=historical_time
                        - pd.Timedelta(minutes=DEFAULT_SIMILAR_CASE_LOOKBACK_MINUTES),
                        top_k=int(normal_reference_top_k),
                        base_forecast_provider=None,
                        local_radius_km=float(no_incident_local_radius_km),
                        report_delay_buffer_minutes=int(report_delay_buffer_minutes),
                        unknown_duration_minutes=int(unknown_duration_minutes),
                        min_score=float(normal_reference_min_score),
                        current_base_forecast=None,
                        timings=timings,
                        timing_metadata=historical_timing_metadata,
                    )
                except NoIncidentCandidatesError:
                    # Very early historical incidents can predate the first complete
                    # traffic window. They cannot supply a counterfactual effect.
                    skipped_no_normal_reference_count += 1
                    continue
            counterfactual = np.asarray(normal_reference["weighted_normal_flow"], dtype=np.float64)
            effect = observed - counterfactual
            direction, threshold = _effect_direction(effect, counterfactual)
            cases.append(
                {
                    "rank": int(rank),
                    "incident_id": row["incident_id"],
                    "retrieval_score": float(row["retrieval_score"]),
                    "source_groups": list(row.get("source_groups", [])),
                    "group_scores": dict(row.get("group_scores", {})),
                    "group_ranks": dict(row.get("group_ranks", {})),
                    **({
                        "paired_state": row["paired_state_diagnostics"],
                        "base_normal_match": row.get("base_normal_match"),
                    } if paired_state_enabled else {}),
                    "candidate_history_flow": [round(float(v), 2) for v in row["_history_flow"]],
                    "retrieval_diagnostics": {
                        **row["retrieval_diagnostics"],
                    },
                    "incident": row["incident"],
                    "time_context": row["time_context"],
                    "target_node": row["target_node"],
                    "matched_spatial_relation": row["matched_spatial_relation"],
                    "historical_effect": {
                        "definition": "observed_future_flow - local_no_incident_normal_counterfactual",
                        "normal_counterfactual_flow": [float(value) for value in counterfactual],
                        "observed_future_flow": [float(value) for value in observed],
                        "effect_residual": [float(value) for value in effect],
                        "mean_effect": float(np.mean(effect)),
                        "direction": direction,
                        "neutral_threshold": float(threshold),
                        "effect_shape": _effect_shape(effect, counterfactual),
                        "normal_reference_quality": normal_reference["evidence_quality"],
                    },
                }
            )
    evidence = None
    if include_normal_reference:
        with timed_section(timings, "incident.effect_aggregation", **timing_metadata):
            evidence = _aggregate_incident_effect_evidence(cases)
    if timings is not None:
        timings.record(
            "incident.counters",
            0.0,
            {
                **timing_metadata,
                "eligible_historical_candidate_count": int(len(historical_candidates)),
                "scored_candidate_count": int(len(rows)),
                "metadata_vector_recall_candidate_count": int(len(metadata_recalled_cases)),
                "history_vector_recall_candidate_count": int(len(history_recalled_cases)),
                "selected_case_count": int(len(cases)),
                "skipped_incomplete_window_count": int(skipped_incomplete_window_count),
                "missing_osrm_route_count": int(missing_osrm_route_count),
                "skipped_no_normal_reference_count": int(skipped_no_normal_reference_count),
            },
        )
    result = {
        "method": (
            "four-component weighted historical incident residual evidence"
            if include_normal_reference
            else "four-component weighted historical continuation evidence"
        ),
        "query_spatial_relation": {
            "target_node_id": int(data.node_order[int(node_index)]),
            "road": str(data.sensors.iloc[int(node_index)]["Fwy"]),
            "direction": str(data.sensors.iloc[int(node_index)]["Direction"]),
            "driving_distance_km": query_route["distance_km"],
            "driving_duration_s": query_route["duration_s"],
            "driving_bearing_deg": query_route["bearing_deg"],
        },
        "selection": {
            "groups": {
                **({BASE_NORMAL_GROUP: {
                    "top_k": incident_base_normal_top_k,
                    "eligible_count": len(state_rows),
                    "ranking": "lexicographic_minimax_of_query_scaled_trajectory_and_step_change_rmse",
                    "dimensions": ["base", "normal", "signed_gap"],
                    "absolute_match_threshold": None,
                    "outcomes_used_for_selection": False,
                }} if paired_state_enabled else {}),
                "incident_network_time_similarity": {
                    "top_k": incident_network_time_top_k,
                    "minimum_score": minimum_score,
                    "score_field": "group_scores.incident_network_time_similarity",
                },
                "history_similarity": {
                    "top_k": incident_history_top_k,
                    "minimum_score": max(0.75, minimum_score),
                    "score_field": "group_scores.history_similarity",
                },
            },
            "selected_count": len(cases),
            **({
                "skipped_no_normal_reference_count": int(skipped_no_normal_reference_count)
            } if include_normal_reference else {}),
            "eligible_historical_candidate_count": int(len(historical_candidates)),
            "scored_candidate_count": int(len(rows)),
            "metadata_vector_recall_candidate_count": int(len(metadata_recalled_cases)),
            "history_vector_recall_candidate_count": int(len(history_recalled_cases)),
            "missing_osrm_route_count": int(missing_osrm_route_count),
            "selection_policy": (
                "independent_metadata_and_history_vector_recall_then_network_time_and_history_thresholds"
                + (
                    f"_plus_independent_base_normal_minimax_top{incident_base_normal_top_k}"
                    if paired_state_enabled else ""
                )
                + "_then_deduplicate_preserving_group_order"
            ),
            **({"paired_state_candidate_audit": [
                {"incident_id": row["incident_id"],
                 "selected": any(case["incident_id"] == row["incident_id"] for case in cases),
                 "base_normal_match": row.get("base_normal_match"),
                 "source_groups": row.get("source_groups", []),
                 "group_ranks": row.get("group_ranks", {}),
                 "paired_state": row.get("paired_state_diagnostics", {
                     "status": "unavailable", "reason": "query_normal_or_base_provider_missing"
                 })}
                for row in rows
            ]} if paired_state_enabled else {}),
            "cases": cases,
        },
    }
    if include_normal_reference:
        result["effect_definition"] = (
            "historical observed future minus local no-incident normal counterfactual"
        )
        result["incident_correction_evidence"] = evidence
    return result


def build_no_incident_normal_reference(
    data: TrafficData,
    query_incident,
    node_index: int,
    base_forecast_provider,
    current_base_forecast: list[float],
    top_k: int = 10,
    index: NoIncidentHistoryIndex | None = None,
    local_radius_km: float = 5.0,
    report_delay_buffer_minutes: int = 30,
    unknown_duration_minutes: int = 60,
    min_score: float = 0.6,
    timings=None,
    timing_metadata: dict | None = None,
) -> dict:
    query_time = query_incident.dt_parsed
    cutoff_time = query_time - pd.Timedelta(minutes=DEFAULT_SIMILAR_CASE_LOOKBACK_MINUTES)
    index = index or get_no_incident_history_index(data)
    return index.retrieve(
        data=data,
        query_incident=query_incident,
        node_index=int(node_index),
        cutoff_time=cutoff_time,
        top_k=int(top_k),
        base_forecast_provider=base_forecast_provider,
        local_radius_km=float(local_radius_km),
        report_delay_buffer_minutes=int(report_delay_buffer_minutes),
        unknown_duration_minutes=int(unknown_duration_minutes),
        min_score=float(min_score),
        current_base_forecast=current_base_forecast,
        cache_model_independent=True,
        timings=timings,
        timing_metadata=timing_metadata,
    )


def build_normal_counterfactual_sequence(*args, **kwargs):
    """Canonical Normal Counterfactual Generation entry point.

    Keyword aliases are translated to the legacy index API at this boundary;
    candidate scoring, filtering, weighting, and caching are unchanged.
    """

    aliases = {
        "foundation_model_provider": "base_forecast_provider",
        "current_foundation_model_prediction": "current_base_forecast",
    }
    for source, target in aliases.items():
        if source in kwargs and target not in kwargs:
            kwargs[target] = kwargs.pop(source)
    return build_no_incident_normal_reference(*args, **kwargs)


def validate_tool_llm_response(
    obj: dict,
    target_node_id: int,
    expected_start_step: int = 1,
    forecast_steps: int = 12,
) -> dict:
    _require_exact_keys(
        obj,
        {"target_node_id", "forecast_explanation", "phase_analysis"},
        "tool_llm_response",
    )
    obj["forecast_explanation"] = _annotation_text(obj["forecast_explanation"])
    if int(obj["target_node_id"]) != int(target_node_id):
        raise ValueError(f"target_node_id mismatch: expected={target_node_id}, actual={obj['target_node_id']}")

    phase_analysis = obj["phase_analysis"]
    if not isinstance(phase_analysis, list) or not phase_analysis:
        raise ValueError(
            "phase_analysis must be a non-empty list of phase objects, "
            f"got {type(phase_analysis).__name__}"
        )
    if len(phase_analysis) > MAX_PHASE_COUNT:
        raise ValueError(f"phase_analysis must contain at most {MAX_PHASE_COUNT} phases, got {len(phase_analysis)}")
    expected_start_step = int(expected_start_step)
    forecast_steps = int(forecast_steps)
    next_step = expected_start_step
    predicted_flow = []
    for phase_index, phase in enumerate(phase_analysis, start=1):
        if not isinstance(phase, dict):
            raise ValueError(f"phase_analysis[{phase_index}] must be an object")
        phase_predicted_flow = _validate_phase_analysis_item(
            phase,
            phase_index=phase_index,
            expected_start_step=next_step,
            forecast_steps=forecast_steps,
        )
        predicted_flow.extend(phase_predicted_flow)
        next_step = int(phase["end_step"]) + 1
    if next_step != forecast_steps + 1:
        raise ValueError(
            f"phase_analysis must cover steps {expected_start_step}..{forecast_steps}, "
            f"ended at step {next_step - 1}"
        )
    if len(predicted_flow) != forecast_steps - expected_start_step + 1:
        raise ValueError(
            f"combined phase_analysis.predicted_flow must contain {forecast_steps - expected_start_step + 1} "
            f"values, got {len(predicted_flow)}"
        )
    obj["predicted_flow"] = predicted_flow
    return obj


def parse_tool_llm_response(
    raw_response: str,
    target_node_id: int,
    expected_start_step: int = 1,
    forecast_steps: int = 12,
    phased_prediction: bool = True,
) -> dict:
    parsed = parse_json_object_response(raw_response, "Tool LLM")
    if not phased_prediction:
        _require_exact_keys(parsed, {"predicted_flow"}, "tool_llm_response")
        predicted_flow = parsed["predicted_flow"]
        if not isinstance(predicted_flow, list) or len(predicted_flow) != forecast_steps:
            raise ValueError(
                f"predicted_flow must contain {forecast_steps} values, got {predicted_flow!r}"
            )
        validated = []
        for step, value in enumerate(predicted_flow, start=1):
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not np.isfinite(value)
                or float(value) < 0
            ):
                raise ValueError(
                    "predicted_flow step "
                    f"{step} must be a finite non-negative number, got {value!r}"
                )
            validated.append(round(float(value), 2))
        parsed["predicted_flow"] = validated
        return parsed
    return validate_tool_llm_response(
        parsed,
        target_node_id,
        expected_start_step=expected_start_step,
        forecast_steps=forecast_steps,
    )


def _require_keys(obj: dict, keys: set[str], label: str) -> None:
    missing = keys - set(obj)
    if missing:
        raise ValueError(f"{label} missing required keys: {sorted(missing)}")


def _require_exact_keys(obj: dict, keys: set[str], label: str) -> None:
    _require_keys(obj, keys, label)
    extra = set(obj) - keys
    if extra:
        raise ValueError(f"{label} has unexpected keys: {sorted(extra)}")


def _annotation_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _validate_phase_analysis_item(
    phase_analysis: dict,
    phase_index: int,
    expected_start_step: int,
    forecast_steps: int,
) -> list[float]:
    label = f"phase_analysis[{phase_index}]"
    _require_keys(
        phase_analysis,
        {
            "start_step",
            "end_step",
            "predicted_flow",
        },
        label,
    )
    phase_analysis["explanation"] = _annotation_text(phase_analysis.get("explanation"))
    references = phase_analysis.get("evidence_references")
    phase_analysis["evidence_references"] = (
        references if isinstance(references, list) else [references] if references else []
    )
    for key in ("start_step", "end_step"):
        value = phase_analysis[key]
        if not isinstance(value, Real) or isinstance(value, bool) or not np.isfinite(value) or int(value) != value:
            raise ValueError(f"{label}.{key} must be an integer, got {value!r}")
        phase_analysis[key] = int(value)
    start_step = phase_analysis["start_step"]
    end_step = phase_analysis["end_step"]
    expected_start_step = int(expected_start_step)
    forecast_steps = int(forecast_steps)
    if start_step != expected_start_step:
        raise ValueError(f"{label}.start_step must be {expected_start_step}, got {start_step}")
    if not start_step <= end_step <= forecast_steps:
        raise ValueError(f"{label}.end_step must be in [{start_step}, {forecast_steps}], got {end_step}")
    predicted_flow = phase_analysis["predicted_flow"]
    expected_length = end_step - start_step + 1
    if not isinstance(predicted_flow, list) or len(predicted_flow) != expected_length:
        raise ValueError(f"{label}.predicted_flow must contain {expected_length} values, got {predicted_flow!r}")
    parsed_flow = []
    for step, value in enumerate(predicted_flow, start=start_step):
        if not isinstance(value, Real) or isinstance(value, bool) or not np.isfinite(value) or float(value) < 0:
            raise ValueError(f"{label}.predicted_flow step {step} must be a finite non-negative number, got {value!r}")
        parsed_flow.append(round(float(value), 2))
    phase_analysis["predicted_flow"] = parsed_flow
    return parsed_flow


def _compact_historical_incident(incident_payload_: dict) -> dict:
    return {
        "incident_id": incident_payload_["incident_id"],
        "time": incident_payload_["time"],
        "description": incident_payload_["description"],
        "type": incident_payload_["type"],
        "area": incident_payload_["area"],
        "location": incident_payload_["location"],
        "freeway": incident_payload_["freeway"],
        "direction": incident_payload_["direction"],
        "duration_minutes": incident_payload_["duration_minutes"],
    }


def _incident_similarity(query_incident, candidate_incident) -> float:
    """Compare incident attributes; route direction belongs to network similarity."""
    fields = ("Type", "DESCRIPTION", "AREA", "LOCATION")
    return float(
        np.mean(
            [
                _text_similarity(
                    _normalized_incident_value(query_incident, field),
                    _normalized_incident_value(candidate_incident, field),
                )
                for field in fields
            ]
        )
    )


def _normalized_incident_value(incident, field: str) -> str:
    value = incident.get(field, "") if isinstance(incident, dict) else getattr(incident, field, "")
    return " ".join(str(value).strip().upper().split())


def _text_similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(None, left, right, autojunk=False).ratio())


def _time_similarity(reference_time_context: dict, historical_time_context: dict) -> float:
    holiday_score = float(
        reference_time_context["is_holiday_or_weekend"] == historical_time_context["is_holiday_or_weekend"]
    )
    period_score = float(reference_time_context["day_period"] == historical_time_context["day_period"])
    hour_delta = abs(int(reference_time_context["hour"]) - int(historical_time_context["hour"])) % 24
    hour_delta = min(hour_delta, 24 - hour_delta)
    hour_score = float((np.cos(2.0 * np.pi * hour_delta / 24.0) + 1.0) / 2.0)
    return float((holiday_score + period_score + hour_score) / 3.0)


def _combine_retrieval_similarity(
    *,
    incident_similarity: float,
    time_similarity: float,
    history_similarity: float,
    network_similarity: float,
) -> float:
    """Return the normalized score from the configured four similarity classes."""
    components = {
        "incident": incident_similarity,
        "time": time_similarity,
        "history": history_similarity,
        "network": network_similarity,
    }
    weighted_total = sum(
        RETRIEVAL_SIMILARITY_WEIGHTS[name] * np.clip(float(value), 0.0, 1.0)
        for name, value in components.items()
    )
    return float(weighted_total / sum(RETRIEVAL_SIMILARITY_WEIGHTS.values()))


def _network_similarity(query_route: dict, candidate_route: dict | None) -> tuple[float, dict]:
    """Compare cached OSRM incident-to-node distance, time, and route bearing."""
    if candidate_route is None:
        return 0.0, {"distance": 0.0, "duration": 0.0, "direction": 0.0}
    components = {
        "distance": _relative_similarity(
            query_route.get("distance_km"), candidate_route.get("distance_km"), minimum_scale=0.5
        ),
        "duration": _relative_similarity(
            query_route.get("duration_s"), candidate_route.get("duration_s"), minimum_scale=60.0
        ),
        "direction": _bearing_similarity(
            query_route.get("bearing_deg"), candidate_route.get("bearing_deg")
        ),
    }
    return float(np.mean(list(components.values()))), components


def _relative_similarity(left, right, *, minimum_scale: float) -> float:
    if left is None or right is None or not np.isfinite(left) or not np.isfinite(right):
        return 0.0
    scale = max(abs(float(left)), abs(float(right)), float(minimum_scale))
    return float(np.exp(-abs(float(left) - float(right)) / scale))


def _bearing_similarity(left, right) -> float:
    if left is None or right is None or not np.isfinite(left) or not np.isfinite(right):
        return 0.0
    difference = abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)
    return float((np.cos(np.deg2rad(difference)) + 1.0) / 2.0)


def _route_relation_payload(
    data: TrafficData,
    node_index: int,
    query_route: dict,
    candidate_route: dict | None,
    network_components: dict,
) -> dict:
    target = data.sensors.iloc[int(node_index)]
    route = candidate_route or {}
    return {
        "same_target_node": True,
        "road": str(target["Fwy"]),
        "direction": str(target["Direction"]),
        "relative_position": "osrm_driving_route",
        "driving_distance_km": route.get("distance_km"),
        "query_driving_distance_km": query_route.get("distance_km"),
        "driving_duration_s": route.get("duration_s"),
        "query_driving_duration_s": query_route.get("duration_s"),
        "driving_bearing_deg": route.get("bearing_deg"),
        "query_driving_bearing_deg": query_route.get("bearing_deg"),
        "network_similarity_components": dict(network_components),
    }


def _effect_shape(effect: np.ndarray, reference: np.ndarray | None = None) -> str:
    """Encode four coarse effect phases as up/down/flat transitions."""
    effect = np.asarray(effect, dtype=np.float64)
    if effect.size != 12:
        raise ValueError(f"Effect shape encoding expects 12 steps, got {effect.shape}")
    reference_scale = (
        float(np.mean(np.abs(np.asarray(reference, dtype=np.float64))))
        if reference is not None
        else float(np.mean(np.abs(effect)))
    )
    scale = max(reference_scale, 1.0)
    phase_means = np.asarray(
        [phase.mean() for phase in np.array_split(effect, EFFECT_SHAPE_PHASE_COUNT)],
        dtype=np.float64,
    )
    changes = np.diff(phase_means) / scale
    return "".join(
        "U" if change > EFFECT_SHAPE_DEADBAND_FRACTION
        else "D" if change < -EFFECT_SHAPE_DEADBAND_FRACTION
        else "F"
        for change in changes
    )


def _prefix_shape_similarity(left: str, right: str) -> float:
    matched = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        matched += 1
    return matched / max(len(left), len(right), 1)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> np.ndarray:
    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    sorted_weights = np.take_along_axis(np.broadcast_to(weights[:, None], values.shape), order, axis=0)
    cumulative = np.cumsum(sorted_weights, axis=0)
    cutoff = float(quantile) * cumulative[-1]
    indices = np.argmax(cumulative >= cutoff, axis=0)
    columns = np.arange(values.shape[1])
    return sorted_values[indices, columns]


def _effect_direction(effect: np.ndarray, counterfactual: np.ndarray) -> tuple[str, float]:
    threshold = max(5.0, 0.02 * float(np.mean(np.abs(counterfactual))))
    mean_effect = float(np.mean(effect))
    if mean_effect > threshold:
        return "positive", threshold
    if mean_effect < -threshold:
        return "negative", threshold
    return "neutral", threshold


def _aggregate_incident_effect_evidence(cases: list[dict]) -> dict:
    if not cases:
        return {
            "status": "no_retrieved_incident_cases",
            "message": "no_retrieved_incident_cases",
            "reason_code": "no_matched_cases",
            "direction_counts": {"positive": 0, "negative": 0, "neutral": 0},
            "direction_weight": {"positive": 0.0, "negative": 0.0, "neutral": 0.0},
            "shape_counts": {},
            "shape_weight": {},
            "shape_medoid": None,
            "shape_agreement": 0.0,
            "case_count": 0,
            "effective_case_count": 0.0,
            "weighted_all_case_effect_residual": [0.0] * 12,
            "effect_residual_std": [0.0] * 12,
            "effect_residual_p10": [0.0] * 12,
            "effect_residual_p50": [0.0] * 12,
            "effect_residual_p90": [0.0] * 12,
        }
    effects = np.asarray(
        [case["historical_effect"]["effect_residual"] for case in cases],
        dtype=np.float64,
    )
    scores = np.asarray(
        [case.get("evidence_weight", case.get("retrieval_score", 1.0)) for case in cases],
        dtype=np.float64,
    )
    weights = np.maximum(scores, 0.0)
    weights = weights / np.sum(weights) if np.sum(weights) > 0.0 else np.full(len(cases), 1.0 / len(cases))
    directions = [case["historical_effect"]["direction"] for case in cases]
    shapes = [
        case["historical_effect"].get(
            "effect_shape",
            _effect_shape(
                np.asarray(case["historical_effect"]["effect_residual"], dtype=np.float64),
                np.asarray(case["historical_effect"].get("normal_counterfactual_flow", []), dtype=np.float64)
                if case["historical_effect"].get("normal_counterfactual_flow")
                else None,
            ),
        )
        for case in cases
    ]
    counts = {direction: directions.count(direction) for direction in ("positive", "negative", "neutral")}
    direction_weight = {
        direction: float(np.sum(weights[np.asarray([value == direction for value in directions], dtype=bool)]))
        for direction in ("positive", "negative", "neutral")
    }
    shape_weight = {
        shape: float(np.sum(weights[np.asarray([value == shape for value in shapes], dtype=bool)]))
        for shape in sorted(set(shapes))
    }
    medoid = max(shape_weight, key=shape_weight.get)
    shape_agreement = float(
        sum(weight * _prefix_shape_similarity(shape, medoid) for shape, weight in zip(shapes, weights))
    )
    all_case_mean = np.sum(effects * weights[:, None], axis=0)
    spread = np.sqrt(np.sum((effects - all_case_mean[None, :]) ** 2 * weights[:, None], axis=0))
    case_effect_magnitude = np.mean(np.abs(effects), axis=1)
    effective_case_count = float(1.0 / np.sum(np.square(weights)))
    return {
        "status": "retrieval_evidence_summary",
        "message": "retrieval_evidence_summary",
        "reason_code": "cases_available",
        "evidence_weight_definition": "retrieval_score from four weighted similarities",
        "direction_counts": counts,
        "direction_weight": direction_weight,
        "shape_counts": {shape: shapes.count(shape) for shape in sorted(set(shapes))},
        "shape_weight": shape_weight,
        "shape_medoid": medoid,
        "shape_agreement": shape_agreement,
        "case_count": len(cases),
        "effective_case_count": effective_case_count,
        "case_effect_magnitude_mean": float(np.mean(case_effect_magnitude)),
        "case_effect_magnitude_std": float(np.std(case_effect_magnitude)),
        "effect_residual_std": [float(value) for value in spread],
        "effect_residual_p10": [float(value) for value in _weighted_quantile(effects, weights, 0.10)],
        "effect_residual_p50": [float(value) for value in _weighted_quantile(effects, weights, 0.50)],
        "effect_residual_p90": [float(value) for value in _weighted_quantile(effects, weights, 0.90)],
        "weighted_all_case_effect_residual": [float(value) for value in all_case_mean],
    }


def _bearing_deg(source_latitude: float, source_longitude: float, target_latitude: float, target_longitude: float) -> float:
    source_latitude_rad = np.radians(float(source_latitude))
    target_latitude_rad = np.radians(float(target_latitude))
    delta_longitude_rad = np.radians(float(target_longitude) - float(source_longitude))
    x = np.sin(delta_longitude_rad) * np.cos(target_latitude_rad)
    y = (
        np.cos(source_latitude_rad) * np.sin(target_latitude_rad)
        - np.sin(source_latitude_rad) * np.cos(target_latitude_rad) * np.cos(delta_longitude_rad)
    )
    return float((np.degrees(np.arctan2(x, y)) + 360.0) % 360.0)


# Paper-facing module entry points.  The legacy names above remain available
# because old evaluation scripts and saved contexts refer to them.
def retrieve_historical_incident_patterns(*args, **kwargs):
    """Historical Incident Retrieval; returns Incident Patterns."""

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


def retrieve_incident_patterns(*args, **kwargs):
    """Canonical Historical Incident Retrieval entry point."""

    return retrieve_historical_incident_patterns(*args, **kwargs)


def generate_normal_counterfactual_sequence(*args, **kwargs):
    """Normal Counterfactual Generation; returns a normal reference sequence."""

    aliases = {
        "foundation_model_provider": "base_forecast_provider",
        "current_foundation_model_prediction": "current_base_forecast",
    }
    for source, target in aliases.items():
        if source in kwargs and target not in kwargs:
            kwargs[target] = kwargs.pop(source)
    return build_no_incident_normal_reference(*args, **kwargs)


# Canonical spelling used by the paper-facing module API.
normal_counterfactual_generation = generate_normal_counterfactual_sequence
