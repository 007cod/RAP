from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from collections import OrderedDict
import time
from pathlib import Path
from threading import Lock

from src.agents.episode_memory import (
    append_episode,
    build_episode,
    build_episode_reflection_messages,
    parse_episode_reflection_response,
)
from src.agents.non_incident_retrieval import NoIncidentHistoryIndex, get_no_incident_history_index
from src.agents.incident_vector_index import IncidentVectorIndex, get_incident_vector_index
from src.agents.incident_history_index import get_incident_history_vector_index
from src.agents.tools import build_chat_messages, parse_tool_llm_response
from src.agents.context import build_traffic_context
from src.agents.terminology import (
    EXPERIENCE_TRAJECTORIES,
    FOUNDATION_MODEL_PREDICTION,
    NORMAL_COUNTERFACTUAL_SEQUENCE,
    PREVIOUS_LAYER_PREDICTIONS,
)
from src.config import DEFAULT_MEMORY_MIN_SIMILARITY, AgentConfig, ImpactScopeConfig, LLMContextConfig
from src.models.forecaster import ForecastingModelForecaster
from src.data.traffic import TrafficData
from src.tool_llm.timing import TimingRecorder, timed_section
from src.tool_llm.gateway import call_and_parse_llm_response
from src.tool_llm.artifacts import (
    incident_aggregate_path,
    incident_node_artifact_stem,
)
from src.utils.json_io import write_json


FORECAST_STEPS = 12


class ReflectionBatch:
    """Run episode reflections independently from the main forecast workers."""

    def __init__(self, max_workers: int):
        if int(max_workers) <= 0:
            raise ValueError(f"reflection worker count must be positive, got {max_workers}")
        self._executor = ThreadPoolExecutor(max_workers=int(max_workers))
        self._futures = []
        self._lock = Lock()
        self._closed = False

    def submit(self, function, *args, **kwargs) -> Future:
        with self._lock:
            if self._closed:
                raise RuntimeError("ReflectionBatch is already closed")
            future = self._executor.submit(function, *args, **kwargs)
            self._futures.append(future)
            return future

    @staticmethod
    def wait_for(futures) -> None:
        first_error = None
        for future in as_completed(list(futures)):
            try:
                future.result()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise RuntimeError("One or more episode reflections failed") from first_error

    def wait(self) -> None:
        with self._lock:
            if self._closed:
                return
            futures = list(self._futures)
            self._closed = True
        try:
            self.wait_for(futures)
        finally:
            self._executor.shutdown(wait=True)


class ForecastService:
    """Serialize model inference while sharing every timestamp forecast.

    The forecasting model is shared by incident workers and is not safe or
    useful to run concurrently on the same device.  A process-wide single
    flight queue avoids making every worker contend on a coarse lock and also
    reuses a historical candidate forecast across incidents and target nodes.
    """

    def __init__(
        self,
        forecaster: ForecastingModelForecaster,
        timings: TimingRecorder | None = None,
        max_memory_entries: int = 512,
    ):
        if int(max_memory_entries) <= 0:
            raise ValueError("max_memory_entries must be positive")
        self.forecaster = forecaster
        self.timings = timings
        self.max_memory_entries = int(max_memory_entries)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="base-forecast")
        self._lock = Lock()
        self._futures: dict[str, Future] = {}
        self._results: OrderedDict[str, list[list[float]]] = OrderedDict()
        self._closed = False

    @staticmethod
    def _key(timestamp) -> str:
        return str(timestamp)

    def _compute(self, timestamp):
        metadata = {"timestamp": str(timestamp)}
        with timed_section(self.timings, "base_model.inference", **metadata):
            # The forecaster's persistent cache is shared with later runs.  The
            # callback is executed only on a genuine cache miss.
            values = self.forecaster.forecast_historical_all(
                timestamp,
                lambda: self.forecaster.forecast_all(timestamp),
                timings=self.timings,
            )
            # HistoricalForecastCache may return an ndarray.  Keep the public
            # service contract as ordinary lists so callers can safely test
            # presence and serialize results without NumPy truth-value errors.
            return [[float(value) for value in row] for row in values]

    def forecast_all(self, timestamp) -> list[list[float]]:
        key = self._key(timestamp)
        with self._lock:
            if self._closed:
                raise RuntimeError("ForecastService is closed")
            cached = self._results.get(key)
            if cached is not None:
                self._results.move_to_end(key)
                return cached
            future = self._futures.get(key)
            if future is None:
                future = self._executor.submit(self._compute, timestamp)
                self._futures[key] = future
        try:
            result = future.result()
        except BaseException:
            with self._lock:
                if self._futures.get(key) is future:
                    self._futures.pop(key, None)
            raise
        with self._lock:
            if self._futures.get(key) is future:
                self._futures.pop(key, None)
                self._results[key] = result
                self._results.move_to_end(key)
                while len(self._results) > self.max_memory_entries:
                    self._results.popitem(last=False)
        return result

    def forecast_target(self, timestamp, node_index: int) -> list[float]:
        return [float(value) for value in self.forecast_all(timestamp)[int(node_index)]]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True)


def run_layer_node(
    data: TrafficData,
    agent_cfg: AgentConfig,
    impact_scope: ImpactScopeConfig,
    llm_context: LLMContextConfig,
    memory_cfg: dict,
    incident_id: str,
    base_forecast,
    base_forecast_source: str | None,
    output_dir: Path,
    call_llm: bool,
    node_index: int,
    layer_index: int,
    previous_layer: list[int],
    llm_predicted_flow: dict[int, list[float]],
    episodes_path: Path | None = None,
    episode_memory_top_k: int = 3,
    normal_reference_base_forecast_provider=None,
    timings: TimingRecorder | None = None,
    llm_request_semaphore=None,
    reflection_batch: ReflectionBatch | None = None,
    no_incident_history_index: NoIncidentHistoryIndex | None = None,
    incident_vector_index: IncidentVectorIndex | None = None,
    context_retrieval_semaphore=None,
    *,
    foundation_model_prediction=None,
    foundation_model_source: str | None = None,
    previous_layer_predictions: list[int] | None = None,
    previous_layer_prediction_values: dict[int, list[float]] | None = None,
    normal_counterfactual_provider=None,
) -> dict:
    # Canonical paper-facing arguments are aliases at this boundary.  The
    # legacy arguments remain accepted for existing scripts and resume jobs.
    if foundation_model_prediction is not None:
        base_forecast = foundation_model_prediction
    if foundation_model_source is not None:
        base_forecast_source = foundation_model_source
    if previous_layer_predictions is not None:
        previous_layer = previous_layer_predictions
    if previous_layer_prediction_values is not None:
        llm_predicted_flow = previous_layer_prediction_values
    if normal_counterfactual_provider is not None:
        normal_reference_base_forecast_provider = normal_counterfactual_provider
    timing_metadata = {
        "incident_id": str(incident_id),
        "layer_index": int(layer_index),
        "node_index": int(node_index),
    }
    node_started = time.perf_counter() if timings is not None else None

    def build_context():
        return build_traffic_context(
            data=data,
            incident_id=incident_id,
            foundation_model_prediction=base_forecast,
            foundation_model_source=base_forecast_source,
            impact_scope=impact_scope,
            llm_context=llm_context,
            node_index=int(node_index),
            layer_index=int(layer_index),
            previous_layer_predictions=previous_layer,
            previous_layer_prediction_values=llm_predicted_flow,
            incident_retrieval_min_score=agent_cfg.incident_retrieval_min_score,
            similar_incident_vector_candidates=agent_cfg.similar_incident_vector_candidates,
            incident_network_time_top_k=agent_cfg.incident_network_time_top_k,
            incident_history_top_k=agent_cfg.incident_history_top_k,
            incident_base_normal_top_k=agent_cfg.incident_base_normal_top_k,
            normal_reference_top_k=agent_cfg.normal_reference_top_k,
            normal_reference_min_score=agent_cfg.normal_reference_min_score,
            episodes_path=str(episodes_path) if episodes_path is not None else None,
            episode_memory_top_k=int(episode_memory_top_k),
            episode_search_window_days=int(memory_cfg.get("search_window_days", 120)),
            episode_recency_half_life_days=int(memory_cfg.get("recency_half_life_days", 30)),
            episode_min_similarity=float(memory_cfg.get("min_similarity", DEFAULT_MEMORY_MIN_SIMILARITY)),
            episode_min_base_mae=float(memory_cfg.get("min_base_mae", 0.0)),
            episode_min_failure_delta=float(memory_cfg.get("min_failure_delta", 10.0)),
            normal_counterfactual_provider=normal_reference_base_forecast_provider,
            no_incident_history_index=no_incident_history_index,
            incident_vector_index=incident_vector_index,
            no_incident_local_radius_km=agent_cfg.no_incident_local_radius_km,
            incident_report_delay_buffer_minutes=agent_cfg.incident_report_delay_buffer_minutes,
            unknown_incident_duration_minutes=agent_cfg.unknown_incident_duration_minutes,
            build_all_evidence=call_llm,
            timings=timings,
            timing_metadata=timing_metadata,
        )

    if context_retrieval_semaphore is None:
        with timed_section(timings, "node.build_context", **timing_metadata):
            context = build_context()
    else:
        with timed_section(timings, "context.retrieval_slot_wait", **timing_metadata):
            context_retrieval_semaphore.acquire()
        try:
            with timed_section(timings, "node.build_context", **timing_metadata):
                context = build_context()
        finally:
            context_retrieval_semaphore.release()
    evaluation = context.pop("_evaluation")
    tool_keys = context.pop("_llm_context_keys")
    name = incident_node_artifact_stem(
        context["sample_id"],
        layer_index,
        context["target_node_id"],
    )

    with timed_section(timings, "node.write_context", **timing_metadata):
        write_json(output_dir / f"{name}_context.json", context)
    with timed_section(timings, "node.build_messages", **timing_metadata):
        initial_messages = build_chat_messages(
            context,
            tool_keys,
            include_foundation_model_prediction=llm_context.foundation_model_prediction,
            phased_prediction=llm_context.phased_prediction,
        )
    with timed_section(timings, "node.write_messages", **timing_metadata):
        write_json(output_dir / f"{name}_messages.json", initial_messages)
    evaluation_record = {
        "sample_id": context["sample_id"],
        "target_node_id": context["target_node_id"],
        "target_node_index": context["target_node_index"],
        "layer_index": int(layer_index),
        "y_true": evaluation["y_true"],
        # A sensor with an all-zero history represents missing records. Keep
        # the node artifact for audit, but exclude every forecast horizon from
        # aggregate metrics.
        "metric_mask": [context.get("target_history_flow_status") != "missing_all_zero"] * len(evaluation["y_true"]),
        "context_path": str(output_dir / f"{name}_context.json"),
        "messages_path": str(output_dir / f"{name}_messages.json"),
    }
    if llm_context.foundation_model_prediction and "base_forecast" in evaluation:
        evaluation_record["base_forecast"] = evaluation["base_forecast"]
    with timed_section(timings, "node.write_evaluation_record", **timing_metadata):
        write_json(output_dir / f"{name}_evaluation_record.json", evaluation_record)

    node_record = {
        "node_index": context["target_node_index"],
        "node_id": context["target_node_id"],
        "layer_index": int(layer_index),
        "context_path": str(output_dir / f"{name}_context.json"),
        "evaluation_record_path": str(output_dir / f"{name}_evaluation_record.json"),
    }
    result = {
        "node_index": int(node_index),
        "node_record": node_record,
        "llm_predicted_flow": None,
        "reflection_future": None,
    }
    # A completely zero history denotes a missing sensor window.  It must not
    # consume an LLM request or enter the forecast/error metrics, but a zero
    # trajectory is still propagated so downstream layer construction remains
    # deterministic and the node is auditable in the aggregate artifact.
    if call_llm and context.get("target_history_flow_status") == "missing_all_zero":
        parsed = {
            "target_node_id": context["target_node_id"],
            "forecast_explanation": "All history values are zero and treated as a missing sensor.",
            "phase_analysis": [{
                "start_step": 1,
                "end_step": FORECAST_STEPS,
                "predicted_flow": [0.0] * FORECAST_STEPS,
                "explanation": "missing target sensor; zero trajectory is a sentinel",
                "evidence_references": [],
            }],
            "predicted_flow": [0.0] * FORECAST_STEPS,
            "missing_target_flow_policy": "skip_llm_and_metrics_zero_sensor",
        }
        result["llm_predicted_flow"] = [0.0] * FORECAST_STEPS
        result_record_path = output_dir / f"{name}_result_record.json"
        with timed_section(timings, "node.write_parsed_response", **timing_metadata):
            write_json(output_dir / f"{name}_parsed.json", parsed)
        result_record = {
            **evaluation_record,
            "llm_predicted_flow": [0.0] * FORECAST_STEPS,
            "parsed_json": parsed,
            "raw_response_path": None,
            "episode_reflection_status": "completed",
            "episode_reflection_skip_reason": "missing_all_zero_sensor",
        }
        with timed_section(timings, "node.write_result_record", **timing_metadata):
            write_json(result_record_path, result_record)
        node_record["result_record_path"] = str(result_record_path)
        if timings is not None and node_started is not None:
            timings.record("node.total", (time.perf_counter() - node_started) * 1000.0, timing_metadata)
        return result
    if call_llm:
        raw_response_path = output_dir / f"{name}_raw_response.json"
        _raw_response, parsed = call_and_parse_llm_response(
            agent_cfg=agent_cfg,
            messages=initial_messages,
            raw_response_path=raw_response_path,
            parse_response=lambda response: parse_tool_llm_response(
                response,
                context["target_node_id"],
                expected_start_step=1,
                forecast_steps=FORECAST_STEPS,
                phased_prediction=llm_context.phased_prediction,
            ),
            label="Tool LLM",
            timings=timings,
            timing_module="tool_llm",
            timing_metadata=timing_metadata,
            llm_request_semaphore=llm_request_semaphore,
        )
        with timed_section(timings, "node.write_parsed_response", **timing_metadata):
            write_json(output_dir / f"{name}_parsed.json", parsed)
        result_record_path = output_dir / f"{name}_result_record.json"
        result["llm_predicted_flow"] = list(parsed["predicted_flow"])
        memory_eligible = (
            episodes_path is not None
            and llm_context.is_enabled("episode_memory")
            and llm_context.foundation_model_prediction
            and context.get("target_history_flow_status") != "missing_all_zero"
        )
        if memory_eligible:
            parsed_for_memory = parsed
            if not llm_context.phased_prediction:
                parsed_for_memory = {
                    **parsed,
                    "phase_analysis": [{
                        "start_step": 1,
                        "end_step": FORECAST_STEPS,
                        "predicted_flow": list(parsed["predicted_flow"]),
                        "evidence_references": [],
                        "explanation": "",
                    }],
                    "forecast_explanation": "",
                    "target_node_id": context["target_node_id"],
                }
            memory_context = {
                **context,
                "layer_index": int(layer_index),
                "_paths": {
                    "context": str(output_dir / f"{name}_context.json"),
                    "messages": str(output_dir / f"{name}_messages.json"),
                },
            }
            reflection_messages, reflection_assessment = build_episode_reflection_messages(
                memory_context,
                evaluation,
                parsed_for_memory,
            )
            reflection_messages_path = output_dir / f"{name}_episode_reflection_messages.json"
            with timed_section(timings, "node.write_reflection_messages", **timing_metadata):
                write_json(reflection_messages_path, reflection_messages)
            reflection_raw_response_path = output_dir / f"{name}_episode_reflection_raw_response.json"
            reflection_parsed_path = output_dir / f"{name}_episode_reflection_parsed.json"
            result_record = {
                **evaluation_record,
                "llm_predicted_flow": parsed["predicted_flow"],
                "parsed_json": parsed,
                "raw_response_path": str(raw_response_path),
                "episode_reflection": None,
                "episode_reflection_status": "pending",
                "episode_reflection_path": str(reflection_parsed_path),
                "episode_reflection_raw_response_path": str(reflection_raw_response_path),
            }
            with timed_section(timings, "node.write_result_record", **timing_metadata):
                write_json(result_record_path, result_record)
            node_record["result_record_path"] = str(result_record_path)
            reflection_job = {
                "agent_cfg": agent_cfg,
                "reflection_messages": reflection_messages,
                "reflection_assessment": reflection_assessment,
                "reflection_raw_response_path": reflection_raw_response_path,
                "reflection_parsed_path": reflection_parsed_path,
                "result_record_path": result_record_path,
                "result_record": result_record,
                "memory_context": memory_context,
                "evaluation": evaluation,
                "parsed": parsed_for_memory,
                "raw_response_path": raw_response_path,
                "episodes_path": episodes_path,
                "memory_cfg": memory_cfg,
                "timings": timings,
                "timing_metadata": timing_metadata,
                "llm_request_semaphore": llm_request_semaphore,
            }
            if reflection_batch is None:
                local_reflection_batch = ReflectionBatch(max_workers=1)
                local_reflection_batch.submit(_run_episode_reflection_job, reflection_job)
                local_reflection_batch.wait()
            else:
                with timed_section(timings, "node.queue_episode_reflection", **timing_metadata):
                    result["reflection_future"] = reflection_batch.submit(
                        _run_episode_reflection_job,
                        reflection_job,
                    )
        else:
            result_record = {
                **evaluation_record,
                "llm_predicted_flow": parsed["predicted_flow"],
                "parsed_json": parsed,
                "raw_response_path": str(raw_response_path),
            }
            if episodes_path is not None and not memory_eligible:
                result_record["episode_reflection_status"] = "completed"
                result_record["episode_reflection_skip_reason"] = (
                    "missing_all_zero_sensor"
                    if context.get("target_history_flow_status") == "missing_all_zero"
                    else "episode_memory_disabled"
                )
            with timed_section(timings, "node.write_result_record", **timing_metadata):
                write_json(result_record_path, result_record)
            node_record["result_record_path"] = str(result_record_path)
    if timings is not None and node_started is not None:
        timings.record("node.total", (time.perf_counter() - node_started) * 1000.0, timing_metadata)
    return result


def _run_episode_reflection_job(job: dict) -> dict:
    """Complete one reflection and atomically publish its episode artifacts."""
    try:
        _raw_response, reflection = call_and_parse_llm_response(
            agent_cfg=job["agent_cfg"],
            messages=job["reflection_messages"],
            raw_response_path=job["reflection_raw_response_path"],
            parse_response=lambda response: parse_episode_reflection_response(
                response,
                job["reflection_assessment"]["verdict"],
            ),
            label="Episode reflection",
            timings=job["timings"],
            timing_module="episode_reflection",
            timing_metadata=job["timing_metadata"],
            llm_request_semaphore=job["llm_request_semaphore"],
        )
        with timed_section(job["timings"], "node.write_reflection_parsed", **job["timing_metadata"]):
            write_json(job["reflection_parsed_path"], reflection)
        completed_record = {
            **job["result_record"],
            "episode_reflection": reflection,
            "episode_reflection_status": "completed",
        }
        with timed_section(job["timings"], "node.write_result_record", **job["timing_metadata"]):
            write_json(job["result_record_path"], completed_record)
        with timed_section(job["timings"], "node.append_episode", **job["timing_metadata"]):
            append_episode(
                job["episodes_path"],
                build_episode(
                    context=job["memory_context"],
                    evaluation=job["evaluation"],
                    parsed=job["parsed"],
                    reflection=reflection,
                    result_record_path=job["result_record_path"],
                    raw_response_path=job["raw_response_path"],
                    reflection_raw_response_path=job["reflection_raw_response_path"],
                ),
                retention_days=int(job["memory_cfg"].get("retention_days", 90)),
                max_entries=int(job["memory_cfg"].get("max_entries", 2000)),
            )
        if job["timings"] is not None:
            job["timings"].record(
                "episode_reflection.completed",
                0.0,
                job["timing_metadata"],
            )
        return {"status": "completed"}
    except Exception as exc:
        failed_record = {
            **job["result_record"],
            "episode_reflection_status": "failed",
            "episode_reflection_error": str(exc),
        }
        write_json(job["result_record_path"], failed_record)
        if job["timings"] is not None:
            job["timings"].record(
                "episode_reflection.failed",
                0.0,
                {
                    **job["timing_metadata"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        # Reflection enriches future episode memory but is not part of the
        # current prediction or its metrics. Preserve the failure for audit
        # without discarding forecasts that have already completed.
        print(
            "Episode reflection failed; continuing evaluation: "
            f"incident={job['timing_metadata'].get('incident_id')} "
            f"node={job['timing_metadata'].get('node_index')} error={exc}",
            flush=True,
        )
        return {"status": "failed", "error": str(exc)}


def _run_case_impl(
    data: TrafficData,
    forecaster: ForecastingModelForecaster | None,
    agent_cfg: AgentConfig,
    impact_scope: ImpactScopeConfig,
    llm_context: LLMContextConfig,
    memory_cfg: dict,
    incident_id: str,
    output_dir: Path,
    call_llm: bool,
    episodes_path: Path | None = None,
    episode_memory_top_k: int = 3,
    timings: TimingRecorder | None = None,
    forecaster_lock=None,
    llm_request_semaphore=None,
    reflection_batch: ReflectionBatch | None = None,
    no_incident_history_index: NoIncidentHistoryIndex | None = None,
    incident_vector_index: IncidentVectorIndex | None = None,
    forecast_service: ForecastService | None = None,
    context_retrieval_semaphore=None,
) -> dict:
    matches = data.incidents.loc[data.incidents["incident_id"].astype(str) == str(incident_id)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one incident: incident_id={incident_id}, matches={len(matches)}")
    incident = matches.iloc[0]
    normal_counterfactual_enabled = llm_context.is_enabled(NORMAL_COUNTERFACTUAL_SEQUENCE)
    foundation_model_prediction_enabled = bool(llm_context.foundation_model_prediction)
    if foundation_model_prediction_enabled and forecaster is None:
        raise ValueError("base_prediction is enabled but no forecasting model was loaded")
    if call_llm and normal_counterfactual_enabled:
        with timed_section(timings, "case.no_incident_history_index", incident_id=str(incident_id)):
            no_incident_history_index = no_incident_history_index or get_no_incident_history_index(
                data,
                agent_cfg.no_incident_cache_path,
                workers=agent_cfg.normal_retrieval_workers,
            )
    if call_llm:
        with timed_section(timings, "case.incident_vector_index", incident_id=str(incident_id)):
            incident_vector_index = incident_vector_index or get_incident_vector_index(
                data,
                agent_cfg.incident_vector_cache_path,
            )
    with timed_section(timings, "case.osrm_impact_scope", incident_id=str(incident_id)):
        osrm_scope = data.osrm_impact_scope(incident, impact_scope)
    target_node_index = osrm_scope["anchor_node_index"]
    if target_node_index is None:
        raise ValueError(f"No cached OSRM route is inside the impact distance bands: incident_id={incident_id}")

    def forecast_all(timestamp) -> list[list[float]]:
        if not foundation_model_prediction_enabled or forecaster is None:
            raise RuntimeError("Base forecasting is disabled")
        if forecast_service is not None:
            return forecast_service.forecast_all(timestamp)
        if forecaster_lock is None:
            with timed_section(timings, "base_model.inference"):
                return forecaster.forecast_all(timestamp)
        wait_started = time.perf_counter()
        forecaster_lock.acquire()
        if timings is not None:
            timings.record(
                "base_model.lock_wait",
                (time.perf_counter() - wait_started) * 1000.0,
                {"incident_id": str(incident_id), "timestamp": str(timestamp)},
            )
        try:
            with timed_section(
                timings,
                "base_model.inference",
                incident_id=str(incident_id),
                timestamp=str(timestamp),
            ):
                return forecaster.forecast_all(timestamp)
        finally:
            forecaster_lock.release()

    foundation_model_prediction = None
    if foundation_model_prediction_enabled:
        with timed_section(timings, "case.base_forecast", incident_id=str(incident_id)):
            foundation_model_prediction = forecast_all(incident["dt_parsed"])
    historical_base_forecast_cache: dict[str, Future] = {}
    historical_base_forecast_lock = Lock()

    def normal_reference_base_forecast_provider(timestamp, historical_node_index: int) -> list[float]:
        if not foundation_model_prediction_enabled or forecaster is None:
            raise RuntimeError("Historical Base forecasting is disabled")
        if forecast_service is not None:
            return forecast_service.forecast_target(timestamp, historical_node_index)
        key = str(timestamp)
        with historical_base_forecast_lock:
            forecast_future = historical_base_forecast_cache.get(key)
            is_owner = forecast_future is None
            if is_owner:
                forecast_future = Future()
                historical_base_forecast_cache[key] = forecast_future
        if is_owner:
            try:
                with timed_section(
                    timings,
                    "context.historical_base_forecast",
                    incident_id=str(incident_id),
                    timestamp=key,
                    node_index=int(historical_node_index),
                ):
                    if hasattr(forecaster, "forecast_historical_all"):
                        cached_forecast = forecaster.forecast_historical_all(
                            timestamp, lambda: forecast_all(timestamp), timings=timings,
                        )
                    else:
                        # Keep lightweight/custom forecasters compatible.
                        cached_forecast = forecast_all(timestamp)
            except BaseException as exc:
                forecast_future.set_exception(exc)
                with historical_base_forecast_lock:
                    historical_base_forecast_cache.pop(key, None)
                raise
            forecast_future.set_result(cached_forecast)
        return [float(value) for value in forecast_future.result()[int(historical_node_index)]]

    target_node_index_set = osrm_scope["node_indices"]
    node_layers = osrm_scope["node_layers"]
    if call_llm:
        # Build all history embeddings before spawning node workers.  The
        # matrices are persisted by the index, so later runs only load them.
        history_index = get_incident_history_vector_index(data, incident_vector_index)
        with timed_section(timings, "case.precompute_history_vectors", incident_id=str(incident_id)):
            history_index.precompute_nodes(target_node_index_set)

    last_layer = []
    llm_predicted_flow = {}
    node_records = []
    reflection_futures = []
    sample_id = int(incident["incident_id"])
    for layer_index, layer in enumerate(node_layers):
        layer_started = time.perf_counter() if timings is not None else None
        print(f"Layer {layer_index}: {layer}")
        previous_layer_predictions = dict(llm_predicted_flow)
        if len(layer) > 1:
            layer_results = [None] * len(layer)
            worker_count = min(agent_cfg.incident_workers, len(layer))
            worker_label = "parallel LLM workers" if call_llm else "parallel context workers"
            print(f"Layer {layer_index}: running {len(layer)} nodes with {worker_count} {worker_label}")
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        run_layer_node,
                        data,
                        agent_cfg,
                        impact_scope,
                        llm_context,
                        memory_cfg,
                        incident_id,
                        foundation_model_prediction,
                        str(forecaster.checkpoint_path) if forecaster is not None else None,
                        output_dir,
                        call_llm,
                        int(node_index),
                        int(layer_index),
                        list(last_layer) if call_llm else [],
                        previous_layer_predictions,
                        episodes_path,
                        episode_memory_top_k,
                        normal_reference_base_forecast_provider if foundation_model_prediction_enabled else None,
                        timings,
                        llm_request_semaphore,
                        reflection_batch,
                        no_incident_history_index,
                        incident_vector_index,
                        context_retrieval_semaphore,
                    ): position
                    for position, node_index in enumerate(layer)
                }
                for future in as_completed(futures):
                    layer_results[futures[future]] = future.result()
        else:
            layer_results = [
                run_layer_node(
                    data=data,
                    agent_cfg=agent_cfg,
                    impact_scope=impact_scope,
                    llm_context=llm_context,
                    memory_cfg=memory_cfg,
                    incident_id=incident_id,
                    base_forecast=foundation_model_prediction,
                    base_forecast_source=(
                        str(forecaster.checkpoint_path) if forecaster is not None else None
                    ),
                    output_dir=output_dir,
                    call_llm=call_llm,
                    node_index=int(node_index),
                    layer_index=int(layer_index),
                    previous_layer=list(last_layer) if call_llm else [],
                    llm_predicted_flow=previous_layer_predictions,
                    episodes_path=episodes_path,
                    episode_memory_top_k=episode_memory_top_k,
                    normal_reference_base_forecast_provider=(
                        normal_reference_base_forecast_provider if foundation_model_prediction_enabled else None
                    ),
                    timings=timings,
                    llm_request_semaphore=llm_request_semaphore,
                    reflection_batch=reflection_batch,
                    no_incident_history_index=no_incident_history_index,
                    incident_vector_index=incident_vector_index,
                    context_retrieval_semaphore=context_retrieval_semaphore,
                )
                for node_index in layer
            ]
        if timings is not None and layer_started is not None:
            timings.record(
                "layer.total",
                (time.perf_counter() - layer_started) * 1000.0,
                {"incident_id": str(incident_id), "layer_index": int(layer_index)},
            )
        for result in layer_results:
            node_records.append(result["node_record"])
            if result["reflection_future"] is not None:
                reflection_futures.append(result["reflection_future"])
            if call_llm:
                llm_predicted_flow[int(result["node_index"])] = result["llm_predicted_flow"]
        last_layer = layer

    if reflection_batch is not None and reflection_futures:
        with timed_section(timings, "case.wait_episode_reflections", incident_id=str(incident_id)):
            reflection_batch.wait_for(reflection_futures)

    aggregate = {
        "incident_id": str(incident_id),
        "sample_id": sample_id,
        "incident_nearest_node_index": int(target_node_index),
        "incident_nearest_node_id": int(data.node_order[target_node_index]),
        "impact_scope": impact_scope.to_dict(),
        "target_node_index_set": [int(index) for index in target_node_index_set],
        "target_node_id_set": [int(data.node_order[int(index)]) for index in target_node_index_set],
        "osrm_routes": osrm_scope["routes"],
        "node_layers": [[int(index) for index in layer] for layer in node_layers],
        "node_layer_ids": [[int(data.node_order[int(index)]) for index in layer] for layer in node_layers],
        "node_records": node_records,
        "evidence_capabilities": {
            "base_prediction": foundation_model_prediction_enabled,
            "normal_reference": normal_counterfactual_enabled,
        },
    }
    aggregate_path = incident_aggregate_path(output_dir, sample_id)
    with timed_section(timings, "case.write_aggregate", incident_id=str(incident_id)):
        write_json(aggregate_path, aggregate)
    return {
        "incident_id": str(incident_id),
        "target_node_id": int(data.node_order[target_node_index]),
        "node_count": len(target_node_index_set),
        "layer_count": len(node_layers),
        "aggregate_path": str(aggregate_path),
    }


def run_case(
    data: TrafficData,
    forecaster: ForecastingModelForecaster | None,
    agent_cfg: AgentConfig,
    impact_scope: ImpactScopeConfig,
    llm_context: LLMContextConfig,
    memory_cfg: dict,
    incident_id: str,
    output_dir: Path,
    call_llm: bool,
    episodes_path: Path | None = None,
    episode_memory_top_k: int = 3,
    timings: TimingRecorder | None = None,
    forecaster_lock=None,
    llm_request_semaphore=None,
    reflection_batch: ReflectionBatch | None = None,
    no_incident_history_index: NoIncidentHistoryIndex | None = None,
    incident_vector_index: IncidentVectorIndex | None = None,
    forecast_service: ForecastService | None = None,
    context_retrieval_semaphore=None,
) -> dict:
    """Run a case, optionally sharing its reflection executor with other cases."""
    run_kwargs = {
        "data": data,
        "forecaster": forecaster,
        "agent_cfg": agent_cfg,
        "impact_scope": impact_scope,
        "llm_context": llm_context,
        "memory_cfg": memory_cfg,
        "incident_id": incident_id,
        "output_dir": output_dir,
        "call_llm": call_llm,
        "episodes_path": episodes_path,
        "episode_memory_top_k": episode_memory_top_k,
        "timings": timings,
        "forecaster_lock": forecaster_lock,
        "llm_request_semaphore": llm_request_semaphore,
        "no_incident_history_index": no_incident_history_index,
        "incident_vector_index": incident_vector_index,
        "forecast_service": forecast_service,
        "context_retrieval_semaphore": context_retrieval_semaphore,
    }
    if reflection_batch is not None:
        return _run_case_impl(**run_kwargs, reflection_batch=reflection_batch)
    if not call_llm or episodes_path is None or not llm_context.foundation_model_prediction:
        return _run_case_impl(**run_kwargs, reflection_batch=None)
    reflection_workers = agent_cfg.reflection_workers
    local_batch = ReflectionBatch(max_workers=reflection_workers)
    try:
        return _run_case_impl(**run_kwargs, reflection_batch=local_batch)
    finally:
        local_batch.wait()


# Paper-facing names for the two stages implemented by this module.
run_phased_prediction = run_layer_node
run_context_aware_phased_reasoning = run_case
