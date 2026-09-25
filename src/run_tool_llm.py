import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from threading import Semaphore

from src.config import default_config_path, load_run_config, use_region_artifacts
from src.data.regions import REGIONS, get_region


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=sorted(REGIONS), default="sacramento")
    parser.add_argument("--data-dir")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", help="Override run.device: auto, cpu, cuda, or cuda:<index>.")
    parser.add_argument("--incident-id", action="append", default=[])
    parser.add_argument("--incident-workers", type=int)
    parser.add_argument("--select-count", type=int, default=3)
    parser.add_argument("--select-month", type=int, default=2)
    parser.add_argument("--call-llm", action="store_true")
    return parser.parse_args()


def _output_dir(region: str, model: str, incident_ids: list[str], output_root: Path, call_llm: bool) -> Path:
    if not incident_ids:
        name = "selected-cases"
    elif len(incident_ids) == 1:
        name = f"incident-{incident_ids[0]}"
    else:
        name = "incidents-" + "_".join(incident_ids)
    root = incident_evaluation_dir(output_root, region, model, call_llm)
    return root / name


def _select_incident_ids(data, eligible_incidents, requested_ids, count, month):
    all_ids = set(data.incidents["incident_id"].astype(str))
    missing = [incident_id for incident_id in requested_ids if incident_id not in all_ids]
    if missing:
        raise ValueError(f"Unknown incident_ids: {missing}")
    eligible_ids = set(eligible_incidents["incident_id"].astype(str))
    if requested_ids:
        selected = [incident_id for incident_id in requested_ids if incident_id in eligible_ids]
        selection_records = None
    else:
        selection_records = select_impact_cases(data, eligible_incidents, count=count, month=month)
        selected = [record["incident_id"] for record in selection_records]
    if len(set(selected)) != len(selected):
        raise ValueError(f"incident_ids must be unique, got {selected}")
    times = dict(zip(eligible_incidents["incident_id"].astype(str), eligible_incidents["dt_parsed"]))
    return sorted(selected, key=times.__getitem__), selection_records


def _run_timed_case(kwargs):
    started = time.perf_counter()
    with timed_section(kwargs["timings"], "case.total", incident_id=str(kwargs["incident_id"])):
        result = run_case(**kwargs)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        **result,
        "processing_time_ms": elapsed_ms,
        "processing_time_seconds": elapsed_ms / 1000.0,
    }


def _run_cases(case_kwargs: list[dict], worker_count: int) -> list[dict]:
    if worker_count <= 1:
        return [_run_timed_case(kwargs) for kwargs in case_kwargs]
    results = [None] * len(case_kwargs)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_run_timed_case, kwargs): position
            for position, kwargs in enumerate(case_kwargs)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def _processing_summary(results: list[dict]) -> dict:
    total_ms = sum(float(result["processing_time_ms"]) for result in results)
    return {
        "incident_count": len(results),
        "total_ms": total_ms,
        "average_ms": total_ms / len(results) if results else 0.0,
        "incidents": [
            {
                "incident_id": result["incident_id"],
                "processing_time_ms": result["processing_time_ms"],
                "processing_time_seconds": result["processing_time_seconds"],
            }
            for result in results
        ],
    }


def main() -> None:
    args = parse_args()
    # Keep ``--help`` and configuration inspection usable without importing
    # torch, kymatio, or the complete retrieval stack. These imports are still
    # loaded before any actual forecast work begins, so runtime behavior is
    # unchanged.
    from src.evaluation.selection import select_impact_cases
    from src.models.forecaster import ForecastingModelForecaster
    from src.models.factory import MODEL_CLASSES
    from src.data.traffic import TrafficData
    from src.tool_llm.execution import ForecastService, ReflectionBatch, run_case
    from src.agents.episode_memory import clear_episode_memory
    from src.tool_llm.timing import TimingRecorder, timed_section
    from src.llmmanager.scheduling import RequestSlots
    from src.llmmanager.config import initialize_provider_settings
    from src.utils.json_io import write_json
    from src.utils.paths import incident_evaluation_dir, incident_evaluation_file
    from src.utils.devices import resolve_torch_device

    region = get_region(args.region)
    started = time.perf_counter()
    timings = TimingRecorder()
    config = load_run_config(args.config or default_config_path())
    if args.call_llm:
        initialize_provider_settings()
    config = use_region_artifacts(config, region.key)
    data = TrafficData(args.data_dir or region.data_dir)
    preprocessing = data.preprocess_incidents_for_impact_scope(config.impact_scope)
    requested_ids = [str(value) for value in (args.incident_id or config.run.incident_ids)]
    incident_ids, selection_records = _select_incident_ids(
        data,
        preprocessing.eligible_incidents,
        requested_ids,
        args.select_count,
        args.select_month,
    )
    checkpoint = Path(args.checkpoint) if args.checkpoint else config.run.checkpoint
    checkpoint_stem = Path(checkpoint).stem.lower()
    model_name = next(
        (name for name in sorted(MODEL_CLASSES, key=len, reverse=True) if checkpoint_stem.startswith(f"{name}_")),
        "graph_wavenet",
    )
    output_dir = Path(args.output_dir) if args.output_dir else _output_dir(
        region.key, model_name,
        requested_ids,
        config.run.output_root,
        args.call_llm,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = replace(
        config,
        memory=replace(
            config.memory,
            # Keep memory inside this experiment directory so runs are
            # independent and self-contained.
            episodes_path=output_dir / "memory" / "episodes_reflection.jsonl",
        ),
    )
    timing_path = incident_evaluation_file(output_dir, "timing")
    timings.configure_report(
        timing_path,
        {"call_llm": args.call_llm, "incident_ids": requested_ids, "output_dir": str(output_dir)},
    )
    preprocessing.eligible_incidents.to_csv(incident_evaluation_file(output_dir, "eligible_incidents"), sep="\t", index=False)
    preprocessing.excluded_incidents.to_csv(incident_evaluation_file(output_dir, "excluded_incidents"), sep="\t", index=False)
    preprocessing_metadata = preprocessing.summary()
    write_json(output_dir / "preprocessing.json", preprocessing_metadata)
    if selection_records is not None:
        write_json(output_dir / "selected-incidents.json", selection_records)

    base_enabled = bool(config.llm_context.base_prediction)
    forecaster = None
    if base_enabled:
        device = resolve_torch_device(args.device or config.run.device)
        forecaster = ForecastingModelForecaster(checkpoint, data, device)
    # Disabling episode memory also disables the separate reflection request.
    episode_memory_enabled = bool(
        args.call_llm and config.llm_context.is_enabled("episode_memory")
        and base_enabled
    )
    if episode_memory_enabled:
        cleared_episode_count = clear_episode_memory(config.memory.episodes_path)
        timings.update_report_metadata(
            episode_memory={
                "path": str(config.memory.episodes_path),
                "cleared_before_run": True,
                "cleared_episode_count": int(cleared_episode_count),
            }
        )
    configured_workers = args.incident_workers or config.agent.incident_workers
    worker_count = min(configured_workers, len(incident_ids))
    request_semaphore = RequestSlots(config.agent.max_concurrent_llm_requests, config.agent.reflection_request_limit or None) if args.call_llm else None
    reflection_batch = ReflectionBatch(config.agent.reflection_workers) if episode_memory_enabled else None
    forecast_service = ForecastService(forecaster, timings) if forecaster is not None else None
    context_retrieval_semaphore = Semaphore(config.agent.context_retrieval_workers)
    case_kwargs = [
        {
            "data": data,
            "forecaster": forecaster,
            "agent_cfg": config.agent,
            "impact_scope": config.impact_scope,
            "llm_context": config.llm_context,
            "memory_cfg": config.memory.policy(),
            "incident_id": incident_id,
            "output_dir": output_dir,
            "call_llm": args.call_llm,
            "episodes_path": config.memory.episodes_path if episode_memory_enabled else None,
            "episode_memory_top_k": config.memory.retrieval_top_k,
            "timings": timings,
            "llm_request_semaphore": request_semaphore,
            "reflection_batch": reflection_batch,
            "forecast_service": forecast_service,
            "context_retrieval_semaphore": context_retrieval_semaphore,
        }
        for incident_id in incident_ids
    ]
    try:
        try:
            results = _run_cases(case_kwargs, worker_count)
        finally:
            if reflection_batch is not None:
                reflection_batch.wait()
    except BaseException as exc:
        timings.flush_report(status="failed", error=f"{type(exc).__name__}: {exc}", force=True)
        raise
    finally:
        if forecast_service is not None:
            forecast_service.close()

    processing = _processing_summary(results)
    concurrency = {
        "incident_workers_configured": configured_workers,
        "incident_workers_used": worker_count,
        "context_retrieval_workers": config.agent.context_retrieval_workers,
        "max_concurrent_llm_requests": config.agent.max_concurrent_llm_requests if args.call_llm else None,
        "provider_request_limits": (
            dict(request_semaphore.provider_limits) if request_semaphore else None
        ),
        "provider_max_active": (
            dict(request_semaphore.provider_max_active) if request_semaphore else None
        ),
        "provider_rate_limit_events": (
            dict(request_semaphore.provider_rate_limit_events)
            if request_semaphore else None
        ),
        "global_rate_limit_events": (
            int(request_semaphore.global_rate_limit_events)
            if request_semaphore else None
        ),
        "reflection_workers": config.agent.reflection_workers if episode_memory_enabled else None,
    }
    timings.record("main.total", (time.perf_counter() - started) * 1000.0)
    timings.update_report_metadata(concurrency=concurrency, incident_processing=processing)
    timings.flush_report(status="completed", force=True)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "cases": results,
                "incident_preprocessing": preprocessing_metadata,
                "concurrency": concurrency,
                "incident_processing": processing,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved timing report to {timing_path}")


if __name__ == "__main__":
    main()
