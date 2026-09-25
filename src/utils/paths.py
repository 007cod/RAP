"""Canonical paths for generated training and incident-evaluation artifacts."""

from __future__ import annotations

from pathlib import Path


INCIDENT_EVALUATION_FILES = {
    "metrics": "metrics.json",
    "timing": "timing.json",
    "split": "split.json",
    "selected_incidents": "selected-incidents.json",
    "evaluated_incidents": "evaluated-incidents.json",
    "skipped_test_incidents": "skipped-test-incidents.tsv",
    "eligible_incidents": "eligible-incidents.tsv",
    "excluded_incidents": "excluded-incidents.tsv",
    "partial_metrics": "case-metrics.partial.json",
    "horizon_metrics": "horizon-metrics.json",
    "horizon_workbook": "horizon-metrics.xlsx",
}


def incident_evaluation_dir(
    output_root: str | Path,
    region: str,
    model: str,
    call_llm: bool,
) -> Path:
    mode = "tool-llm" if call_llm else "baseline"
    return Path(output_root) / "incident-evaluations" / str(region) / str(model) / mode


def incident_evaluation_file(output_dir: str | Path, key: str) -> Path:
    try:
        filename = INCIDENT_EVALUATION_FILES[key]
    except KeyError as error:
        raise ValueError(f"Unknown incident evaluation file key: {key}") from error
    return Path(output_dir) / filename


def model_artifact_paths(artifacts_dir: str | Path, model: str, region: str) -> dict[str, Path]:
    """Keep the existing regional artifact layout while standardizing its names."""
    root = Path(artifacts_dir)
    stem = f"{model}_{region}"
    return {
        "checkpoint": root / "models" / f"{stem}.pt",
        "last_checkpoint": root / "models" / f"{stem}_last.pt",
        "metrics": root / "reports" / f"{stem}_metrics.json",
        "history": root / "reports" / f"{stem}_history.json",
        "config": root / "reports" / f"{stem}_train_config.json",
        "split": root / "reports" / f"{stem}_split.json",
    }
