from pathlib import Path


def incident_artifact_stem(incident_id: str | int) -> str:
    return f"incident_{incident_id}"


def incident_node_artifact_stem(
    incident_id: str | int,
    layer_index: int,
    node_id: int,
) -> str:
    return f"{incident_artifact_stem(incident_id)}_layer_{layer_index}_node_{node_id}"


def incident_aggregate_filename(incident_id: str | int) -> str:
    return f"{incident_artifact_stem(incident_id)}_aggregate.json"


def incident_aggregate_path(output_dir: Path, incident_id: str | int) -> Path:
    return output_dir / incident_aggregate_filename(incident_id)
