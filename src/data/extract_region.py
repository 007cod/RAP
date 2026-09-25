import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


from src.data.regions import RAW_DATA_DIR, get_region


def validate_source(raw_dir: Path, *, require_adjacency: bool) -> None:
    source_paths = sorted((raw_dir / "year_2024").glob("2024_p*.npy"))
    missing = []
    if len(source_paths) != 12:
        missing.append(f"12 monthly arrays under {raw_dir / 'year_2024'} (found {len(source_paths)})")
    if require_adjacency and not (raw_dir / "adj_matrix.npy").exists():
        missing.append(str(raw_dir / "adj_matrix.npy"))
    if missing:
        raise FileNotFoundError("Cannot perform regional extraction; missing: " + "; ".join(missing))


def load_region_sensors(raw_dir: Path, county: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    sensors = pd.read_csv(raw_dir / "sensor_meta_feature.csv", sep="\t")
    node_order = np.load(raw_dir / "node_order.npy")
    sensor_ids = set(sensors.loc[sensors["County"] == county, "station_id"].astype(int))
    indices = np.asarray([idx for idx, station_id in enumerate(node_order) if int(station_id) in sensor_ids], dtype=np.int64)
    if indices.size != len(sensor_ids):
        raise ValueError(
            f"{county} sensor count mismatch: node_indices={indices.size}, sensor_ids={len(sensor_ids)}"
        )
    by_station = sensors.set_index("station_id", drop=False)
    ordered_sensors = by_station.loc[node_order[indices]].reset_index(drop=True)
    ordered_sensors.insert(0, "node_index", indices)
    return ordered_sensors, node_order[indices], indices


def extract_monthly_arrays(raw_dir: Path, output_dir: Path, indices: np.ndarray) -> list[dict]:
    source_year_dir = raw_dir / "year_2024"
    source_paths = sorted(source_year_dir.glob("2024_p*.npy"))
    if len(source_paths) != 12:
        raise FileNotFoundError(
            f"Expected 12 statewide monthly arrays in {source_year_dir}, found {len(source_paths)}"
        )
    out_dir = output_dir / "year_2024"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source_path in source_paths:
        source = np.load(source_path, mmap_mode="r")
        subset = np.asarray(source[indices, :, :])
        out_path = out_dir / source_path.name
        np.save(out_path, subset)
        records.append(
            {
                "file": str(out_path),
                "source_file": str(source_path),
                "shape": list(subset.shape),
                "dtype": str(subset.dtype),
            }
        )
    return records


def extract_adjacency(raw_dir: Path, output_dir: Path, indices: np.ndarray) -> dict:
    source_path = raw_dir / "adj_matrix.npy"
    if not source_path.exists():
        raise FileNotFoundError(f"Statewide adjacency matrix does not exist: {source_path}")
    adj = np.load(source_path, mmap_mode="r")
    subset = np.asarray(adj[np.ix_(indices, indices)])
    out_path = output_dir / "adj_matrix.npy"
    np.save(out_path, subset)
    return {
        "file": str(out_path),
        "shape": list(subset.shape),
        "dtype": str(subset.dtype),
        "nonzero": int(np.count_nonzero(subset)),
    }


def extract_incidents(raw_dir: Path, output_dir: Path, incident_areas: tuple[str, ...]) -> dict:
    incidents = pd.read_csv(raw_dir / "incidents_y2024.csv", sep="\t", dtype=str)
    area = incidents["AREA"].fillna("").str.strip()
    region_incidents = incidents.loc[area.isin(incident_areas)].copy()
    out_path = output_dir / "incidents_y2024.csv"
    region_incidents.to_csv(out_path, sep="\t", index=False)
    return {
        "file": str(out_path),
        "rows": int(len(region_incidents)),
        "area_counts": region_incidents["AREA"].fillna("").str.strip().value_counts().to_dict(),
    }


def write_manifest(
    output_dir: Path,
    raw_dir: Path,
    region,
    months: list[dict] | None,
    adjacency: dict | None,
    incidents: dict,
    node_count: int,
) -> None:
    if months is not None and adjacency is not None:
        status = "complete"
    elif months is not None:
        status = "missing_adjacency"
    else:
        status = "metadata_only"
    manifest = {
        "dataset": region.dataset_name,
        "region": region.key,
        "status": status,
        "source": str(raw_dir),
        "sensor_filter": f"sensor_meta_feature.County == {region.county!r}",
        "incident_filter": {"column": "AREA", "allowed_values": list(region.incident_areas)},
        "incident_sensor_alignment": "not performed",
        "node_count": int(node_count),
        "monthly_arrays": months,
        "adjacency": adjacency,
        "incidents": incidents,
        "channels": {
            "0": "flow",
            "1": "occupancy",
            "2": "speed",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Extract one county dataset from statewide California data")
    parser.add_argument("--region", required=True, choices=("sacramento", "fresno", "kern", "san_francisco"))
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument(
        "--skip-adjacency",
        action="store_true",
        help="Extract traffic arrays but leave the dataset marked incomplete until adjacency is available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    region = get_region(args.region)
    if args.metadata_only and args.skip_adjacency:
        raise ValueError("--metadata-only and --skip-adjacency cannot be used together")
    if not args.metadata_only:
        validate_source(args.raw_dir, require_adjacency=not args.skip_adjacency)
    output_dir = region.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sensors, node_order, indices = load_region_sensors(args.raw_dir, region.county)
    sensors.to_csv(output_dir / "sensor_meta_feature.csv", sep="\t", index=False)
    np.save(output_dir / "node_order.npy", node_order)
    incidents = extract_incidents(args.raw_dir, output_dir, region.incident_areas)
    months = None if args.metadata_only else extract_monthly_arrays(args.raw_dir, output_dir, indices)
    adjacency = (
        None
        if args.metadata_only or args.skip_adjacency
        else extract_adjacency(args.raw_dir, output_dir, indices)
    )
    write_manifest(output_dir, args.raw_dir, region, months, adjacency, incidents, len(node_order))
    print(json.dumps({"output_dir": str(output_dir), "nodes": len(node_order), **incidents}, indent=2))


if __name__ == "__main__":
    main()
