"""Precompute regional OSRM route matrices for offline incident evaluation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from src.config import default_config_path, load_run_config
from src.data.osrm import OSRMClient, OSRMError, OSRMPoint, OSRMRouteCache, save_route_cache
from src.data.regions import REGIONS, get_region
from src.data.traffic import TrafficData


def precompute(data: TrafficData, *, base_url: str, profile: str, timeout_seconds: float,
               output_path: Path, table_chunk_size: int = 99, limit: int | None = None,
               resume: bool = False, checkpoint_every: int = 25, workers: int = 1) -> Path:
    client = OSRMClient(base_url, profile, timeout_seconds)
    if table_chunk_size <= 0:
        raise ValueError("table_chunk_size must be positive")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    # OSRM table performs road snapping internally. Every valid sensor is a
    # destination so the runtime scope can be based only on driving distance.
    sensor_points: list[OSRMPoint | None] = []
    sensor_snapped = np.full((len(data.sensors), 2), np.nan, dtype=np.float64)
    sensor_road_names = []
    for node_index, row in enumerate(
        tqdm(data.sensors.itertuples(index=False), total=len(data.sensors), desc="Load sensors")
    ):
        sensor_road_names.append(str(getattr(row, "Fwy_Name", "")))
        try:
            longitude = float(row.Lng)
            latitude = float(row.Lat)
        except (TypeError, ValueError):
            sensor_points.append(None)
            continue
        if not np.isfinite(longitude) or not np.isfinite(latitude):
            sensor_points.append(None)
            continue
        sensor_points.append(OSRMPoint(longitude, latitude, longitude, latitude, sensor_road_names[-1]))
        sensor_snapped[node_index] = [longitude, latitude]
    valid_sensor_indices = [
        index for index, point in enumerate(sensor_points) if point is not None
    ]
    valid_sensor_points = [sensor_points[index] for index in valid_sensor_indices]
    incident_rows = list(data.incidents.itertuples(index=False))
    if limit is not None:
        incident_rows = incident_rows[: int(limit)]
    count = len(incident_rows)
    node_count = len(sensor_points)
    distance_m = np.full((count, node_count), np.nan, dtype=np.float32)
    duration_s = np.full((count, node_count), np.nan, dtype=np.float32)
    bearing_deg = np.full((count, node_count), np.nan, dtype=np.float32)
    incident_snapped = np.full((count, 2), np.nan, dtype=np.float64)
    incident_ids = [str(row.incident_id) for row in incident_rows]
    incident_road_names = [""] * count
    start_index = 0
    partial_path = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    if resume and partial_path.exists():
        partial = OSRMRouteCache.load(partial_path)
        expected_ids = tuple(incident_ids)
        if partial.node_ids != tuple(int(value) for value in data.node_order):
            raise ValueError(f"Partial OSRM cache node order mismatch: {partial_path}")
        if partial.incident_ids != expected_ids:
            raise ValueError(f"Partial OSRM cache incident order mismatch: {partial_path}")
        start_index = partial.completed_count
        if not 0 <= start_index <= count:
            raise ValueError(f"Partial OSRM cache completed count is invalid: {partial_path}")
        if partial.distance_m.shape != (count, node_count):
            raise ValueError(f"Partial OSRM cache shape mismatch: {partial_path}")
        distance_m[:] = partial.distance_m
        duration_s[:] = partial.duration_s
        bearing_deg[:] = partial.bearing_deg
        incident_snapped[:start_index] = partial.incident_snapped[:start_index]
        incident_road_names[:] = partial.incident_road_names

    def route_incident(incident):
        if not np.isfinite(float(incident.Longitude_num)) or not np.isfinite(float(incident.Latitude_num)):
            return None, [], []
        longitude = float(incident.Longitude_num)
        latitude = float(incident.Latitude_num)
        source = OSRMPoint(longitude, latitude, longitude, latitude, "")
        route_indices = valid_sensor_indices
        route_rows = []
        for start in range(0, len(route_indices), int(table_chunk_size)):
            destinations = valid_sensor_points[start : start + int(table_chunk_size)]
            rows = client.table(source, destinations)
            route_rows.extend(rows)
        return source, route_rows, route_indices

    pending_incidents = incident_rows[start_index:]
    def save_partial(completed_count: int) -> None:
        if completed_count <= 0:
            return
        save_route_cache(
            partial_path,
            incident_ids=incident_ids,
            node_ids=data.node_order,
            distance_m=distance_m,
            duration_s=duration_s,
            bearing_deg=bearing_deg,
            incident_snapped=incident_snapped,
            sensor_snapped=sensor_snapped,
            incident_road_names=incident_road_names,
            sensor_road_names=sensor_road_names,
            completed_count=completed_count,
        )

    executor = ThreadPoolExecutor(max_workers=int(workers)) if workers > 1 else None
    completed_count = start_index
    try:
        route_results = executor.map(route_incident, pending_incidents) if executor else map(route_incident, pending_incidents)
        completed_results = tqdm(
            zip(range(start_index, count), route_results),
            desc="Route incidents",
            initial=start_index,
            total=count,
        )
        for row_index, (source, rows, route_indices) in completed_results:
            incident_road_names[row_index] = "" if source is None else source.name
            if source is not None:
                incident_snapped[row_index] = [source.snapped_longitude, source.snapped_latitude]
                for node_index, item in zip(route_indices, rows):
                    distance_m[row_index, node_index] = np.nan if item["distance_m"] is None else item["distance_m"]
                    duration_s[row_index, node_index] = np.nan if item["duration_s"] is None else item["duration_s"]
                    bearing_deg[row_index, node_index] = np.nan if item["bearing_deg"] is None else item["bearing_deg"]
            completed_count = row_index + 1
            if completed_count % checkpoint_every == 0 and completed_count < count:
                save_partial(completed_count)
    except OSRMError as exc:
        save_partial(completed_count)
        raise OSRMError(
            f"{exc}. The partial cache was saved at {partial_path} with "
            f"{completed_count}/{count} completed incidents. Restore the OSRM service, then rerun with --resume."
        ) from exc
    except BaseException:
        save_partial(completed_count)
        raise
    finally:
        if executor:
            executor.shutdown(wait=True)
    save_route_cache(
        output_path,
        incident_ids=incident_ids,
        node_ids=data.node_order,
        distance_m=distance_m,
        duration_s=duration_s,
        bearing_deg=bearing_deg,
        incident_snapped=incident_snapped,
        sensor_snapped=sensor_snapped,
        incident_road_names=incident_road_names,
        sensor_road_names=sensor_road_names,
        completed_count=count,
    )
    if partial_path.exists():
        partial_path.unlink()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=sorted(REGIONS), default="sacramento")
    parser.add_argument("--data-dir")
    parser.add_argument("--output")
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument("--base-url", help="Override impact_scope.osrm_base_url (for a local OSRM server).")
    parser.add_argument("--profile", help="Override impact_scope.osrm_profile.")
    parser.add_argument("--table-chunk-size", type=int, default=99)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", help="Resume from the adjacent .partial.npz checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--workers", type=int, default=8, help="Concurrent incident route workers; use 1 for a local OSRM server.")
    args = parser.parse_args()
    region = get_region(args.region)
    config = load_run_config(args.config)
    data = TrafficData(args.data_dir or region.data_dir)
    output = Path(args.output) if args.output else data.data_dir / "osrm_routes.npz"
    result = precompute(
        data,
        base_url=args.base_url or config.impact_scope.osrm_base_url,
        profile=args.profile or config.impact_scope.osrm_profile,
        timeout_seconds=config.impact_scope.osrm_timeout_seconds,
        output_path=output,
        table_chunk_size=args.table_chunk_size,
        limit=args.limit,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        workers=args.workers,
    )
    print(result)


if __name__ == "__main__":
    main()
