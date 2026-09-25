"""Small OSRM HTTP client used by the Fresno incident scope."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class OSRMPoint:
    longitude: float
    latitude: float
    snapped_longitude: float
    snapped_latitude: float
    name: str


class OSRMError(RuntimeError):
    pass


@dataclass(frozen=True)
class OSRMRouteCache:
    incident_ids: tuple[str, ...]
    node_ids: tuple[int, ...]
    distance_m: object
    duration_s: object
    bearing_deg: object
    incident_snapped: object
    sensor_snapped: object
    incident_road_names: tuple[str, ...]
    sensor_road_names: tuple[str, ...]
    incident_positions: dict[str, int]
    completed_count: int

    @classmethod
    def load(cls, path: str | Path) -> "OSRMRouteCache":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"OSRM route cache does not exist: {path}. "
                "Run `python -m src.data.precompute_osrm --region fresno` first."
            )
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            incident_ids = tuple(str(value) for value in metadata["incident_ids"])
            node_ids = tuple(int(value) for value in metadata["node_ids"])
            distance_m = payload["distance_m"]
            duration_s = payload["duration_s"]
            if distance_m.shape != duration_s.shape or distance_m.shape != (len(incident_ids), len(node_ids)):
                raise ValueError(
                    f"Invalid OSRM route cache shape: path={path}, "
                    f"distance={distance_m.shape}, duration={duration_s.shape}, "
                    f"incidents={len(incident_ids)}, nodes={len(node_ids)}"
                )
            return cls(
                incident_ids=incident_ids,
                node_ids=node_ids,
                distance_m=distance_m,
                duration_s=duration_s,
                bearing_deg=payload["bearing_deg"],
                incident_snapped=payload["incident_snapped"],
                sensor_snapped=payload["sensor_snapped"],
                incident_road_names=tuple(str(value) for value in metadata["incident_road_names"]),
                sensor_road_names=tuple(str(value) for value in metadata["sensor_road_names"]),
                incident_positions={incident_id: index for index, incident_id in enumerate(incident_ids)},
                completed_count=int(metadata.get("completed_count", len(incident_ids))),
            )

    def incident_position(self, incident_id: str) -> int:
        try:
            return self.incident_positions[str(incident_id)]
        except ValueError as exc:
            raise KeyError(f"Incident is absent from OSRM route cache: {incident_id}") from exc


def save_route_cache(path: str | Path, *, incident_ids, node_ids, distance_m, duration_s,
                     bearing_deg, incident_snapped, sensor_snapped,
                     incident_road_names, sensor_road_names, completed_count: int | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 2,
        "route_scope": "all_sensors_osrm_driving_distance",
        "incident_ids": [str(value) for value in incident_ids],
        "node_ids": [int(value) for value in node_ids],
        "incident_road_names": [str(value) for value in incident_road_names],
        "sensor_road_names": [str(value) for value in sensor_road_names],
        "completed_count": int(len(incident_ids) if completed_count is None else completed_count),
    }
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(metadata), dtype=str),
        distance_m=np.asarray(distance_m, dtype=np.float32),
        duration_s=np.asarray(duration_s, dtype=np.float32),
        bearing_deg=np.asarray(bearing_deg, dtype=np.float32),
        incident_snapped=np.asarray(incident_snapped, dtype=np.float64),
        sensor_snapped=np.asarray(sensor_snapped, dtype=np.float64),
    )


def _bearing_deg(start: OSRMPoint, end: OSRMPoint) -> float:
    lat1 = math.radians(start.snapped_latitude)
    lat2 = math.radians(end.snapped_latitude)
    delta_lon = math.radians(end.snapped_longitude - start.snapped_longitude)
    value = math.atan2(
        math.sin(delta_lon) * math.cos(lat2),
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon),
    )
    return float((math.degrees(value) + 360.0) % 360.0)


class OSRMClient:
    def __init__(
        self,
        base_url: str,
        profile: str = "driving",
        timeout_seconds: float = 30.0,
        retry_count: int = 4,
        retry_backoff_seconds: float = 1.0,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.profile = str(profile)
        self.timeout_seconds = float(timeout_seconds)
        self.retry_count = int(retry_count)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        if self.retry_count < 0:
            raise ValueError(f"retry_count must be non-negative, got {self.retry_count}")
        if self.retry_backoff_seconds < 0:
            raise ValueError(
                f"retry_backoff_seconds must be non-negative, got {self.retry_backoff_seconds}"
            )

    def _get(self, path: str, params: dict[str, str]) -> dict:
        query = urlencode(params)
        request = Request(f"{self.base_url}{path}?{query}", headers={"User-Agent": "incident-osrm/1.0"})
        attempts = self.retry_count + 1
        for attempt in range(attempts):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
            except (URLError, TimeoutError, OSError) as exc:
                retryable = True
            except Exception as exc:
                raise OSRMError(f"OSRM request failed: {request.full_url}: {exc}") from exc
            if not retryable or attempt + 1 >= attempts:
                raise OSRMError(
                    f"OSRM request failed after {attempt + 1}/{attempts} attempt(s): "
                    f"{request.full_url}: {exc}"
                ) from exc
            time.sleep(self.retry_backoff_seconds * (2 ** attempt))
        if payload.get("code") != "Ok":
            raise OSRMError(f"OSRM returned {payload.get('code')}: {payload}")
        return payload

    def table(self, source: OSRMPoint, destinations: list[OSRMPoint]) -> list[dict]:
        points = [source, *destinations]
        coordinates = ";".join(
            f"{point.snapped_longitude},{point.snapped_latitude}" for point in points
        )
        payload = self._get(
            f"/table/v1/{self.profile}/{coordinates}",
            {"annotations": "duration,distance"},
        )
        distances = payload.get("distances", [[]])[0]
        durations = payload.get("durations", [[]])[0]
        if len(distances) != len(destinations) + 1 or len(durations) != len(destinations) + 1:
            raise OSRMError("OSRM table response has an unexpected shape")
        rows = []
        for index, destination in enumerate(destinations, start=1):
            distance = distances[index]
            duration = durations[index]
            rows.append(
                {
                    "distance_m": None if distance is None else float(distance),
                    "duration_s": None if duration is None else float(duration),
                    "bearing_deg": _bearing_deg(source, destination),
                    "snapped_longitude": destination.snapped_longitude,
                    "snapped_latitude": destination.snapped_latitude,
                    "road_name": destination.name,
                }
            )
        return rows
