from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.osrm import OSRMRouteCache


DEFAULT_ONE_HOP_MIN_EDGE_WEIGHT = 0.5


class IncidentSensorMatchError(ValueError):
    """Raised when an incident has no valid sensor on its road and direction."""


@dataclass(frozen=True)
class TrafficWindow:
    history_flow: list[float]
    future_flow: list[float]


@dataclass(frozen=True)
class IncidentScopePreprocessing:
    eligible_incidents: pd.DataFrame
    excluded_incidents: pd.DataFrame
    max_distance_km: float

    def summary(self) -> dict:
        excluded_by_reason = self.excluded_incidents["skip_reason"].value_counts().to_dict()
        return {
            "scope_method": "osrm_driving_distance",
            "max_distance_km": float(self.max_distance_km),
            "eligible_incident_count": int(len(self.eligible_incidents)),
            "excluded_incident_count": int(len(self.excluded_incidents)),
            "excluded_by_reason": {str(key): int(value) for key, value in excluded_by_reason.items()},
        }


class TrafficData:
    def __init__(self, data_dir: str | Path = "data/processed/sacramento_2024"):
        self.data_dir = Path(data_dir)
        self.node_order = np.load(self.data_dir / "node_order.npy")
        self.sensors = pd.read_csv(self.data_dir / "sensor_meta_feature.csv", sep="\t")
        self._incident_sensor_indices = self._build_incident_sensor_index()
        self.incidents = self._load_incidents()
        self.adj_matrix = np.load(self.data_dir / "adj_matrix.npy", mmap_mode="r")
        self._network_distance_matrix: np.ndarray | None = None
        self.month_arrays = {
            month: np.load(self.data_dir / "year_2024" / f"2024_p{month:02d}.npy", mmap_mode="r")
            for month in range(1, 13)
        }
        self._incident_scope_preprocessing_cache = {}
        self.osrm_cache_path = self.data_dir / "osrm_routes.npz"
        self._osrm_route_cache: OSRMRouteCache | None = None
        self._osrm_scope_cache = {}

    def network_distance_matrix(self) -> np.ndarray:
        """Directed network distances for regional nodes, in meters.

        The statewide matrix uses the statewide ``node_order.npy`` ordering, so
        regional data must be remapped by node ID before it is used.
        """
        cached = getattr(self, "_network_distance_matrix", None)
        if cached is not None:
            return cached
        source_dir = self.data_dir.parents[1] / "raw" / "california_2024"
        source_matrix = source_dir / "dis_matrix.npy"
        source_order = source_dir / "node_order.npy"
        if not source_matrix.exists() or not source_order.exists():
            raise FileNotFoundError(
                "Distance-based incident layers require statewide distance data: "
                f"{source_matrix} and {source_order}"
            )
        statewide_node_order = np.load(source_order)
        statewide_positions = {int(node_id): index for index, node_id in enumerate(statewide_node_order)}
        try:
            regional_positions = np.asarray(
                [statewide_positions[int(node_id)] for node_id in self.node_order], dtype=np.int64
            )
        except KeyError as exc:
            raise ValueError(
                f"Regional node ID is absent from statewide distance matrix: {exc.args[0]}"
            ) from exc
        statewide_distances = np.load(source_matrix, mmap_mode="r")
        distances = np.asarray(
            statewide_distances[np.ix_(regional_positions, regional_positions)], dtype=np.float64
        )
        distances.setflags(write=False)
        self._network_distance_matrix = distances
        return distances

    def network_distance_km(self, source_index: int, target_index: int) -> float | None:
        source_index = int(source_index)
        target_index = int(target_index)
        if source_index == target_index:
            return 0.0
        distance_m = float(self.network_distance_matrix()[source_index, target_index])
        # The source matrix encodes unreachable/invalid pairs as non-positive.
        return distance_m / 1000.0 if np.isfinite(distance_m) and distance_m > 0 else None

    def _load_incidents(self) -> pd.DataFrame:
        incidents = pd.read_csv(self.data_dir / "incidents_y2024.csv", sep="\t", dtype=str)
        incidents["dt_parsed"] = pd.to_datetime(incidents["dt"], format="%m/%d/%Y %H:%M:%S")
        incidents["duration_minutes"] = pd.to_numeric(incidents["duration"])
        incidents["Latitude_num"] = pd.to_numeric(incidents["Latitude"])
        incidents["Longitude_num"] = pd.to_numeric(incidents["Longitude"])
        return incidents.sort_values("dt_parsed").reset_index(drop=True)

    def preprocess_incidents_for_impact_scope(self, impact_scope) -> IncidentScopePreprocessing:
        """Select incidents with at least one cached OSRM route in the configured bands."""
        rules = tuple(
            (float(rule["min_exclusive_km"]), float(rule["max_inclusive_km"]))
            for rule in impact_scope.layer_rules
        )
        cached = self._incident_scope_preprocessing_cache.get(rules)
        if cached is not None:
            return cached

        incidents = self.incidents
        count = len(incidents)
        target_indices = np.full(count, -1, dtype=np.int64)
        target_distances = np.full(count, np.nan, dtype=np.float64)
        impact_node_counts = np.zeros(count, dtype=np.int64)
        skip_reasons = np.full(count, "missing_osrm_route_cache", dtype=object)
        cache = self._load_osrm_route_cache()
        for position, incident_id in enumerate(incidents["incident_id"].astype(str)):
            cache_position = cache.incident_positions.get(incident_id)
            if cache_position is None:
                continue
            distances_km = np.asarray(cache.distance_m[cache_position], dtype=np.float64) / 1000.0
            selected_mask = np.zeros(distances_km.shape, dtype=bool)
            for layer_index, (lower, upper) in enumerate(rules):
                selected_mask |= np.isfinite(distances_km) & (distances_km <= upper) & (
                    distances_km >= lower if layer_index == 0 else distances_km > lower
                )
            selected = np.flatnonzero(selected_mask)
            if selected.size == 0:
                skip_reasons[position] = "no_osrm_route_in_distance_bands"
                continue
            anchor = int(selected[np.argmin(distances_km[selected])])
            target_indices[position] = anchor
            target_distances[position] = float(distances_km[anchor])
            impact_node_counts[position] = int(selected.size)
            skip_reasons[position] = ""

        eligible_mask = skip_reasons == ""
        eligible = incidents.loc[eligible_mask].copy()
        eligible["impact_target_node_index"] = target_indices[eligible_mask]
        eligible["impact_target_node_id"] = self.node_order[target_indices[eligible_mask]]
        eligible["impact_target_distance_km"] = target_distances[eligible_mask]
        eligible["impact_node_count"] = impact_node_counts[eligible_mask]
        excluded = incidents.loc[~eligible_mask].copy()
        excluded["skip_reason"] = skip_reasons[~eligible_mask]
        excluded["osrm_anchor_node_index"] = target_indices[~eligible_mask]
        excluded["osrm_anchor_distance_km"] = target_distances[~eligible_mask]
        excluded["impact_node_count"] = impact_node_counts[~eligible_mask]
        result = IncidentScopePreprocessing(eligible, excluded, rules[-1][1])
        self._incident_scope_preprocessing_cache[rules] = result
        return result

    def _load_osrm_route_cache(self) -> OSRMRouteCache:
        cached = getattr(self, "_osrm_route_cache", None)
        if cached is not None:
            return cached
        path = getattr(self, "osrm_cache_path", self.data_dir / "osrm_routes.npz")
        cache = OSRMRouteCache.load(path)
        expected_node_ids = tuple(int(value) for value in self.node_order)
        if cache.node_ids != expected_node_ids:
            raise ValueError(
                f"OSRM route cache node order does not match dataset: cache={path}, "
                f"cache_nodes={len(cache.node_ids)}, dataset_nodes={len(expected_node_ids)}"
            )
        self._osrm_route_cache = cache
        return cache

    def osrm_incident_node_route(self, incident, node_index: int) -> dict:
        node_index = int(node_index)
        if not 0 <= node_index < len(self.node_order):
            raise IndexError(f"Invalid node index for OSRM route lookup: {node_index}")
        incident_id = str(getattr(incident, "incident_id"))
        cache = self._load_osrm_route_cache()
        incident_position = cache.incident_position(incident_id)
        distance_m = float(cache.distance_m[incident_position, node_index])
        duration_s = float(cache.duration_s[incident_position, node_index])
        sensor = self.sensors.iloc[node_index]
        incident_road = _road_key(getattr(incident, "Fwy", ""))
        incident_direction = _direction_key(getattr(incident, "Freeway_direction", ""))
        return {
            "node_index": node_index,
            "node_id": int(self.node_order[node_index]),
            "distance_km": distance_m / 1000.0 if np.isfinite(distance_m) and distance_m >= 0 else None,
            "duration_s": duration_s if np.isfinite(duration_s) and duration_s >= 0 else None,
            "bearing_deg": float(cache.bearing_deg[incident_position, node_index]),
            "road_metadata_relation": {
                "same_road": _road_key(sensor["Fwy"]) == incident_road,
                "same_direction": _direction_key(sensor["Direction"]) == incident_direction,
                "sensor_freeway": str(sensor["Fwy"]),
                "sensor_direction": str(sensor["Direction"]),
                "sensor_type": str(sensor.get("Type", "")),
            },
        }

    def osrm_impact_scope(self, incident, impact_scope) -> dict:
        incident_id = str(getattr(incident, "incident_id"))
        rules = tuple(
            (float(rule["min_exclusive_km"]), float(rule["max_inclusive_km"]))
            for rule in impact_scope.layer_rules
        )
        cache_key = (incident_id, rules)
        cached = getattr(self, "_osrm_scope_cache", {}).get(cache_key)
        if cached is not None:
            return cached
        routes = [self.osrm_incident_node_route(incident, node_index) for node_index in range(len(self.node_order))]
        node_layers = []
        for layer_index, (lower, upper) in enumerate(rules):
            layer = []
            for route in routes:
                distance_km = route["distance_km"]
                if distance_km is None or distance_km > upper:
                    continue
                if (layer_index == 0 and distance_km >= lower) or (
                    layer_index > 0 and distance_km > lower
                ):
                    layer.append(route)
            node_layers.append(sorted(layer, key=lambda route: (route["distance_km"], route["node_index"])))
        selected_routes = [route for layer in node_layers for route in layer]
        if not selected_routes:
            result = {"anchor_node_index": None, "anchor_distance_km": None, "node_indices": [], "node_layers": [[] for _ in rules], "routes": []}
        else:
            anchor = min(selected_routes, key=lambda route: (route["distance_km"], route["node_index"]))
            result = {
                "anchor_node_index": int(anchor["node_index"]),
                "anchor_distance_km": float(anchor["distance_km"]),
                "node_indices": [int(route["node_index"]) for route in selected_routes],
                "node_layers": [[int(route["node_index"]) for route in layer] for layer in node_layers],
                "routes": selected_routes,
            }
        self._osrm_scope_cache[cache_key] = result
        return result

    def _build_incident_sensor_index(self) -> dict[tuple[str, str], np.ndarray]:
        buckets: dict[tuple[str, str], list[int]] = {}
        freeway_keys = self.sensors["Fwy"].map(_road_key)
        direction_keys = self.sensors["Direction"].map(_direction_key)
        for node_index, (freeway_key, direction_key) in enumerate(zip(freeway_keys, direction_keys)):
            buckets.setdefault((freeway_key, direction_key), []).append(int(node_index))
        index = {}
        for key, node_indices in buckets.items():
            values = np.asarray(node_indices, dtype=np.int64)
            values.setflags(write=False)
            index[key] = values
        return index

    def nearest_node_index(self, latitude: float, longitude: float) -> int:
        lat = self.sensors["Lat"].to_numpy(dtype=np.float64)
        lng = self.sensors["Lng"].to_numpy(dtype=np.float64)
        distance = (lat - float(latitude)) ** 2 + (lng - float(longitude)) ** 2
        valid = np.isfinite(distance)
        if not np.any(valid):
            raise ValueError(f"No sensors have valid coordinates: latitude={latitude}, longitude={longitude}")
        valid_indices = np.flatnonzero(valid)
        return int(valid_indices[np.argmin(distance[valid])])

    def incident_road_sensor_indices(self, freeway, direction) -> np.ndarray:
        freeway_key = _road_key(freeway)
        direction_key = _direction_key(direction)
        node_indices = self._incident_sensor_indices.get((freeway_key, direction_key))
        if node_indices is None:
            return np.empty(0, dtype=np.int64)
        return node_indices

    def nearest_incident_node_index(self, latitude: float, longitude: float, freeway, direction) -> int:
        freeway_key = _road_key(freeway)
        direction_key = _direction_key(direction)
        road_indices = self.incident_road_sensor_indices(freeway, direction)
        if road_indices.size == 0:
            raise IncidentSensorMatchError(
                f"No sensors match incident road and direction: freeway={freeway_key}, direction={direction_key}, "
                f"latitude={latitude}, longitude={longitude}"
            )
        lat = self.sensors["Lat"].to_numpy(dtype=np.float64)
        lng = self.sensors["Lng"].to_numpy(dtype=np.float64)
        distance = (lat - float(latitude)) ** 2 + (lng - float(longitude)) ** 2
        valid_indices = road_indices[np.isfinite(distance[road_indices])]
        if valid_indices.size == 0:
            raise IncidentSensorMatchError(
                f"No same-road sensors have valid coordinates: freeway={freeway_key}, direction={direction_key}, "
                f"latitude={latitude}, longitude={longitude}"
            )
        return int(valid_indices[np.argmin(distance[valid_indices])])

    def node_distance_km(self, source_index: int, target_index: int) -> float:
        source = self.sensors.iloc[int(source_index)]
        target = self.sensors.iloc[int(target_index)]
        return _haversine_km(float(source["Lat"]), float(source["Lng"]), float(target["Lat"]), float(target["Lng"]))

    def incident_node_distance_km(self, latitude: float, longitude: float, node_index: int) -> float:
        sensor = self.sensors.iloc[int(node_index)]
        return _haversine_km(float(latitude), float(longitude), float(sensor["Lat"]), float(sensor["Lng"]))

    def incident_distances_to_node_km(self, node_index: int) -> np.ndarray:
        sensor = self.sensors.iloc[int(node_index)]
        incident_lat = self.incidents["Latitude_num"].to_numpy(dtype=np.float64)
        incident_lng = self.incidents["Longitude_num"].to_numpy(dtype=np.float64)
        distances = np.full(len(self.incidents), np.nan, dtype=np.float64)
        valid = np.isfinite(incident_lat) & np.isfinite(incident_lng)
        distances[valid] = _haversine_array_km(
            float(sensor["Lat"]),
            float(sensor["Lng"]),
            incident_lat[valid],
            incident_lng[valid],
        )
        return distances

    def nearby_nodes(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
        max_nodes: int,
        target_node_index: int | None = None,
        min_edge_weight: float = 0.0,
    ) -> dict:
        if int(max_nodes) <= 0:
            raise ValueError(f"max_nodes must be positive, got {max_nodes}")
        lat = self.sensors["Lat"].to_numpy(dtype=np.float64)
        lng = self.sensors["Lng"].to_numpy(dtype=np.float64)
        valid = np.isfinite(lat) & np.isfinite(lng)
        distances = _haversine_array_km(float(latitude), float(longitude), lat[valid], lng[valid])
        valid_indices = np.flatnonzero(valid)
        within_order = np.argsort(distances[distances <= float(radius_km)])
        within_indices = valid_indices[distances <= float(radius_km)][within_order]
        within_distances = distances[distances <= float(radius_km)][within_order]
        if target_node_index is None:
            selected_indices = within_indices[: int(max_nodes)]
            selected_distances = within_distances[: int(max_nodes)]
            edge_weights = np.zeros(selected_indices.size, dtype=np.float64)
        else:
            target_node_index = int(target_node_index)
            edge_weights = np.asarray(self.adj_matrix[target_node_index, within_indices], dtype=np.float64)
            connected_indices = self.one_hop_neighbor_indices(target_node_index, min_edge_weight)
            keep = (within_indices == target_node_index) | np.isin(within_indices, connected_indices)
            filtered_indices = within_indices[keep]
            filtered_distances = within_distances[keep]
            filtered_weights = edge_weights[keep]
            target_position = np.flatnonzero(filtered_indices == target_node_index)
            non_target = np.flatnonzero(filtered_indices != target_node_index)
            ordered_non_target = non_target[np.argsort(filtered_weights[non_target])[::-1]]
            ordered_positions = (
                np.concatenate([target_position, ordered_non_target])
                if target_position.size
                else ordered_non_target
            )
            selected_positions = ordered_positions[: int(max_nodes)]
            selected_indices = filtered_indices[selected_positions]
            selected_distances = filtered_distances[selected_positions]
            edge_weights = filtered_weights[selected_positions]
        nodes = []
        for node_index, distance_km, edge_weight in zip(selected_indices, selected_distances, edge_weights):
            sensor = self.sensors.iloc[int(node_index)]
            nodes.append(
                {
                    "node_index": int(node_index),
                    "node_id": int(self.node_order[int(node_index)]),
                    "distance_km": float(distance_km),
                    "target_edge_weight": float(edge_weight),
                    "freeway": str(sensor["Fwy"]),
                    "road_direction": str(sensor["Direction"]),
                    "name": str(sensor["Name"]),
                }
            )
        return {
            "radius_km": float(radius_km),
            "max_nodes": int(max_nodes),
            "min_edge_weight": float(min_edge_weight),
            "candidate_node_count": int(within_indices.size),
            "selected_node_count": int(len(nodes)),
            "nodes": nodes,
        }

    def nearest_node_index_set(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
        max_nodes: int,
        min_edge_weight: float,
        target_node_index: int | None = None,
    ) -> list[int]:
        target_node_index = self.nearest_node_index(latitude, longitude) if target_node_index is None else int(target_node_index)
        nearby = self.nearby_nodes(latitude, longitude, radius_km, max_nodes, target_node_index, min_edge_weight)
        return [int(node["node_index"]) for node in nearby["nodes"]]

    def get_adjacency_matrix(self, node_indices: list[int]) -> np.ndarray:
        indices = np.asarray(node_indices, dtype=np.int64)
        return np.asarray(self.adj_matrix[np.ix_(indices, indices)], dtype=np.float64)

    def get_node_layer(
        self,
        target_node_index: int,
        adjacency_matrix: np.ndarray,
        target_node_index_set: list[int],
        layer_rules: list[dict],
    ) -> list[list[int]]:
        """Group the selected impact nodes by anchor-to-node edge strength."""
        node_indices = [int(index) for index in target_node_index_set]
        target_node_index = int(target_node_index)
        if target_node_index not in node_indices:
            raise ValueError(
                "Target node is not included in impact node set: "
                f"target_node_index={target_node_index}, target_node_index_set={node_indices}"
            )
        target_local_index = node_indices.index(target_node_index)
        adjacency = np.asarray(adjacency_matrix, dtype=np.float64)
        if adjacency.shape != (len(node_indices), len(node_indices)):
            raise ValueError(
                "Impact adjacency shape does not match node set: "
                f"shape={adjacency.shape}, node_count={len(node_indices)}"
            )
        target_weights = adjacency[target_local_index]
        layers = [[target_node_index]]
        assigned_positions = {target_local_index}
        for rule in layer_rules:
            lower = float(rule["min_exclusive"])
            upper = float(rule["max_inclusive"])
            positions = np.flatnonzero(
                (target_weights > lower) & (target_weights <= upper)
            )
            layer = [
                node_indices[int(position)]
                for position in positions
                if int(position) not in assigned_positions
            ]
            layers.append(layer)
            assigned_positions.update(int(position) for position in positions)
        return layers

    def one_hop_neighbor_indices(
        self,
        node_index: int,
        min_edge_weight: float = DEFAULT_ONE_HOP_MIN_EDGE_WEIGHT,
        adjacency_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        adjacency = self.adj_matrix if adjacency_matrix is None else np.asarray(adjacency_matrix)
        node_index = int(node_index)
        min_edge_weight = float(min_edge_weight)
        return np.flatnonzero(adjacency[node_index] > min_edge_weight).astype(np.int64)

    def get_distance_node_layers(
        self,
        target_node_index: int,
        layer_rules: list[dict],
        max_nodes: int,
    ) -> tuple[list[list[int]], list[int]]:
        """Build legacy distance layers for diagnostics; impact runs use ``get_node_layer``."""
        target_node_index = int(target_node_index)
        if int(max_nodes) <= 0:
            raise ValueError(f"max_nodes must be positive, got {max_nodes}")
        target = self.sensors.iloc[target_node_index]
        candidate_indices = self.incident_road_sensor_indices(target["Fwy"], target["Direction"])
        distances_m = self.network_distance_matrix()[target_node_index, candidate_indices]
        maximum_distance_km = float(layer_rules[-1]["max_inclusive_km"]) if layer_rules else 0.0
        valid = np.isfinite(distances_m) & (distances_m > 0) & (distances_m <= maximum_distance_km * 1000.0)
        candidates = [
            (int(index), float(distance_m) / 1000.0)
            for index, distance_m in zip(candidate_indices[valid], distances_m[valid])
        ]
        candidates.sort(key=lambda item: item[1])
        # The anchor occupies one slot; keep the closest reachable nodes when a
        # run-level request cap is configured.
        candidates = candidates[: max(0, int(max_nodes) - 1)]
        layers = [[target_node_index]]
        for rule in layer_rules:
            min_exclusive_km = float(rule["min_exclusive_km"])
            max_inclusive_km = float(rule["max_inclusive_km"])
            layer = [
                node_index
                for node_index, distance_km in candidates
                if min_exclusive_km < distance_km <= max_inclusive_km
            ]
            layers.append(layer)
        return layers, [target_node_index, *[node_index for node_index, _ in candidates]]

    def has_network_predecessor(
        self,
        anchor_node_index: int,
        predecessor_node_index: int,
        target_node_index: int,
    ) -> tuple[bool, float | None]:
        """Return whether a prior-layer node has a directed path to the target."""
        predecessor_to_target_km = self.network_distance_km(predecessor_node_index, target_node_index)
        predecessor_from_anchor_km = self.network_distance_km(anchor_node_index, predecessor_node_index)
        target_from_anchor_km = self.network_distance_km(anchor_node_index, target_node_index)
        valid = (
            predecessor_to_target_km is not None
            and predecessor_from_anchor_km is not None
            and target_from_anchor_km is not None
            and predecessor_from_anchor_km < target_from_anchor_km
        )
        return bool(valid), predecessor_to_target_km

    def shortest_path(self, source_index: int, target_index: int) -> list[int]:
        source_index = int(source_index)
        target_index = int(target_index)
        if source_index == target_index:
            return [source_index]
        visited = np.zeros(int(self.node_order.size), dtype=bool)
        previous = np.full(int(self.node_order.size), -1, dtype=np.int64)
        queue = [source_index]
        visited[source_index] = True
        cursor = 0
        while cursor < len(queue):
            current = queue[cursor]
            cursor += 1
            neighbors = self.one_hop_neighbor_indices(current)
            for neighbor in neighbors:
                neighbor = int(neighbor)
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                previous[neighbor] = current
                if neighbor == target_index:
                    path = [target_index]
                    while path[-1] != source_index:
                        path.append(int(previous[path[-1]]))
                    return list(reversed(path))
                queue.append(neighbor)
        raise ValueError(f"No graph path between nodes: source_index={source_index}, target_index={target_index}")

    def month_step(self, timestamp: pd.Timestamp) -> tuple[int, int]:
        month_start = pd.Timestamp(year=2024, month=int(timestamp.month), day=1)
        minutes = (timestamp - month_start).total_seconds() / 60.0
        return int(timestamp.month), int(minutes // 5)

    def traffic_window(self, timestamp: pd.Timestamp, node_index: int) -> TrafficWindow:
        month, step = self.month_step(timestamp)
        array = self.month_arrays[month]
        history_start = step - 11
        future_start = step + 1
        future_end = future_start + 12
        if history_start < 0 or future_end > array.shape[1]:
            raise ValueError(f"Window crosses month boundary: timestamp={timestamp}, month={month}, step={step}")
        history = np.asarray(array[node_index, history_start : step + 1, 0], dtype=np.float64)
        future = np.asarray(array[node_index, future_start:future_end, 0], dtype=np.float64)
        return TrafficWindow(
            history_flow=[float(value) for value in history],
            future_flow=[float(value) for value in future],
        )

    def traffic_history_tensor(self, timestamp: pd.Timestamp) -> np.ndarray:
        month, step = self.month_step(timestamp)
        array = self.month_arrays[month]
        history_start = step - 11
        future_end = step + 13
        if history_start < 0 or future_end > array.shape[1]:
            raise ValueError(f"Window crosses month boundary: timestamp={timestamp}, month={month}, step={step}")
        return np.asarray(array[:, history_start : step + 1, :], dtype=np.float32)

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    coordinates = [lat1, lon1, lat2, lon2]
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"Coordinates must be finite for haversine distance, got {coordinates}")
    radius_km = 6371.0088
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    return float(2.0 * radius_km * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def _road_key(value) -> str:
    text = str(value).strip()
    numeric = pd.to_numeric(text, errors="coerce")
    if pd.notna(numeric) and float(numeric).is_integer():
        return str(int(numeric))
    return text


def _direction_key(value) -> str:
    return str(value).strip().upper()


def _haversine_array_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius_km = 6371.0088
    lat1_rad = np.radians(float(lat1))
    lon1_rad = np.radians(float(lon1))
    lat2_rad = np.radians(lat2.astype(np.float64))
    lon2_rad = np.radians(lon2.astype(np.float64))
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    return 2.0 * radius_km * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _haversine_pairwise_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    radius_km = 6371.0088
    lat1_rad = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1_rad = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2_rad = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2_rad = np.radians(np.asarray(lon2, dtype=np.float64))
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    return 2.0 * radius_km * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
