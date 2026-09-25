"""Per-target-node vector indexes for historical traffic dynamics."""

from __future__ import annotations

from threading import Lock
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from src.agents.history_similarity import history_dynamics_embeddings
from src.agents.incident_vector_index import IncidentVectorIndex, get_incident_vector_index


DEFAULT_HISTORY_VECTOR_RECALL_TOP_K = 128
_INDEX_CACHE: dict[tuple[str, str], "IncidentHistoryVectorIndex"] = {}
_INDEX_CACHE_LOCK = Lock()


class IncidentHistoryVectorIndex:
    """Lazily materialize one compact dynamics matrix for each sensor node."""

    def __init__(self, data, incident_vector_index: IncidentVectorIndex | None = None):
        incident_vector_index = incident_vector_index or get_incident_vector_index(data)
        self.data = data
        self.row_positions = np.asarray(incident_vector_index.row_positions, dtype=np.int64)
        self.timestamps_ns = np.asarray(incident_vector_index.timestamps_ns, dtype=np.int64)
        self._node_vectors: dict[int, np.ndarray] = {}
        self._node_lock = Lock()
        digest = hashlib.sha1(str(Path(getattr(data, "data_dir", ""))).encode()).hexdigest()[:12]
        self._cache_dir = Path("artifacts") / "retrieval" / f"history_vectors_{digest}"

    def _history_matrix(self, node_index: int) -> np.ndarray:
        histories = np.empty((self.row_positions.size, 12), dtype=np.float64)
        incidents = self.data.incidents.iloc[self.row_positions]
        months = incidents["dt_parsed"].dt.month.to_numpy(dtype=np.int16)
        for month in np.unique(months):
            local = np.flatnonzero(months == month)
            timestamps = incidents.iloc[local]["dt_parsed"]
            steps = np.asarray(
                [self.data.month_step(timestamp)[1] for timestamp in timestamps], dtype=np.int64
            )
            indices = steps[:, None] - np.arange(11, -1, -1, dtype=np.int64)[None, :]
            histories[local] = np.asarray(self.data.month_arrays[int(month)][int(node_index), indices, 0])
        return histories

    def _vectors_for_node(self, node_index: int) -> np.ndarray:
        node_index = int(node_index)
        cached = self._node_vectors.get(node_index)
        if cached is not None:
            return cached
        with self._node_lock:
            cached = self._node_vectors.get(node_index)
            if cached is None:
                cache_path = self._cache_dir / f"node_{node_index}.npy"
                try:
                    cached = np.load(cache_path, allow_pickle=False)
                    if cached.ndim != 2 or cached.shape[0] != self.row_positions.size:
                        raise ValueError("stale history vector cache")
                except (OSError, ValueError):
                    cached = history_dynamics_embeddings(self._history_matrix(node_index))
                    self._cache_dir.mkdir(parents=True, exist_ok=True)
                    temporary = cache_path.with_suffix(".tmp.npy")
                    np.save(temporary, cached)
                    temporary.replace(cache_path)
                cached.setflags(write=False)
                self._node_vectors[node_index] = cached
            return cached

    def recall(
        self,
        history_flow: list[float],
        cutoff_time,
        node_index: int,
        top_k: int = DEFAULT_HISTORY_VECTOR_RECALL_TOP_K,
    ) -> list[tuple[int, float]]:
        top_k = int(top_k)
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        cutoff_ns = int(pd.Timestamp(cutoff_time).value)
        eligible = np.flatnonzero(self.timestamps_ns <= cutoff_ns)
        if eligible.size == 0:
            return []
        query = history_dynamics_embeddings(
            np.asarray(history_flow, dtype=np.float64).reshape(1, -1)
        )[0]
        scores = self._vectors_for_node(node_index)[eligible] @ query
        count = min(top_k, int(eligible.size))
        selected = (
            np.argpartition(-scores, count - 1)[:count]
            if count < eligible.size
            else np.arange(eligible.size)
        )
        selected = selected[np.argsort(-scores[selected], kind="stable")]
        return [
            (int(self.row_positions[eligible[index]]), float(scores[index]))
            for index in selected
        ]

    def precompute_nodes(self, node_indices) -> None:
        """Materialize and persist all requested node matrices serially.

        This is intentionally called before the incident worker pool starts,
        so expensive scattering transforms are not duplicated by competing
        workers.
        """
        for node_index in sorted({int(index) for index in node_indices}):
            self._vectors_for_node(node_index)


def get_incident_history_vector_index(
    data,
    incident_vector_index: IncidentVectorIndex | None = None,
) -> IncidentHistoryVectorIndex:
    data_dir = str(getattr(data, "data_dir", ""))
    fingerprint = str(len(data.incidents))
    key = (data_dir, fingerprint)
    with _INDEX_CACHE_LOCK:
        index = _INDEX_CACHE.get(key)
        if index is None:
            index = IncidentHistoryVectorIndex(data, incident_vector_index)
            _INDEX_CACHE[key] = index
    return index


# Historical Incident Retrieval terminology; implementation and index cache
# remain shared with the legacy API.
HistoricalIncidentRetrievalIndex = IncidentHistoryVectorIndex
get_historical_incident_retrieval_index = get_incident_history_vector_index



__all__ = [
    "DEFAULT_HISTORY_VECTOR_RECALL_TOP_K",
    "IncidentHistoryVectorIndex",
    "get_incident_history_vector_index",
    "HistoricalIncidentRetrievalIndex",
    "get_historical_incident_retrieval_index",
]
