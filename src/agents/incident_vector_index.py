"""Cached metadata vectors used to preselect historical incident candidates."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

from src.agents.time_context import time_context


VECTOR_CACHE_VERSION = 1
DEFAULT_VECTOR_CACHE_PATH = Path("artifacts_sacramento/retrieval/incident_vectors_v1.npz")
DEFAULT_VECTOR_RECALL_TOP_K = 50
_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_TEXT_FIELD_SPECS = (
    ("type", "Type", 64, False),
    ("description", "DESCRIPTION", 256, True),
    ("area", "AREA", 64, True),
    ("location", "LOCATION", 256, True),
    ("freeway", "Fwy", 32, False),
    ("direction", "Freeway_direction", 32, False),
)
_TIME_VECTOR_DIMENSION = 20
_VECTOR_DIMENSION = sum(item[2] for item in _TEXT_FIELD_SPECS) + _TIME_VECTOR_DIMENSION
_PER_INDEX_LOCK = Lock()
_INDEX_CACHE: dict[tuple[str, str, str], "IncidentVectorIndex"] = {}


def _record_value(record, field: str):
    return record[field] if isinstance(record, dict) else getattr(record, field)


def _normalized_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(str(value).strip().lower().split())


def _hashed_index(field: str, token: str, dimension: int) -> int:
    digest = hashlib.blake2b(f"{field}\x00{token}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % int(dimension)


def _field_tokens(value: str, rich_text: bool) -> list[str]:
    if not value:
        return []
    words = _TOKEN_PATTERN.findall(value)
    tokens = {value, *words}
    if rich_text:
        compact = "".join(words)
        for ngram_size in (3, 4):
            tokens.update(compact[index:index + ngram_size] for index in range(max(0, len(compact) - ngram_size + 1)))
    return sorted(tokens)


def _normalized_block(field: str, value, dimension: int, rich_text: bool) -> np.ndarray:
    block = np.zeros(int(dimension), dtype=np.float32)
    for token in _field_tokens(_normalized_text(value), rich_text):
        block[_hashed_index(field, token, dimension)] += 1.0
    norm = float(np.linalg.norm(block))
    return block / norm if norm > 0.0 else block


def _time_block(timestamp) -> np.ndarray:
    context = time_context(pd.Timestamp(timestamp))
    block = np.zeros(_TIME_VECTOR_DIMENSION, dtype=np.float32)
    block[0 if context["day_period"] == "night" else {"morning_peak": 1, "midday": 2, "evening_peak": 3}[context["day_period"]]] = 1.0
    block[4 + int(context["weekday"])] = 1.0
    block[11] = float(context["is_weekend"])
    block[12] = float(context["is_holiday"])
    block[13] = float(context["is_holiday_or_weekend"])
    angle = 2.0 * np.pi * float(context["hour"]) / 24.0
    block[14:16] = [np.sin(angle), np.cos(angle)]
    norm = float(np.linalg.norm(block))
    return block / norm if norm > 0.0 else block


def incident_feature_vector(incident) -> np.ndarray:
    blocks = [
        _normalized_block(field, _record_value(incident, source), dimension, rich_text)
        for field, source, dimension, rich_text in _TEXT_FIELD_SPECS
    ]
    vector = np.concatenate([*blocks, _time_block(_record_value(incident, "dt_parsed"))]).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def _active_incident_fingerprint(data) -> dict:
    rows = data.incidents[
        [
            "incident_id",
            "dt_parsed",
            "Type",
            "DESCRIPTION",
            "AREA",
            "LOCATION",
            "Fwy",
            "Freeway_direction",
        ]
    ]
    digest = pd.util.hash_pandas_object(rows, index=False).to_numpy(dtype=np.uint64)
    return {"count": int(len(rows)), "sha256": hashlib.sha256(digest.tobytes()).hexdigest()}


def _cache_metadata(data) -> dict:
    source = Path(data.data_dir) / "incidents_y2024.csv"
    return {
        "cache_version": VECTOR_CACHE_VERSION,
        "incident_source": {"size": int(source.stat().st_size)},
        "active_incidents": _active_incident_fingerprint(data),
        "vector_dimension": _VECTOR_DIMENSION,
        "text_field_specs": [list(item) for item in _TEXT_FIELD_SPECS],
        "time_vector_dimension": _TIME_VECTOR_DIMENSION,
    }


class IncidentVectorIndex:
    """A persisted metadata-vector index for complete historical incidents."""

    def __init__(self, data, cache_path: str | Path | None = None):
        self.data_dir = Path(data.data_dir).resolve()
        self.cache_path = Path(cache_path or DEFAULT_VECTOR_CACHE_PATH).expanduser()
        self.metadata = _cache_metadata(data)
        try:
            self._load_cache()
            self.cache_hit = True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            self._build(data)
            self._write_cache()
            self.cache_hit = False

    def _load_cache(self):
        with np.load(self.cache_path, allow_pickle=False) as payload:
            if json.loads(str(payload["metadata"].item())) != self.metadata:
                raise ValueError("Incident vector cache metadata mismatch")
            self.row_positions = np.asarray(payload["row_positions"], dtype=np.int64)
            self.timestamps_ns = np.asarray(payload["timestamps_ns"], dtype=np.int64)
            self.incident_ids = np.asarray(payload["incident_ids"], dtype=str)
            self.vectors = np.asarray(payload["vectors"], dtype=np.float32)
        self._validate()

    def _build(self, data):
        rows = []
        for position, incident in enumerate(data.incidents.itertuples(index=False)):
            timestamp = pd.Timestamp(incident.dt_parsed)
            month, step = data.month_step(timestamp)
            if step - 11 < 0 or step + 13 > data.month_arrays[month].shape[1]:
                continue
            rows.append((int(position), int(timestamp.value), str(incident.incident_id), incident_feature_vector(incident)))
        if not rows:
            raise ValueError(f"No incidents with complete traffic windows found in {data.data_dir}")
        rows.sort(key=lambda item: item[1])
        self.row_positions = np.asarray([item[0] for item in rows], dtype=np.int64)
        self.timestamps_ns = np.asarray([item[1] for item in rows], dtype=np.int64)
        self.incident_ids = np.asarray([item[2] for item in rows], dtype=str)
        self.vectors = np.asarray([item[3] for item in rows], dtype=np.float32)
        self._validate()

    def _validate(self):
        count = self.row_positions.size
        if count == 0 or self.timestamps_ns.shape != (count,) or self.incident_ids.shape != (count,):
            raise ValueError("Invalid incident vector cache row shapes")
        if self.vectors.shape != (count, _VECTOR_DIMENSION) or np.any(self.timestamps_ns[1:] < self.timestamps_ns[:-1]):
            raise ValueError("Invalid incident vector cache arrays")

    def _write_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.cache_path.with_name(f".{self.cache_path.name}.tmp")
        try:
            with temporary_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    metadata=np.asarray(json.dumps(self.metadata, sort_keys=True), dtype=str),
                    row_positions=self.row_positions,
                    timestamps_ns=self.timestamps_ns,
                    incident_ids=self.incident_ids,
                    vectors=self.vectors,
                )
            os.replace(temporary_path, self.cache_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def recall(self, incident, cutoff_time, top_k: int = DEFAULT_VECTOR_RECALL_TOP_K) -> list[tuple[int, float]]:
        top_k = int(top_k)
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        eligible_end = int(np.searchsorted(self.timestamps_ns, int(pd.Timestamp(cutoff_time).value), side="right"))
        if eligible_end == 0:
            return []
        scores = self.vectors[:eligible_end] @ incident_feature_vector(incident)
        query_id = str(_record_value(incident, "incident_id"))
        eligible = np.flatnonzero(self.incident_ids[:eligible_end] != query_id)
        if eligible.size == 0:
            return []
        count = min(top_k, int(eligible.size))
        pool_scores = scores[eligible]
        selected = np.argpartition(-pool_scores, count - 1)[:count] if count < eligible.size else np.arange(eligible.size)
        selected = eligible[selected[np.argsort(-pool_scores[selected], kind="stable")]]
        return [(int(self.row_positions[index]), float(scores[index])) for index in selected]


def get_incident_vector_index(data, cache_path: str | Path | None = None) -> IncidentVectorIndex:
    requested = Path(cache_path or DEFAULT_VECTOR_CACHE_PATH).expanduser()
    digest = _active_incident_fingerprint(data)["sha256"]
    key = (str(Path(data.data_dir).resolve()), str(requested.resolve()), digest)
    with _PER_INDEX_LOCK:
        index = _INDEX_CACHE.get(key)
        if index is None:
            index = IncidentVectorIndex(data, requested)
            _INDEX_CACHE[key] = index
    return index


# Historical Incident Retrieval naming alias.
HistoricalIncidentMetadataIndex = IncidentVectorIndex


__all__ = [
    "DEFAULT_VECTOR_CACHE_PATH",
    "DEFAULT_VECTOR_RECALL_TOP_K",
    "IncidentVectorIndex",
    "HistoricalIncidentMetadataIndex",
    "get_incident_vector_index",
    "incident_feature_vector",
]
