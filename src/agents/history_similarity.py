from functools import lru_cache
import warnings

import numpy as np
from kymatio import Scattering1D


HISTORY_SIMILARITY_WEIGHTS = {
    "wavelet_scattering": 0.45,
    "level": 0.20,
    "recent": 0.20,
    "changepoint": 0.15,
}
RECENT_STEPS = 3
CHANGEPOINT_MIN_RELATIVE_JUMP = 0.20


def history_dynamics_embeddings(histories: np.ndarray) -> np.ndarray:
    """Encode history windows for independent ANN retrieval.

    Each row combines robustly normalized levels, first differences, and the
    cached scattering embedding, then is L2-normalized for cosine ANN search.
    """
    values = np.asarray(histories, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("histories must be a finite 2D array with at least two steps")
    rows = []
    for row in values:
        level = _robust_normalize(row).astype(np.float64)
        delta = _robust_normalize(np.diff(row)).astype(np.float64)
        scattering = _wavelet_embedding(tuple(float(value) for value in row))
        embedding = np.concatenate([level, delta, scattering])
        norm = float(np.linalg.norm(embedding))
        rows.append(embedding / norm if norm > 0.0 else embedding)
    return np.asarray(rows, dtype=np.float32)


@lru_cache(maxsize=8)
def _scattering_transform(size: int):
    if size < 8:
        raise ValueError(f"Wavelet history encoding requires at least 8 steps, got {size}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Signal support is too small")
        return Scattering1D(
            J=3,
            shape=int(size),
            Q=(2, 1),
            max_order=2,
            frontend="numpy",
        )


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, 1.0)
    return np.clip((values - median) / scale, -8.0, 8.0).astype(np.float32)


@lru_cache(maxsize=65536)
def _wavelet_embedding(values: tuple[float, ...]) -> np.ndarray:
    normalized = _robust_normalize(np.asarray(values, dtype=np.float64))
    coefficients = np.asarray(
        _scattering_transform(normalized.size)(normalized[None, :])
    )[0]
    if coefficients.ndim > 1:
        coefficients = coefficients.mean(axis=-1)
    embedding = np.log1p(np.abs(coefficients)).astype(np.float64).reshape(-1)
    embedding.setflags(write=False)
    return embedding


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.clip(np.dot(left, right) / (left_norm * right_norm), 0.0, 1.0))


def _level_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_level = float(np.median(left))
    right_level = float(np.median(right))
    scale = max(abs(left_level), abs(right_level), 1.0)
    return float(np.exp(-2.0 * abs(left_level - right_level) / scale))


def _recent_similarity(left: np.ndarray, right: np.ndarray) -> float:
    recent_steps = min(RECENT_STEPS, left.size)
    left_recent = left[-recent_steps:]
    right_recent = right[-recent_steps:]
    rmse = float(np.sqrt(np.mean((left_recent - right_recent) ** 2)))
    scale = max(
        float(np.mean(np.abs(left_recent))),
        float(np.mean(np.abs(right_recent))),
        1.0,
    )
    return float(np.exp(-2.0 * rmse / scale))


def _dominant_changepoint(values: np.ndarray) -> dict:
    differences = np.diff(values)
    index = int(np.argmax(np.abs(differences)))
    jump = float(differences[index])
    baseline = max(float(np.median(np.abs(values[: index + 1]))), 1.0)
    return {
        "index": index,
        "direction": int(np.sign(jump)),
        "relative_magnitude": abs(jump) / baseline,
    }


def _changepoint_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_change = _dominant_changepoint(left)
    right_change = _dominant_changepoint(right)
    left_strong = left_change["relative_magnitude"] >= CHANGEPOINT_MIN_RELATIVE_JUMP
    right_strong = right_change["relative_magnitude"] >= CHANGEPOINT_MIN_RELATIVE_JUMP
    if not left_strong and not right_strong:
        return 1.0
    if left_strong != right_strong or left_change["direction"] != right_change["direction"]:
        return 0.0
    position_similarity = np.exp(
        -abs(left_change["index"] - right_change["index"]) / 2.0
    )
    magnitude_similarity = np.exp(
        -abs(
            np.log1p(left_change["relative_magnitude"])
            - np.log1p(right_change["relative_magnitude"])
        )
    )
    return float(position_similarity * magnitude_similarity)


def history_similarity_components(
    reference_history_flow: list[float],
    historical_history_flow: list[float],
) -> dict[str, float]:
    reference = np.asarray(reference_history_flow, dtype=np.float64)
    historical = np.asarray(historical_history_flow, dtype=np.float64)
    if reference.ndim != 1 or historical.shape != reference.shape:
        raise ValueError(
            "History flow shape mismatch: "
            f"reference={reference.shape}, historical={historical.shape}"
        )
    if reference.size < 2 or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(historical)):
        raise ValueError("History flow must contain at least two finite values")

    components = {
        "wavelet_scattering": _cosine_similarity(
            _wavelet_embedding(tuple(float(value) for value in reference)),
            _wavelet_embedding(tuple(float(value) for value in historical)),
        ),
        "level": _level_similarity(reference, historical),
        "recent": _recent_similarity(reference, historical),
        "changepoint": _changepoint_similarity(reference, historical),
    }
    combined = sum(
        HISTORY_SIMILARITY_WEIGHTS[name] * components[name]
        for name in HISTORY_SIMILARITY_WEIGHTS
    )
    return {
        **{name: float(value) for name, value in components.items()},
        "combined": float(combined),
    }
