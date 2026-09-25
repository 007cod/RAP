"""Historical forecast errors. Horizon bands are statistics, not traffic phases."""
from __future__ import annotations

import numpy as np

HORIZON_BANDS = ((1, 3), (4, 6), (7, 12))


def historical_error_profile(predictions, truth, weights=None) -> dict:
    pred = np.atleast_2d(np.asarray(predictions, dtype=np.float64))
    actual = np.atleast_2d(np.asarray(truth, dtype=np.float64))
    if pred.shape != actual.shape:
        raise ValueError("Historical predictions and truth must have matching shapes")
    w = np.ones(len(pred)) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (len(pred),) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Historical weights must be finite, nonnegative and nonempty")
    valid = np.isfinite(actual) & (actual != 0) & np.isfinite(pred)
    error = np.where(valid, pred - actual, 0.0)
    weighted = valid * w[:, None]

    def aggregate(start, stop):
        v, ew = valid[:, start:stop], weighted[:, start:stop]
        e = error[:, start:stop]
        denominator = float(ew.sum())
        support = ew.sum(axis=1)
        return {
            "start_step": start + 1, "end_step": stop,
            "mae": float((abs(e) * ew).sum() / denominator) if denominator else None,
            "bias": float((e * ew).sum() / denominator) if denominator else None,
            "valid_points": int(v.sum()),
            "effective_windows": float(support.sum() ** 2 / (support @ support)) if denominator else 0.0,
        }

    return {
        "source": "matched historical windows; not current-future accuracy",
        "bias_definition": "prediction minus observed flow; positive means historical overprediction",
        "by_step": [aggregate(h, h + 1) for h in range(pred.shape[1])],
        "bands": [aggregate(a - 1, min(b, pred.shape[1])) for a, b in HORIZON_BANDS if a <= pred.shape[1]],
        "overall": aggregate(0, pred.shape[1]),
    }


def local_outcomes(base, normal, prediction, truth, intervals=None) -> list[dict]:
    """Keep local wins AND losses, including in an overall improved episode."""
    arrays = {"base": base, "normal": normal, "prediction": prediction}
    y = np.asarray(truth, dtype=float)
    outcomes = []
    for start, end in intervals or HORIZON_BANDS:
        end = min(int(end), len(y)); start = max(1, int(start))
        if end < start:
            continue
        mask = np.isfinite(y[start-1:end]) & (y[start-1:end] != 0)
        if not mask.any():
            continue
        metrics = {}
        for name, values in arrays.items():
            if values is None or len(values) != len(y):
                continue
            e = np.asarray(values, dtype=float)[start-1:end][mask] - y[start-1:end][mask]
            if not np.isfinite(e).all():
                continue
            metrics[name] = {"mae": float(abs(e).mean()), "bias": float(e.mean())}
        delta = metrics["prediction"]["mae"] - metrics["base"]["mae"]
        outcomes.append({
            "start_step": start, "end_step": end, "valid_points": int(mask.sum()),
            "metrics": metrics, "delta_mae_vs_base": delta,
            "outcome_vs_base": "improved" if delta < -1e-6 else "degraded" if delta > 1e-6 else "equal",
        })
    return outcomes
