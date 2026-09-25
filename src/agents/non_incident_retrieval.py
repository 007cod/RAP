from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

from src.agents.time_context import time_context
from src.agents.horizon_evidence import historical_error_profile
from src.agents.normal_runtime import NormalRuntime
from src.tool_llm.timing import timed_section


INDEX_VERSION = 2
DEFAULT_NON_INCIDENT_CACHE_PATH = Path(
    "artifacts_sacramento/retrieval/no_incident_reference_index_v2.npz"
)
HISTORY_STEPS = 12
FORECAST_STEPS = 12
HISTORY_MINUTES = 55
FUTURE_MINUTES = 60
MINUTES_TO_NS = np.int64(60 * 1_000_000_000)
DAY_TO_NS = np.int64(24 * 60 * 60 * 1_000_000_000)
CHANNEL_NAMES = ("flow", "occupancy", "speed")
CHANNEL_SCALES = np.asarray([1.0, 0.001, 1.0], dtype=np.float32)

_INDEX_CACHE_LOCK = Lock()
_INDEX_CACHE: dict[tuple[str, str], "NoIncidentHistoryIndex"] = {}
COARSE_CANDIDATE_LIMIT = 16384
COARSE_TIME_TOP_K = 4096


class NoIncidentCandidatesError(ValueError):
    """Raised when Normal Counterfactual Generation finds no valid window."""


def _cache_metadata(data) -> dict:
    month_shapes = {
        str(month): [int(value) for value in array.shape]
        for month, array in sorted(data.month_arrays.items())
    }
    return {
        "version": INDEX_VERSION,
        "month_shapes": month_shapes,
        "history_steps": HISTORY_STEPS,
        "forecast_steps": FORECAST_STEPS,
        "channel_names": list(CHANNEL_NAMES),
    }


def _cache_matches(metadata: dict, expected: dict) -> bool:
    metadata = dict(metadata)
    metadata.pop("data_dir", None)
    return metadata == expected


def _time_arrays(timestamps_ns: np.ndarray) -> dict[str, np.ndarray]:
    timestamps = pd.DatetimeIndex(pd.to_datetime(timestamps_ns))
    hours = timestamps.hour.to_numpy(dtype=np.uint8)
    weekdays = timestamps.weekday.to_numpy(dtype=np.uint8)
    months = timestamps.month.to_numpy(dtype=np.uint8)
    periods = np.select(
        [hours < 6, hours < 10, hours < 16, hours < 20],
        [0, 1, 2, 3],
        default=0,
    ).astype(np.uint8)
    holidays = {
        "2024-01-01",
        "2024-01-15",
        "2024-02-19",
        "2024-05-27",
        "2024-06-19",
        "2024-07-04",
        "2024-09-02",
        "2024-10-14",
        "2024-11-11",
        "2024-11-28",
        "2024-12-25",
    }
    is_holiday = np.asarray(
        [date in holidays for date in timestamps.strftime("%Y-%m-%d")],
        dtype=bool,
    )
    return {
        "hours": hours,
        "weekdays": weekdays,
        "calendar_months": months,
        "periods": periods,
        "is_weekend": weekdays >= 5,
        "is_holiday": is_holiday,
    }


def _incident_free_mask(
    timestamps_ns: np.ndarray,
    incident_starts_ns: np.ndarray,
    incident_ends_ns: np.ndarray,
) -> np.ndarray:
    changes = np.zeros(timestamps_ns.size + 1, dtype=np.int32)
    candidate_starts = incident_starts_ns - FUTURE_MINUTES * MINUTES_TO_NS
    candidate_ends = incident_ends_ns + HISTORY_MINUTES * MINUTES_TO_NS
    left = np.searchsorted(timestamps_ns, candidate_starts, side="left")
    right = np.searchsorted(timestamps_ns, candidate_ends, side="right")
    np.add.at(changes, left, 1)
    np.add.at(changes, right, -1)
    return np.cumsum(changes[:-1]) == 0


def _normalized_traffic_features(history: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = np.maximum(np.mean(np.abs(history), axis=-2), CHANNEL_SCALES)
    x = np.arange(HISTORY_STEPS, dtype=np.float32)
    centered_x = x - np.mean(x)
    slope = np.sum(history * centered_x.reshape((1,) * (history.ndim - 2) + (-1, 1)), axis=-2)
    slope /= float(np.sum(centered_x**2))
    trend = slope / scale
    volatility = np.std(history, axis=-2) / scale
    return scale, trend, volatility


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> list[float]:
    result = []
    for horizon in range(values.shape[1]):
        order = np.argsort(values[:, horizon], kind="stable")
        cumulative = np.cumsum(weights[order])
        position = min(int(np.searchsorted(cumulative, quantile, side="left")), len(order) - 1)
        result.append(float(values[order[position], horizon]))
    return result


def _series_slope(values: np.ndarray) -> float:
    x = np.arange(values.size, dtype=np.float64)
    centered_x = x - np.mean(x)
    return float(np.sum(values * centered_x) / np.sum(centered_x**2))


class NormalCounterfactualCandidatesError(NoIncidentCandidatesError):
    """Canonical error name for an empty Normal candidate set."""


class NormalCounterfactualIndex:
    """Index used by Normal Counterfactual Generation for same-node windows."""

    def __init__(self, data, cache_path: str | Path | None = None, *, workers: int = 4):
        self.data_dir = Path(data.data_dir).resolve()
        self.cache_path = Path(cache_path) if cache_path is not None else DEFAULT_NON_INCIDENT_CACHE_PATH
        self.cache_path = self.cache_path.expanduser()
        self.metadata = _cache_metadata(data)
        self._local_free_cache: dict[tuple[int, float, int, int], tuple[np.ndarray, int, int]] = {}
        self._local_free_cache_lock = Lock()
        self.runtime = NormalRuntime(data, self.cache_path, workers)
        self._normal_counterfactual_cache = self.runtime._normal
        if self.cache_path.exists():
            self._load_cache()
        else:
            self._build(data)
            self._write_cache()

    def _build(self, data) -> None:
        timestamps_ns = []
        month_values = []
        step_values = []
        for month, array in sorted(data.month_arrays.items()):
            month_start_ns = pd.Timestamp(year=2024, month=int(month), day=1).value
            valid_steps = np.arange(11, int(array.shape[1]) - 12, dtype=np.int32)
            timestamps_ns.append(month_start_ns + valid_steps.astype(np.int64) * 5 * MINUTES_TO_NS)
            month_values.append(np.full(valid_steps.shape, int(month), dtype=np.uint8))
            step_values.append(valid_steps)
        self.timestamps_ns = np.concatenate(timestamps_ns)
        self.months = np.concatenate(month_values)
        self.steps = np.concatenate(step_values)
        self.date_days = self.timestamps_ns // DAY_TO_NS
        self.time_arrays = _time_arrays(self.timestamps_ns)
        self._validate()

    def _load_cache(self) -> None:
        with np.load(self.cache_path, allow_pickle=False) as payload:
            cached_metadata = json.loads(str(payload["metadata"].item()))
            if not _cache_matches(cached_metadata, self.metadata):
                raise ValueError(f"No-incident index metadata mismatch: {self.cache_path}")
            self.timestamps_ns = np.asarray(payload["timestamps_ns"], dtype=np.int64)
            self.months = np.asarray(payload["months"], dtype=np.uint8)
            self.steps = np.asarray(payload["steps"], dtype=np.int32)
            self.date_days = np.asarray(payload["date_days"], dtype=np.int64)
            self.time_arrays = {
                key: np.asarray(payload[key])
                for key in (
                    "hours",
                    "weekdays",
                    "calendar_months",
                    "periods",
                    "is_weekend",
                    "is_holiday",
                )
            }
        self._validate()

    def _validate(self) -> None:
        count = int(self.timestamps_ns.size)
        if count == 0:
            raise ValueError(f"No complete traffic windows found: {self.cache_path}")
        if np.any(self.timestamps_ns[1:] < self.timestamps_ns[:-1]):
            raise ValueError(f"No-incident index timestamps are not sorted: {self.cache_path}")
        expected = (count,)
        if self.months.shape != expected or self.steps.shape != expected or self.date_days.shape != expected:
            raise ValueError(f"Invalid no-incident index row shapes: {self.cache_path}")
        if any(values.shape != expected for values in self.time_arrays.values()):
            raise ValueError(f"Invalid no-incident index time feature shapes: {self.cache_path}")

    def _write_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.cache_path.with_name(f".{self.cache_path.name}.tmp")
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.asarray(json.dumps(self.metadata, sort_keys=True), dtype=str),
                timestamps_ns=self.timestamps_ns,
                months=self.months,
                steps=self.steps,
                date_days=self.date_days,
                **self.time_arrays,
            )
        os.replace(temporary_path, self.cache_path)

    def _history_tensor(self, data, node_index: int, positions: np.ndarray) -> np.ndarray:
        result = np.empty((positions.size, HISTORY_STEPS, len(CHANNEL_NAMES)), dtype=np.float32)
        candidate_months = self.months[positions]
        for month in np.unique(candidate_months):
            output_positions = np.flatnonzero(candidate_months == month)
            index_positions = positions[output_positions]
            steps = self.steps[index_positions]
            values = np.asarray(data.month_arrays[int(month)][int(node_index)], dtype=np.float32)
            windows = np.lib.stride_tricks.sliding_window_view(values, HISTORY_STEPS, axis=0)
            result[output_positions] = np.moveaxis(windows[steps - HISTORY_STEPS + 1], -1, 1)
        return result

    def _local_incident_free_positions(
        self,
        data,
        node_index: int,
        positions: np.ndarray,
        local_radius_km: float,
        report_delay_buffer_minutes: int,
        unknown_duration_minutes: int,
    ) -> tuple[np.ndarray, dict]:
        cache_key = (
            int(node_index),
            float(local_radius_km),
            int(report_delay_buffer_minutes),
            int(unknown_duration_minutes),
        )
        with self._local_free_cache_lock:
            cached = self._local_free_cache.get(cache_key)
        if cached is None:
            route_cache = data._load_osrm_route_cache()
            incident_ids = data.incidents["incident_id"].astype(str).to_numpy()
            route_positions = np.asarray(
                [route_cache.incident_positions.get(incident_id, -1) for incident_id in incident_ids],
                dtype=np.int64,
            )
            distances = np.full(incident_ids.shape, np.nan, dtype=np.float64)
            cached_routes = route_positions >= 0
            distances[cached_routes] = (
                np.asarray(route_cache.distance_m[route_positions[cached_routes], int(node_index)], dtype=np.float64)
                / 1000.0
            )
            local = np.isfinite(distances) & (distances >= 0.0) & (distances <= float(local_radius_km))
            incidents = data.incidents.loc[local]
            durations = incidents["duration_minutes"].to_numpy(dtype=np.float64)
            unknown = ~np.isfinite(durations) | (durations <= 0)
            report_times_ns = pd.to_datetime(incidents["dt_parsed"]).to_numpy(
                dtype="datetime64[ns]"
            ).astype(np.int64)
            effective_durations = durations.copy()
            effective_durations[unknown] = int(unknown_duration_minutes)
            starts = report_times_ns - int(report_delay_buffer_minutes) * MINUTES_TO_NS
            ends = report_times_ns + effective_durations.astype(np.int64) * MINUTES_TO_NS
            cached_mask = _incident_free_mask(self.timestamps_ns, starts, ends)
            cached = (
                cached_mask,
                int(local.sum()),
                int(unknown.sum()),
                int(cached_routes.sum()),
            )
            with self._local_free_cache_lock:
                self._local_free_cache[cache_key] = cached
        cached_mask, local_incident_count, unknown_duration_count, cached_route_count = cached
        return positions[cached_mask[positions]], {
            "method": "cached_osrm_driving_distance",
            "max_driving_distance_km": float(local_radius_km),
            "report_delay_buffer_minutes": int(report_delay_buffer_minutes),
            "unknown_duration_minutes": int(unknown_duration_minutes),
            "cached_route_incident_count": int(cached_route_count),
            "local_incident_count": int(local_incident_count),
            "unknown_or_nonpositive_duration_count": int(unknown_duration_count),
        }

    def _coarse_time_candidates(
        self,
        positions: np.ndarray,
        query_context: dict,
        query_timestamp: pd.Timestamp,
        limit: int = COARSE_CANDIDATE_LIMIT,
    ) -> tuple[np.ndarray, dict]:
        """Reduce the expensive traffic-window scoring set without changing final scoring."""
        if positions.size <= int(limit):
            return positions, {"applied": False, "input_count": int(positions.size), "output_count": int(positions.size)}
        values = {key: array[positions] for key, array in self.time_arrays.items()}
        holiday_or_weekend = values["is_holiday"] | values["is_weekend"]
        weekday = values["weekdays"] == int(query_context["weekday"])
        period = values["periods"] == {"night": 0, "morning_peak": 1, "midday": 2, "evening_peak": 3}[query_context["day_period"]]
        hour_delta = np.abs(values["hours"].astype(np.int16) - int(query_context["hour"]))
        hour_delta = np.minimum(hour_delta, 24 - hour_delta)
        month_delta = np.abs(values["calendar_months"].astype(np.int16) - int(query_timestamp.month))
        month_delta = np.minimum(month_delta, 12 - month_delta)
        time_score = (
            weekday.astype(np.float32)
            + period.astype(np.float32)
            + ((np.cos(2.0 * np.pi * hour_delta / 24.0) + 1.0) / 2.0)
            + ((np.cos(2.0 * np.pi * month_delta / 12.0) + 1.0) / 2.0)
            + (holiday_or_weekend == bool(query_context["is_holiday_or_weekend"])).astype(np.float32)
        ) / 5.0
        broad = weekday | period | (hour_delta <= 3) | (month_delta <= 1) | (
            holiday_or_weekend == bool(query_context["is_holiday_or_weekend"])
        )
        broad_positions = np.flatnonzero(broad)
        top_time_count = min(int(COARSE_TIME_TOP_K), int(limit), int(positions.size))
        top_time = np.argpartition(-time_score, top_time_count - 1)[:top_time_count]
        selected_local = np.unique(np.concatenate([broad_positions, top_time]))
        if selected_local.size > int(limit):
            order = np.argsort(-time_score[selected_local], kind="stable")[: int(limit)]
            selected_local = selected_local[order]
        selected = positions[selected_local]
        return selected, {
            "applied": True,
            "input_count": int(positions.size),
            "output_count": int(selected.size),
            "limit": int(limit),
            "broad_candidate_count": int(broad_positions.size),
            "time_top_k": int(top_time_count),
        }

    def retrieve(
        self,
        data,
        query_incident,
        node_index: int,
        cutoff_time,
        top_k: int,
        base_forecast_provider,
        local_radius_km: float,
        report_delay_buffer_minutes: int,
        unknown_duration_minutes: int,
        current_base_forecast: list[float] | None = None,
        min_score: float = 0.6,
        timings=None,
        timing_metadata: dict | None = None,
        cache_model_independent: bool = False,
    ) -> dict:
        # Preserve the legacy direct-call contract: callers that explicitly
        # provide model evidence can still request a fully uncached result.
        # The evaluation path opts into cache_model_independent so Normal
        # selection is reused before Base diagnostics are attached.
        if not cache_model_independent and (
            base_forecast_provider is not None or current_base_forecast is not None
        ):
            return self._retrieve_uncached(
                data=data,
                query_incident=query_incident,
                node_index=node_index,
                cutoff_time=cutoff_time,
                top_k=top_k,
                base_forecast_provider=base_forecast_provider,
                local_radius_km=local_radius_km,
                report_delay_buffer_minutes=report_delay_buffer_minutes,
                unknown_duration_minutes=unknown_duration_minutes,
                current_base_forecast=current_base_forecast,
                min_score=min_score,
                timings=timings,
                timing_metadata=timing_metadata,
            )
        kwargs = dict(data=data, query_incident=query_incident, node_index=node_index,
                      cutoff_time=cutoff_time, top_k=top_k, base_forecast_provider=None,
                      local_radius_km=local_radius_km, report_delay_buffer_minutes=report_delay_buffer_minutes,
                      unknown_duration_minutes=unknown_duration_minutes, current_base_forecast=None,
                      min_score=min_score, timings=timings, timing_metadata=timing_metadata)
        key = (int(node_index), int(pd.Timestamp(query_incident.dt_parsed).value),
               int(pd.Timestamp(cutoff_time).value), int(top_k), float(min_score),
               float(local_radius_km), int(report_delay_buffer_minutes), int(unknown_duration_minutes))
        result = self.runtime.normal(key, lambda: self._retrieve_uncached(**kwargs), timings, timing_metadata)
        if base_forecast_provider is None and current_base_forecast is None:
            # Sanitize results produced by older model-independent cache
            # versions so a strict Base ablation cannot expose stale fields.
            result.pop("base_forecast_normal_gap", None)
            result.pop("historical_base_error", None)
            result.pop("weighted_average_base_error", None)
            result.pop("weighted_average_base_error_definition", None)
            result.pop("base_residual_distribution", None)
            for case in result.get("selection", {}).get("selected_cases", []):
                case.pop("mean_base_residual", None)
            result.pop("_selected_timestamps", None)
            result.pop("_future_flow_matrix", None)
            result.pop("_selected_weights", None)
            return result
        return self._attach_base_evidence(
            result,
            node_index=int(node_index),
            base_forecast_provider=base_forecast_provider,
            current_base_forecast=current_base_forecast,
            timings=timings,
            timing_metadata=timing_metadata,
        )

    @staticmethod
    def _attach_base_evidence(
        result: dict,
        *,
        node_index: int,
        base_forecast_provider,
        current_base_forecast: list[float] | None,
        timings=None,
        timing_metadata: dict | None = None,
    ) -> dict:
        """Add model-dependent diagnostics to a cached Normal reference.

        Normal candidate selection is independent of the Base model.  Keeping
        this enrichment outside the cached computation prevents every query
        from rerunning the expensive traffic-window search merely because its
        Base trajectory differs.
        """
        timing_metadata = dict(timing_metadata or {})
        selected_timestamps = [pd.Timestamp(value) for value in result.pop("_selected_timestamps")]
        future_flow = np.asarray(result.pop("_future_flow_matrix"), dtype=np.float64)
        weights = np.asarray(result.pop("_selected_weights"), dtype=np.float64)
        weighted_normal_flow = np.asarray(result["weighted_normal_flow"], dtype=np.float64)
        if current_base_forecast is not None:
            current_base_array = np.asarray(current_base_forecast, dtype=np.float64)
            if current_base_array.shape != weighted_normal_flow.shape:
                raise ValueError("current_base_forecast and weighted_normal_flow must have the same shape")
            result["base_forecast_normal_gap"] = {
                "G": float(np.mean(np.abs(current_base_array - weighted_normal_flow))),
                "definition": (
                    "Mean absolute per-step difference between base_forecast and "
                    "weighted_normal_flow; a small G means the two references agree. "
                    "G is a diagnostic for correction strength, not a hard forecast bound."
                ),
            }
        if (base_forecast_provider is None) != (current_base_forecast is None):
            raise ValueError(
                "base_forecast_provider and current_base_forecast must either both be provided or both be omitted"
            )
        if base_forecast_provider is None:
            return result
        with timed_section(timings, "no_incident.historical_forecast", **timing_metadata):
            historical_base = np.asarray(
                [base_forecast_provider(timestamp, int(node_index)) for timestamp in selected_timestamps],
                dtype=np.float64,
            )
        residuals = future_flow - historical_base
        residual_mean = np.sum(residuals * weights[:, None], axis=0)
        residual_std = np.sqrt(
            np.sum((residuals - residual_mean[None, :]) ** 2 * weights[:, None], axis=0)
        )
        weighted_average_base_error = float(
            np.mean(np.sum(np.abs(residuals) * weights[:, None], axis=0))
        )
        for case, residual in zip(result["selection"]["selected_cases"], residuals):
            case["mean_base_residual"] = float(np.mean(residual))
        result["historical_base_error"] = historical_error_profile(historical_base, future_flow, weights)
        result["weighted_average_base_error"] = weighted_average_base_error
        result["weighted_average_base_error_definition"] = (
            "Weighted mean absolute error of historical base forecasts against "
            "observed future flow across the selected incident-free normal windows."
        )
        result["base_residual_distribution"] = {
            "weighted_mean": [float(value) for value in residual_mean],
            "weighted_std": [float(value) for value in residual_std],
            "p10": _weighted_quantile(residuals, weights, 0.10),
            "p50": _weighted_quantile(residuals, weights, 0.50),
            "p90": _weighted_quantile(residuals, weights, 0.90),
        }
        return result

    def _candidate_features(self, data, node_index, timings=None, metadata=None):
        def compute():
            count = len(self.timestamps_ns)
            trend = np.empty((count, 3), dtype=np.float32)
            volatility = np.empty_like(trend)
            for start in range(0, count, 4096):
                positions = np.arange(start, min(start + 4096, count))
                _, trend[positions], volatility[positions] = _normalized_traffic_features(
                    self._history_tensor(data, node_index, positions))
            return trend, volatility
        return self.runtime.features(int(node_index), len(self.timestamps_ns), compute, timings, metadata)

    def _retrieve_uncached(
        self,
        data,
        query_incident,
        node_index: int,
        cutoff_time,
        top_k: int,
        base_forecast_provider,
        local_radius_km: float,
        report_delay_buffer_minutes: int,
        unknown_duration_minutes: int,
        current_base_forecast: list[float] | None = None,
        min_score: float = 0.6,
        timings=None,
        timing_metadata: dict | None = None,
    ) -> dict:
        timing_metadata = dict(timing_metadata or {})
        eligible_end = int(
            np.searchsorted(self.timestamps_ns, int(pd.Timestamp(cutoff_time).value), side="right")
        )
        positions = np.arange(eligible_end, dtype=np.int64)
        with timed_section(timings, "no_incident.local_filter", **timing_metadata):
            positions, local_filter = self._local_incident_free_positions(
                data,
                int(node_index),
                positions,
                local_radius_km,
                report_delay_buffer_minutes,
                unknown_duration_minutes,
            )
        if positions.size == 0:
            raise NormalCounterfactualCandidatesError(
                "No local incident-free candidates: "
                f"node_index={node_index}, cutoff_time={cutoff_time}, "
                f"max_driving_distance_km={local_radius_km}"
            )
        local_eligible_candidate_count = int(positions.size)

        month, step = data.month_step(query_incident.dt_parsed)
        reference = np.asarray(
            data.month_arrays[month][int(node_index), step - HISTORY_STEPS + 1 : step + 1],
            dtype=np.float32,
        )
        query_context = time_context(query_incident.dt_parsed)
        with timed_section(timings, "no_incident.coarse_filter", **timing_metadata):
            positions, coarse_filter = self._coarse_time_candidates(
                positions,
                query_context,
                pd.Timestamp(query_incident.dt_parsed),
            )
        with timed_section(timings, "no_incident.history_tensor", **timing_metadata):
            history = self._history_tensor(data, int(node_index), positions)
        with timed_section(timings, "no_incident.feature_scoring", **timing_metadata):
            reference_scale, reference_trend, reference_volatility = _normalized_traffic_features(
                reference[None, :, :]
            )
            all_trend, all_volatility = self._candidate_features(data, node_index, timings, timing_metadata)
            candidate_trend, candidate_volatility = all_trend[positions], all_volatility[positions]
            rmse = np.sqrt(np.mean((history - reference[None, :, :]) ** 2, axis=1))
            channel_similarity = np.exp(-rmse / reference_scale[0])
            trend_similarity = np.exp(-np.mean(np.abs(candidate_trend - reference_trend), axis=1))
            volatility_similarity = np.exp(
                -np.mean(np.abs(candidate_volatility - reference_volatility), axis=1)
            )
            traffic_similarity = (
                channel_similarity[:, 0]
                + channel_similarity[:, 1]
                + channel_similarity[:, 2]
                + trend_similarity
                + volatility_similarity
            ) / 5.0

            time_values = {key: values[positions] for key, values in self.time_arrays.items()}
            holiday_or_weekend = time_values["is_holiday"] | time_values["is_weekend"]
            holiday_score = (
                holiday_or_weekend == bool(query_context["is_holiday_or_weekend"])
            ).astype(np.float32)
            weekday_score = (time_values["weekdays"] == int(query_context["weekday"])).astype(np.float32)
            period_score = (
                time_values["periods"]
                == {"night": 0, "morning_peak": 1, "midday": 2, "evening_peak": 3}[
                    query_context["day_period"]
                ]
            ).astype(np.float32)
            hour_delta = np.abs(time_values["hours"].astype(np.int16) - int(query_context["hour"]))
            hour_delta = np.minimum(hour_delta, 24 - hour_delta)
            hour_score = (np.cos(2.0 * np.pi * hour_delta / 24.0) + 1.0) / 2.0
            month_delta = np.abs(
                time_values["calendar_months"].astype(np.int16) - int(query_incident.dt_parsed.month)
            )
            month_delta = np.minimum(month_delta, 12 - month_delta)
            month_score = (np.cos(2.0 * np.pi * month_delta / 12.0) + 1.0) / 2.0
            time_similarity = (
                holiday_score + weekday_score + period_score + hour_score + month_score
            ) / 5.0
            scores = 0.75 * traffic_similarity + 0.25 * time_similarity

        with timed_section(timings, "no_incident.selection", **timing_metadata):
            min_score = float(min_score)
            if not 0.0 <= min_score <= 1.0:
                raise ValueError(f"min_score must be between 0 and 1, got {min_score}")
            eligible_local = np.flatnonzero(scores >= min_score)
            if eligible_local.size == 0:
                raise NormalCounterfactualCandidatesError(
                    "No normal reference candidates meet minimum score: "
                    f"min_score={min_score}, max_score={float(np.max(scores)):.4f}"
                )
            selected_local = []
            selected_dates = set()
            for local_position in eligible_local[np.argsort(-scores[eligible_local], kind="stable")]:
                date_day = int(self.date_days[positions[int(local_position)]])
                if date_day in selected_dates:
                    continue
                selected_dates.add(date_day)
                selected_local.append(int(local_position))
                if len(selected_local) == int(top_k):
                    break
            selected_local = np.asarray(selected_local, dtype=np.int64)
            selected_positions = positions[selected_local]
            # Retrieval scores are the raw candidate weights. Filter them before
            # normalizing so weak matches cannot influence the reference.
            selected_scores = scores[selected_local]
            weights = selected_scores / np.sum(selected_scores)

        with timed_section(timings, "no_incident.future_flow", **timing_metadata):
            future_flow = np.asarray(
                [
                    data.traffic_window(
                        pd.Timestamp(int(self.timestamps_ns[position])), int(node_index)
                    ).future_flow
                    for position in selected_positions
                ],
                dtype=np.float64,
            )
        weighted_normal_flow = np.sum(future_flow * weights[:, None], axis=0)
        last_flow = float(reference[-1, 0])
        base_normal_gap = None
        if current_base_forecast is not None:
            current_base_array = np.asarray(current_base_forecast, dtype=np.float64)
            if current_base_array.shape != weighted_normal_flow.shape:
                raise ValueError(
                    "current_base_forecast and weighted_normal_flow must have the same shape"
                )
            # G measures disagreement between the model baseline and the local
            # incident-free reference. It is evidence for correction strength,
            # not a forecast bound.
            base_normal_gap = float(
                np.mean(np.abs(current_base_array - weighted_normal_flow))
            )

        residuals = None
        residual_mean = None
        residual_std = None
        weighted_average_base_error = None
        if (base_forecast_provider is None) != (current_base_forecast is None):
            raise ValueError(
                "base_forecast_provider and current_base_forecast must either both be provided or both be omitted"
            )
        if base_forecast_provider is not None:
            with timed_section(timings, "no_incident.historical_forecast", **timing_metadata):
                historical_base = np.asarray(
                    [
                        base_forecast_provider(
                            pd.Timestamp(int(self.timestamps_ns[position])), int(node_index)
                        )
                        for position in selected_positions
                    ],
                    dtype=np.float64,
                )
            residuals = future_flow - historical_base
            residual_mean = np.sum(residuals * weights[:, None], axis=0)
            residual_std = np.sqrt(
                np.sum((residuals - residual_mean[None, :]) ** 2 * weights[:, None], axis=0)
            )
            weighted_average_base_error = float(
                np.mean(np.sum(np.abs(residuals) * weights[:, None], axis=0))
            )

        selected_cases = []
        for rank, (local_position, index_position, weight) in enumerate(
            zip(selected_local, selected_positions, weights), start=1
        ):
            timestamp = pd.Timestamp(int(self.timestamps_ns[index_position]))
            components = {
                "traffic_aggregate": float(traffic_similarity[local_position]),
                "flow_history": float(channel_similarity[local_position, 0]),
                "occupancy_history": float(channel_similarity[local_position, 1]),
                "speed_history": float(channel_similarity[local_position, 2]),
                "trend": float(trend_similarity[local_position]),
                "volatility": float(volatility_similarity[local_position]),
                "time_aggregate": float(time_similarity[local_position]),
                "exact_weekday": float(weekday_score[local_position]),
                "month": float(month_score[local_position]),
                "hour": float(hour_score[local_position]),
                "day_period": float(period_score[local_position]),
                "holiday_or_weekend": float(holiday_score[local_position]),
            }
            selected_cases.append(
                {
                    "rank": int(rank),
                    "sample_time": str(timestamp),
                    "retrieval_score": float(scores[local_position]),
                    "normalized_weight": float(weight),
                    "score_components": components,
                    "mean_future_flow": float(np.mean(future_flow[rank - 1])),
                    **(
                        {"mean_base_residual": float(np.mean(residuals[rank - 1]))}
                        if residuals is not None
                        else {}
                    ),
                }
            )

        result = {
            "method": "same-node local-incident-free weighted reference",
            "target_node": {
                "node_id": int(data.node_order[int(node_index)]),
                "freeway": str(data.sensors.iloc[int(node_index)]["Fwy"]),
                "road_direction": str(data.sensors.iloc[int(node_index)]["Direction"]),
            },
            "selection": {
                "requested_top_k": int(top_k),
                "minimum_retrieval_score": min_score,
                "selected_count": int(len(selected_cases)),
                "one_candidate_per_date": True,
                "selected_cases": selected_cases,
            },
            "weighted_normal_flow": [float(value) for value in weighted_normal_flow],
            "weighted_normal_flow_definition": (
                "Weighted observed future flow from matched incident-free windows; "
                "this is the no-incident reference."
            ),
            **({"base_forecast_normal_gap": {
                "G": base_normal_gap,
                "definition": (
                    "Mean absolute per-step difference between base_forecast and "
                    "weighted_normal_flow; a small G means the two references agree. "
                    "G is a diagnostic for correction strength, not a hard forecast bound."
                ),
            }} if base_normal_gap is not None else {}),
            "recovery_evidence": {
                "last_observed_flow": last_flow,
                "recent_history_flow_slope_per_step": _series_slope(reference[:, 0]),
                "weighted_normal_flow_slope_per_step": _series_slope(weighted_normal_flow),
                "weighted_normal_change_from_last_observation": [
                    float(value - last_flow) for value in weighted_normal_flow
                ],
            },
            "evidence_quality": {
                "candidate_weight_threshold": min_score,
                "lowest_selected_candidate_weight": float(np.min(selected_scores)),
                "average_candidate_weight": float(np.mean(selected_scores)),
                "effective_sample_size": float(1.0 / np.sum(weights**2)),
                "normal_future_spread_by_step": np.sqrt(np.sum(
                    weights[:, None] * (future_flow - weighted_normal_flow) ** 2, axis=0
                )).tolist(),
                "spread_definition": "Weighted population standard deviation of reference futures; not a calibrated confidence interval.",
            },
            # Internal arrays are removed by retrieve() after model-dependent
            # Base diagnostics are attached.  They remain in the persistent
            # Normal cache so later paired-state requests do not rescan data.
            "_selected_timestamps": [str(pd.Timestamp(int(self.timestamps_ns[position]))) for position in selected_positions],
            "_future_flow_matrix": future_flow.tolist(),
            "_selected_weights": weights.tolist(),
        }
        if residuals is not None:
            result["historical_base_error"] = historical_error_profile(historical_base, future_flow, weights)
            # Raw audit compatibility only; the model sees the horizon profile.
            result["weighted_average_base_error"] = weighted_average_base_error
            result["weighted_average_base_error_definition"] = (
                "Weighted mean absolute error of historical base forecasts against "
                "observed future flow across the selected incident-free normal windows."
            )
            result["base_residual_distribution"] = {
                "weighted_mean": [float(value) for value in residual_mean],
                "weighted_std": [float(value) for value in residual_std],
                "p10": _weighted_quantile(residuals, weights, 0.10),
                "p50": _weighted_quantile(residuals, weights, 0.50),
                "p90": _weighted_quantile(residuals, weights, 0.90),
            }
        return result


def get_no_incident_history_index(data, cache_path: str | Path | None = None, *, workers: int = 4) -> NoIncidentHistoryIndex:
    requested_path = Path(cache_path) if cache_path is not None else DEFAULT_NON_INCIDENT_CACHE_PATH
    key = (str(Path(data.data_dir).resolve()), str(requested_path.expanduser().resolve()), int(workers))
    with _INDEX_CACHE_LOCK:
        index = _INDEX_CACHE.get(key)
        if index is None:
            index = NormalCounterfactualIndex(data, requested_path, workers=workers)
            _INDEX_CACHE[key] = index
    return index


# Paper-facing aliases.  The legacy names are retained because cache files and
# existing recovery scripts use them.
NoIncidentHistoryIndex = NormalCounterfactualIndex
get_normal_counterfactual_index = get_no_incident_history_index


__all__ = [
    "DEFAULT_NON_INCIDENT_CACHE_PATH",
    "NormalCounterfactualCandidatesError",
    "NormalCounterfactualIndex",
    "NoIncidentHistoryIndex",
    "NoIncidentCandidatesError",
    "get_normal_counterfactual_index",
    "get_no_incident_history_index",
]
