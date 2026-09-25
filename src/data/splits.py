import numpy as np


DEFAULT_MONTHS = tuple(range(1, 13))
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15


def validate_months(months) -> list[int]:
    values = [int(month) for month in months]
    if not values:
        raise ValueError("At least one split month is required")
    if len(values) != len(set(values)):
        raise ValueError(f"Split months must be unique, got {values}")
    if values != sorted(values) or any(month < 1 or month > 12 for month in values):
        raise ValueError(f"Split months must be ordered values from 1 through 12, got {values}")
    return values


def window_starts_for_steps(time_steps: int, history_steps: int, forecast_steps: int) -> np.ndarray:
    total_windows = int(time_steps) - int(history_steps) - int(forecast_steps) + 1
    if total_windows <= 0:
        raise ValueError(
            f"Data array is too short for windows: time_steps={time_steps}, "
            f"history_steps={history_steps}, forecast_steps={forecast_steps}"
        )
    return np.arange(total_windows, dtype=np.int64)


def split_windows(
    starts: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratio_sum = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            f"Split ratios must sum to 1.0, got train={train_ratio}, val={val_ratio}, "
            f"test={test_ratio}, sum={ratio_sum}"
        )
    starts = np.asarray(starts, dtype=np.int64)
    total = int(starts.size)
    train_count = int(round(total * float(train_ratio)))
    val_count = int(round(total * float(val_ratio)))
    test_count = total - train_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError(
            f"Split produced an empty partition: total={total}, train={train_count}, "
            f"val={val_count}, test={test_count}"
        )
    train_starts = starts[:train_count]
    val_starts = starts[train_count : train_count + val_count]
    test_starts = starts[train_count + val_count :]
    return train_starts, val_starts, test_starts


def start_range(starts: np.ndarray) -> dict[str, int]:
    starts = np.asarray(starts, dtype=np.int64)
    if starts.size == 0:
        raise ValueError("Cannot summarize empty starts")
    return {
        "window_count": int(starts.size),
        "start_min": int(starts[0]),
        "start_max": int(starts[-1]),
    }
