from __future__ import annotations

import pandas as pd


def time_context(timestamp) -> dict:
    """Return the calendar features shared by incident and normal retrieval."""
    timestamp = pd.Timestamp(timestamp)
    holidays = {
        "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27", "2024-06-19",
        "2024-07-04", "2024-09-02", "2024-10-14", "2024-11-11", "2024-11-28", "2024-12-25",
    }
    hour = int(timestamp.hour)
    period = (
        "night" if hour < 6 or hour >= 20
        else "morning_peak" if hour < 10
        else "midday" if hour < 16
        else "evening_peak"
    )
    is_weekend = int(timestamp.weekday()) >= 5
    is_holiday = timestamp.date().isoformat() in holidays
    return {
        "date": timestamp.date().isoformat(),
        "hour": hour,
        "weekday": int(timestamp.weekday()),
        "is_weekend": bool(is_weekend),
        "is_holiday": bool(is_holiday),
        "is_holiday_or_weekend": bool(is_holiday or is_weekend),
        "day_period": period,
    }
