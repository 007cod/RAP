import numpy as np

from src.agents.tools import has_full_window
from src.data.traffic import IncidentSensorMatchError, TrafficData


def select_impact_cases(
    data: TrafficData,
    incidents,
    count: int = 3,
    month: int = 2,
) -> list[dict]:
    rows = []
    for incident in incidents.itertuples(index=False):
        timestamp = incident.dt_parsed
        if int(timestamp.month) != int(month):
            continue
        if not has_full_window(data, timestamp):
            continue
        try:
            node_index = data.nearest_incident_node_index(
                float(incident.Latitude_num),
                float(incident.Longitude_num),
                incident.Fwy,
                incident.Freeway_direction,
            )
        except IncidentSensorMatchError:
            continue
        window = data.traffic_window(timestamp, node_index)
        history_last = float(window.history_flow[-1])
        future_min = float(np.min(window.future_flow))
        abs_drop = history_last - future_min
        relative_drop = abs_drop / max(history_last, 1.0)
        if history_last < 50 or abs_drop < 20 or relative_drop < 0.1:
            continue
        rows.append(
            {
                "incident_id": str(incident.incident_id),
                "timestamp": str(timestamp),
                "node_index": int(node_index),
                "node_id": int(data.node_order[node_index]),
                "history_last": history_last,
                "future_min": future_min,
                "abs_drop": float(abs_drop),
                "relative_drop": float(relative_drop),
                "score": float(abs_drop * relative_drop),
                "description": str(incident.DESCRIPTION),
                "type": str(incident.Type),
                "area": str(incident.AREA),
            }
        )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows[:count]
