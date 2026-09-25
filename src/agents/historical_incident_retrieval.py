"""Historical Incident Retrieval public API for RAP."""

from src.agents.incident_history_index import (
    IncidentHistoryVectorIndex,
    get_incident_history_vector_index,
)
from src.agents.incident_vector_index import (
    IncidentVectorIndex,
    get_incident_vector_index,
)
from src.agents.tools import retrieve_incident_patterns as _retrieve_incident_patterns


HistoricalIncidentIndex = IncidentHistoryVectorIndex


def retrieve_incident_patterns(*args, **kwargs):
    """Retrieve the Incident Patterns evidence set for a query node."""

    return _retrieve_incident_patterns(*args, **kwargs)


__all__ = [
    "HistoricalIncidentIndex",
    "IncidentHistoryVectorIndex",
    "IncidentVectorIndex",
    "get_incident_history_vector_index",
    "get_incident_vector_index",
    "retrieve_incident_patterns",
]
