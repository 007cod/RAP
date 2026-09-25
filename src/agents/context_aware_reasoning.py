"""Context-Aware Phased Reasoning public API for RAP."""

from src.agents.context import build_traffic_context
from src.tool_llm.execution import run_phased_prediction as _run_phased_prediction


def build_context_analysis(*args, **kwargs):
    """Build the compact evidence context consumed by the prediction LLM."""

    return build_traffic_context(*args, **kwargs)


def run_phased_prediction(*args, **kwargs):
    """Run one target node's phased prediction within an influence layer."""

    return _run_phased_prediction(*args, **kwargs)


__all__ = ["build_context_analysis", "run_phased_prediction"]
