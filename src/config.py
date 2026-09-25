from dataclasses import dataclass, replace
import math
from pathlib import Path

from src.utils.json_io import read_json
from src.agents.terminology import (
    CANONICAL_CONTEXT_ALIASES,
    canonical_context_key,
    legacy_context_key,
)


# Memory is an optional evidence source.  The default must allow the vector
# retriever to return genuinely similar experiences; 0.90 filtered every
# candidate in normal runs because the calibrated score range is lower.
DEFAULT_MEMORY_MIN_SIMILARITY = 0.50


LLM_CONTEXT_KEYS = frozenset(
    {
        "retrieve_similar_cases_top5",
        "non_incident_normal_reference",
        "previous_layer_nodes",
        "episode_memory",
    }
)
LLM_CONTEXT_OPTION_KEYS = frozenset({"foundation_model_prediction", "base_prediction", "phased_prediction"})

# The persisted context-field set above is intentionally unchanged.  This
# parallel set is used by paper-facing APIs and keeps configuration vocabulary
# independent from the historical JSON keys.
PAPER_LLM_CONTEXT_KEYS = frozenset({
    "incident_patterns",
    "normal_counterfactual_sequence",
    "previous_layer_predictions",
    "experience_trajectories",
})

CONTEXT_FIELD_MAP = {
    "similar_incident_cases": "retrieve_similar_cases_top5",
    "normal_reference": "non_incident_normal_reference",
    "previous_layer_nodes": "previous_layer_nodes",
    "episode_memory": "episode_memory",
}
LEGACY_RUNTIME_CONTEXT_KEYS = frozenset(CONTEXT_FIELD_MAP.values())

# Paper-facing option names.  ``CONTEXT_FIELD_MAP`` remains the persisted
# compatibility map because existing configs and experiment artifacts use it.
CANONICAL_CONTEXT_FIELD_MAP = {
    canonical_context_key(key): key
    for key, value in CONTEXT_FIELD_MAP.items()
}

CONFIG_KEYS = frozenset(
    {"run", "impact_scope", "context", "memory", "concurrency", "retrieval"}
)
CONCURRENCY_KEYS = frozenset(
    {
        "incident_workers",
        "max_concurrent_llm_requests",
        "reflection_workers",
        "reflection_request_limit",
        "normal_retrieval_workers",
        "context_retrieval_workers",
    }
)
RETRIEVAL_KEYS = frozenset(
    {
        "similar_incident_vector_candidates",
        "incident_network_time_top_k",
        "incident_history_top_k",
        "incident_base_normal_top_k",
        "incident_vector_cache_path",
        "no_incident_cache_path",
        "no_incident_local_radius_km",
        "incident_report_delay_buffer_minutes",
        "unknown_incident_duration_minutes",
        "incident_retrieval_min_score",
        "normal_reference_top_k",
        "normal_reference_min_score",
        # Paper-facing aliases normalized to the persisted configuration keys.
        "incident_patterns_vector_candidates",
        "normal_counterfactual_top_k",
        "normal_counterfactual_min_score",
        "incident_similarity_top_k",
        "sequence_similarity_top_k",
        "foundation_normal_state_top_k",
    }
)


@dataclass(frozen=True)
class LLMContextConfig:
    enabled_keys: frozenset[str]
    foundation_model_prediction: bool = True
    phased_prediction: bool = True

    @classmethod
    def from_dict(cls, values: dict) -> "LLMContextConfig":
        values = dict(values)
        foundation_model_prediction = values.pop(
            "foundation_model_prediction",
            values.pop("base_prediction", True),
        )
        phased_prediction = values.pop("phased_prediction", True)
        # Accept both the paper terminology and the legacy configuration
        # names.  The internal set remains backward compatible with callers
        # that inspect ``enabled_keys``.
        normalized_values = {}
        config_field_for_runtime_key = {
            runtime_key: config_key
            for config_key, runtime_key in CONTEXT_FIELD_MAP.items()
        }
        for key, enabled in values.items():
            paper_key = canonical_context_key(key)
            if key in LEGACY_RUNTIME_CONTEXT_KEYS and key not in CONTEXT_FIELD_MAP:
                raise ValueError(
                    f"Unknown context keys: {[key]}. Runtime context field names "
                    "are not configuration options."
                )
            runtime_key = legacy_context_key(paper_key)
            config_key = config_field_for_runtime_key.get(runtime_key, paper_key)
            normalized_values[config_key] = enabled
        values = normalized_values
        unknown = set(values) - set(CONTEXT_FIELD_MAP)
        if unknown:
            raise ValueError(
                f"Unknown context keys: {sorted(unknown)}. "
                f"Allowed keys: {sorted(set(CONTEXT_FIELD_MAP) | LLM_CONTEXT_OPTION_KEYS)}"
            )
        option_values = {
            **values,
            "foundation_model_prediction": foundation_model_prediction,
            "phased_prediction": phased_prediction,
        }
        non_boolean = {
            key: value for key, value in option_values.items()
            if not isinstance(value, bool)
        }
        if non_boolean:
            raise TypeError(f"context values must be booleans, got {non_boolean}")
        return cls(
            frozenset(
                CONTEXT_FIELD_MAP[key]
                for key, enabled in values.items()
                if enabled
            ),
            foundation_model_prediction=foundation_model_prediction,
            phased_prediction=phased_prediction,
        )

    def is_enabled(self, key: str) -> bool:
        """Check a runtime context field or output mode."""
        if key in {"base_prediction", "foundation_model_prediction"}:
            return self.foundation_model_prediction
        if key == "phased_prediction":
            return self.phased_prediction
        return legacy_context_key(key) in self.enabled_keys

    @property
    def canonical_enabled_keys(self) -> frozenset[str]:
        """Enabled context fields using the terminology of the paper."""

        return frozenset(canonical_context_key(key) for key in self.enabled_keys)

    @property
    def base_prediction(self) -> bool:
        """Legacy alias for the Foundation Model Prediction switch."""

        return self.foundation_model_prediction

@dataclass(frozen=True)
class AgentConfig:
    incident_workers: int
    max_concurrent_llm_requests: int
    reflection_workers: int
    similar_incident_vector_candidates: int
    incident_vector_cache_path: str | None
    no_incident_cache_path: str | None
    no_incident_local_radius_km: float
    incident_report_delay_buffer_minutes: int
    unknown_incident_duration_minutes: int
    incident_retrieval_min_score: float
    normal_reference_top_k: int
    normal_reference_min_score: float
    reflection_request_limit: int = 0
    normal_retrieval_workers: int = 4
    context_retrieval_workers: int = 2
    incident_network_time_top_k: int = 3
    incident_history_top_k: int = 3
    incident_base_normal_top_k: int = 3

    @property
    def normal_counterfactual_top_k(self) -> int:
        return self.normal_reference_top_k

    @property
    def normal_counterfactual_min_score(self) -> float:
        return self.normal_reference_min_score

    @property
    def incident_patterns_vector_candidates(self) -> int:
        return self.similar_incident_vector_candidates

    @property
    def incident_similarity_top_k(self) -> int:
        return self.incident_network_time_top_k

    @property
    def sequence_similarity_top_k(self) -> int:
        return self.incident_history_top_k

    @property
    def foundation_normal_state_top_k(self) -> int:
        return self.incident_base_normal_top_k

    @property
    def historical_incident_vector_candidates(self) -> int:
        """Paper-facing alias for the rough-recall candidate pool size."""

        return self.similar_incident_vector_candidates

    @property
    def historical_incident_retrieval_top_k(self) -> int:
        """Maximum cases selected by the incident-similarity view."""

        return self.incident_network_time_top_k

    @property
    def historical_incident_retrieval_vector_candidates(self) -> int:
        """Candidate-pool size for Historical Incident Retrieval rough recall."""

        return self.similar_incident_vector_candidates

    @property
    def normal_counterfactual_generation_top_k(self) -> int:
        """Maximum incident-free windows used by Normal Counterfactual Generation."""

        return self.normal_reference_top_k

    @property
    def normal_counterfactual_generation_min_score(self) -> float:
        """Minimum window similarity for Normal Counterfactual Generation."""

        return self.normal_reference_min_score

    @classmethod
    def from_dict(cls, values: dict) -> "AgentConfig":
        concurrency = dict(values.get("concurrency", {}))
        retrieval = dict(values.get("retrieval", {}))
        retrieval_aliases = {
            "incident_patterns_vector_candidates": "similar_incident_vector_candidates",
            "normal_counterfactual_top_k": "normal_reference_top_k",
            "normal_counterfactual_min_score": "normal_reference_min_score",
            "incident_similarity_top_k": "incident_network_time_top_k",
            "sequence_similarity_top_k": "incident_history_top_k",
            "foundation_normal_state_top_k": "incident_base_normal_top_k",
        }
        for canonical_key, legacy_key in retrieval_aliases.items():
            if canonical_key in retrieval and legacy_key not in retrieval:
                retrieval[legacy_key] = retrieval[canonical_key]
        unknown_concurrency = set(concurrency) - CONCURRENCY_KEYS
        unknown_retrieval = set(retrieval) - RETRIEVAL_KEYS
        if unknown_concurrency or unknown_retrieval:
            raise ValueError(
                "Unknown agent settings: "
                f"concurrency={sorted(unknown_concurrency)}, "
                f"retrieval={sorted(unknown_retrieval)}"
            )
        values = {
            **concurrency,
            **retrieval,
        }
        incident_workers = int(values.get("incident_workers", 1))
        max_requests = int(values.get("max_concurrent_llm_requests", incident_workers))
        group_limits = {}
        for key in ("incident_network_time_top_k", "incident_history_top_k", "incident_base_normal_top_k"):
            value = values.get(key, 3)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"retrieval.{key} must be a positive integer, got {value!r}")
            group_limits[key] = value
        config = cls(
            **group_limits,
            incident_workers=incident_workers,
            max_concurrent_llm_requests=max_requests,
            reflection_workers=int(values.get("reflection_workers", max_requests)),
            similar_incident_vector_candidates=int(values.get("similar_incident_vector_candidates", 50)),
            incident_vector_cache_path=values.get("incident_vector_cache_path"),
            no_incident_cache_path=values.get("no_incident_cache_path"),
            no_incident_local_radius_km=float(values["no_incident_local_radius_km"]),
            incident_report_delay_buffer_minutes=int(
                values["incident_report_delay_buffer_minutes"]
            ),
            unknown_incident_duration_minutes=int(values["unknown_incident_duration_minutes"]),
            incident_retrieval_min_score=float(
                values.get("incident_retrieval_min_score", 0.8)
            ),
            normal_reference_top_k=int(values["normal_reference_top_k"]),
            normal_reference_min_score=float(values.get("normal_reference_min_score", 0.6)),
            reflection_request_limit=int(values.get('reflection_request_limit', max(1, max_requests // 3))),
            normal_retrieval_workers=int(values.get('normal_retrieval_workers', 4)),
            context_retrieval_workers=int(values.get('context_retrieval_workers', 2)),
        )
        positive = {
            "normal_retrieval_workers": config.normal_retrieval_workers,
            "context_retrieval_workers": config.context_retrieval_workers,
            "incident_workers": config.incident_workers,
            "max_concurrent_llm_requests": config.max_concurrent_llm_requests,
            "reflection_workers": config.reflection_workers,
            "similar_incident_vector_candidates": config.similar_incident_vector_candidates,
            "incident_report_delay_buffer_minutes": config.incident_report_delay_buffer_minutes,
            "unknown_incident_duration_minutes": config.unknown_incident_duration_minutes,
            "normal_reference_top_k": config.normal_reference_top_k,
        }
        invalid = {key: value for key, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"Agent integer settings must be positive, got {invalid}")
        if not 1 <= config.reflection_request_limit <= max_requests:
            raise ValueError('reflection_request_limit must be between 1 and max_concurrent_llm_requests')
        if config.no_incident_local_radius_km <= 0:
            raise ValueError(
                "agent.no_incident_local_radius_km must be positive, "
                f"got {config.no_incident_local_radius_km}"
            )
        if not 0.0 <= config.normal_reference_min_score <= 1.0:
            raise ValueError(
                "retrieval.normal_reference_min_score must be between 0 and 1, "
                f"got {config.normal_reference_min_score}"
            )
        if not 0.0 <= config.incident_retrieval_min_score <= 1.0:
            raise ValueError(
                "retrieval.incident_retrieval_min_score must be between 0 and 1, "
                f"got {config.incident_retrieval_min_score}"
            )
        return config


@dataclass(frozen=True)
class ImpactScopeConfig:
    osrm_base_url: str
    osrm_profile: str
    osrm_timeout_seconds: float
    layer_rules: tuple[dict, ...]

    @classmethod
    def from_dict(cls, values: dict) -> "ImpactScopeConfig":
        config = cls(
            osrm_base_url=str(values["osrm_base_url"]),
            osrm_profile=str(values["osrm_profile"]),
            osrm_timeout_seconds=float(values["osrm_timeout_seconds"]),
            layer_rules=tuple(dict(rule) for rule in values["layer_rules"]),
        )
        if not config.osrm_base_url:
            raise ValueError("impact_scope.osrm_base_url must be non-empty")
        if not config.osrm_profile:
            raise ValueError("impact_scope.osrm_profile must be non-empty")
        if config.osrm_timeout_seconds <= 0:
            raise ValueError(
                "impact_scope.osrm_timeout_seconds must be positive, "
                f"got {config.osrm_timeout_seconds}"
            )
        previous_upper = 0.0
        for index, rule in enumerate(config.layer_rules, start=1):
            lower = float(rule["min_exclusive_km"])
            upper = float(rule["max_inclusive_km"])
            if lower < 0 or lower >= upper:
                raise ValueError(f"Invalid impact_scope layer rule {index}: {rule}")
            if lower != previous_upper:
                raise ValueError(
                    "impact_scope.layer_rules must be contiguous distance bands: "
                    f"{config.layer_rules}"
                )
            previous_upper = upper
        if not config.layer_rules:
            raise ValueError("impact_scope.layer_rules must define at least one OSRM distance band")
        return config

    def to_dict(self) -> dict:
        return {
            "osrm_base_url": self.osrm_base_url,
            "osrm_profile": self.osrm_profile,
            "osrm_timeout_seconds": self.osrm_timeout_seconds,
            "layer_rules": [dict(rule) for rule in self.layer_rules],
        }


@dataclass(frozen=True)
class MemoryConfig:
    episodes_path: Path
    retrieval_top_k: int
    retention_days: int
    search_window_days: int
    recency_half_life_days: int
    max_entries: int
    min_similarity: float = DEFAULT_MEMORY_MIN_SIMILARITY
    min_base_mae: float = 0.0
    min_failure_delta: float = 10.0

    @property
    def experience_trajectory_top_k(self) -> int:
        return self.retrieval_top_k

    def __post_init__(self):
        if not 0.0 <= self.min_similarity <= 1.0:
            raise ValueError("memory.min_similarity must be between 0 and 1")
        for name in ("min_base_mae", "min_failure_delta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"memory.{name} must be finite and nonnegative")

    @classmethod
    def from_dict(cls, values: dict) -> "MemoryConfig":
        return cls(
            episodes_path=Path(values.get("episodes_path", "artifacts/memory/episodes_reflection.jsonl")),
            retrieval_top_k=int(values.get("retrieval_top_k", 3)),
            retention_days=int(values.get("retention_days", 90)),
            search_window_days=int(values.get("search_window_days", 120)),
            recency_half_life_days=int(values.get("recency_half_life_days", 30)),
            max_entries=int(values.get("max_entries", 2000)),
            min_similarity=float(values.get("min_similarity", DEFAULT_MEMORY_MIN_SIMILARITY)),
            min_base_mae=float(values.get("min_base_mae", 0.0)),
            min_failure_delta=float(values.get("min_failure_delta", 10.0)),
        )

    def policy(self) -> dict:
        return {
            "retention_days": self.retention_days,
            "search_window_days": self.search_window_days,
            "recency_half_life_days": self.recency_half_life_days,
            "max_entries": self.max_entries,
            "min_similarity": self.min_similarity,
            "min_base_mae": self.min_base_mae,
            "min_failure_delta": self.min_failure_delta,
        }


@dataclass(frozen=True)
class RunOptions:
    incident_ids: tuple[str, ...]
    checkpoint: Path | None
    output_root: Path
    device: str

    @classmethod
    def from_dict(cls, values: dict) -> "RunOptions":
        return cls(
            incident_ids=tuple(str(value) for value in values["incident_ids"]),
            checkpoint=(Path(values["checkpoint"]) if values.get("checkpoint") else None),
            output_root=Path(values["output_root"]),
            device=str(values.get("device", "auto")),
        )


@dataclass(frozen=True)
class RunConfig:
    run: RunOptions
    impact_scope: ImpactScopeConfig
    llm_context: LLMContextConfig
    memory: MemoryConfig
    agent: AgentConfig


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    values = read_json(config_path)
    unknown_keys = set(values) - CONFIG_KEYS
    if unknown_keys:
        raise ValueError(
            f"Unknown configuration sections in {config_path}: {sorted(unknown_keys)}; "
            f"allowed={sorted(CONFIG_KEYS)}"
        )
    context_values = values.get("context", {})
    agent_values = {
        "concurrency": values.get("concurrency", {}),
        "retrieval": values.get("retrieval", {}),
    }
    run_values = dict(values.get("run", {}))
    run_values.setdefault("incident_ids", [])
    run_values.setdefault("output_root", "outputs")
    run_values.setdefault("device", "auto")
    return RunConfig(
        run=RunOptions.from_dict(run_values),
        impact_scope=ImpactScopeConfig.from_dict(values["impact_scope"]),
        llm_context=LLMContextConfig.from_dict(context_values),
        memory=MemoryConfig.from_dict(values.get("memory", {})),
        agent=AgentConfig.from_dict(agent_values),
    )


def use_region_artifacts(config: RunConfig, region: str) -> RunConfig:
    """Derive storage paths so caches and models cannot leak across node sets."""
    region = str(region).strip().lower()
    artifacts_dir = Path(f"artifacts_{region}")
    return replace(
        config,
        run=replace(
            config.run,
            incident_ids=config.run.incident_ids if region == "sacramento" else (),
            checkpoint=artifacts_dir / "models" / f"graph_wavenet_{region}.pt",
        ),
        memory=replace(
            config.memory,
            episodes_path=artifacts_dir / "memory" / "episodes_reflection.jsonl",
        ),
        agent=replace(
            config.agent,
            no_incident_cache_path=str(
                artifacts_dir / "retrieval" / "no_incident_reference_index_v2.npz"
            ),
            incident_vector_cache_path=str(
                artifacts_dir / "retrieval" / "incident_vectors_v1.npz"
            ),
        ),
    )


def default_config_path() -> Path:
    return Path("configs/default.json")
