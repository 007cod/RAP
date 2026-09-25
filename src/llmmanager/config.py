from dataclasses import dataclass
from copy import deepcopy
import json
import os
import re
from pathlib import Path
import threading


_PROVIDERS_PATH = Path(__file__).with_name("providers.json")
_provider_settings_lock = threading.Lock()
_provider_settings_snapshot = None
_ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _expand_environment_values(value):
    """Resolve exact ``${ENV_VAR}`` values without changing other settings."""
    if isinstance(value, dict):
        return {key: _expand_environment_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment_values(item) for item in value]
    if isinstance(value, str):
        match = _ENV_PATTERN.fullmatch(value.strip())
        return os.environ.get(match.group(1), "") if match else value
    return value


def provider_settings() -> dict:
    """Read provider settings once per process; edits apply to the next run.

    CLI entry points initialize this snapshot before starting work. The lock
    also covers callers that first create managers concurrently. Return a copy
    so a caller cannot change settings used by later forecasts or reflections.
    """
    return initialize_provider_settings()


def initialize_provider_settings(
    *, provider_pool: list[str] | None = None,
) -> dict:
    """Freeze configuration; explicit pools replace both routes and fallbacks."""
    global _provider_settings_snapshot
    with _provider_settings_lock:
        if _provider_settings_snapshot is not None:
            if provider_pool is not None:
                raise RuntimeError(
                    "Provider settings were already initialized; provider selection must be "
                    "applied before creating any LLMManager"
                )
            return deepcopy(_provider_settings_snapshot)
        snapshot = _expand_environment_values(
            json.loads(_PROVIDERS_PATH.read_text(encoding="utf-8"))
        )
        providers = snapshot.get("providers", {})
        unknown = sorted(set(provider_pool or []) - set(providers))
        if unknown:
            raise ValueError(f"Unknown LLM provider(s): {unknown}")
        if provider_pool is not None:
            if not provider_pool:
                raise ValueError("LLM provider pool cannot be empty")
            for route in ("forecast", "reflection"):
                snapshot.setdefault("routes", {})[route] = {
                    "pool": list(dict.fromkeys(provider_pool)), "fallback": [],
                }
        _provider_settings_snapshot = snapshot
        return deepcopy(_provider_settings_snapshot)

@dataclass(frozen=True)
class ProviderConfig:
    name: str
    type: str
    base_url: str
    model: str
    api_key: str
    temperature: float = 0.0
    timeout_seconds: float = 300.0
    max_attempts: int = 5
    retry_backoff_seconds: float = 1.0
    retry_backoff_max_seconds: float = 16.0
    
def provider_configs(values: dict) -> dict[str, ProviderConfig]:
    result = {}
    for name, raw in values.items():
        config = ProviderConfig(
            name=name,
            type=str(raw.get("type", "openai_compatible")),
            base_url=str(raw["base_url"]),
            model=str(raw["model"]),
            api_key=str(raw.get("api_key", "")),
            temperature=float(raw.get("temperature", 0.0)),
            timeout_seconds=float(raw.get("timeout_seconds", 300.0)),
            max_attempts=int(raw.get("max_attempts", 5)),
            retry_backoff_seconds=float(raw.get("retry_backoff_seconds", 1.0)),
            retry_backoff_max_seconds=float(raw.get("retry_backoff_max_seconds", 16.0)),
        )
        if config.max_attempts <= 0:
            raise ValueError(f"Provider {name} max_attempts must be positive")
        if config.retry_backoff_seconds < 0 or config.retry_backoff_max_seconds < 0:
            raise ValueError(f"Provider {name} retry backoff values must be non-negative")
        result[name] = config
    return result
