from dataclasses import dataclass
import email.utils
import threading
import time
import requests
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
from src.tool_llm.timing import timed_section
from .config import provider_configs, provider_settings

@dataclass(frozen=True)
class LLMRequest:
    messages: list[dict]
    model: str | None = None
    temperature: float | None = None
    timeout_seconds: float | None = None
    response_format: dict | None = None

@dataclass(frozen=True)
class LLMResponse:
    content: str
    provider: str
    model: str
    attempts: int

class LLMManager:
    """Single interface for OpenAI-compatible providers, including DeepSeek."""
    _pool_lock = threading.Lock()
    _pool_cursors = {}

    def __init__(self, agent_cfg, *, provider=None, route=None, fallback=(), max_concurrent=None, providers=None):
        raw_config = provider_settings()
        self.providers = dict(providers) if providers is not None else provider_configs(raw_config["providers"])
        routes = raw_config.get("routes", {})
        selected_route = routes.get(route or "forecast", {})
        configured_pool = selected_route.get("pool") or selected_route.get("providers")
        if provider is not None:
            configured_pool = provider if isinstance(provider, (list, tuple)) else [provider]
        elif not configured_pool:
            configured_pool = [selected_route.get("primary") or raw_config.get("default_provider", "sudorelay")]
        self.provider_pool = tuple(dict.fromkeys(configured_pool))
        if not self.provider_pool:
            raise ValueError("LLM provider pool cannot be empty")
        unknown = [name for name in self.provider_pool if name not in self.providers]
        if unknown:
            raise ValueError(f"Unknown LLM provider(s) in pool: {unknown}")
        self.provider = self.provider_pool[0]
        self.fallback = tuple(fallback or selected_route.get("fallback", []))
        unknown_fallback = [name for name in self.fallback if name not in self.providers]
        if unknown_fallback:
            raise ValueError(f"Unknown fallback LLM provider(s): {unknown_fallback}")
        self._slots = threading.BoundedSemaphore(max_concurrent or agent_cfg.max_concurrent_llm_requests)

    def _balanced_provider_order(self):
        """Rotate the first-choice API while retaining other pool members as fallbacks."""
        pool = self.provider_pool
        if len(pool) == 1:
            return pool
        with self._pool_lock:
            cursor = self._pool_cursors.get(pool, 0)
            self._pool_cursors[pool] = (cursor + 1) % len(pool)
        return pool[cursor:] + pool[:cursor]

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(exc, requests.HTTPError):
            response = exc.response
            if response is None:
                return True
            status = int(response.status_code)
            return status in {408, 409, 425, 429} or 500 <= status < 600
        # A 200 response with a malformed OpenAI-compatible envelope is often
        # produced by a transient proxy/upstream failure.
        return isinstance(exc, (KeyError, IndexError, TypeError, ValueError))

    @staticmethod
    def _retry_after_seconds(response) -> float | None:
        if response is None:
            return None
        value = response.headers.get("Retry-After")
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_at = email.utils.parsedate_to_datetime(value)
                return max(0.0, retry_at.timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                return None

    def complete(self, request: LLMRequest, *, request_slots=None, request_kind='forecast',
                 timings=None, timing_module='llm', timing_metadata=None) -> LLMResponse:
        last = None
        total_attempts = 0
        balanced_pool = self._balanced_provider_order()
        for name in (*balanced_pool, *self.fallback):
            provider_cfg = self.providers[name]
            key = provider_cfg.api_key
            base = provider_cfg.base_url
            default_model = provider_cfg.model
            if not key:
                last = ValueError(f"Missing api_key for provider {name}")
                continue
            model = request.model or default_model
            temperature = provider_cfg.temperature if request.temperature is None else request.temperature
            timeout_seconds = provider_cfg.timeout_seconds if request.timeout_seconds is None else request.timeout_seconds
            payload = {"model": model, "messages": request.messages, "temperature": float(temperature), "response_format": request.response_format or {"type": "json_object"}}
            provider_type = provider_cfg.type
            thinking_disabled = (
                name == "deepseek"
                or provider_type in {"doubao", "minimax"}
                or provider_type.startswith("gemini")
            )
            if thinking_disabled:
                # Keep reasoning disabled for deterministic structured forecasts.
                payload["thinking"] = {"type": "disabled"}
                payload.pop("reasoning_effort", None)
            max_attempts = provider_cfg.max_attempts
            backoff = provider_cfg.retry_backoff_seconds
            max_backoff = provider_cfg.retry_backoff_max_seconds
            for attempt in range(1, max_attempts + 1):
                total_attempts += 1
                response = None
                metadata = {**(timing_metadata or {}), 'provider': name, 'model': model,
                            'request_kind': request_kind, 'http_attempt': attempt,
                            'input_chars': sum(len(str(m.get('content', ''))) for m in request.messages)}
                if timings is not None:
                    timings.record(f'{timing_module}.queued', 0, metadata)
                try:
                    if hasattr(request_slots, 'slot'):
                        shared_slot = request_slots.slot(
                            request_kind,
                            providers=self.provider_pool if name in self.provider_pool else None,
                            preferred_provider=name if name in self.provider_pool else None,
                        )
                    else:
                        shared_slot = request_slots if request_slots is not None else nullcontext()
                    with ExitStack() as slots:
                        with timed_section(timings, f'{timing_module}.slot_wait', **metadata):
                            slots.enter_context(shared_slot)
                            slots.enter_context(self._slots)
                        started = time.perf_counter()
                        if timings is not None:
                            timings.record(f'{timing_module}.http_start', 0, metadata)
                        try:
                            response = requests.post(
                                base.rstrip("/") + "/chat/completions",
                                headers={
                                    "Authorization": f"Bearer {key}",
                                    "Content-Type": "application/json",
                                    "Connection": "close",
                                },
                                json=payload,
                                timeout=float(timeout_seconds),
                            )
                            response.raise_for_status()
                        finally:
                            elapsed = (time.perf_counter() - started) * 1000
                            status = int(response.status_code) if response is not None else None
                            if timings is not None:
                                timings.record(f'{timing_module}.http', elapsed, {**metadata, 'status_code': status})
                            print(f"{datetime.now(timezone.utc).isoformat()} LLM HTTP: kind={request_kind} "
                                  f"incident={metadata.get('incident_id')} node={metadata.get('node_index')} "
                                  f"attempt={attempt}/{max_attempts} status={status} seconds={elapsed/1000:.2f}", flush=True)
                    with timed_section(timings, f'{timing_module}.decode', **metadata):
                        content = response.json()["choices"][0]["message"]["content"]
                    if hasattr(request_slots, 'report_success'):
                        request_slots.report_success(name)
                    if timings is not None:
                        timings.record(f'{timing_module}.response', 0, {**metadata, 'output_chars': len(content),
                                       'total_http_attempts': total_attempts})
                    return LLMResponse(content, name, model, total_attempts)
                except Exception as exc:
                    last = exc
                    retryable = self._is_retryable(exc)
                    retry_after = self._retry_after_seconds(response)
                    delay = min(max_backoff, backoff * (2 ** (attempt - 1)))
                    if retry_after is not None:
                        # Retry-After is the provider's explicit earliest retry
                        # time and must not be shortened by the local cap.
                        delay = max(delay, retry_after)
                    status = int(response.status_code) if response is not None else None
                    if status == 429:
                        # Providers frequently omit Retry-After. A one-second
                        # retry loop makes concurrent workers collide again
                        # before a minute/token bucket has recovered.
                        if retry_after is None:
                            delay = min(max_backoff, max(delay, 15.0))
                        if hasattr(request_slots, 'report_rate_limit'):
                            request_slots.report_rate_limit(name, delay)
                    if not retryable or attempt >= max_attempts:
                        break
                    print(
                        f"{datetime.now(timezone.utc).isoformat()} LLM request failed, retrying: "
                        f"kind={request_kind}, incident={metadata.get('incident_id')}, node={metadata.get('node_index')}, "
                        f"provider={name}, attempt={attempt}/{max_attempts}, "
                        f"delay_seconds={delay:.1f}, error={exc}",
                        flush=True,
                    )
                    # Both the shared quota and per-manager slot were released
                    # before entering this backoff; another request can progress.
                    with timed_section(timings, f'{timing_module}.backoff', delay_seconds=delay, **metadata):
                        time.sleep(delay)
        if last is None:
            last = RuntimeError("No LLM provider was configured")
        raise RuntimeError(f"All LLM providers failed: {last}") from last

    def complete_chat(self, messages, *, request_slots=None, request_kind='forecast',
                      timings=None, timing_module='llm', timing_metadata=None, **kwargs):
        return self.complete(LLMRequest(messages, **kwargs), request_slots=request_slots,
                             request_kind=request_kind, timings=timings,
                             timing_module=timing_module, timing_metadata=timing_metadata).content
