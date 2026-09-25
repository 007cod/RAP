"""Coordinate forecast and reflection HTTP capacity and rate-limit pauses."""

import time
from contextlib import contextmanager
from threading import Condition


class RequestSlots:
    def __init__(self, total: int, reflection_limit: int | None = None):
        self.total = int(total)
        self.reflection_limit = int(reflection_limit if reflection_limit is not None else max(1, total // 3))
        if not 1 <= self.reflection_limit <= self.total:
            raise ValueError('Require 1 <= reflection_limit <= total')
        self._condition = Condition()
        self.active = {'forecast': 0, 'reflection': 0}
        self.waiting = {'forecast': 0, 'reflection': 0}
        self.provider_active = {}
        self.provider_limits = {}
        self.provider_max_active = {}
        self.provider_rate_limit_events = {}
        self.global_rate_limit_events = 0
        self._provider_configured_limits = {}
        self._provider_cooldown_until = {}
        self._global_cooldown_until = 0.0
        self._provider_success_streak = {}

    def _available(self, kind, provider=None, provider_limit=None):
        occupied = sum(self.active.values())
        if occupied >= self.total:
            return False
        # A provider-level 429 is treated as a shared upstream quota signal.
        # Do not let another provider or request kind bypass the cooldown.
        if time.monotonic() < self._global_cooldown_until:
            return False
        if provider is not None and time.monotonic() < self._provider_cooldown_until.get(provider, 0.0):
            return False
        if provider is not None and self.provider_active.get(provider, 0) >= provider_limit:
            return False
        if kind == 'reflection':
            return self.active[kind] < self.reflection_limit
        # Forecasts borrow unused reflection capacity, but let queued reflections
        # obtain their quota when an HTTP attempt finishes.
        reserved = max(0, self.reflection_limit - self.active['reflection']) if self.waiting['reflection'] else 0
        return occupied < self.total - reserved

    def report_rate_limit(self, provider: str, cooldown_seconds: float) -> None:
        """Apply shared backpressure after one worker receives HTTP 429."""
        with self._condition:
            configured = self._provider_configured_limits.get(provider, self.total)
            current = self.provider_limits.get(provider, configured)
            self.provider_limits[provider] = max(1, current // 2)
            self.provider_rate_limit_events[provider] = (
                self.provider_rate_limit_events.get(provider, 0) + 1
            )
            self.global_rate_limit_events += 1
            self._provider_success_streak[provider] = 0
            cooldown_until = time.monotonic() + max(0.0, float(cooldown_seconds))
            self._provider_cooldown_until[provider] = max(
                self._provider_cooldown_until.get(provider, 0.0), cooldown_until
            )
            self._global_cooldown_until = max(self._global_cooldown_until, cooldown_until)
            self._condition.notify_all()

    def report_success(self, provider: str) -> None:
        """Recover provider concurrency cautiously after sustained success."""
        with self._condition:
            configured = self._provider_configured_limits.get(provider)
            if configured is None:
                return
            current = self.provider_limits.get(provider, configured)
            if current >= configured:
                self._provider_success_streak[provider] = 0
                return
            streak = self._provider_success_streak.get(provider, 0) + 1
            if streak >= 10:
                self.provider_limits[provider] = min(configured, current + 1)
                streak = 0
            self._provider_success_streak[provider] = streak
            self._condition.notify_all()

    def _cooldown_wait_seconds(self, providers) -> float | None:
        now = time.monotonic()
        waits = [self._global_cooldown_until - now]
        waits.extend(
            self._provider_cooldown_until.get(name, 0.0) - now
            for name in providers
        )
        positive = [wait for wait in waits if wait > 0]
        return min(positive) if positive else None

    @contextmanager
    def slot(self, kind='forecast', *, providers=None, preferred_provider=None):
        if kind not in self.active:
            raise ValueError(f'Unknown request kind: {kind}')
        providers = tuple(dict.fromkeys(providers or ()))
        if preferred_provider is not None and preferred_provider not in providers:
            raise ValueError('preferred_provider must be included in providers')
        provider_limit = max(1, self.total // len(providers)) if providers else None
        with self._condition:
            for name in providers:
                self._provider_configured_limits[name] = provider_limit
                self.provider_limits.setdefault(name, provider_limit)
            self.waiting[kind] += 1
            try:
                def available_provider():
                    if not providers:
                        return None if not self._available(kind) else ''
                    eligible = [
                        name for name in providers
                        if self._available(
                            kind,
                            name,
                            self.provider_limits.get(name, provider_limit),
                        )
                    ]
                    if preferred_provider is not None:
                        return preferred_provider if preferred_provider in eligible else None
                    return min(
                        eligible,
                        key=lambda name: (self.provider_active.get(name, 0), providers.index(name)),
                    ) if eligible else None

                while available_provider() is None:
                    # A timeout is required so waiters wake when a provider
                    # cooldown expires even if no in-flight request finishes.
                    self._condition.wait(timeout=self._cooldown_wait_seconds(providers))
                selected_provider = available_provider()
                self.active[kind] += 1
                if providers:
                    self.provider_active[selected_provider] = self.provider_active.get(selected_provider, 0) + 1
                    self.provider_max_active[selected_provider] = max(
                        self.provider_max_active.get(selected_provider, 0),
                        self.provider_active[selected_provider],
                    )
            finally:
                self.waiting[kind] -= 1
        try:
            yield selected_provider or None
        finally:
            with self._condition:
                self.active[kind] -= 1
                if providers:
                    self.provider_active[selected_provider] -= 1
                self._condition.notify_all()
