from collections import OrderedDict, deque
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Lock, current_thread
import time


class TimingRecorder:
    """Record thread-safe module timings and retain per-invocation events."""

    def __init__(self, max_recent_events: int = 10_000):
        if int(max_recent_events) <= 0:
            raise ValueError("max_recent_events must be positive")
        self._lock = Lock()
        self._events = deque(maxlen=int(max_recent_events))
        self._modules = OrderedDict()
        self._next_event_id = 0
        self._report_lock = Lock()
        self._report_path: Path | None = None
        self._report_metadata: dict = {}
        self._report_status = "running"
        self._report_error: str | None = None
        self._flush_interval_seconds = 2.0
        self._last_flush_monotonic = 0.0

    @contextmanager
    def measure(self, module: str, **metadata):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(
                module=module,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                metadata=metadata,
            )

    def record(self, module: str, elapsed_ms: float, metadata: dict | None = None) -> None:
        module = str(module)
        elapsed_ms = float(elapsed_ms)
        should_flush = False
        with self._lock:
            self._events.append(
                {
                    "event_id": int(self._next_event_id),
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "module": module,
                    "elapsed_ms": elapsed_ms,
                    "thread": current_thread().name,
                    "metadata": dict(metadata or {}),
                }
            )
            aggregate = self._modules.get(module)
            if aggregate is None:
                self._modules[module] = {
                    "calls": 1,
                    "total_ms": elapsed_ms,
                    "min_ms": elapsed_ms,
                    "max_ms": elapsed_ms,
                }
            else:
                aggregate["calls"] += 1
                aggregate["total_ms"] += elapsed_ms
                aggregate["min_ms"] = min(aggregate["min_ms"], elapsed_ms)
                aggregate["max_ms"] = max(aggregate["max_ms"], elapsed_ms)
            self._next_event_id += 1
            if self._report_path is not None:
                now = time.monotonic()
                if now - self._last_flush_monotonic >= self._flush_interval_seconds:
                    self._last_flush_monotonic = now
                    should_flush = True
        if should_flush:
            self.flush_report()

    def summary(self) -> dict:
        with self._lock:
            events = list(self._events)
            aggregates = {
                module: dict(values)
                for module, values in self._modules.items()
            }
            event_count = int(self._next_event_id)
        modules = {
            module: {
                **values,
                "mean_ms": values["total_ms"] / values["calls"],
            }
            for module, values in sorted(aggregates.items())
        }
        return {
            "modules": modules,
            "events": events,
            "retained_event_count": len(events),
            "dropped_event_count": max(0, event_count - len(events)),
        }

    def configure_report(
        self,
        path: str | Path,
        metadata: dict | None = None,
        flush_interval_seconds: float = 2.0,
    ) -> None:
        """Persist a readable timing snapshot while a long-running job is active."""
        with self._lock:
            self._report_path = Path(path)
            self._report_metadata = dict(metadata or {})
            self._report_status = "running"
            self._report_error = None
            self._flush_interval_seconds = max(float(flush_interval_seconds), 0.1)
            self._last_flush_monotonic = 0.0
        self.flush_report(force=True)

    def update_report_metadata(self, metadata: dict | None = None, **extra_metadata) -> None:
        with self._lock:
            if metadata:
                self._report_metadata.update(metadata)
            self._report_metadata.update(extra_metadata)

    def flush_report(
        self,
        status: str | None = None,
        error: str | None = None,
        force: bool = False,
    ) -> bool:
        """Write a complete atomic snapshot without blocking timing producers."""
        if not force and not self._report_lock.acquire(blocking=False):
            return False
        if force:
            self._report_lock.acquire()
        try:
            with self._lock:
                if self._report_path is None:
                    return False
                if status is not None:
                    self._report_status = str(status)
                if error is not None:
                    self._report_error = str(error)
                report_path = self._report_path
                metadata = dict(self._report_metadata)
                report_status = self._report_status
                report_error = self._report_error
                event_count = int(self._next_event_id)
            payload = {
                **metadata,
                "status": report_status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "event_count": event_count,
                "summary": self.summary(),
            }
            if report_error is not None:
                payload["error"] = report_error
            report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = report_path.with_name(f".{report_path.name}.tmp")
            try:
                temporary_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                os.replace(temporary_path, report_path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
            return True
        finally:
            self._report_lock.release()


def timed_section(timings: TimingRecorder | None, module: str, **metadata):
    if timings is None:
        return nullcontext()
    return timings.measure(module, **metadata)
