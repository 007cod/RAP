"""Bounded parallel Normal computation, single-flight and versioned disk cache."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from threading import Lock

import numpy as np
import pandas as pd

from src.tool_llm.timing import timed_section


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def data_version(data):
    """File identity for immutable mmap data; content identity for in-memory fixtures."""
    digest = hashlib.sha256()
    for month, array in sorted(data.month_arrays.items()):
        filename = getattr(array, 'filename', None)
        if filename:
            path = Path(filename).resolve()
            stat = path.stat()
            digest.update(str((month, str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)).encode())
        else:
            digest.update(str((month, array.shape, str(array.dtype))).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
    for name in ('incidents', 'sensors'):
        table = getattr(data, name, None)
        if table is not None:
            digest.update(pd.util.hash_pandas_object(table, index=True).values.tobytes())
    digest.update(np.asarray(data.node_order).tobytes())
    route_path = Path(getattr(data, 'osrm_cache_path', Path(data.data_dir) / 'osrm_routes.npz'))
    if route_path.exists():
        stat = route_path.stat()
        digest.update(str((str(route_path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)).encode())
    for name in ('normal_runtime.py', 'non_incident_retrieval.py', 'time_context.py', 'horizon_evidence.py'):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


class NormalRuntime:
    def __init__(self, data, cache_path, workers=4):
        if workers < 1:
            raise ValueError('Normal workers must be positive')
        self.data, self.cache_path = data, Path(cache_path)
        self.workers = workers
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='normal')
        self._root = None
        self._root_lock = Lock()
        self._locks = [Lock() for _ in range(128)]
        self._feature_locks = [Lock() for _ in range(64)]
        self._features = OrderedDict()
        self._features_lock = Lock()
        self._normal = OrderedDict()
        self._normal_lock = Lock()

    @property
    def root(self):
        with self._root_lock:
            if self._root is None:
                self._root = self.cache_path.parent / (self.cache_path.stem + '_runtime') / data_version(self.data)
            return self._root

    def _lock(self, key):
        return self._locks[int(_digest(key)[:8], 16) % len(self._locks)]

    def map(self, function, items):
        # executor.map yields in input order, so completion order never reranks.
        return list(self.executor.map(function, items))

    @staticmethod
    def _atomic(path, writer):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as handle:
                writer(handle)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def normal(self, key, compute, timings=None, metadata=None):
        metadata = metadata or {}
        # Concurrent identical Normal requests compute only once in this process.
        lock = self._lock(('normal', key))
        with timed_section(timings, 'normal_cache.singleflight_wait', **metadata):
            lock.acquire()
        try:
            with self._normal_lock:
                cached = self._normal.get(key)
                if cached is not None:
                    self._normal.move_to_end(key)
            if cached is not None:
                with timed_section(timings, 'normal_cache.memory_hit', **metadata):
                    return copy.deepcopy(cached)
            path = self.root / 'normal' / (_digest(key) + '.json')
            cached = None
            with timed_section(timings, 'normal_cache.disk_lookup', **metadata):
                try:
                    envelope = json.loads(path.read_text())
                    if envelope['checksum'] == _digest(envelope['result']):
                        cached = envelope['result']
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            if cached is None:
                with timed_section(timings, 'normal_cache.compute', **metadata):
                    cached = compute()
                with timed_section(timings, 'normal_cache.disk_write', **metadata):
                    payload = json.dumps({'checksum': _digest(cached), 'result': cached}).encode()
                    self._atomic(path, lambda handle: handle.write(payload))
            elif timings is not None:
                timings.record('normal_cache.disk_hit', 0, metadata)
            with self._normal_lock:
                self._normal[key] = copy.deepcopy(cached)
                # Paired-state retrieval revisits candidates across nearby
                # incidents.  Keep enough completed references to bridge many
                # query nodes instead of immediately falling back to JSON I/O.
                while len(self._normal) > 8192:
                    self._normal.popitem(last=False)
            return cached
        finally:
            lock.release()

    def features(self, node_index, count, compute, timings=None, metadata=None):
        with timed_section(timings, 'normal_features.singleflight_wait', **(metadata or {})):
            lock = self._feature_locks[node_index % len(self._feature_locks)]
            lock.acquire()
        try:
            return self._features_locked(node_index, count, compute, timings, metadata)
        finally:
            lock.release()

    def _features_locked(self, node_index, count, compute, timings=None, metadata=None):
        # Feature locks must be distinct from normal locks: compute is nested.
        with self._features_lock:
            item = self._features.get(node_index)
            if item is not None:
                self._features.move_to_end(node_index)
                return item
        path = self.root / 'features' / f'node_{node_index}.npz'
        # Per-node single-flight avoids recomputing features for parallel
        # candidates. Atomic writes also protect readers in other processes.
        with timed_section(timings, 'normal_features.load_or_build', **(metadata or {})):
            try:
                with np.load(path, allow_pickle=False) as payload:
                    item = (payload['trend'], payload['volatility'])
                if any(a.shape != (count, 3) or a.dtype != np.float32 or not np.isfinite(a).all() for a in item):
                    raise ValueError('Invalid feature cache')
            except (OSError, ValueError, EOFError, KeyError):
                item = compute()
                self._atomic(path, lambda handle: np.savez(handle, trend=item[0], volatility=item[1]))
        for array in item:
            array.setflags(write=False)
        with self._features_lock:
            self._features[node_index] = item
            # Candidate features are several orders of magnitude smaller than
            # the raw traffic data and are reused by every incident at a node.
            while len(self._features) > 64:
                self._features.popitem(last=False)
        return item
