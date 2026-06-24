"""Predictive working-set cache with hard RAM budget."""

from __future__ import annotations

import time
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Set

import torch

from sws.types import Forecast, ShardId, ShardPayload


@dataclass
class CacheEntry:
    payload: ShardPayload
    priority: float = 0.0
    pinned: bool = False
    last_access: float = field(default_factory=time.time)
    is_exact: bool = True


class PredictiveCache:
    """32GB-class byte-budget cache; LRU baseline, prediction-driven eviction in Phase 4."""

    def __init__(self, ram_budget_mb: int = 32_000, pin_lower_layers: bool = False):
        self.ram_budget_bytes = ram_budget_mb * 1024 * 1024
        self.pin_lower_layers = pin_lower_layers
        self._entries: OrderedDict[ShardId, CacheEntry] = OrderedDict()
        self._resident_bytes = 0
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "peak_bytes": 0,
            "stalls": 0,
        }
        self._pending: Dict[ShardId, Future[ShardPayload]] = {}

    @property
    def peak_bytes(self) -> int:
        return self._stats["peak_bytes"]

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def resident_set(self) -> Set[ShardId]:
        return set(self._entries.keys())

    def resident_bytes(self) -> int:
        return self._resident_bytes

    def _track_peak(self) -> None:
        self._stats["peak_bytes"] = max(self._stats["peak_bytes"], self._resident_bytes)

    def _entry_bytes(self, entry: CacheEntry) -> int:
        return entry.payload.byte_size

    def _touch(self, shard_id: ShardId) -> None:
        if shard_id in self._entries:
            self._entries.move_to_end(shard_id)
            self._entries[shard_id].last_access = time.time()

    def _remove(self, shard_id: ShardId) -> None:
        entry = self._entries.pop(shard_id, None)
        if entry is not None:
            self._resident_bytes -= self._entry_bytes(entry)

    def _ensure_budget(self, incoming_bytes: int, forecast: Optional[Forecast] = None) -> None:
        while self._resident_bytes + incoming_bytes > self.ram_budget_bytes and self._entries:
            if forecast is not None:
                evict_id = self._lowest_priority_shard(forecast)
            else:
                evict_id = self._lru_evictable()
            if evict_id is None:
                break
            self._remove(evict_id)
            self._stats["evictions"] += 1

    def _lru_evictable(self) -> Optional[ShardId]:
        for shard_id, entry in self._entries.items():
            if entry.pinned:
                continue
            return shard_id
        return None

    def _lowest_priority_shard(self, forecast: Forecast) -> Optional[ShardId]:
        candidates = [
            (forecast.priority(sid), sid)
            for sid, entry in self._entries.items()
            if not entry.pinned
        ]
        if not candidates:
            return self._lru_evictable()
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def pin_shard(self, shard_id: ShardId, pinned: bool = True) -> None:
        if shard_id in self._entries:
            self._entries[shard_id].pinned = pinned

    def pin_lower_layer_shards(self, layer_threshold: int = 2) -> None:
        for shard_id in list(self._entries.keys()):
            if shard_id.startswith("layer_"):
                try:
                    layer_num = int(shard_id.split("/")[0].split("_")[1])
                except (IndexError, ValueError):
                    continue
                if layer_num <= layer_threshold:
                    self.pin_shard(shard_id, pinned=True)

    def put(self, payload: ShardPayload, priority: float = 0.0, is_exact: bool = True) -> None:
        if payload.shard_id in self._entries:
            self._remove(payload.shard_id)
        self._ensure_budget(payload.byte_size)
        self._entries[payload.shard_id] = CacheEntry(
            payload=payload,
            priority=priority,
            is_exact=is_exact,
        )
        self._resident_bytes += payload.byte_size
        self._track_peak()
        self._touch(payload.shard_id)

    def prefetch(self, future: Future[ShardPayload], priority: float = 1.0) -> None:
        shard_id = None
        if hasattr(future, "_shard_id"):
            shard_id = future._shard_id  # type: ignore[attr-defined]
        self._pending[id(future)] = future

        def _apply(payload: ShardPayload) -> None:
            self.put(payload, priority=priority, is_exact=True)

        future.add_done_callback(lambda f: _apply(f.result()) if f.exception() is None else None)

    def prefetch_shard(self, future: Future[ShardPayload], shard_id: ShardId, priority: float = 1.0) -> None:
        future._shard_id = shard_id  # type: ignore[attr-defined]
        self.prefetch(future, priority=priority)

    def wait_prefetch(self, shard_id: ShardId, timeout: float = 30.0) -> bool:
        for fut in list(self._pending.values()):
            sid = getattr(fut, "_shard_id", None)
            if sid == shard_id:
                fut.result(timeout=timeout)
                return shard_id in self._entries
        return shard_id in self._entries

    def get(self, shard_id: ShardId) -> Optional[ShardPayload]:
        if shard_id in self._entries:
            self._stats["hits"] += 1
            self._touch(shard_id)
            return self._entries[shard_id].payload
        self._stats["misses"] += 1
        return None

    def load_exact(self, payloads: Iterable[ShardPayload], forecast: Optional[Forecast] = None) -> None:
        """Synchronous exact load after verifier rejection (stall)."""
        self._stats["stalls"] += 1
        for payload in payloads:
            priority = forecast.priority(payload.shard_id) if forecast else 1.0
            self._ensure_budget(payload.byte_size, forecast=forecast)
            self.put(payload, priority=priority, is_exact=True)

    def evict_lowest_priority(self, forecast: Forecast, count: int = 1) -> None:
        for _ in range(count):
            evict_id = self._lowest_priority_shard(forecast)
            if evict_id is None:
                break
            self._remove(evict_id)
            self._stats["evictions"] += 1

    def trim_to_budget(self, forecast: Optional[Forecast] = None) -> int:
        """Evict unpinned shards until within budget; returns eviction count."""
        evicted = 0
        while self._resident_bytes > self.ram_budget_bytes and self._entries:
            if forecast is not None:
                evict_id = self._lowest_priority_shard(forecast)
            else:
                evict_id = self._lru_evictable()
            if evict_id is None:
                break
            self._remove(evict_id)
            self._stats["evictions"] += 1
            evicted += 1
        return evicted

    def assert_within_budget(self) -> None:
        assert self._resident_bytes <= self.ram_budget_bytes, (
            f"RAM budget exceeded: {self._resident_bytes} > {self.ram_budget_bytes}"
        )