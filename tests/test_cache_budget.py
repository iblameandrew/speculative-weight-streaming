"""Tests for PredictiveCache RAM budget enforcement."""

from __future__ import annotations

from sws.cache import PredictiveCache
from sws.types import ShardPayload


def _payload(sid: str, kb: int) -> ShardPayload:
    import torch

    t = torch.zeros(kb * 256, dtype=torch.float32)
    return ShardPayload(shard_id=sid, tensors={"t": t}, byte_size=t.numel() * 4)


def test_cache_eviction_respects_budget():
    cache = PredictiveCache(ram_budget_mb=1)
    cache.put(_payload("a", 256))
    cache.put(_payload("b", 256))
    cache.put(_payload("c", 256))
    cache.assert_within_budget()
    assert cache.resident_bytes() <= cache.ram_budget_bytes