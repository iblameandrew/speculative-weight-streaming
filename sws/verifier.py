"""Verifier — ensures assembled sub-model matches reference giant activation path."""

from __future__ import annotations

from typing import Set

from sws.cache import PredictiveCache
from sws.lazy_model import LazyMoERunner
from sws.types import RealPath, ReassemblyBlueprint, ShardId


class Verifier:
    """Strict safety net: detect misses between blueprint assembly and realized path."""

    def detect_miss(
        self,
        real_path: RealPath,
        cache: PredictiveCache,
        runner: LazyMoERunner,
        blueprint: ReassemblyBlueprint,
    ) -> Set[ShardId]:
        """
        Return shard IDs that must be fetched/reassembled for functional equivalence.
        Includes absent pieces and inexact (approximated) stand-ins.
        """
        needed = runner.required_shards_for_path(real_path)
        resident = cache.resident_set()
        miss = set(needed) - resident

        for sid in needed:
            entry = cache._entries.get(sid)
            if entry is not None and not entry.is_exact:
                miss.add(sid)

        return miss

    def accepts(self, miss: Set[ShardId]) -> bool:
        return len(miss) == 0