"""DynamicAssembler — reassembly engine and RAM budget enforcer."""

from __future__ import annotations

import time
from typing import List, Optional, Set

import torch

from sws.cache import PredictiveCache
from sws.lazy_model import LazyMoERunner
from sws.store import NVMeWeightStore
from sws.synthetic_moe import SyntheticMoEConfig
from sws.types import AssembledSubgraph, RealPath, ReassemblyBlueprint, ShardId, ShardPayload


class DynamicAssembler:
    """
    Takes a reassembly blueprint, fetches selected raw pieces from NVMe,
    and materializes an executable subgraph within the RAM budget.
    """

    def __init__(
        self,
        cfg: SyntheticMoEConfig,
        ram_budget_mb: int = 32_000,
        pin_lower_layers: bool = False,
    ):
        self.cfg = cfg
        self.cache = PredictiveCache(ram_budget_mb, pin_lower_layers=pin_lower_layers)
        self.runner = LazyMoERunner(cfg, self.cache)
        self._subgraph = AssembledSubgraph()
        self._step_counter = 0

    @property
    def resident_set(self) -> Set[ShardId]:
        return self.cache.resident_set()

    @property
    def peak_bytes(self) -> int:
        return self.cache.peak_bytes

    @property
    def stats(self) -> dict:
        return self.cache.stats

    @property
    def assembled(self) -> AssembledSubgraph:
        return self._subgraph

    def assemble_from_blueprint(
        self,
        blueprint: ReassemblyBlueprint,
        store: NVMeWeightStore,
        use_approx: bool = False,
    ) -> AssembledSubgraph:
        """Materialize selected pieces into an executable subgraph."""
        self._step_counter += 1
        pieces: Set[ShardId] = set()
        is_exact = True

        futures: list[tuple[ShardId, object]] = []
        for shard in blueprint.high_priority_pieces:
            if self.cache.get(shard) is None:
                fut = store.fetch(shard)
                self.cache.prefetch_shard(fut, shard, priority=blueprint.priority(shard))
                futures.append((shard, fut))
            pieces.add(shard)

        for shard in blueprint.low_priority_pieces:
            if self.cache.get(shard) is None:
                if use_approx:
                    self.cache.put(
                        store.extract_pieces({shard}, exact=False)[0],
                        priority=blueprint.priority(shard),
                        is_exact=False,
                    )
                    is_exact = False
                else:
                    fut = store.fetch(shard)
                    self.cache.prefetch_shard(fut, shard, priority=blueprint.priority(shard))
                    futures.append((shard, fut))
            else:
                entry = self.cache._entries.get(shard)
                if entry and not entry.is_exact:
                    is_exact = False
            pieces.add(shard)

        for shard, fut in futures:
            if self.cache.get(shard) is None:
                try:
                    self.cache.put(fut.result(), priority=blueprint.priority(shard), is_exact=True)
                except Exception:
                    self.cache.load_exact(store.extract_pieces({shard}, exact=True), blueprint)

        self._subgraph = AssembledSubgraph(
            piece_ids=pieces,
            byte_size=self.cache.resident_bytes(),
            is_exact=is_exact,
            step_id=self._step_counter,
        )
        return self._subgraph

    def assemble_essential(
        self,
        store: NVMeWeightStore,
    ) -> ReassemblyBlueprint:
        """Baseline assembly: infrastructure pieces only; experts on demand."""
        from sws.sharding import attention_shard_id, layer_norm_shard_id, router_shard_id

        essential: set[ShardId] = {"embed_weight", "final_norm_weight", "lm_head_weight"}
        for layer in range(self.cfg.num_layers):
            essential.add(layer_norm_shard_id(layer))
            essential.add(attention_shard_id(layer))
            essential.add(router_shard_id(layer))

        blueprint = ReassemblyBlueprint(layers=[], high_priority_pieces=essential)
        for shard_id in essential:
            if self.cache.get(shard_id) is None:
                self.cache.load_exact(store.extract_pieces({shard_id}, exact=True))
        self._subgraph = AssembledSubgraph(piece_ids=essential, byte_size=self.cache.resident_bytes())
        return blueprint

    def execute_assembled(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, RealPath]:
        """Run forward on the currently assembled subgraph."""
        return self.runner.forward(input_ids)

    def fetch_missing_pieces(
        self,
        miss: Set[ShardId],
        store: NVMeWeightStore,
        blueprint: Optional[ReassemblyBlueprint] = None,
        use_approx: bool = False,
    ) -> None:
        """Synchronous exact fetch for verifier fallback."""
        if not miss:
            return
        payloads: List[ShardPayload] = []
        for sid in miss:
            if use_approx and blueprint and sid in blueprint.low_priority_pieces:
                payloads.extend(store.extract_pieces({sid}, exact=False))
            else:
                payloads.extend(store.extract_pieces({sid}, exact=True))
        self.cache.load_exact(payloads, blueprint)
        self._subgraph.piece_ids |= miss
        self._subgraph.byte_size = self.cache.resident_bytes()
        self._subgraph.is_exact = True

    def fallback_reassemble(
        self,
        miss: Set[ShardId],
        blueprint: ReassemblyBlueprint,
        store: NVMeWeightStore,
    ) -> AssembledSubgraph:
        """Fetch additional exact pieces and rebuild a more complete subgraph."""
        self.fetch_missing_pieces(miss, store, blueprint, use_approx=False)
        self._subgraph = AssembledSubgraph(
            piece_ids=self.cache.resident_set(),
            byte_size=self.cache.resident_bytes(),
            is_exact=True,
            step_id=self._step_counter,
        )
        return self._subgraph

    def execute_with_resilience(
        self,
        input_ids: torch.Tensor,
        store: NVMeWeightStore,
        blueprint: ReassemblyBlueprint,
        max_retries: int = 32,
    ) -> tuple[torch.Tensor, RealPath, int]:
        """
        Execute assembled subgraph, incrementally fetching missing pieces.
        Returns (output, real_path, reject_count).
        """
        rejects = 0
        pinned: list[ShardId] = []

        def _is_infrastructure(sid: ShardId) -> bool:
            return (
                "attention" in sid
                or "norm" in sid
                or "router" in sid
                or sid in {"embed_weight", "final_norm_weight", "lm_head_weight"}
            )

        for shard in blueprint.high_priority_pieces:
            if _is_infrastructure(shard) and shard in self.cache.resident_set():
                self.cache.pin_shard(shard, pinned=True)
                pinned.append(shard)

        out: Optional[torch.Tensor] = None
        real_path: Optional[RealPath] = None
        try:
            for _ in range(max_retries):
                self.runner.last_misses = set()
                try:
                    out, real_path = self.runner.forward(input_ids)
                    break
                except KeyError:
                    miss = set(self.runner.last_misses) - self.cache.resident_set()
                    if not miss:
                        raise
                    rejects += 1
                    self.fetch_missing_pieces(miss, store, blueprint)
        finally:
            for sid in pinned:
                if sid in self.cache.resident_set():
                    self.cache.pin_shard(sid, pinned=False)
            self.cache.trim_to_budget(blueprint)

        if out is None or real_path is None:
            raise RuntimeError("Reassembly loop failed to produce a valid forward pass")

        return out, real_path, rejects

    def pin_lower_layer_shards(self, layer_threshold: int = 1) -> None:
        self.cache.pin_lower_layer_shards(layer_threshold)

    def assert_within_budget(self) -> None:
        self.cache.assert_within_budget()