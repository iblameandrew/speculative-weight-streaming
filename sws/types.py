"""Shared types for Speculative Weight Streaming."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch

ShardId = str


@dataclass(frozen=True)
class ShardPayload:
    """Weight tensors loaded from a single on-disk shard (raw clay piece)."""

    shard_id: ShardId
    tensors: Dict[str, torch.Tensor]
    byte_size: int
    is_approx: bool = False

    def clone_tensors(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.tensors.items()}


@dataclass
class LayerBlueprint:
    """Per-layer selection plan from the micro draft model."""

    layer_idx: int
    expert_probs: torch.Tensor
    attention_prob: float = 1.0
    selected_experts: Set[int] = field(default_factory=set)


@dataclass
class ReassemblyBlueprint:
    """Micro draft model output: which raw pieces to fetch and how to compose them."""

    layers: List[LayerBlueprint]
    high_priority_pieces: Set[ShardId] = field(default_factory=set)
    low_priority_pieces: Set[ShardId] = field(default_factory=set)
    expert_shard_ids: Dict[Tuple[int, int], ShardId] = field(default_factory=dict)
    attention_shard_ids: Dict[int, ShardId] = field(default_factory=dict)

    @property
    def high_confidence(self) -> Set[ShardId]:
        return self.high_priority_pieces

    @property
    def low_confidence(self) -> Set[ShardId]:
        return self.low_priority_pieces

    def priority(self, shard_id: ShardId) -> float:
        for layer_bp in self.layers:
            for e_idx, prob in enumerate(layer_bp.expert_probs.tolist()):
                sid = self.expert_shard_ids.get((layer_bp.layer_idx, e_idx))
                if sid == shard_id:
                    return float(prob)
            attn_sid = self.attention_shard_ids.get(layer_bp.layer_idx)
            if attn_sid == shard_id:
                return layer_bp.attention_prob
        return 0.0

    def all_pieces(self) -> Set[ShardId]:
        return self.high_priority_pieces | self.low_priority_pieces


# Backward-compatible alias
LayerForecast = LayerBlueprint
Forecast = ReassemblyBlueprint


@dataclass
class AssembledSubgraph:
    """Executable sub-model instance materialized from selected weight pieces."""

    piece_ids: Set[ShardId] = field(default_factory=set)
    byte_size: int = 0
    is_exact: bool = True
    step_id: int = 0


@dataclass
class RealPath:
    """Actual activation footprint captured from router hooks (reference giant)."""

    fired_experts: Dict[int, Set[int]] = field(default_factory=dict)
    used_attention: Set[int] = field(default_factory=set)

    def shard_ids(
        self,
        expert_shard_fn,
        attention_shard_fn,
    ) -> FrozenSet[ShardId]:
        shards: Set[ShardId] = set()
        for layer_idx, experts in self.fired_experts.items():
            for e_idx in experts:
                shards.add(expert_shard_fn(layer_idx, e_idx))
        for layer_idx in self.used_attention:
            shards.add(attention_shard_fn(layer_idx))
        return frozenset(shards)


@dataclass
class StepMetrics:
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    verifier_accepts: int = 0
    verifier_rejects: int = 0
    stalls_ms: float = 0.0
    used_approx: int = 0
    reassembly_count: int = 0