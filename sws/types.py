"""Shared types for Speculative Weight Streaming."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set

import torch

ShardId = str


@dataclass(frozen=True)
class ShardPayload:
    """Weight tensors loaded from a single on-disk shard."""

    shard_id: ShardId
    tensors: Dict[str, torch.Tensor]
    byte_size: int
    is_approx: bool = False

    def clone_tensors(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.tensors.items()}


@dataclass
class LayerForecast:
    layer_idx: int
    expert_probs: torch.Tensor
    attention_prob: float = 1.0


@dataclass
class Forecast:
    """Predictor output for the next forward step."""

    layers: List[LayerForecast]
    high_confidence: Set[ShardId] = field(default_factory=set)
    low_confidence: Set[ShardId] = field(default_factory=set)
    expert_shard_ids: Dict[tuple[int, int], ShardId] = field(default_factory=dict)
    attention_shard_ids: Dict[int, ShardId] = field(default_factory=dict)

    def priority(self, shard_id: ShardId) -> float:
        for layer_fc in self.layers:
            for e_idx, prob in enumerate(layer_fc.expert_probs.tolist()):
                sid = self.expert_shard_ids.get((layer_fc.layer_idx, e_idx))
                if sid == shard_id:
                    return float(prob)
            attn_sid = self.attention_shard_ids.get(layer_fc.layer_idx)
            if attn_sid == shard_id:
                return layer_fc.attention_prob
        return 0.0


@dataclass
class RealPath:
    """Actual activation footprint captured from router hooks."""

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