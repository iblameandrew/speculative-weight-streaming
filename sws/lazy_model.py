"""Lazy MoE model — weights pulled from PredictiveCache on forward."""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from sws.cache import PredictiveCache
from sws.sharding import attention_shard_id, expert_shard_id, layer_norm_shard_id, router_shard_id
from sws.synthetic_moe import SyntheticMoEConfig
from sws.types import RealPath, ShardId


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(x, weight)


def _require_tensors(
    cache: PredictiveCache,
    shard_id: ShardId,
    keys: Tuple[str, ...],
    missed: Optional[Set[ShardId]] = None,
) -> Dict[str, torch.Tensor]:
    payload = cache.get(shard_id)
    if payload is None:
        if missed is not None:
            missed.add(shard_id)
        raise KeyError(shard_id)
    missing = [k for k in keys if k not in payload.tensors]
    if missing:
        raise KeyError(f"Shard {shard_id} missing keys: {missing}")
    return payload.tensors


class LazyMoERunner:
    """Runs one forward pass using shards resident in cache."""

    def __init__(self, cfg: SyntheticMoEConfig, cache: PredictiveCache):
        self.cfg = cfg
        self.cache = cache
        self.last_misses: Set[ShardId] = set()

    def _attn(self, layer: int, x: torch.Tensor, missed: Set[ShardId]) -> torch.Tensor:
        sid = attention_shard_id(layer)
        t = _require_tensors(
            self.cache, sid, ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"), missed,
        )
        bsz, seq, hidden = x.shape
        num_heads = self.cfg.num_heads
        head_dim = hidden // num_heads
        q = _linear(x, t["q_proj.weight"]).view(bsz, seq, num_heads, head_dim).transpose(1, 2)
        k = _linear(x, t["k_proj.weight"]).view(bsz, seq, num_heads, head_dim).transpose(1, 2)
        v = _linear(x, t["v_proj.weight"]).view(bsz, seq, num_heads, head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq, hidden)
        return _linear(attn, t["o_proj.weight"])

    def _expert(self, layer: int, expert: int, x: torch.Tensor, missed: Set[ShardId]) -> torch.Tensor:
        sid = expert_shard_id(layer, expert)
        t = _require_tensors(
            self.cache, sid, ("gate_proj.weight", "up_proj.weight", "down_proj.weight"), missed,
        )
        gate = _linear(x, t["gate_proj.weight"])
        up = _linear(x, t["up_proj.weight"])
        return _linear(F.silu(gate) * up, t["down_proj.weight"])

    def _moe(self, layer: int, x: torch.Tensor, missed: Set[ShardId]) -> Tuple[torch.Tensor, Set[int]]:
        router_sid = router_shard_id(layer)
        rt = _require_tensors(self.cache, router_sid, ("weight",), missed)
        bsz, seq, hidden = x.shape
        flat = x.view(-1, hidden)
        router_logits = _linear(flat, rt["weight"])
        routing = F.softmax(router_logits, dim=-1)
        topk = torch.topk(routing, self.cfg.num_experts_per_tok, dim=-1)

        out = torch.zeros_like(flat)
        fired: Set[int] = set()
        for expert_idx in range(self.cfg.num_experts):
            mask = (topk.indices == expert_idx).any(dim=-1)
            if not mask.any():
                continue
            fired.add(expert_idx)
            token_idx = mask.nonzero(as_tuple=False).squeeze(-1)
            expert_weights = routing[token_idx, expert_idx].unsqueeze(-1)
            out[token_idx] += expert_weights * self._expert(layer, expert_idx, flat[token_idx], missed)
        return out.view(bsz, seq, hidden), fired

    def _layer_norm(self, layer: int, x: torch.Tensor, missed: Set[ShardId]) -> torch.Tensor:
        sid = layer_norm_shard_id(layer)
        t = _require_tensors(self.cache, sid, ("weight",), missed)
        return F.layer_norm(x, (self.cfg.hidden_size,), t["weight"])

    def forward(
        self,
        input_ids: torch.Tensor,
        embed_shard: ShardId = "embed_weight",
        final_norm_shard: ShardId = "final_norm_weight",
        lm_head_shard: ShardId = "lm_head_weight",
    ) -> Tuple[torch.Tensor, RealPath]:
        missed: Set[ShardId] = set()
        self.last_misses = missed
        embed = _require_tensors(self.cache, embed_shard, ("embed.weight",), missed)
        x = F.embedding(input_ids, embed["embed.weight"])

        real_path = RealPath(used_attention=set(range(self.cfg.num_layers)))
        for layer_idx in range(self.cfg.num_layers):
            h = self._layer_norm(layer_idx, x, missed)
            x = x + self._attn(layer_idx, h, missed)
            h = self._layer_norm(layer_idx, x, missed)
            moe_out, fired = self._moe(layer_idx, h, missed)
            real_path.fired_experts[layer_idx] = fired
            x = x + moe_out

        fn = _require_tensors(self.cache, final_norm_shard, ("final_norm.weight",), missed)
        x = F.layer_norm(x, (self.cfg.hidden_size,), fn["final_norm.weight"])
        lh = _require_tensors(self.cache, lm_head_shard, ("lm_head.weight",), missed)
        logits = _linear(x, lh["lm_head.weight"])
        return logits, real_path

    def required_shards_for_path(self, real_path: RealPath) -> Set[ShardId]:
        needed: Set[ShardId] = {"embed_weight", "final_norm_weight", "lm_head_weight"}
        for layer_idx in range(self.cfg.num_layers):
            needed.add(layer_norm_shard_id(layer_idx))
            needed.add(router_shard_id(layer_idx))
            needed.add(attention_shard_id(layer_idx))
            for e_idx in real_path.fired_experts.get(layer_idx, set()):
                needed.add(expert_shard_id(layer_idx, e_idx))
        return needed