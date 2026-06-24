"""Minimal MoE transformer for local SWS gates without multi-GB downloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SyntheticMoEConfig:
    vocab_size: int = 512
    hidden_size: int = 128
    intermediate_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_experts: int = 8
    num_experts_per_tok: int = 2
    max_seq_len: int = 64


class ExpertFFN(nn.Module):
    def __init__(self, cfg: SyntheticMoEConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEBlock(nn.Module):
    def __init__(self, cfg: SyntheticMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList(ExpertFFN(cfg) for _ in range(cfg.num_experts))
        self._last_expert_indices: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Set[int]]:
        bsz, seq, hidden = x.shape
        flat = x.view(-1, hidden)
        router_logits = self.router(flat)
        routing = F.softmax(router_logits, dim=-1)
        topk = torch.topk(routing, self.cfg.num_experts_per_tok, dim=-1)
        self._last_expert_indices = topk.indices

        out = torch.zeros_like(flat)
        fired: Set[int] = set()
        for expert_idx, expert in enumerate(self.experts):
            mask = (topk.indices == expert_idx).any(dim=-1)
            if not mask.any():
                continue
            fired.add(expert_idx)
            token_idx = mask.nonzero(as_tuple=False).squeeze(-1)
            expert_weights = routing[token_idx, expert_idx].unsqueeze(-1)
            out[token_idx] += expert_weights * expert(flat[token_idx])

        return out.view(bsz, seq, hidden), fired


class AttentionBlock(nn.Module):
    def __init__(self, cfg: SyntheticMoEConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape
        q = self.q_proj(x).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq, -1)
        return self.o_proj(attn)


class TransformerLayer(nn.Module):
    def __init__(self, cfg: SyntheticMoEConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.attn = AttentionBlock(cfg)
        self.moe = MoEBlock(cfg)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Set[int]]:
        h = self.norm(x)
        x = x + self.attn(h)
        h = self.norm(x)
        moe_out, fired = self.moe(h)
        return x + moe_out, fired


class SyntheticMoEModel(nn.Module):
    """Reference (vanilla) model — all weights resident."""

    def __init__(self, cfg: Optional[SyntheticMoEConfig] = None):
        super().__init__()
        self.cfg = cfg or SyntheticMoEConfig()
        self.embed = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_size)
        self.layers = nn.ModuleList(TransformerLayer(self.cfg) for _ in range(self.cfg.num_layers))
        self.final_norm = nn.LayerNorm(self.cfg.hidden_size)
        self.lm_head = nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_paths: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[int, Set[int]]]]:
        x = self.embed(input_ids)
        paths: Dict[int, Set[int]] = {}
        for layer_idx, layer in enumerate(self.layers):
            x, fired = layer(x)
            paths[layer_idx] = fired
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if return_paths:
            return logits, paths
        return logits, None

    def state_dict_named(self) -> Dict[str, torch.Tensor]:
        sd: Dict[str, torch.Tensor] = {
            "embed.weight": self.embed.weight,
            "final_norm.weight": self.final_norm.weight,
            "lm_head.weight": self.lm_head.weight,
        }
        for i, layer in enumerate(self.layers):
            for name, param in layer.norm.named_parameters():
                sd[f"layers.{i}.norm.{name}"] = param
            for name, param in layer.attn.named_parameters():
                sd[f"layers.{i}.attn.{name}"] = param
            for name, param in layer.moe.router.named_parameters():
                sd[f"layers.{i}.moe.router.{name}"] = param
            for e_idx, expert in enumerate(layer.moe.experts):
                for name, param in expert.named_parameters():
                    sd[f"layers.{i}.moe.experts.{e_idx}.{name}"] = param
        return sd


def init_deterministic_model(cfg: Optional[SyntheticMoEConfig] = None, seed: int = 0) -> SyntheticMoEModel:
    torch.manual_seed(seed)
    model = SyntheticMoEModel(cfg)
    return model