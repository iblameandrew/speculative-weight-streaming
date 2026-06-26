"""Architecture constants for deepreinforce-ai/Ornith-1.0-397B (Qwen3.5-MoE)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

ORNITH_MODEL_ID = "deepreinforce-ai/Ornith-1.0-397B"
ORNITH_BASE_ARCH = "Qwen3_5MoeForConditionalGeneration"

AttentionKind = Literal["linear_attention", "full_attention"]


@dataclass(frozen=True)
class OrnithMoEConfig:
    """Text backbone config from Ornith-1.0-397B config.json."""

    model_id: str = ORNITH_MODEL_ID
    hidden_size: int = 4096
    num_hidden_layers: int = 60
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 1024
    shared_expert_intermediate_size: int = 1024
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    full_attention_interval: int = 4
    layer_types: tuple[AttentionKind, ...] = field(default_factory=lambda: _default_layer_types())

    # SWS operational defaults for 32 GB single-node target
    ram_budget_mb: int = 32_000
    draft_history_tokens: int = 128
    draft_speculative_experts: int = 4  # prefetch k + ε beyond router top-k

    @property
    def active_expert_ratio(self) -> float:
        return self.num_experts_per_tok / self.num_experts

    @property
    def experts_per_forward(self) -> int:
        return self.num_hidden_layers * self.num_experts_per_tok

    def attention_kind(self, layer_idx: int) -> AttentionKind:
        return self.layer_types[layer_idx]

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "full_attention"


def _default_layer_types() -> tuple[AttentionKind, ...]:
    kinds: List[AttentionKind] = []
    for i in range(60):
        kinds.append("full_attention" if (i + 1) % 4 == 0 else "linear_attention")
    return tuple(kinds)


def ornith_expert_shard_bytes(cfg: OrnithMoEConfig, dtype_bytes: int = 2) -> int:
    """BF16 bytes per routed expert FFN shard (gate, up, down)."""
    elems = 3 * cfg.hidden_size * cfg.moe_intermediate_size
    return elems * dtype_bytes


def ornith_working_set_estimate_mb(cfg: OrnithMoEConfig) -> float:
    """Rough lower bound: active experts + shared + attention infra one forward."""
    expert_b = ornith_expert_shard_bytes(cfg)
    routed = cfg.experts_per_forward * expert_b
    shared = cfg.num_hidden_layers * expert_b  # shared expert every layer
    attn_overhead = cfg.num_hidden_layers * 80 * 1024 * 1024  # ~80 MB/layer placeholder
    return (routed + shared + attn_overhead) / (1024 * 1024)