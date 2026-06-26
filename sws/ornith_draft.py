"""Ornith-1.0-397B micro draft model — hierarchical selector architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sws.ornith_config import ORNITH_MODEL_ID, OrnithMoEConfig, ornith_working_set_estimate_mb
from sws.sharding import (
    attention_shard_id,
    expert_shard_id,
    layer_norm_shard_id,
    router_shard_id,
)
from sws.types import LayerBlueprint, RealPath, ReassemblyBlueprint, ShardId

# Ornith-specific shard naming
SHARED_EXPERT_SHARD = "shared_expert"
LINEAR_ATTN_SUFFIX = "linear_attention"
FULL_ATTN_SUFFIX = "full_attention"


def ornith_attention_shard_id(layer: int, kind: str) -> ShardId:
    return f"layer_{layer}/{kind}"


def ornith_shared_expert_shard_id(layer: int) -> ShardId:
    return f"layer_{layer}/{SHARED_EXPERT_SHARD}"


@dataclass
class DraftModelSpec:
    """Proposed parameter budget for the Ornith micro draft model."""

    history_embed_dim: int = 128
    history_bucket_size: int = 8192
    trunk_dim: int = 2048
    layer_embed_dim: int = 64
    cross_layer_layers: int = 2
    cross_layer_heads: int = 4
    expert_super_buckets: int = 64
    estimated_params_millions: float = 0.0
    estimated_resident_mb_bf16: float = 0.0


class _HistoryEncoder(nn.Module):
    """Compress recent token IDs into a fixed context vector."""

    def __init__(self, bucket_size: int, embed_dim: int, gru_hidden: int = 128):
        super().__init__()
        self.embed = nn.Embedding(bucket_size, embed_dim)
        self.gru = nn.GRU(embed_dim, gru_hidden, batch_first=True)
        self.out_dim = gru_hidden

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (seq,) int — bucketed externally
        x = self.embed(token_ids.unsqueeze(0))
        _, h = self.gru(x)
        return h[-1, 0]


class _CrossLayerRefiner(nn.Module):
    """Lightweight transformer over layer dimension — experts correlate across depth."""

    def __init__(self, num_layers: int, num_experts: int, dim: int, n_heads: int, n_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 2,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(dim, num_experts)

    def forward(self, layer_feats: torch.Tensor) -> torch.Tensor:
        # layer_feats: (num_layers, dim)
        refined = self.encoder(layer_feats.unsqueeze(0)).squeeze(0)
        return self.out_proj(refined)


class OrnithDraftSelector(nn.Module):
    """
    Hierarchical micro draft model for Ornith-1.0-397B.

    Stage 1 — fuse pooled giant hidden state + token-history encoding.
    Stage 2 — per-layer trunk produces expert logits (512 per layer × 60 layers).
    Stage 3 — cross-layer refiner exploits depth-wise routing correlation.
    Stage 4 — super-bucket auxiliary head for coarse 64-way pre-filter (optional).

    Design constraint: output space is 60 × 512 = 30,720 expert activations per step;
    a flat MLP is infeasible. This hierarchy keeps the resident footprint ~150–350 MB (BF16).
    """

    def __init__(self, cfg: OrnithMoEConfig, spec: Optional[DraftModelSpec] = None):
        super().__init__()
        self.cfg = cfg
        self.spec = spec or DraftModelSpec()
        h = cfg.hidden_size
        d = self.spec.trunk_dim

        self.history_enc = _HistoryEncoder(
            self.spec.history_bucket_size,
            self.spec.history_embed_dim,
        )
        hist_dim = self.history_enc.out_dim

        self.input_proj = nn.Sequential(
            nn.Linear(h + hist_dim, d),
            nn.SiLU(),
            nn.Linear(d, d),
            nn.SiLU(),
        )

        self.layer_embed = nn.Embedding(cfg.num_hidden_layers, self.spec.layer_embed_dim)
        self.layer_trunk = nn.Sequential(
            nn.Linear(d + self.spec.layer_embed_dim, d),
            nn.SiLU(),
        )
        self.expert_heads = nn.ModuleList(
            [nn.Linear(d, cfg.num_experts) for _ in range(cfg.num_hidden_layers)]
        )

        self.cross_layer = _CrossLayerRefiner(
            cfg.num_hidden_layers,
            cfg.num_experts,
            d,
            self.spec.cross_layer_heads,
            self.spec.cross_layer_layers,
        )

        self.super_bucket_head = nn.Linear(d, cfg.num_hidden_layers * self.spec.expert_super_buckets)
        self.attn_load_head = nn.Linear(d, cfg.num_hidden_layers)

        self._estimate_size()

    def _estimate_size(self) -> None:
        params = sum(p.numel() for p in self.parameters())
        self.spec.estimated_params_millions = params / 1e6
        self.spec.estimated_resident_mb_bf16 = params * 2 / (1024 * 1024)

    def _bucket_tokens(self, history: List[int]) -> torch.Tensor:
        bucket = self.spec.history_bucket_size
        ids = [t % bucket for t in history[-self.cfg.draft_history_tokens :]]
        if not ids:
            ids = [0]
        return torch.tensor(ids, dtype=torch.long)

    def forward(
        self,
        hidden: torch.Tensor,
        history: Optional[List[int]] = None,
        device: torch.device | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            expert_logits: (num_layers, num_experts)
            attn_load: (num_layers,) — predicted attention-block load
        """
        device = device or hidden.device
        pooled = hidden.mean(dim=1).squeeze(0).to(device)

        if history:
            hist_ids = self._bucket_tokens(history).to(device)
            hist_vec = self.history_enc(hist_ids)
        else:
            hist_vec = torch.zeros(self.history_enc.out_dim, device=device)

        ctx = self.input_proj(torch.cat([pooled, hist_vec], dim=-1))

        layer_feats = []
        expert_logits_list = []
        for layer_idx in range(self.cfg.num_hidden_layers):
            le = self.layer_embed(torch.tensor(layer_idx, device=device))
            feat = self.layer_trunk(torch.cat([ctx, le], dim=-1))
            layer_feats.append(feat)
            expert_logits_list.append(self.expert_heads[layer_idx](feat))

        layer_stack = torch.stack(layer_feats, dim=0)
        cross_logits = self.cross_layer(layer_stack)

        expert_logits = torch.stack(expert_logits_list, dim=0)
        expert_logits = expert_logits + 0.5 * cross_logits

        attn_load = torch.sigmoid(self.attn_load_head(ctx))
        return expert_logits, attn_load

    def super_bucket_logits(self, hidden: torch.Tensor, history: Optional[List[int]] = None) -> torch.Tensor:
        device = hidden.device
        pooled = hidden.mean(dim=1).squeeze(0).to(device)
        if history:
            hist_vec = self.history_enc(self._bucket_tokens(history).to(device))
        else:
            hist_vec = torch.zeros(self.history_enc.out_dim, device=device)
        ctx = self.input_proj(torch.cat([pooled, hist_vec], dim=-1))
        return self.super_bucket_head(ctx).view(
            self.cfg.num_hidden_layers, self.spec.expert_super_buckets
        )


class OrnithMicroDraftModel:
    """
    Production micro draft model wrapper for Ornith-1.0-397B SWS deployment.
    Emits ReassemblyBlueprint with Ornith-specific shard topology.
    """

    def __init__(
        self,
        cfg: Optional[OrnithMoEConfig] = None,
        confidence_threshold: float = 0.12,
        device: torch.device | str = "cpu",
        spec: Optional[DraftModelSpec] = None,
    ):
        self.cfg = cfg or OrnithMoEConfig()
        self.tau = confidence_threshold
        self.device = torch.device(device)
        self.spec = spec or DraftModelSpec()
        self._selector = OrnithDraftSelector(self.cfg, self.spec).to(self.device)
        self._optimizer = torch.optim.AdamW(self._selector.parameters(), lr=5e-4, weight_decay=0.01)
        self._history: List[int] = []

        # Reasoning-phase token markers (Qwen/Ornith template)
        self._think_open_id: Optional[int] = None
        self._think_close_id: Optional[int] = None

    @property
    def architecture_spec(self) -> DraftModelSpec:
        return self.spec

    def _build_shard_maps(self) -> Tuple[Dict[Tuple[int, int], ShardId], Dict[int, ShardId]]:
        expert_map = {
            (layer, expert): expert_shard_id(layer, expert)
            for layer in range(self.cfg.num_hidden_layers)
            for expert in range(self.cfg.num_experts)
        }
        attn_map = {
            layer: ornith_attention_shard_id(layer, self.cfg.attention_kind(layer))
            for layer in range(self.cfg.num_hidden_layers)
        }
        return expert_map, attn_map

    def _in_reasoning_phase(self, history: List[int]) -> bool:
        """Ornith is a reasoning model — routing differs inside <think> blocks."""
        if not history or self._think_open_id is None:
            return False
        opens = sum(1 for t in history if t == self._think_open_id)
        closes = sum(1 for t in history if t == self._think_close_id)
        return opens > closes

    def select_and_plan(
        self,
        hidden: torch.Tensor,
        history: Optional[List[int]] = None,
    ) -> ReassemblyBlueprint:
        if history is not None:
            self._history = history

        prefetch_k = self.cfg.num_experts_per_tok + self.cfg.draft_speculative_experts
        reasoning_boost = 1.15 if self._in_reasoning_phase(self._history) else 1.0

        with torch.no_grad():
            expert_logits, attn_load = self._selector(
                hidden.to(self.device), self._history, device=self.device,
            )
            probs = torch.sigmoid(expert_logits * reasoning_boost)

        expert_map, attn_map = self._build_shard_maps()
        layers: List[LayerBlueprint] = []
        high: Set[ShardId] = set()
        low: Set[ShardId] = set()

        for layer_idx in range(self.cfg.num_hidden_layers):
            layer_probs = probs[layer_idx]
            topk = torch.topk(layer_probs, prefetch_k)
            selected = set(topk.indices.tolist())

            layers.append(
                LayerBlueprint(
                    layer_idx=layer_idx,
                    expert_probs=layer_probs.cpu(),
                    attention_prob=float(attn_load[layer_idx].item()),
                    selected_experts=selected,
                )
            )

            high.add(attn_map[layer_idx])
            high.add(layer_norm_shard_id(layer_idx))
            high.add(router_shard_id(layer_idx))
            high.add(ornith_shared_expert_shard_id(layer_idx))

            for expert_idx, prob in zip(topk.indices.tolist(), topk.values.tolist()):
                sid = expert_map[(layer_idx, expert_idx)]
                if prob >= self.tau:
                    high.add(sid)
                else:
                    low.add(sid)

        high.update({"embed_weight", "final_norm_weight", "lm_head_weight"})

        return ReassemblyBlueprint(
            layers=layers,
            high_priority_pieces=high,
            low_priority_pieces=low - high,
            expert_shard_ids=expert_map,
            attention_shard_ids=attn_map,
        )

    def adapt(
        self,
        blueprint: ReassemblyBlueprint,
        real_path: RealPath,
        hidden: Optional[torch.Tensor] = None,
    ) -> float:
        if hidden is None:
            return 0.0

        target = torch.zeros(
            self.cfg.num_hidden_layers,
            self.cfg.num_experts,
            device=self.device,
        )
        for layer_idx, experts in real_path.fired_experts.items():
            for e_idx in experts:
                if e_idx < self.cfg.num_experts:
                    target[layer_idx, e_idx] = 1.0

        self._selector.train()
        with torch.enable_grad():
            logits, _ = self._selector(hidden.detach().to(self.device), self._history, self.device)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._selector.parameters(), 1.0)
            self._optimizer.step()
        self._selector.eval()
        return float(loss.item())

    def train_offline_router_distillation(
        self,
        traces: List[Tuple[torch.Tensor, RealPath, Optional[List[int]]]],
        epochs: int = 3,
    ) -> List[float]:
        """Distill Ornith giant router decisions into the micro draft selector."""
        losses: List[float] = []
        self._selector.train()
        for _ in range(epochs):
            epoch_loss = 0.0
            for hidden, real_path, hist in traces:
                if hist:
                    self._history = hist
                target = torch.zeros(
                    self.cfg.num_hidden_layers,
                    self.cfg.num_experts,
                    device=self.device,
                )
                for layer_idx, experts in real_path.fired_experts.items():
                    for e_idx in experts:
                        if e_idx < self.cfg.num_experts:
                            target[layer_idx, e_idx] = 1.0
                logits, _ = self._selector(hidden.to(self.device), self._history, self.device)
                loss = F.binary_cross_entropy_with_logits(logits, target)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                epoch_loss += float(loss.item())
            losses.append(epoch_loss / max(len(traces), 1))
        self._selector.eval()
        return losses

    def bind_tokenizer(self, tokenizer) -> None:
        """Resolve reasoning marker token IDs from Ornith tokenizer."""
        for marker in ("<think>", "</think>"):
            try:
                tid = tokenizer.convert_tokens_to_ids(marker)
                if marker == "<think>":
                    self._think_open_id = tid
                else:
                    self._think_close_id = tid
            except Exception:
                pass


def propose_architecture_summary(cfg: Optional[OrnithMoEConfig] = None) -> dict:
    """Return architecture proposal metrics without instantiating on GPU."""
    cfg = cfg or OrnithMoEConfig()
    model = OrnithDraftSelector(cfg)
    return {
        "target_giant": ORNITH_MODEL_ID,
        "giant_params_b": 397,
        "giant_experts_per_layer": cfg.num_experts,
        "giant_active_experts_per_token": cfg.num_experts_per_tok,
        "giant_layers": cfg.num_hidden_layers,
        "attention_pattern": "3× linear_attention + 1× full_attention (repeating)",
        "draft_params_m": round(model.spec.estimated_params_millions, 2),
        "draft_resident_mb_bf16": round(model.spec.estimated_resident_mb_bf16, 2),
        "draft_output_space": cfg.num_hidden_layers * cfg.num_experts,
        "estimated_working_set_mb": round(ornith_working_set_estimate_mb(cfg), 1),
    }


