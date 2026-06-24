"""MicroDraftModel — intelligent selector and reassembly planner."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sws.sharding import attention_shard_id, expert_shard_id, layer_norm_shard_id, router_shard_id
from sws.synthetic_moe import SyntheticMoEConfig
from sws.types import LayerBlueprint, RealPath, ReassemblyBlueprint, ShardId


class _SelectorMLP(nn.Module):
    """Compact permanently-resident network (~hundreds of KB) for piece selection."""

    def __init__(self, hidden_size: int, num_experts: int, num_layers: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_layers * num_experts),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        pooled = hidden.mean(dim=1)
        logits = self.net(pooled)
        return logits.view(-1, self.num_layers, self.num_experts)


class MicroDraftModel:
    """
    Micro draft model: examines hidden state + history, selects raw weight pieces
    from the giant repository, and emits a reassembly blueprint for the subgraph
    that should execute the next forward pass.
    """

    def __init__(
        self,
        cfg: SyntheticMoEConfig,
        confidence_threshold: float = 0.15,
        device: torch.device | str = "cpu",
    ):
        self.cfg = cfg
        self.tau = confidence_threshold
        self.device = torch.device(device)
        self._selector = _SelectorMLP(cfg.hidden_size, cfg.num_experts, cfg.num_layers).to(self.device)
        self._optimizer = torch.optim.Adam(self._selector.parameters(), lr=1e-3)
        self._history: List[int] = []

    def _build_shard_maps(self) -> Tuple[Dict[Tuple[int, int], ShardId], Dict[int, ShardId]]:
        expert_map = {
            (layer, expert): expert_shard_id(layer, expert)
            for layer in range(self.cfg.num_layers)
            for expert in range(self.cfg.num_experts)
        }
        attn_map = {layer: attention_shard_id(layer) for layer in range(self.cfg.num_layers)}
        return expert_map, attn_map

    def select_and_plan(
        self,
        hidden: torch.Tensor,
        history: Optional[List[int]] = None,
    ) -> ReassemblyBlueprint:
        """Select raw clay pieces and design the executable subgraph blueprint."""
        if history is not None:
            self._history = history[-self.cfg.max_seq_len :]

        with torch.no_grad():
            probs = torch.sigmoid(self._selector(hidden.to(self.device)))
        probs = probs.squeeze(0)

        expert_map, attn_map = self._build_shard_maps()
        layers: List[LayerBlueprint] = []
        high: Set[ShardId] = set()
        low: Set[ShardId] = set()

        for layer_idx in range(self.cfg.num_layers):
            layer_probs = probs[layer_idx]
            topk = torch.topk(layer_probs, self.cfg.num_experts_per_tok + 2)
            selected = set(topk.indices.tolist())

            layers.append(
                LayerBlueprint(
                    layer_idx=layer_idx,
                    expert_probs=layer_probs.cpu(),
                    selected_experts=selected,
                )
            )
            high.add(attn_map[layer_idx])
            high.add(layer_norm_shard_id(layer_idx))
            high.add(router_shard_id(layer_idx))

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
        """Online adaptation: refine selection from (blueprint, realized path) pairs."""
        if hidden is None:
            return 0.0

        target = torch.zeros(
            self.cfg.num_layers,
            self.cfg.num_experts,
            device=self.device,
        )
        for layer_idx, experts in real_path.fired_experts.items():
            for e_idx in experts:
                target[layer_idx, e_idx] = 1.0

        self._selector.train()
        with torch.enable_grad():
            logits = self._selector(hidden.detach().to(self.device)).squeeze(0)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
        self._selector.eval()
        return float(loss.item())

    def train_offline(
        self,
        traces: List[Tuple[torch.Tensor, RealPath]],
        epochs: int = 5,
    ) -> List[float]:
        """Train selector on logged (hidden_state, router top-k) traces."""
        losses: List[float] = []
        self._selector.train()
        for _ in range(epochs):
            epoch_loss = 0.0
            for hidden, real_path in traces:
                target = torch.zeros(
                    self.cfg.num_layers,
                    self.cfg.num_experts,
                    device=self.device,
                )
                for layer_idx, experts in real_path.fired_experts.items():
                    for e_idx in experts:
                        target[layer_idx, e_idx] = 1.0
                logits = self._selector(hidden.to(self.device)).squeeze(0)
                loss = F.binary_cross_entropy_with_logits(logits, target)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                epoch_loss += float(loss.item())
            losses.append(epoch_loss / max(len(traces), 1))
        self._selector.eval()
        return losses

    def state_dict(self) -> dict:
        return self._selector.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self._selector.load_state_dict(state)


# Backward-compatible alias
TinyFootprintPredictor = MicroDraftModel