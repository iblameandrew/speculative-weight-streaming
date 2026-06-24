"""Speculative Weight Streamer — select, reassemble, verify, adapt loop."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import torch

from sws.assembler import DynamicAssembler
from sws.micro_draft import MicroDraftModel
from sws.store import NVMeWeightStore
from sws.synthetic_moe import SyntheticMoEConfig
from sws.types import RealPath, ReassemblyBlueprint, ShardId, StepMetrics
from sws.verifier import Verifier


class SpeculativeWeightStreamer:
    """
    Dynamic model-recomposition system. The micro draft model selects raw clay
    pieces from NVMe; the assembler materializes an executable subgraph; the
    verifier guarantees equivalence to the reference giant.
    """

    def __init__(
        self,
        cfg: SyntheticMoEConfig,
        store: NVMeWeightStore,
        ram_budget_mb: int = 32_000,
        confidence_threshold: float = 0.15,
        use_predictor: bool = True,
        use_micro_draft: Optional[bool] = None,
        use_approx: bool = False,
        prediction_eviction: bool = False,
        pin_lower_layers: bool = False,
        device: torch.device | str = "cpu",
    ):
        self.cfg = cfg
        self.store = store
        self.device = torch.device(device)
        self.assembler = DynamicAssembler(cfg, ram_budget_mb, pin_lower_layers=pin_lower_layers)
        self.micro_draft = MicroDraftModel(cfg, confidence_threshold, device=device)
        self.verifier = Verifier()
        self.use_micro_draft = use_micro_draft if use_micro_draft is not None else use_predictor
        self.use_approx = use_approx
        self.prediction_eviction = prediction_eviction
        self._history: List[int] = []
        self._step_metrics: List[StepMetrics] = []
        self._trace_buffer: List[Tuple[torch.Tensor, RealPath]] = []

        if pin_lower_layers:
            self.assembler.pin_lower_layer_shards(layer_threshold=1)

    # Backward-compatible aliases
    @property
    def cache(self):
        return self.assembler.cache

    @property
    def predictor(self):
        return self.micro_draft

    @property
    def use_predictor(self) -> bool:
        return self.use_micro_draft

    @use_predictor.setter
    def use_predictor(self, value: bool) -> None:
        self.use_micro_draft = value

    def forward_step(
        self,
        input_ids: torch.Tensor,
        history: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, StepMetrics]:
        metrics = StepMetrics()
        t0 = time.perf_counter()
        hidden = self._embed_hidden(input_ids)

        if history is not None:
            self._history = history
        elif input_ids.numel() > 0:
            self._history.extend(input_ids.view(-1).tolist())

        if self.use_micro_draft:
            blueprint = self.micro_draft.select_and_plan(hidden, self._history)
            self.assembler.assemble_from_blueprint(blueprint, self.store, use_approx=self.use_approx)
        else:
            blueprint = self.assembler.assemble_essential(self.store)

        stall_start = time.perf_counter()
        out, real_path, rejects = self.assembler.execute_with_resilience(
            input_ids, self.store, blueprint,
        )
        metrics.verifier_rejects += rejects
        metrics.reassembly_count = 1
        metrics.stalls_ms += (time.perf_counter() - stall_start) * 1000

        miss = self.verifier.detect_miss(real_path, self.cache, self.assembler.runner, blueprint)
        if not self.verifier.accepts(miss):
            metrics.verifier_rejects += 1
            metrics.reassembly_count += 1
            self.assembler.fallback_reassemble(miss, blueprint, self.store)
            out, real_path = self.assembler.execute_assembled(input_ids)
        else:
            metrics.verifier_accepts += 1

        self._trace_buffer.append((hidden.detach().cpu(), real_path))
        if self.use_micro_draft:
            self.micro_draft.adapt(blueprint, real_path, hidden=hidden)

        self.assembler.assert_within_budget()
        metrics.prefetch_hits = self.cache.stats["hits"]
        metrics.prefetch_misses = self.cache.stats["misses"]
        self._step_metrics.append(metrics)
        _ = t0
        return out, metrics

    def _embed_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed_sid = "embed_weight"
        if self.cache.get(embed_sid) is None:
            self.cache.put(self.store.load_sync(embed_sid))
        payload = self.cache.get(embed_sid)
        assert payload is not None
        return torch.nn.functional.embedding(input_ids, payload.tensors["embed.weight"])

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 16,
    ) -> torch.Tensor:
        tokens = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits, _ = self.forward_step(tokens, history=tokens.view(-1).tolist())
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens

    @property
    def aggregate_metrics(self) -> dict:
        total = len(self._step_metrics)
        if total == 0:
            return {}
        accepts = sum(m.verifier_accepts for m in self._step_metrics)
        rejects = sum(m.verifier_rejects for m in self._step_metrics)
        return {
            "steps": total,
            "accept_rate": accepts / total,
            "reject_rate": rejects / total,
            "miss_rate": self.cache.stats["misses"] / max(self.cache.stats["hits"] + self.cache.stats["misses"], 1),
            "evictions": self.cache.stats["evictions"],
            "peak_ram_mb": self.cache.peak_bytes / (1024 * 1024),
            "stalls": self.cache.stats["stalls"],
            "reassembly_count": sum(m.reassembly_count for m in self._step_metrics),
        }

    def collect_traces(self) -> List[Tuple[torch.Tensor, RealPath]]:
        return list(self._trace_buffer)