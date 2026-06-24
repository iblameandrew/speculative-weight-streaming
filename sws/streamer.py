"""Speculative Weight Streamer — draft, materialize, verify, adapt loop."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

# Optional already imported

import torch

from sws.cache import PredictiveCache
from sws.lazy_model import LazyMoERunner
from sws.predictor import TinyFootprintPredictor
from sws.store import NVMeWeightStore
from sws.synthetic_moe import SyntheticMoEConfig
from sws.types import Forecast, RealPath, ShardId, StepMetrics


class SpeculativeWeightStreamer:
    def __init__(
        self,
        cfg: SyntheticMoEConfig,
        store: NVMeWeightStore,
        ram_budget_mb: int = 32_000,
        confidence_threshold: float = 0.15,
        use_predictor: bool = True,
        use_approx: bool = False,
        prediction_eviction: bool = False,
        pin_lower_layers: bool = False,
        device: torch.device | str = "cpu",
    ):
        self.cfg = cfg
        self.store = store
        self.device = torch.device(device)
        self.cache = PredictiveCache(ram_budget_mb, pin_lower_layers=pin_lower_layers)
        self.predictor = TinyFootprintPredictor(cfg, confidence_threshold, device=device)
        self.runner = LazyMoERunner(cfg, self.cache)
        self.use_predictor = use_predictor
        self.use_approx = use_approx
        self.prediction_eviction = prediction_eviction
        self._history: List[int] = []
        self._step_metrics: List[StepMetrics] = []
        self._trace_buffer: List[Tuple[torch.Tensor, RealPath]] = []

        if pin_lower_layers:
            self.cache.pin_lower_layer_shards(layer_threshold=1)

    def _materialize_baseline(self, input_ids: torch.Tensor) -> Forecast:
        """Phase 1: resident infrastructure shards; experts loaded on demand."""
        from sws.sharding import attention_shard_id, layer_norm_shard_id, router_shard_id

        essential: set[ShardId] = {"embed_weight", "final_norm_weight", "lm_head_weight"}
        for layer in range(self.cfg.num_layers):
            essential.add(layer_norm_shard_id(layer))
            essential.add(attention_shard_id(layer))
            essential.add(router_shard_id(layer))
        forecast = Forecast(layers=[], high_confidence=essential, low_confidence=set())
        for shard_id in essential:
            if self.cache.get(shard_id) is None:
                self.cache.load_exact([self.store.load_sync(shard_id)])
        return forecast

    def _materialize_predicted(self, forecast: Forecast) -> None:
        futures = []
        for shard in forecast.high_confidence:
            if self.cache.get(shard) is None:
                fut = self.store.fetch(shard)
                self.cache.prefetch_shard(fut, shard, priority=forecast.priority(shard))
                futures.append((shard, fut))
        for shard in forecast.low_confidence:
            if self.cache.get(shard) is None:
                if self.use_approx:
                    self.cache.put(
                        self.store.reconstruct_approx(shard),
                        priority=forecast.priority(shard),
                        is_exact=False,
                    )
                else:
                    fut = self.store.fetch(shard)
                    self.cache.prefetch_shard(fut, shard, priority=forecast.priority(shard))
                    futures.append((shard, fut))

        for shard, fut in futures:
            if self.cache.get(shard) is None:
                try:
                    self.cache.put(fut.result(), priority=forecast.priority(shard), is_exact=True)
                except Exception:
                    self.cache.load_exact([self.store.load_sync(shard)], forecast=forecast)

    def _ensure_resident(self, shard_ids: set[ShardId], forecast: Optional[Forecast] = None) -> List[ShardId]:
        miss = [sid for sid in shard_ids if self.cache.get(sid) is None]
        if miss:
            payloads = []
            for sid in miss:
                if forecast and sid in forecast.low_confidence and self.use_approx:
                    payloads.append(self.store.reconstruct_approx(sid))
                else:
                    payloads.append(self.store.load_sync(sid))
            self.cache.load_exact(payloads, forecast=forecast)
        return miss

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

        if self.use_predictor:
            forecast = self.predictor(hidden, self._history)
            self._materialize_predicted(forecast)
        else:
            forecast = self._materialize_baseline(input_ids)

        out: Optional[torch.Tensor] = None
        real_path: Optional[RealPath] = None
        stall_start = time.perf_counter()
        pinned_this_step: list[ShardId] = []

        def _is_infrastructure(sid: ShardId) -> bool:
            return (
                "attention" in sid
                or "norm" in sid
                or "router" in sid
                or sid in {"embed_weight", "final_norm_weight", "lm_head_weight"}
            )

        for shard in forecast.high_confidence:
            if _is_infrastructure(shard) and shard in self.cache.resident_set():
                self.cache.pin_shard(shard, pinned=True)
                pinned_this_step.append(shard)
        try:
            for _ in range(32):
                self.runner.last_misses = set()
                try:
                    out, real_path = self.runner.forward(input_ids)
                    break
                except KeyError:
                    miss = set(self.runner.last_misses) - self.cache.resident_set()
                    if not miss:
                        raise
                    metrics.verifier_rejects += 1
                    self._ensure_resident(miss, forecast=forecast)
        finally:
            for sid in pinned_this_step:
                if sid in self.cache.resident_set():
                    self.cache.pin_shard(sid, pinned=False)
            self.cache.trim_to_budget(forecast)
        metrics.stalls_ms += (time.perf_counter() - stall_start) * 1000

        if out is None or real_path is None:
            raise RuntimeError("Verifier fallback failed to obtain a valid forward pass")

        needed = self.runner.required_shards_for_path(real_path)
        resident = self.cache.resident_set()
        post_miss = set(needed) - resident
        for sid in needed:
            entry = self.cache._entries.get(sid)
            if entry is not None and not entry.is_exact:
                post_miss.add(sid)
        if post_miss:
            metrics.verifier_rejects += 1
            self._ensure_resident(post_miss, forecast=forecast)
            out, real_path = self.runner.forward(input_ids)
        else:
            metrics.verifier_accepts += 1

        if real_path is not None:
            self._trace_buffer.append((hidden.detach().cpu(), real_path))
        if self.use_predictor and real_path is not None:
            loss = self.predictor.update(forecast, real_path, hidden=hidden)
            _ = loss

        self.cache.assert_within_budget()
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
        }

    def collect_traces(self) -> List[Tuple[torch.Tensor, RealPath]]:
        return list(self._trace_buffer)