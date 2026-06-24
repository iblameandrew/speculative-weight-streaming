"""Phase 2 gate: predictor + prefetch; stall rate drops; outputs still exact."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.common import free_memory, make_stressed_fixture, shard_tiny_model
from sws.streamer import SpeculativeWeightStreamer


def _collect_traces(cfg, store, prompt, steps: int = 10):
    streamer = SpeculativeWeightStreamer(
        cfg, store, ram_budget_mb=12, use_predictor=False,
    )
    tokens = prompt.clone()
    for _ in range(steps):
        logits, _ = streamer.forward_step(tokens, history=tokens.view(-1).tolist())
        nxt = torch.argmax(logits[:, -1, :], keepdim=True)
        tokens = torch.cat([tokens, nxt], dim=1)
    return streamer.collect_traces()


def run_gate() -> bool:
    print("=" * 60)
    print("PHASE 2 GATE: Predictor + Async Prefetch")
    print("=" * 60)

    cfg, vanilla = make_stressed_fixture()
    _, _, store = shard_tiny_model(vanilla)
    prompt = torch.tensor([[3, 7, 15, 31]])

    traces = _collect_traces(cfg, store, prompt, steps=10)

    ram_budget_mb = 12
    baseline = SpeculativeWeightStreamer(
        cfg, store, ram_budget_mb=ram_budget_mb, use_predictor=False,
    )
    predictor_streamer = SpeculativeWeightStreamer(
        cfg, store, ram_budget_mb=ram_budget_mb, use_predictor=True, use_approx=False,
    )
    predictor_streamer.predictor.train_offline(traces, epochs=10)

    with torch.no_grad():
        vanilla_tokens = prompt.clone()
        for _ in range(6):
            logits, _ = vanilla(vanilla_tokens)
            nxt = torch.argmax(logits[:, -1, :], keepdim=True)
            vanilla_tokens = torch.cat([vanilla_tokens, nxt], dim=1)

    def run_once(streamer):
        streamer.cache._entries.clear()
        streamer.cache._resident_bytes = 0
        streamer.cache._stats = {"hits": 0, "misses": 0, "evictions": 0, "peak_bytes": 0, "stalls": 0}
        streamer._step_metrics.clear()
        tokens = streamer.generate(prompt, max_new_tokens=6)
        return tokens, streamer.aggregate_metrics

    del vanilla
    free_memory()

    base_tokens, base_metrics = run_once(baseline)
    pred_tokens, pred_metrics = run_once(predictor_streamer)

    exact_match = torch.equal(vanilla_tokens, base_tokens) and torch.equal(base_tokens, pred_tokens)
    stalls_improved = pred_metrics.get("stalls", 1e9) < base_metrics.get("stalls", 1e9)

    print(f"  Baseline miss rate: {base_metrics.get('miss_rate', 0):.4f}")
    print(f"  Predictor miss rate: {pred_metrics.get('miss_rate', 0):.4f}")
    print(f"  Baseline stalls: {base_metrics.get('stalls', 0)}")
    print(f"  Predictor stalls: {pred_metrics.get('stalls', 0)}")
    print(f"  Baseline peak RAM MB: {base_metrics.get('peak_ram_mb', 0):.2f}")
    print(f"  Predictor peak RAM MB: {pred_metrics.get('peak_ram_mb', 0):.2f}")
    print(f"  Token exact match: {exact_match}")
    print(f"  Traces used for offline training: {len(traces)}")

    passed = exact_match and stalls_improved
    print(f"  GATE {'PASSED' if passed else 'FAILED'}")
    return passed


if __name__ == "__main__":
    ok = run_gate()
    sys.exit(0 if ok else 1)