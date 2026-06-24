"""Phase 4 gate: online adaptation + prediction-driven eviction."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.common import free_memory, make_stressed_fixture, shard_tiny_model
from sws.streamer import SpeculativeWeightStreamer


def _run_long(streamer, prompt, steps: int) -> list[float]:
    miss_rates = []
    tokens = prompt.clone()
    for _ in range(steps):
        logits, _ = streamer.forward_step(tokens, history=tokens.view(-1).tolist())
        stats = streamer.cache.stats
        total = stats["hits"] + stats["misses"]
        miss_rates.append(stats["misses"] / max(total, 1))
        nxt = torch.argmax(logits[:, -1, :], keepdim=True)
        tokens = torch.cat([tokens, nxt], dim=1)
    return miss_rates


def run_gate() -> bool:
    print("=" * 60)
    print("PHASE 4 GATE: Online Adaptation + Prediction Eviction")
    print("=" * 60)

    cfg, vanilla = make_stressed_fixture()
    _, _, store = shard_tiny_model(vanilla)
    prompt = torch.tensor([[3, 7, 15, 31]])

    del vanilla
    free_memory()

    phase1 = SpeculativeWeightStreamer(cfg, store, ram_budget_mb=13, use_predictor=False)
    t0 = time.perf_counter()
    _ = phase1.generate(prompt, max_new_tokens=6)
    phase1_tps = 6 / max(time.perf_counter() - t0, 1e-6)

    adaptive = SpeculativeWeightStreamer(
        cfg,
        store,
        ram_budget_mb=13,
        use_predictor=True,
        use_approx=True,
        prediction_eviction=True,
        pin_lower_layers=True,
        confidence_threshold=0.2,
    )

    miss_curve = _run_long(adaptive, prompt, steps=12)
    first_third = sum(miss_curve[:4]) / 4
    last_third = sum(miss_curve[-4:]) / 4
    improved = last_third <= first_third

    t0 = time.perf_counter()
    _ = adaptive.generate(prompt, max_new_tokens=6)
    phase4_tps = 6 / max(time.perf_counter() - t0, 1e-6)

    faster = phase4_tps >= phase1_tps * 0.9

    print(f"  Phase 1 tokens/sec: {phase1_tps:.2f}")
    print(f"  Phase 4 tokens/sec: {phase4_tps:.2f}")
    print(f"  Miss rate early: {first_third:.4f}")
    print(f"  Miss rate late:  {last_third:.4f}")
    print(f"  Online improvement: {improved}")
    print(f"  Peak RAM MB: {adaptive.cache.peak_bytes / (1024 * 1024):.2f}")

    passed = improved and faster
    print(f"  GATE {'PASSED' if passed else 'FAILED'}")
    return passed


if __name__ == "__main__":
    ok = run_gate()
    sys.exit(0 if ok else 1)