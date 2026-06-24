"""Phase 1 gate: LRU cache + on-demand; exact generation; peak RAM <= budget."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.common import free_memory, make_tiny_fixture, max_abs_diff, shard_tiny_model
from sws.streamer import SpeculativeWeightStreamer


def run_gate() -> bool:
    print("=" * 60)
    print("PHASE 1 GATE: PredictiveCache (LRU) + On-Demand")
    print("=" * 60)

    cfg, vanilla = make_tiny_fixture()
    _, _, store = shard_tiny_model(vanilla)

    prompt = torch.tensor([[3, 7, 15, 31]])
    ram_budget_mb = 8

    with torch.no_grad():
        vanilla_tokens = prompt.clone()
        for _ in range(8):
            logits, _ = vanilla(vanilla_tokens)
            nxt = torch.argmax(logits[:, -1, :], keepdim=True)
            vanilla_tokens = torch.cat([vanilla_tokens, nxt], dim=1)

    del vanilla
    free_memory()

    streamer = SpeculativeWeightStreamer(
        cfg,
        store,
        ram_budget_mb=ram_budget_mb,
        use_predictor=False,
        use_approx=False,
    )

    t0 = time.perf_counter()
    sws_tokens = streamer.generate(prompt, max_new_tokens=8)
    elapsed = time.perf_counter() - t0
    tokens_per_sec = 8 / max(elapsed, 1e-6)

    exact_match = torch.equal(vanilla_tokens, sws_tokens)
    within_budget = streamer.cache.peak_bytes <= ram_budget_mb * 1024 * 1024
    metrics = streamer.aggregate_metrics

    print(f"  Vanilla tokens: {vanilla_tokens.tolist()}")
    print(f"  SWS tokens:     {sws_tokens.tolist()}")
    print(f"  Exact match: {exact_match}")
    print(f"  Peak cache: {metrics.get('peak_ram_mb', 0):.2f} MB (budget {ram_budget_mb} MB)")
    print(f"  Miss rate: {metrics.get('miss_rate', 0):.4f}")
    print(f"  Tokens/sec: {tokens_per_sec:.2f}")
    print(f"  Evictions: {metrics.get('evictions', 0)}")

    passed = exact_match and within_budget
    print(f"  GATE {'PASSED' if passed else 'FAILED'}")
    return passed


if __name__ == "__main__":
    ok = run_gate()
    sys.exit(0 if ok else 1)