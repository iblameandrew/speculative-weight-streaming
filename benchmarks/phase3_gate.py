"""Phase 3 gate: approximation path + verifier fallback."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.common import free_memory, make_tiny_fixture, max_abs_diff, shard_tiny_model
from sws.streamer import SpeculativeWeightStreamer


def run_gate() -> bool:
    print("=" * 60)
    print("PHASE 3 GATE: Approximation + Verifier Fallback")
    print("=" * 60)

    cfg, vanilla = make_tiny_fixture()
    _, _, store = shard_tiny_model(vanilla)
    prompt = torch.tensor([[3, 7, 15, 31]])

    baseline = SpeculativeWeightStreamer(cfg, store, ram_budget_mb=32, use_predictor=False)
    tokens = prompt.clone()
    for _ in range(12):
        logits, _ = baseline.forward_step(tokens)
        nxt = torch.argmax(logits[:, -1, :], keepdim=True)
        tokens = torch.cat([tokens, nxt], dim=1)
    traces = baseline.collect_traces()

    with torch.no_grad():
        vanilla_logits, _ = vanilla(prompt)
        vanilla_tokens = prompt.clone()
        for _ in range(8):
            logits, _ = vanilla(vanilla_tokens)
            nxt = torch.argmax(logits[:, -1, :], keepdim=True)
            vanilla_tokens = torch.cat([vanilla_tokens, nxt], dim=1)

    del vanilla
    free_memory()

    approx_streamer = SpeculativeWeightStreamer(
        cfg,
        store,
        ram_budget_mb=4,
        use_predictor=True,
        use_approx=True,
        confidence_threshold=0.5,
    )
    approx_streamer.predictor.train_offline(traces, epochs=6)

    sws_tokens = approx_streamer.generate(prompt, max_new_tokens=8)

    with torch.no_grad():
        sws_logits, _ = approx_streamer.forward_step(prompt)

    logit_diff = max_abs_diff(vanilla_logits, sws_logits)
    token_match = torch.equal(vanilla_tokens, sws_tokens)
    metrics = approx_streamer.aggregate_metrics
    verifier_active = metrics.get("reject_rate", 0) > 0 or metrics.get("accept_rate", 0) > 0

    print(f"  Logit max-abs-diff: {logit_diff:.2e} (tolerance 1e-4)")
    print(f"  Token exact match: {token_match}")
    print(f"  Accept rate: {metrics.get('accept_rate', 0):.4f}")
    print(f"  Reject/stall events: {metrics.get('stalls', 0)}")
    print(f"  Verifier active: {verifier_active}")
    print(f"  Approx path enabled: True (low-confidence shards)")

    passed = logit_diff < 1e-4 and token_match
    print(f"  GATE {'PASSED' if passed else 'FAILED'}")
    return passed


if __name__ == "__main__":
    ok = run_gate()
    sys.exit(0 if ok else 1)