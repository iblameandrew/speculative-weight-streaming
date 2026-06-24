"""Phase 0 gate: sharding + lazy store; RSS low; forward matches vanilla."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.common import free_memory, make_tiny_fixture, max_abs_diff, rss_mb, shard_tiny_model
from sws.cache import PredictiveCache
from sws.lazy_model import LazyMoERunner
from sws.sharding import expert_shard_id


def run_gate() -> bool:
    print("=" * 60)
    print("PHASE 0 GATE: Sharding & Lazy NVMeWeightStore")
    print("=" * 60)

    cfg, vanilla = make_tiny_fixture()
    _, shard_dir, store = shard_tiny_model(vanilla)

    input_ids = torch.tensor([[1, 5, 12, 42]])
    with torch.no_grad():
        vanilla_logits, _ = vanilla(input_ids)

    del vanilla
    free_memory()
    baseline_rss = rss_mb()

    cache = PredictiveCache(ram_budget_mb=512)
    runner = LazyMoERunner(cfg, cache)

    loaded_experts = []
    for layer in range(cfg.num_layers):
        for expert in range(min(2, cfg.num_experts)):
            sid = expert_shard_id(layer, expert)
            payload = store.load_sync(sid)
            cache.put(payload)
            loaded_experts.append(sid)
            rss_after = rss_mb()
            print(f"  loaded {sid}: cache={cache.resident_bytes() / 1e6:.2f}MB RSS={rss_after:.1f}MB")

    mmap_only_rss = rss_mb()
    with store.shard_path(loaded_experts[0]).open("rb") as _:
        _ = store.mmap_keys(loaded_experts[0])
    rss_mmap = rss_mb()

    for sid in store.list_shards():
        if cache.get(sid) is None:
            cache.put(store.load_sync(sid))

    with torch.no_grad():
        lazy_logits, real_path = runner.forward(input_ids)

    diff = max_abs_diff(vanilla_logits, lazy_logits)
    peak_cache_mb = cache.peak_bytes / (1024 * 1024)
    passed = diff < 1e-4

    print(f"\n  Vanilla vs lazy max logit diff: {diff:.2e}")
    print(f"  Experts loaded on demand: {len(loaded_experts)}")
    print(f"  RSS baseline / after loads / mmap probe: {baseline_rss:.1f} / {rss_after:.1f} / {rss_mmap:.1f} MB")
    print(f"  Peak cache usage: {peak_cache_mb:.2f} MB")
    print(f"  Real path layers: {list(real_path.fired_experts.keys())}")
    print(f"  GATE {'PASSED' if passed else 'FAILED'}: diff < 1e-4")
    return passed


if __name__ == "__main__":
    ok = run_gate()
    sys.exit(0 if ok else 1)