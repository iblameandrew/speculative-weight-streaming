"""Shared benchmark utilities for SWS phase gates."""

from __future__ import annotations

import gc
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sws.metrics import MemoryTracker
from sws.sharding import shard_model_state_dict
from sws.store import NVMeWeightStore
from sws.synthetic_moe import SyntheticMoEConfig, init_deterministic_model


def make_stressed_fixture(seed: int = 0) -> tuple:
    """Larger MoE for cache-pressure benchmarks (many expert shards)."""
    cfg = SyntheticMoEConfig(
        vocab_size=256,
        hidden_size=96,
        intermediate_size=192,
        num_layers=5,
        num_heads=4,
        num_experts=12,
        num_experts_per_tok=2,
    )
    model = init_deterministic_model(cfg, seed=seed)
    model.eval()
    return cfg, model


def make_tiny_fixture(seed: int = 0) -> tuple:
    cfg = SyntheticMoEConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_layers=3,
        num_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
    )
    model = init_deterministic_model(cfg, seed=seed)
    model.eval()
    return cfg, model


def shard_tiny_model(model, tmp_dir: Path | None = None) -> tuple[SyntheticMoEConfig, Path, NVMeWeightStore]:
    cfg = model.cfg
    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="sws_shards_"))
    shard_model_state_dict(model.state_dict_named(), tmp_dir)
    store = NVMeWeightStore(tmp_dir)
    return cfg, tmp_dir, store


def rss_mb() -> float:
    tracker = MemoryTracker()
    return tracker.sample()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()