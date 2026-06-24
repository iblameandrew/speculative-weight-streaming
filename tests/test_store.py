"""Tests for NVMeWeightStore and sharding."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from sws.lazy_model import LazyMoERunner
from sws.sharding import expert_shard_id, shard_model_state_dict
from sws.store import NVMeWeightStore
from sws.synthetic_moe import init_deterministic_model


def test_shard_and_lazy_forward_matches_vanilla():
    model = init_deterministic_model()
    model.eval()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shard_model_state_dict(model.state_dict_named(), tmp_path)
        store = NVMeWeightStore(tmp_path)

        from sws.cache import PredictiveCache

        cache = PredictiveCache(ram_budget_mb=64)
        for sid in store.list_shards():
            cache.put(store.load_sync(sid))

        input_ids = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            ref_logits, _ = model(input_ids)
            lazy_logits, _ = LazyMoERunner(model.cfg, cache).forward(input_ids)

        diff = float((ref_logits - lazy_logits).abs().max())
        assert diff < 1e-4


def test_fetch_single_expert_low_footprint():
    model = init_deterministic_model()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shard_model_state_dict(model.state_dict_named(), tmp_path)
        store = NVMeWeightStore(tmp_path)
        payload = store.load_sync(expert_shard_id(0, 0))
        assert "gate_proj.weight" in payload.tensors
        assert payload.byte_size > 0