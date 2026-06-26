"""Tests for Ornith-1.0-397B micro draft model architecture."""

from __future__ import annotations

import torch

from sws.ornith_config import ORNITH_MODEL_ID, OrnithMoEConfig
from sws.ornith_draft import OrnithMicroDraftModel, propose_architecture_summary


def test_ornith_config_sparsity():
    cfg = OrnithMoEConfig()
    assert cfg.num_experts == 512
    assert cfg.num_experts_per_tok == 10
    assert cfg.num_hidden_layers == 60
    assert cfg.active_expert_ratio < 0.02


def test_draft_architecture_budget():
    summary = propose_architecture_summary()
    assert summary["target_giant"] == ORNITH_MODEL_ID
    assert summary["draft_params_m"] < 500
    assert summary["draft_resident_mb_bf16"] < 1024
    assert summary["draft_output_space"] == 60 * 512


def test_ornith_draft_blueprint_shape():
    cfg = OrnithMoEConfig()
    draft = OrnithMicroDraftModel(cfg, device="cpu")
    hidden = torch.randn(1, 8, cfg.hidden_size)
    history = [100, 200, 300, 248044]
    bp = draft.select_and_plan(hidden, history)
    assert len(bp.layers) == 60
    assert "embed_weight" in bp.high_priority_pieces
    assert any("shared_expert" in sid for sid in bp.high_priority_pieces)
    assert any("linear_attention" in sid or "full_attention" in sid for sid in bp.high_priority_pieces)