"""Hugging Face transformers integration for real MoE models."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Set

import torch
import torch.nn as nn

from sws.cache import PredictiveCache
from sws.sharding import expert_shard_id, shard_model_state_dict
from sws.store import NVMeWeightStore

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None  # type: ignore[misc, assignment]
    AutoTokenizer = None  # type: ignore[misc, assignment]


class LazyLinear(nn.Module):
    """Proxy linear layer that pulls weights from PredictiveCache on forward."""

    def __init__(self, shard_id: str, weight_key: str, cache: PredictiveCache, in_features: int, out_features: int):
        super().__init__()
        self.shard_id = shard_id
        self.weight_key = weight_key
        self.cache = cache
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        payload = self.cache.get(self.shard_id)
        if payload is None:
            raise KeyError(f"Cache miss: {self.shard_id}")
        weight = payload.tensors[self.weight_key]
        return torch.nn.functional.linear(x, weight)


class RouterHook:
    """Captures real expert indices from MoE router top-k."""

    def __init__(self):
        self.real_path: Dict[int, Set[int]] = {}
        self._layer_idx = 0

    def __call__(self, module, inputs, output) -> None:
        if isinstance(output, tuple) and len(output) >= 2:
            router_logits = output[0] if output[0].dim() >= 2 else inputs[0]
        else:
            router_logits = inputs[0] if isinstance(inputs, tuple) else output
        if not isinstance(router_logits, torch.Tensor):
            return
        topk = torch.topk(router_logits, k=min(2, router_logits.shape[-1]), dim=-1)
        self.real_path[self._layer_idx] = set(topk.indices.unique().tolist())
        self._layer_idx += 1


def shard_hf_model(model_id: str, output_dir: Path, dtype: torch.dtype = torch.float16) -> Path:
    """Download (if needed), shard per-expert weights, never keep full model in RAM after write."""
    if AutoModelForCausalLM is None:
        raise ImportError("transformers is required for HF integration")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    remapped: Dict[str, torch.Tensor] = {}
    expert_re = re.compile(r"(.*)\.experts\.(\d+)\.(.*)")

    for key, tensor in state.items():
        m = expert_re.search(key)
        if m:
            prefix, expert_idx, suffix = m.groups()
            layer_m = re.search(r"\.(\d+)\.", prefix)
            layer_idx = int(layer_m.group(1)) if layer_m else 0
            new_key = f"layers.{layer_idx}.moe.experts.{expert_idx}.{suffix}"
            remapped[new_key] = tensor
        elif "self_attn" in key or "attention" in key:
            layer_m = re.search(r"\.(\d+)\.", key)
            layer_idx = int(layer_m.group(1)) if layer_m else 0
            suffix = key.split(f".{layer_idx}.")[-1].replace("self_attn.", "").replace("attention.", "")
            remapped[f"layers.{layer_idx}.attn.{suffix}"] = tensor
        elif "gate" in key or "router" in key:
            layer_m = re.search(r"\.(\d+)\.", key)
            layer_idx = int(layer_m.group(1)) if layer_m else 0
            suffix = key.split(f".{layer_idx}.")[-1].split(".")[-1]
            remapped[f"layers.{layer_idx}.moe.router.{suffix}"] = tensor
        elif key in ("model.embed_tokens.weight", "transformer.wte.weight"):
            remapped["embed.weight"] = tensor
        elif key.endswith("norm.weight") and "layers" not in key:
            remapped["final_norm.weight"] = tensor
        elif key == "lm_head.weight" or key.endswith("embed_out.weight"):
            remapped["lm_head.weight"] = tensor

    shard_model_state_dict(remapped, output_dir)
    return output_dir


def build_lazy_store(shard_dir: Path, ram_budget_mb: int = 32_000) -> tuple[NVMeWeightStore, PredictiveCache]:
    store = NVMeWeightStore(shard_dir)
    cache = PredictiveCache(ram_budget_mb)
    return store, cache


def recommended_test_models() -> list[str]:
    """Smallest genuine MoE models with per-expert weight exposure."""
    return [
        "trl-internal-testing/tiny-Mixtral-8x7B-Instruct-v0.1",
        "Qwen/Qwen1.5-MoE-A2.7B",
        "mistralai/Mixtral-8x7B-v0.1",
    ]