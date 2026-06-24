"""Shard MoE model weights into per-expert / per-layer safetensors files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from safetensors.torch import save_file

from sws.types import ShardId

EXPERT_PATTERN = re.compile(
    r"^layers\.(\d+)\.moe\.experts\.(\d+)\.(.+)$"
)
ATTN_PATTERN = re.compile(r"^layers\.(\d+)\.attn\.(.+)$")
ROUTER_PATTERN = re.compile(r"^layers\.(\d+)\.moe\.router\.(.+)$")
NORM_PATTERN = re.compile(r"^layers\.(\d+)\.norm\.(.+)$")
GLOBAL_SHARDS = ("embed.weight", "final_norm.weight", "lm_head.weight")


def expert_shard_id(layer: int, expert: int) -> ShardId:
    return f"layer_{layer}/expert_{expert}"


def attention_shard_id(layer: int) -> ShardId:
    return f"layer_{layer}/attention"


def router_shard_id(layer: int) -> ShardId:
    return f"layer_{layer}/router"


def layer_norm_shard_id(layer: int) -> ShardId:
    return f"layer_{layer}/norm"


def global_shard_id(name: str) -> ShardId:
    return name.replace(".", "_")


def tensor_byte_size(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _accumulate(
    buckets: Dict[ShardId, Dict[str, torch.Tensor]],
    shard_id: ShardId,
    key: str,
    tensor: torch.Tensor,
) -> None:
    buckets.setdefault(shard_id, {})[key] = tensor.contiguous()


def partition_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[ShardId, Dict[str, torch.Tensor]]:
    """Split a flat state dict into SWS shard buckets."""
    buckets: Dict[ShardId, Dict[str, torch.Tensor]] = {}

    for key, tensor in state_dict.items():
        if key in GLOBAL_SHARDS:
            _accumulate(buckets, global_shard_id(key), key, tensor)
            continue

        m = EXPERT_PATTERN.match(key)
        if m:
            layer, expert, suffix = m.groups()
            _accumulate(
                buckets,
                expert_shard_id(int(layer), int(expert)),
                suffix,
                tensor,
            )
            continue

        m = ATTN_PATTERN.match(key)
        if m:
            layer, suffix = m.groups()
            _accumulate(buckets, attention_shard_id(int(layer)), suffix, tensor)
            continue

        m = ROUTER_PATTERN.match(key)
        if m:
            layer, suffix = m.groups()
            _accumulate(buckets, router_shard_id(int(layer)), suffix, tensor)
            continue

        m = NORM_PATTERN.match(key)
        if m:
            layer, suffix = m.groups()
            _accumulate(buckets, layer_norm_shard_id(int(layer)), suffix, tensor)
            continue

        raise KeyError(f"Unrecognized state_dict key for sharding: {key}")

    return buckets


def quantize_int8(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Cheap stand-in quantization for approximation shards."""
    t = tensor.float()
    scale = t.abs().max().clamp(min=1e-8) / 127.0
    q = torch.round(t / scale).to(torch.int8)
    return {"weight": q, "scale": scale.to(torch.float32)}


def dequantize_int8(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        if key.endswith("_q"):
            base = key[:-2]
            scale_key = f"{base}_scale"
            if scale_key in tensors:
                out[base] = value.float() * tensors[scale_key]
        elif key.endswith("_scale"):
            continue
        else:
            out[key] = value
    return out


def build_approx_shard(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    approx: Dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        if tensor.ndim >= 2 and "weight" in key:
            q = quantize_int8(tensor)
            approx[f"{key}_q"] = q["weight"]
            approx[f"{key}_scale"] = q["scale"]
        else:
            approx[key] = tensor.clone()
    return approx


def write_shards(
    buckets: Dict[ShardId, Dict[str, torch.Tensor]],
    output_dir: Path,
    write_approx: bool = True,
) -> Dict[ShardId, int]:
    """Persist shard buckets to safetensors; returns byte sizes per shard."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes: Dict[ShardId, int] = {}

    for shard_id, tensors in buckets.items():
        rel_path = shard_id.replace("/", "__") + ".safetensors"
        path = output_dir / rel_path
        contiguous = {k: v.contiguous() for k, v in tensors.items()}
        save_file(contiguous, path)
        sizes[shard_id] = sum(tensor_byte_size(t) for t in contiguous.values())

        if write_approx:
            approx = build_approx_shard(contiguous)
            approx_path = output_dir / (rel_path.replace(".safetensors", "_approx.safetensors"))
            save_file({k: v.contiguous() for k, v in approx.items()}, approx_path)

    manifest = {
        "shards": [
            {"id": sid, "file": sid.replace("/", "__") + ".safetensors", "bytes": sz}
            for sid, sz in sizes.items()
        ]
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return sizes


def shard_model_state_dict(
    state_dict: Dict[str, torch.Tensor],
    output_dir: Path,
    write_approx: bool = True,
) -> Dict[ShardId, int]:
    buckets = partition_state_dict(state_dict)
    return write_shards(buckets, output_dir, write_approx=write_approx)


def iter_moe_expert_shards(num_layers: int, num_experts: int) -> Iterable[Tuple[int, int, ShardId]]:
    for layer in range(num_layers):
        for expert in range(num_experts):
            yield layer, expert, expert_shard_id(layer, expert)