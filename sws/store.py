"""NVMe-backed weight store with lazy safetensors mmap and async fetch."""

from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import torch
from safetensors import safe_open

from sws.sharding import dequantize_int8
from sws.types import ShardId, ShardPayload


class NVMeWeightStore:
    """Disk-resident full model; never materializes all shards at once."""

    def __init__(
        self,
        shard_dir: Path,
        max_workers: int = 4,
        device: torch.device | str = "cpu",
    ):
        self.shard_dir = Path(shard_dir)
        self.device = torch.device(device)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sws-fetch")
        self._lock = threading.Lock()
        self._index: Dict[ShardId, Path] = {}
        self._approx_index: Dict[ShardId, Path] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        manifest_path = self.shard_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for entry in manifest["shards"]:
                sid = entry["id"]
                self._index[sid] = self.shard_dir / entry["file"]
        else:
            for path in self.shard_dir.glob("*.safetensors"):
                if path.name.endswith("_approx.safetensors"):
                    sid = path.stem.replace("__", "/").replace("_approx", "")
                    self._approx_index[sid] = path
                else:
                    sid = path.stem.replace("__", "/")
                    self._index[sid] = path

        for path in self.shard_dir.glob("*_approx.safetensors"):
            sid = path.stem.replace("__", "/").replace("_approx", "")
            self._approx_index[sid] = path

    def list_shards(self) -> list[ShardId]:
        return sorted(self._index.keys())

    def shard_path(self, shard_id: ShardId) -> Path:
        if shard_id in self._index:
            return self._index[shard_id]
        rel = shard_id.replace("/", "__") + ".safetensors"
        path = self.shard_dir / rel
        if path.exists():
            self._index[shard_id] = path
            return path
        raise FileNotFoundError(f"Shard not found: {shard_id}")

    def _read_shard_file(self, path: Path, is_approx: bool) -> Dict[str, torch.Tensor]:
        tensors: Dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt", device=str(self.device)) as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)
        if is_approx:
            tensors = dequantize_int8(tensors)
        return tensors

    def _payload_from_file(self, shard_id: ShardId, path: Path, is_approx: bool) -> ShardPayload:
        tensors = self._read_shard_file(path, is_approx=is_approx)
        byte_size = sum(t.numel() * t.element_size() for t in tensors.values())
        return ShardPayload(shard_id=shard_id, tensors=tensors, byte_size=byte_size, is_approx=is_approx)

    def load_sync(self, shard_id: ShardId) -> ShardPayload:
        """Blocking exact shard load (used by verifier fallback)."""
        path = self.shard_path(shard_id)
        return self._payload_from_file(shard_id, path, is_approx=False)

    def fetch(self, shard_id: ShardId) -> Future[ShardPayload]:
        """Async exact shard load; overlaps with compute via thread pool."""
        path = self.shard_path(shard_id)

        def _work() -> ShardPayload:
            return self._payload_from_file(shard_id, path, is_approx=False)

        return self._executor.submit(_work)

    def extract_pieces(
        self,
        selection: Iterable[ShardId],
        exact: bool = True,
    ) -> List[ShardPayload]:
        """Return selected raw weight pieces for dynamic reassembly."""
        payloads: List[ShardPayload] = []
        for shard_id in selection:
            if exact:
                payloads.append(self.load_sync(shard_id))
            else:
                payloads.append(self.reconstruct_approx(shard_id))
        return payloads

    def reconstruct_approx(self, shard_id: ShardId) -> ShardPayload:
        """Cheap draft weight stand-in (int8-quantized offline)."""
        if shard_id in self._approx_index:
            path = self._approx_index[shard_id]
        else:
            path = self.shard_dir / (shard_id.replace("/", "__") + "_approx.safetensors")
        if not path.exists():
            return self.load_sync(shard_id)
        return self._payload_from_file(shard_id, path, is_approx=True)

    def mmap_keys(self, shard_id: ShardId) -> list[str]:
        """Expose lazy mmap keys without materializing tensors (for RSS checks)."""
        path = self.shard_path(shard_id)
        with safe_open(path, framework="pt", device="cpu") as handle:
            return list(handle.keys())

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass