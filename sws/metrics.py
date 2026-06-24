"""Runtime metrics for SWS benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


@dataclass
class BenchmarkResult:
    phase: str
    peak_rss_mb: float
    peak_cache_mb: float
    tokens_per_sec: float
    miss_rate: float
    accept_rate: float
    max_logit_diff: float
    passed: bool
    notes: str = ""


@dataclass
class MemoryTracker:
    samples_mb: List[float] = field(default_factory=list)

    def sample(self) -> float:
        if psutil is None:
            return 0.0
        rss = psutil.Process().memory_info().rss / (1024 * 1024)
        self.samples_mb.append(rss)
        return rss

    @property
    def peak_mb(self) -> float:
        return max(self.samples_mb) if self.samples_mb else 0.0


def measure_tokens_per_sec(fn, *args, warmup: int = 1, **kwargs) -> tuple[float, float]:
    for _ in range(warmup):
        fn(*args, **kwargs)
    tracker = MemoryTracker()
    tracker.sample()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    tracker.sample()
    return result, (1.0 / elapsed if elapsed > 0 else 0.0)