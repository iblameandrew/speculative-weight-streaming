"""Run all SWS phase verification gates sequentially."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.phase0_gate import run_gate as gate0
from benchmarks.phase1_gate import run_gate as gate1
from benchmarks.phase2_gate import run_gate as gate2
from benchmarks.phase3_gate import run_gate as gate3
from benchmarks.phase4_gate import run_gate as gate4


def main() -> int:
    gates = [
        ("Phase 0", gate0),
        ("Phase 1", gate1),
        ("Phase 2", gate2),
        ("Phase 3", gate3),
        ("Phase 4", gate4),
    ]
    results = []
    for name, fn in gates:
        print()
        ok = fn()
        results.append((name, ok))
        if not ok:
            print(f"\nStopping: {name} gate failed.")
            break

    print("\n" + "=" * 60)
    print("SUMMARY")
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print("=" * 60)
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())