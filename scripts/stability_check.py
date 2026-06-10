"""v10 P1 acceptance: N consecutive deterministic cache_eviction runs (same seed,
same config) must produce IDENTICAL op_utility maps (both tiers).

Usage: python scripts/stability_check.py [N]   (default 5)
Exit 0 = stable; 1 = any divergence (diff printed).
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    maps = []
    for i in range(n):
        before = set(p.name for p in (ROOT / "runs").iterdir())
        r = subprocess.run(
            [sys.executable, "-m", "rde", "--backend", "deterministic",
             "--domain", "cache_eviction", "--quiet"],
            cwd=ROOT, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            print(f"run {i + 1}: engine FAILED rc={r.returncode}\n{r.stderr[-500:]}")
            return 1
        new = [p for p in (ROOT / "runs").iterdir()
               if p.name not in before and "cache_eviction" in p.name]
        claims = json.loads((new[0] / "claims.json").read_text(encoding="utf-8"))
        snap = {"static": claims["op_utility_static"],
                "adaptive": claims["op_utility_adaptive"],
                "incomplete": claims["incomplete"],
                "run": claims["run_id"]}
        maps.append(snap)
        print(f"run {i + 1}/{n}: {claims['run_id']}  "
              f"beats(adaptive)={sum(1 for v in snap['adaptive'].values() if v == 'beats_human')}")
    ref = {k: maps[0][k] for k in ("static", "adaptive")}
    stable = True
    for i, m in enumerate(maps[1:], start=2):
        for tier in ("static", "adaptive"):
            if m[tier] != ref[tier]:
                stable = False
                diff = {op: (ref[tier].get(op), m[tier].get(op))
                        for op in set(ref[tier]) | set(m[tier])
                        if ref[tier].get(op) != m[tier].get(op)}
                print(f"DIVERGENCE run 1 vs run {i} [{tier}]: {diff}")
    print("STABLE: identical op_utility maps across "
          f"{n} consecutive runs" if stable else "UNSTABLE")
    return 0 if stable else 1


if __name__ == "__main__":
    sys.exit(main())
