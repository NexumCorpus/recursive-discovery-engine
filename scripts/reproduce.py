"""Re-derive the v11 headline from a fresh clone and diff it against the cited run.

Usage:  python scripts/reproduce.py              # runs the engine (~3 min), then compares
        python scripts/reproduce.py runs/<id>    # compares an existing run

The deterministic backend is reproducible, so every headline number in the cited
run's claims.json must come back identical. The holdout is the exception by
design: it is redrawn from a fresh seed each run, so its margins move. What must
NOT move is its verdict: every in-run winner still loses on the holdout (overfit).
Exit 0 = reproduced; exit 1 = drift (each difference printed).
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CITED = ROOT / "evidence" / "runs" / "20260610_115502_cache_eviction_deterministic"
# run_id names the run; holdout is redrawn per run; template_hash was seed-dependent
# before the fix that ships with this script, so the cited value cannot reproduce.
VARIES = {"run_id", "holdout", "template_hash"}


def fresh_run() -> Path:
    def runs() -> set:
        return {p for p in (ROOT / "runs").glob("*_cache_eviction_*") if p.is_dir()}

    before = runs() if (ROOT / "runs").exists() else set()
    subprocess.run([sys.executable, "-m", "rde", "--backend", "deterministic",
                    "--domain", "cache_eviction", "--quiet"], cwd=ROOT, check=True)
    new = sorted(runs() - before)
    if not new:
        sys.exit("FAIL: the engine wrote no run directory")
    return new[-1]


def main() -> int:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else fresh_run()
    if not run.is_absolute():
        run = ROOT / run
    cited = json.loads((CITED / "claims.json").read_text(encoding="utf-8"))
    fresh = json.loads((run / "claims.json").read_text(encoding="utf-8"))

    bad = [k for k in cited if k not in VARIES and cited[k] != fresh.get(k)]
    for k in bad:
        print(f"DRIFT {k}: cited={cited[k]!r} fresh={fresh.get(k)!r}")

    margins = fresh.get("holdout", {}).get("margins", {})
    for name, m in sorted(margins.items()):
        print(f"holdout {name}: bench {m['bench_margin']:+.4f}  holdout {m['holdout_margin']:+.4f}"
              f"  vs {m['comparator']}  overfit={m['overfit']}")
    if not margins or not all(m["overfit"] for m in margins.values()):
        print("DRIFT holdout: a winner survived the fresh holdout (the cited verdict was OVERFIT)")
        bad.append("holdout")

    same = len(cited) - len(VARIES & set(cited))
    if bad:
        print(f"FAIL: {len(bad)} difference(s) vs {CITED.name}")
        return 1
    print(f"REPRODUCED: {same}/{same} headline fields identical to {CITED.name}; "
          f"fresh holdout (seed {fresh['holdout']['seed']}) marks all "
          f"{len(margins)} winners OVERFIT again")
    return 0


if __name__ == "__main__":
    sys.exit(main())
