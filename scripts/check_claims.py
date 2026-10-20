"""v10 P5: diff the README's verified-result numbers against a named run's
claims.json -- the mechanism that prevents declared-vs-actual drift between the
docs and what the code produces.

Usage:  python scripts/check_claims.py evidence/runs/<run_id>

Checks:
  1. the README cites this exact run id (the Evidence section must name its run);
  2. every headline number the README quotes for that run appears in the README
     as claims.json states it:
       - adaptive-tier beats/matches/below counts, n_graded and selection factor
       - robust_beats count out of n_graded
       - the recommended artifact's adaptive grade and margin
       - the benchmark and holdout margins of each in-run winner, and that
         every one of them is marked OVERFIT.
Exit 0 = consistent; exit 1 = drift (each mismatch printed).

This checks the README against the run the README cites, i.e. the committed
evidence/runs/20260610_115502_cache_eviction_deterministic. It is not a check of
your own fresh run: the README does not cite that run's id, so it fails by
design. To check a fresh run, use scripts/reproduce.py.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def signed(x: float, nd: int) -> list:
    """The README's spelling of a signed margin: '+0.035', or '−0.171' with a
    true minus sign (an ASCII hyphen is accepted too)."""
    text = f"{abs(x):.{nd}f}"
    return [f"+{text}"] if x >= 0 else [f"−{text}", f"-{text}"]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    run_dir = Path(sys.argv[1])
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    claims_p = run_dir / "claims.json"
    if not claims_p.exists():
        print(f"FAIL: no claims.json in {run_dir}")
        return 1
    claims = json.loads(claims_p.read_text(encoding="utf-8"))
    # Collapse whitespace so a number is found even where the README wraps a line.
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())

    failures = []
    run_id = claims["run_id"]
    if run_id not in readme:
        failures.append(f"README does not cite run id {run_id}")

    ua = claims.get("utility_adaptive") or {}
    if ua:
        expect = (f"{ua.get('beats_human', 0)} beats, "
                  f"{ua.get('matches_human', 0)} matches, "
                  f"{ua.get('below_human', 0)} below")
        if expect not in readme:
            failures.append(
                f"README does not contain the adaptive-tier counts '{expect}'")

    rg = claims.get("recommended_grade") or {}
    if rg.get("adaptive") and rg["adaptive"] not in readme:
        failures.append(f"README does not mention the recommended artifact's "
                        f"adaptive grade '{rg['adaptive']}'")
    if rg.get("adaptive_margin") is not None:
        if not any(m in readme for m in signed(rg["adaptive_margin"], 4)):
            failures.append(f"README does not state the recommended artifact's "
                            f"adaptive margin {rg['adaptive_margin']:+.4f}")

    n_graded = claims.get("n_graded")
    if n_graded is not None:
        for expect in (f"n_graded={n_graded}",
                       f"{len(claims.get('robust_beats') or [])}/{n_graded} in the citable run"):
            if expect not in readme:
                failures.append(f"README does not contain '{expect}'")
    if claims.get("selection_factor") is not None:
        expect = f"selection factor {claims['selection_factor']:.2f}"
        if expect not in readme:
            failures.append(f"README does not contain '{expect}'")

    margins = (claims.get("holdout") or {}).get("margins") or {}
    for op, m in sorted(margins.items()):
        for key, label in (("bench_margin", "benchmark"), ("holdout_margin", "holdout")):
            if not any(t in readme for t in signed(m[key], 3)):
                failures.append(f"README does not state {op}'s {label} margin "
                                f"{m[key]:+.3f}")
        if not m.get("overfit"):
            failures.append(f"claims.json does not mark {op} OVERFIT, but the "
                            f"README says every in-run winner is")

    if failures:
        print(f"CLAIMS DRIFT vs {run_id}:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"OK: README is consistent with {run_id}/claims.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
