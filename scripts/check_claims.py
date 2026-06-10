"""v10 P5: diff the README's verified-result numbers against a named run's
claims.json -- the mechanism that prevents declared-vs-actual drift between the
docs and what the code produces.

Usage:  python scripts/check_claims.py runs/<run_id>

Checks:
  1. the README cites this exact run id (the v10 section must name its run);
  2. every headline number in claims.json appears in the README:
     adaptive-tier beats/matches/below counts and the recommended-artifact grade.
Exit 0 = consistent; exit 1 = drift (each mismatch printed).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


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
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

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

    if failures:
        print(f"CLAIMS DRIFT vs {run_id}:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"OK: README is consistent with {run_id}/claims.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
