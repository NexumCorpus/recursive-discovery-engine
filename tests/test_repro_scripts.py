"""scripts/check_claims.py and scripts/reproduce.py: the README-vs-evidence checks.

Fast (no engine loop): both scripts run against the committed cited run, in a
scratch copy of the repo so the README and the claims can be perturbed.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "20260610_115502_cache_eviction_deterministic"
RUN = f"evidence/runs/{RUN_ID}"


@pytest.fixture()
def scratch(tmp_path):
    shutil.copytree(ROOT / "scripts", tmp_path / "scripts")
    shutil.copytree(ROOT / "evidence", tmp_path / "evidence")
    shutil.copy(ROOT / "README.md", tmp_path / "README.md")
    return tmp_path


def run(scratch, script, *args):
    return subprocess.run([sys.executable, str(scratch / "scripts" / script), *args],
                          capture_output=True, text=True, encoding="utf-8")


def test_check_claims_passes_on_the_cited_run(scratch):
    r = run(scratch, "check_claims.py", RUN)
    assert r.returncode == 0, r.stdout
    assert RUN_ID in r.stdout


# Each README number the script now pins, changed to a plausible wrong value, and
# the line the script must print about it.
@pytest.mark.parametrize("old,new,expect", [
    ("n_graded=14", "n_graded=13", "'n_graded=14'"),
    ("selection factor 1.95", "selection factor 1.94", "'selection factor 1.95'"),
    ("0/14 in the citable run", "1/14 in the citable run", "'0/14 in the citable run'"),
    ("(+0.0126", "(+0.0127", "adaptive margin +0.0126"),
    ("+0.035 / +0.059", "+0.036 / +0.059", "ad_thrash's benchmark margin +0.035"),
    ("−0.171 / −0.137", "−0.172 / −0.137", "ad_thrash's holdout margin -0.171"),
    ("2 beats, 6 matches, 6 below", "3 beats, 6 matches, 5 below", "'2 beats, 6 matches, 6 below'"),
])
def test_check_claims_catches_a_changed_readme_number(scratch, old, new, expect):
    readme = (scratch / "README.md").read_text(encoding="utf-8")
    flat = " ".join(readme.split())
    assert old in flat, f"README no longer contains {old!r}; update this test with it"
    # The script collapses whitespace before matching, so a phrase the README wraps
    # across lines is mutated in the collapsed text.
    mutated = readme.replace(old, new) if old in readme else flat.replace(old, new)
    (scratch / "README.md").write_text(mutated, encoding="utf-8")
    r = run(scratch, "check_claims.py", RUN)
    assert r.returncode == 1 and not r.stderr, (r.stdout, r.stderr)
    assert "CLAIMS DRIFT" in r.stdout and expect in r.stdout


def test_check_claims_fails_on_a_run_the_readme_does_not_cite(scratch):
    other = scratch / "evidence" / "runs" / "20990101_000000_cache_eviction_deterministic"
    shutil.copytree(scratch / RUN, other)
    claims = json.loads((other / "claims.json").read_text(encoding="utf-8"))
    claims["run_id"] = other.name
    (other / "claims.json").write_text(json.dumps(claims), encoding="utf-8")
    r = run(scratch, "check_claims.py", f"evidence/runs/{other.name}")
    assert r.returncode == 1
    assert "does not cite run id" in r.stdout


def test_reproduce_accepts_the_cited_run_compared_with_itself(scratch):
    r = run(scratch, "reproduce.py", RUN)
    assert r.returncode == 0, r.stdout
    assert "REPRODUCED: 24/24" in r.stdout


def test_reproduce_reports_drift_and_the_timing_note(scratch):
    drifted = scratch / "evidence" / "runs" / "drifted"
    shutil.copytree(scratch / RUN, drifted)
    claims = json.loads((drifted / "claims.json").read_text(encoding="utf-8"))
    claims["frontier_hypervolume"] = 0.4465
    (drifted / "claims.json").write_text(json.dumps(claims), encoding="utf-8")
    r = run(scratch, "reproduce.py", "evidence/runs/drifted")
    assert r.returncode == 1, r.stdout
    assert "DRIFT frontier_hypervolume" in r.stdout
    assert "FAIL: 1 difference(s)" in r.stdout
    assert "wall-clock" in r.stdout and "idle machine" in r.stdout
