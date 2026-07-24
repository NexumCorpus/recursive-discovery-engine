"""Independent checker for the frozen causal-xenosoma domain."""
from __future__ import annotations

from .domains.causal_xenosoma import canonical, run


def check(artifact, report):
    expected = run(artifact)
    problems = []
    if canonical(report.get("measurements")) != canonical(expected.get("measurements")):
        problems.append("measurements-mismatch")
    if canonical(report.get("families")) != canonical(expected.get("families")):
        problems.append("holdout-family-mismatch")
    if canonical(report.get("audit")) != canonical(expected.get("audit")):
        problems.append("audit-output-mismatch")
    if report.get("falsifiers") != expected.get("falsifiers"):
        problems.append("falsifier-mismatch")
    if report.get("ok") is not expected.get("ok"):
        problems.append("verdict-mismatch")
    return {"ok": not problems and expected.get("ok") is True, "problems": problems, "recomputed": expected.get("ok") is True}
