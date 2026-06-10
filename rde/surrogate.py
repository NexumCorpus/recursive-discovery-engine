"""v7 surrogate pre-filter -- a CHEAP value estimate for a proposed op before paying
for full grounding. As proposal volume grows, grounding every proposal on the full
hardened suite gets expensive; this runs the op on ONE small workload and lets the
engine skip the ones that are clearly low-value.

It deliberately COMPOSES the extracted ``grounded-code-execution`` SKILL (the
decoupled harness from the skill-extraction phase) rather than the engine's own
grounding -- a real "engine uses the skills" link, and a clean separation: the
surrogate path is a lightweight, parameterised harness with no Config/Evidence deps.
If the skill is unavailable for any reason, the surrogate is a no-op (never blocks).
"""

from __future__ import annotations

import pathlib
import sys

_SKILL_DIR = (pathlib.Path(__file__).resolve().parent.parent
              / "skills" / "grounded-code-execution")
if _SKILL_DIR.is_dir() and str(_SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(_SKILL_DIR))

try:
    from grounding import (GroundingSpec as _SkillSpec,           # the SKILL's harness
                           measure_quality_on as _skill_measure)
    HAVE_SKILL = True
except Exception:                                                 # noqa: BLE001
    HAVE_SKILL = False


def surrogate_quality(code: str, engine_spec, workload: dict, *,
                      timeout_s: float = 6.0, mem_cap_mb: int = 512):
    """Cheap quality estimate: run `code` on ONE small workload via the SKILL harness
    (a separate, decoupled grounded execution). Returns the measured quality, or None
    if the skill is unavailable or the code did not run.

    v9: the skill harness predates the trusted-driver contract -- it requires a
    top-level solve() and its safety screen (correctly) rejects the driver's own
    text. For a TRUSTED-DRIVER domain the surrogate therefore falls back to the
    ENGINE harness on the same single tiny workload (equally cheap: one small
    subprocess; the engine runner hosts the driver natively). The skill path remains
    the surrogate for legacy solve-style domains. Without this, the pre-filter had
    been a silent no-op for driver domains since v8 (symbol-only candidates have no
    solve, so the skill returned None -> never blocked)."""
    if getattr(engine_spec, "driver_src", ""):
        from .config import Config as _Cfg
        from .grounding import measure_recall_on as _engine_measure
        cfg = _Cfg(bench_repeats=1, subprocess_timeout_s=timeout_s,
                   subprocess_mem_cap_mb=mem_cap_mb)
        rec = _engine_measure(code, [workload], engine_spec, cfg)
        return rec.get(workload["name"])
    if not HAVE_SKILL:
        return None
    spec = _SkillSpec(
        oracle_src=engine_spec.oracle_src,
        workload_gen_src=engine_spec.workload_gen_src,
        criteria_cases=[],
        workloads=[],
        mem_workload=workload,
        verdict_src=engine_spec.verdict_src or "")
    rec = _skill_measure(code, [workload], spec,
                         timeout_s=timeout_s, mem_cap_mb=mem_cap_mb)
    return rec.get(workload["name"])
