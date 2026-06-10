"""v10: close the seams. P1 completeness gate (penalise, never renormalise);
P2 candidate statelessness (module-shape screen + cost cross-check); P3 adaptive
human baselines (ARC / LeCaR, one ghost-charging rule, two-tier flags); P4
loop/verdict re-coupling (frontier on the weighted benchmark CR, in-round
composition, coherence); P5 cleanups (memory_ref, BLENDED, fragments, claims).
Every soundness rule is pinned with a test that ATTEMPTS the exploit.
"""

import dataclasses
import json

from rde.config import Config
from rde.domains import cache_eviction as ce
from rde.domains import heavy_hitters as hh
from rde.engine import Engine
from rde.grounding import driver_shape_violations, ground, measure_recall_on
from rde.roles import make_roles
from rde.types import AlgoClass, Evidence

CFG = Config(bench_repeats=1)
_BASE = ("def _evict_score(la, fr, ins, t):\n    return la\n"
         "def _on_hit(la, fr, ins, t):\n    return [t, fr + 1, ins]\n")


def _fast_cache_domain():
    small = [dict(w, n=2500) for w in ce.WORKLOADS]
    spec = dataclasses.replace(ce.GROUNDING_SPEC, workloads=small,
                               mem_workload=dict(ce.MEM_WORKLOAD, n=2500))
    return dataclasses.replace(
        ce.DOMAIN, grounding_spec=spec,
        adversary_workloads=[dict(w, n=2500) for w in ce.ADVERSARY_WORKLOADS],
        benchmark_suite=[dict(w, n=2500) for w in ce.BENCHMARK_SUITE])


# --------------------------------------------------------------------------- #
# P1 -- completeness gate: retry once, penalise 0.0 + flag, NEVER renormalise
# --------------------------------------------------------------------------- #

def test_completeness_gate_penalises_never_renormalises(monkeypatch):
    eng = Engine(CFG, _fast_cache_domain(), make_roles(CFG))
    suite = eng._utility_workloads()
    calls = {"n": 0}

    def fake_measure(code, wls, spec, cfg):
        calls["n"] += 1
        return {w["name"]: 1.0 for w in wls if w["name"] != "bench_loop"}

    monkeypatch.setattr("rde.engine.measure_recall_on", fake_measure)
    rec, missing = eng._measure_suite_complete("whatever", suite)
    assert calls["n"] == 2, "must retry the suite exactly once"
    assert missing == ["bench_loop"], "the missing workload must be FLAGGED"
    assert rec["bench_loop"] == 0.0, "missing must be PENALISED as 0.0"
    q = eng._weighted_quality(rec, suite)
    # full denominator: 6/7 measured at 1.0 with bench_loop weight 0.10 -> 0.90.
    # Silent renormalisation over the measured subset would give 1.0.
    assert abs(q - 0.90) < 1e-9, f"renormalised! got {q}"


# --------------------------------------------------------------------------- #
# P2 -- statelessness: every stash vector attempted and rejected
# --------------------------------------------------------------------------- #

def test_module_level_stash_is_rejected():
    stash = _BASE + "_history = []\nCMUL = 2\n"
    assert any(v.startswith("module-binding:_history")
               for v in driver_shape_violations(stash, ce.GROUNDING_SPEC))
    ev = ground(stash, ce.GROUNDING_SPEC, CFG)
    assert not ev.correct and "driver-contract" in (ev.error or "")


def test_mutable_default_param_is_rejected():
    stash = (_BASE.replace("def _on_hit(la, fr, ins, t):",
                           "def _on_hit(la, fr, ins, t, _m={}):") + "CMUL = 2\n")
    assert any(v.startswith("mutable-default")
               for v in driver_shape_violations(stash, ce.GROUNDING_SPEC))
    assert not ground(stash, ce.GROUNDING_SPEC, CFG).correct


def test_attribute_store_is_rejected():
    stash = (_BASE + "CMUL = 2\n"
             "def _helper(la, fr, ins, t):\n"
             "    _evict_score.cache = 1\n    return la\n")
    assert any(v.startswith("attribute-store")
               for v in driver_shape_violations(stash, ce.GROUNDING_SPEC))
    assert not ground(stash, ce.GROUNDING_SPEC, CFG).correct


def test_global_ban_composes_with_shape_rules():
    g = ("def _evict_score(la, fr, ins, t):\n    return la\n"
         "def _on_hit(la, fr, ins, t):\n    global CMUL\n    CMUL = 4\n"
         "    return [t, fr + 1, ins]\nCMUL = 1\n")
    assert not ground(g, ce.GROUNDING_SPEC, CFG).correct   # screen + census reject


def test_legacy_domains_are_untouched_by_shape_rules():
    stash = ("STATE = []\n"
             "def solve(items, k):\n    return sorted(set(items))[:k]\n")
    assert driver_shape_violations(stash, hh.GROUNDING_SPEC) == []


def test_cost_cross_check_backstop_trips():
    eng = Engine(CFG, _fast_cache_domain(), make_roles(CFG))
    code = ce._assemble_policy("lru", "bump_both", 2)        # declared 3200B
    ev = Evidence(grounded=True, correct=True)
    ev.peak_memory_bytes = 3200 * Engine._COST_XCHECK_MULT + 1
    eng._apply_cost_axis(ev, code)
    assert not ev.correct and "cost-cross-check" in (ev.error or "")
    # honest measurement (1.35x declared) must NOT trip the backstop
    ev2 = Evidence(grounded=True, correct=True)
    ev2.peak_memory_bytes = 4316
    eng._apply_cost_axis(ev2, code)
    assert ev2.correct and ev2.peak_memory_bytes == 3200


# --------------------------------------------------------------------------- #
# P3 -- adaptive human baselines + one ghost rule + two tiers
# --------------------------------------------------------------------------- #

def test_ghost_charging_is_one_rule_applied_once():
    assert ce._baseline_cost_units(2) == 2 * (1 + ce.GHOST_ENTRY_COST_FRAC)
    for ref in ce.adaptive_baselines():
        cmul = int(ref["label"].lstrip("arclecr"))
        assert ref["cost_units"] == ce._baseline_cost_units(cmul), ref["label"]


def test_arc_reference_sanity_vs_lru():
    cfg = Config(bench_repeats=1)
    suite = [dict(w, n=2500) for w in ce.BENCHMARK_SUITE
             if w["name"] in ("mixed", "drift", "bench_scan", "bench_loop")]
    spec_arc = dataclasses.replace(ce.GROUNDING_SPEC, driver_src=ce.ARC_DRIVER_SRC)
    stub = ce._assemble_policy("lru", "bump_both", 2)
    arc = measure_recall_on(stub, suite, spec_arc, cfg)
    lru = measure_recall_on(stub, suite, ce.GROUNDING_SPEC, cfg)
    assert arc["mixed"] >= lru["mixed"] - 0.01          # ARC >= LRU on mixed
    assert arc["drift"] >= lru["drift"] - 0.01          # ... and drift
    assert arc["bench_scan"] >= lru["bench_scan"] - 0.02   # survives the scan
    assert 0.0 <= arc["bench_loop"] <= 1.0              # sane on the loop


def test_two_tier_flags_and_versioned_history(tmp_path):
    cfg = Config(backend="deterministic", bench_repeats=1, max_iterations=6,
                 utility_history_path=str(tmp_path / "uh.jsonl"))
    eng = Engine(cfg, _fast_cache_domain(), make_roles(cfg))
    result = eng.run()
    v6 = result.v6_summary
    assert "utility_adaptive" in v6 and "op_utility_adaptive" in v6
    # static tier preserved (v6 semantics) alongside the adaptive headline tier
    assert set(v6["op_utility"]) == set(v6["op_utility_adaptive"])
    # completeness flags present and (on a healthy machine) empty
    assert "incomplete" in v6 and "human_baseline_incomplete" in v6
    # history records are VERSIONED: every op appears once per tier
    rows = [json.loads(l) for l in
            (tmp_path / "uh.jsonl").read_text().splitlines() if l.strip()]
    by_op = {}
    for r in rows:
        by_op.setdefault(r["op"], set()).add(r["baseline"])
    assert all(v == {"static", "adaptive"} for v in by_op.values()), by_op
    # P4 coherence: the assertion must not fire on a healthy run
    assert v6.get("incoherent") is None
    # P4: the recommended artifact is GRADED next to the headline
    assert "recommended_grade" in result.improvement
    # P5: fragments persisted for provenance
    assert result.vocab_provenance.get("proposed_op_fragments")
    # P5: claims.json pinned
    claims = json.loads((__import__("pathlib").Path(result.run_dir)
                         / "claims.json").read_text())
    assert claims["run_id"] in result.run_dir
    assert claims["utility_adaptive"] == v6["utility_adaptive"]


# --------------------------------------------------------------------------- #
# P4 -- the frontier optimises the SAME metric the verdict judges
# --------------------------------------------------------------------------- #

def test_frontier_quality_axis_is_the_weighted_benchmark():
    eng = Engine(CFG, _fast_cache_domain(), make_roles(CFG))

    class _A:                                  # minimal Approach stand-in
        pass
    a = _A()
    a.evidence = Evidence(grounded=True, correct=True)
    bs = eng.domain.benchmark_suite
    a.evidence.quality_by_workload = {w["name"]: 1.0 for w in bs}
    a.evidence.quality_by_workload[bs[0]["name"]] = 0.0    # zipf (weight .20) -> 0
    assert abs(eng._suite_quality(a) - 0.80) < 1e-9
    # a missing benchmark measurement is PENALISED as 0.0, never renormalised
    del a.evidence.quality_by_workload[bs[1]["name"]]      # recency (weight .15)
    assert abs(eng._suite_quality(a) - 0.65) < 1e-9


# --------------------------------------------------------------------------- #
# P5 -- memory_ref from the declared capacity range; BLENDED label
# --------------------------------------------------------------------------- #

def test_memory_ref_gives_max_capacity_points_headroom():
    cfg = Config(backend="deterministic", bench_repeats=1, max_iterations=1)
    eng = Engine(cfg, _fast_cache_domain(), make_roles(cfg))
    eng.run()
    assert eng.frontier.memory_ref == 1.2 * 4 * 10 * Engine._BYTES_PER_ENTRY
    from rde.frontier import FrontierPoint
    seed_q = max(p.quality for p in eng.frontier.points)
    gain = eng.frontier.gain(FrontierPoint("probe", min(seed_q + 0.05, 1.0),
                                           4 * 10 * Engine._BYTES_PER_ENTRY, 0.0))
    assert gain > 0, "a frontier-extending max-capacity point must score gain > 0"


def test_blended_vs_adaptive_labels():
    blended = ce._assemble_core("la + 30 * fr.bit_length()",
                                ce._UPDATE_OPS["bump_both"], 2)
    assert ce.classify(blended) == AlgoClass.BLENDED
    adaptive = ce._assemble_adaptive(
        "(-la) if aux[0] < 0.08 else (la + 30 * fr.bit_length())", 2)
    assert ce.classify(adaptive) == AlgoClass.ADAPTIVE
    noop_aux = (_BASE + "AUX0 = [0.5]\n"
                "def _on_access(aux, hit, t):\n    return [aux[0]]\n"
                "def _evict_score_aux(la, fr, ins, t, aux):\n    return la\n"
                "CMUL = 2\n")
    # active score ignores aux -> NOT adaptive (it is plain recency)
    assert ce.classify(noop_aux) == AlgoClass.RECENCY
