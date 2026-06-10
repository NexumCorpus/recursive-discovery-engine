"""v2 streaming heavy-hitters: continuous recall verdict, the exact oracle, and a
full frontier run that (a) synthesises a bounded high-recall hybrid which advances
the frontier and (b) lets the Adversary expose metric-gaming."""

from rde.config import Config
from rde.domains import heavy_hitters as hh
from rde.engine import Engine
from rde.grounding import ground
from rde.roles import make_roles
from rde.types import AlgoClass

CFG = Config(bench_repeats=1)


def test_oracle_exact_topk():
    assert hh.oracle([1, 1, 1, 2, 2, 3], 2) == [1, 2]
    assert hh.oracle([], 3) == []


def test_verdict_recall_and_wellformedness():
    ok, q = hh.verdict([1, 2], [1, 2, 3], [1, 2, 3], 3)
    assert ok and abs(q - 2 / 3) < 1e-9              # recall@3
    assert hh.verdict([1, 1], [1, 2], [1, 2], 2)[0] is False     # duplicates
    assert hh.verdict([9], [1], [1, 2], 1)[0] is False           # not in stream
    assert hh.verdict([1, 2, 3], [1], [1, 2, 3], 1)[0] is False  # longer than k


def test_exact_counter_has_perfect_recall():
    ev = ground(hh.STRATEGIES["exact_counter"].code, hh.GROUNDING_SPEC, CFG)
    assert ev.correct and ev.quality > 0.99


def test_approximate_method_is_admissible_but_lower_recall():
    # Misra-Gries is WELL-FORMED (admissible) yet approximate -> recall < 1.
    ev = ground(hh.STRATEGIES["misra_gries"].code, hh.GROUNDING_SPEC, CFG)
    assert ev.correct                                # passes the well-formed gate
    assert ev.quality < 0.99                         # but not perfect recall
    assert ev.peak_memory_bytes < 100 * 1024         # and far cheaper than exact


def _cls(eng, point):
    a = eng.memory.get(point.approach_id)
    return a.algo_class if a else None


_SYNTH = {AlgoClass.HYBRID, AlgoClass.INVENTED}


def test_v3_synthesis_invents_frontier_advancing_program(hh_cfg, fast_hh_domain):
    eng = Engine(hh_cfg, fast_hh_domain, make_roles(hh_cfg))
    result = eng.run()

    assert result.status in {"accepted", "terminated"}
    assert eng.frontier is not None

    pts = eng.frontier.points
    exact = [p for p in pts if _cls(eng, p) == AlgoClass.EXACT]
    invented = [p for p in pts if _cls(eng, p) in _SYNTH]
    assert exact, "exact should hold the high-recall/high-memory end"
    assert invented, "synthesis should put an INVENTED/composed program on frontier"

    inv = max(invented, key=lambda p: p.quality)
    e = exact[0]
    # The discovery: a composed program reaching high recall at strictly LESS
    # memory than exact -- a region no single LIBRARY strategy occupied.
    assert inv.quality > 0.6
    assert inv.memory < e.memory
    # Its core logic was NOT in the initial library (it was assembled / invented).
    inv_a = eng.memory.get(inv.approach_id)
    assert inv_a.origin.value == "synthesis"
    assert not any(inv_a.code.strip() == s.code.strip()
                   for s in fast_hh_domain.strategies.values())
    # v4: the winning structure is a SYNTHESISED genome policy whose core op lies
    # OUTSIDE the v3 grammar (e.g. halved inheritance `vv // 2`, second-min victim,
    # or a global decay pre-step) -- not a recombination of named v3 policies.
    assert "# v4-synth-policy" in inv_a.code
    assert any(sig in inv_a.code for sig in
               ("vv // 2", "max(1, vv - 1)", "vk = max(counts",
                "_rest = [kk for kk in counts", "counts[_kk]"))
    # And it is the recommended shippable point within the budget.
    rec = eng.frontier.recommended(eng.budget)
    assert rec is not None and _cls(eng, rec) in _SYNTH


def test_v2_adversary_pressures_the_metric(hh_cfg, fast_hh_domain):
    n_base_workloads = len(fast_hh_domain.grounding_spec.workloads)
    eng = Engine(hh_cfg, fast_hh_domain, make_roles(hh_cfg))
    result = eng.run()

    # The Adversary exposed at least one overfitter/metric-gamer ...
    assert any("Adversary exposed" in i.text for i in result.insights)
    # ... and HARDENED the evaluation suite with the workload(s) it failed.
    assert len(eng.spec.workloads) > n_base_workloads
