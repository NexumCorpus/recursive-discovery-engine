"""The Pareto-frontier reward: dominated points earn nothing, points that reach a
new (recall, memory) region earn positive hypervolume, and the frontier only
grows."""

from rde.frontier import Frontier, FrontierPoint, _dominates, non_dominated


def test_dominance():
    a = FrontierPoint("a", 1.0, 100)
    b = FrontierPoint("b", 0.5, 200)
    assert _dominates(a, b) and not _dominates(b, a)


def test_non_dominated_filters_and_dedups():
    pts = [FrontierPoint("a", 1.0, 900), FrontierPoint("b", 0.3, 100),
           FrontierPoint("c", 0.2, 150)]               # c dominated by b
    nd = non_dominated(pts)
    assert {p.approach_id for p in nd} == {"a", "b"}


def test_gain_zero_for_dominated_point():
    f = Frontier(memory_ref=1000)
    f.add(FrontierPoint("a", 1.0, 100))
    # A point with less recall and more memory is dominated -> no reward.
    assert f.gain(FrontierPoint("b", 0.5, 200)) == 0.0


def test_gain_positive_for_new_region():
    f = Frontier(memory_ref=1000)
    f.add(FrontierPoint("exact", 1.0, 900))
    # A low-memory point opens a region the frontier did not cover.
    assert f.gain(FrontierPoint("mg", 0.3, 100)) > 0.0


def test_gain_positive_for_gap_filling_hybrid():
    f = Frontier(memory_ref=1000)
    f.add(FrontierPoint("exact", 1.0, 900))
    f.add(FrontierPoint("mg", 0.3, 100))
    # A bounded high-recall hybrid fills the middle -> rewarded even though it is
    # worse than exact on recall and worse than MG on memory.
    assert f.gain(FrontierPoint("hybrid", 0.85, 400)) > 0.0


def test_hypervolume_monotonic_nondecreasing():
    f = Frontier(memory_ref=1000)
    hv0 = f.hypervolume()
    f.add(FrontierPoint("a", 1.0, 900))
    hv1 = f.hypervolume()
    f.add(FrontierPoint("b", 0.3, 100))
    hv2 = f.hypervolume()
    assert hv0 < hv1 <= hv2


def test_recommended_respects_budget():
    f = Frontier(memory_ref=1000)
    f.add(FrontierPoint("exact", 1.0, 900))
    f.add(FrontierPoint("hybrid", 0.8, 400))
    rec = f.recommended(budget=500)        # exact is over budget
    assert rec.approach_id == "hybrid"
