"""Drift/stall/patching detection and the Synthesizer's response policy.

These tests pin the *control logic* the spec demands: stall awareness, the
recombination/patching check, recurring-failure detection, and the rule that
only the Synthesizer terminates -- and only for defensible reasons."""

from rde.config import Config
from rde.domains import topk
from rde.memory import Memory
from rde.roles import DetSynthesizer, RoundContext, untried_classes
from rde import signals as sig
from rde.types import (Action, AlgoClass, Approach, Evidence)

CFG = Config(backend="deterministic")
ALL_CLASSES = ["full-sort", "heap", "selection", "bucket"]


# --------------------------------------------------------------------------- #
# signals.compute
# --------------------------------------------------------------------------- #

def _incorrect(id_, cls=AlgoClass.BUCKET):
    a = Approach(id=id_, code="", rationale="", algo_class=cls)
    a.evidence = Evidence(grounded=True, correct=False)
    return a


def test_stall_detected_when_no_gain_over_window():
    s = sig.compute([0.5, 0.5, 0.5, 0.5], 0, [], [], CFG)
    assert s.stalled is True
    assert s.fitness_gain_recent == 0.0


def test_progress_not_flagged_as_stall():
    s = sig.compute([0.5, 0.6, 0.7, 0.85], 0, [], [], CFG)
    assert s.stalled is False
    assert s.fitness_gain_recent > CFG.stall_epsilon


def test_patching_when_recombination_streak_hits_limit():
    s = sig.compute([0.8, 0.8], CFG.patching_limit, [], [], CFG)
    assert s.patching is True
    assert s.consecutive_recombinations == CFG.patching_limit


def test_recurring_failure_class_detected():
    approaches = [_incorrect("a1"), _incorrect("a2")]   # 2 incorrect == limit
    s = sig.compute([0.5], 0, approaches, approaches, CFG)
    assert "incorrect-output" in s.recurring_failures


# --------------------------------------------------------------------------- #
# DetSynthesizer policy
# --------------------------------------------------------------------------- #

def _memory_with_best():
    mem = Memory(CFG)
    a = Approach(id="best", code="", rationale="", algo_class=AlgoClass.HEAP)
    a.evidence = Evidence(grounded=True, correct=True, perf_score=0.9,
                          mem_score=0.7, simplicity_score=0.8)
    a.fitness = 0.85
    mem.add_approach(a)
    assert mem.best is a
    return mem


def _ctx(signals, mem=None, current=None, winners=None):
    return RoundContext(cfg=CFG, domain=topk.DOMAIN,
                        memory=mem or Memory(CFG), iteration=3,
                        signals=signals, current_round=current or [],
                        winners=winners or {})


def _signals(**kw):
    s = sig.Signals()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_untried_classes_helper():
    s = _signals(classes_tried=["full-sort", "heap"])
    assert [c.value for c in untried_classes(s)] == ["selection", "bucket"]


def test_patching_with_untried_explores():
    # v1: patching with untried classes and no synthesis opportunity -> EXPLORE
    # a new niche (rather than RESET to a variation).
    s = _signals(patching=True, consecutive_recombinations=2,
                 classes_tried=["full-sort"])
    d = DetSynthesizer().decide(_ctx(s, _memory_with_best()))
    assert d.action == Action.EXPLORE
    assert any("patching" in f for f in d.drift_flags)


def test_patching_with_all_classes_explored_accepts_best():
    # v1: patching, nothing new to try, but a correct best exists -> ACCEPT it.
    s = _signals(patching=True, consecutive_recombinations=2,
                 classes_tried=ALL_CLASSES)
    d = DetSynthesizer().decide(_ctx(s, _memory_with_best()))
    assert d.action == Action.ACCEPT


def test_patching_with_synthesis_available_synthesizes():
    # v1: the headline upgrade -- recombination/patching + a synthesis
    # opportunity -> SYNTHESIZE instead of patching.
    s = _signals(patching=True, consecutive_recombinations=2,
                 synthesis_available=True, classes_tried=ALL_CLASSES)
    mem = _memory_with_best()
    winners = {"small-k": mem.best}
    d = DetSynthesizer().decide(_ctx(s, mem, winners=winners))
    assert d.action == Action.SYNTHESIZE
    assert d.parent_ids


def test_stall_with_untried_keeps_exploring():
    # v1: an untried class still exists -> breadth-first CONTINUE preempts the
    # stall handling (exploring a new class IS the response to the stall).
    s = _signals(stalled=True, fitness_gain_recent=0.0,
                 classes_tried=["full-sort"])
    d = DetSynthesizer().decide(_ctx(s, _memory_with_best()))
    assert d.action == Action.CONTINUE
    assert d.target_class is not None


def test_stall_with_all_explored_accepts_best():
    s = _signals(stalled=True, classes_tried=ALL_CLASSES)
    d = DetSynthesizer().decide(_ctx(s, _memory_with_best()))
    assert d.action == Action.ACCEPT


def test_healthy_with_untried_continues():
    s = _signals(classes_tried=["full-sort"], best_fitness=0.8)
    d = DetSynthesizer().decide(_ctx(s, _memory_with_best()))
    assert d.action == Action.CONTINUE
    assert d.target_class is not None


def test_all_explored_healthy_accepts():
    s = _signals(classes_tried=ALL_CLASSES, best_fitness=0.85)
    d = DetSynthesizer().decide(_ctx(s, _memory_with_best()))
    assert d.action == Action.ACCEPT


def test_recurring_failure_with_untried_keeps_exploring():
    # v1: classes still unexplored -> CONTINUE to a new class (which IS the
    # redirect away from the recurring failure). EXPLORE/redirect only once
    # breadth is exhausted.
    s = _signals(recurring_failures=["incorrect-output"],
                 classes_tried=["full-sort"])
    d = DetSynthesizer().decide(_ctx(s, _memory_with_best(),
                                     current=[_incorrect("x")]))
    assert d.action == Action.CONTINUE
    assert d.target_class is not None


def test_recurring_failure_all_explored_redirects():
    # With no classes left, a recurring failure and no fresh success -> EXPLORE.
    s = _signals(recurring_failures=["incorrect-output"], classes_tried=ALL_CLASSES)
    d = DetSynthesizer().decide(_ctx(s, _memory_with_best(),
                                     current=[_incorrect("x")]))
    assert d.action in (Action.EXPLORE, Action.ACCEPT)
