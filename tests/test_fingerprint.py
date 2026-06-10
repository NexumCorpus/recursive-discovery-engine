"""Novelty / recombination check: the engine must tell a cosmetic variation from
a genuinely new algorithmic idea."""

from rde.domains import topk
from rde.fingerprint import (assess_novelty, extract_code, fingerprint,
                             similarity)
from rde.types import AlgoClass, Approach


def _approach(name, code, cls):
    a = Approach(id=name, code=code, rationale="", algo_class=cls)
    a.fingerprint = fingerprint(code, cls)
    return a


def test_identical_code_is_maximally_similar():
    code = topk.STRATEGIES["sorted_topk"].code
    fp = fingerprint(code, AlgoClass.FULL_SORT)
    assert similarity(fp, fp) == 1.0


def test_cosmetic_wrapper_flagged_as_recombination():
    # nlargest vs nlargest-wrapped-in-list(): same class, trivially different.
    orig = _approach("heapq", topk.STRATEGIES["heapq_nlargest"].code, AlgoClass.HEAP)
    wrapped = _approach("wrapped", topk.STRATEGIES["nlargest_wrapped"].code,
                        AlgoClass.HEAP)
    sim = similarity(orig.fingerprint, wrapped.fingerprint)
    assert sim >= 0.85, f"expected high similarity, got {sim}"
    novelty, is_recomb, nearest = assess_novelty(wrapped, [orig], 0.85)
    assert is_recomb is True
    assert nearest == "heapq"
    assert novelty < 0.15


def test_different_classes_not_recombination():
    sort = _approach("sort", topk.STRATEGIES["sorted_topk"].code, AlgoClass.FULL_SORT)
    heap = _approach("heap", topk.STRATEGIES["heapq_nlargest"].code, AlgoClass.HEAP)
    novelty, is_recomb, _ = assess_novelty(heap, [sort], 0.85)
    assert is_recomb is False          # different algorithmic class
    assert novelty > 0.3               # and structurally distinct


def test_no_priors_is_fully_novel():
    a = _approach("a", topk.STRATEGIES["sorted_topk"].code, AlgoClass.FULL_SORT)
    novelty, is_recomb, nearest = assess_novelty(a, [], 0.85)
    assert (novelty, is_recomb, nearest) == (1.0, False, "")


def test_extract_code_from_fence():
    reply = "Here you go:\n```python\ndef solve(items, k):\n    return []\n```\nDone."
    assert extract_code(reply) == "def solve(items, k):\n    return []"
