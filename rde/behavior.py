"""Behavioural descriptors -- v1's measured (not asserted) novelty axis.

v0 judged novelty purely on source STRUCTURE (the fingerprint). That misses two
things the spec's v1 brief cares about:

  1. "Sophisticated local patching": code that is heavily rewritten (low
     structural similarity) but performs identically -- a disguised non-result.
  2. Genuine synthesis: a dispatcher that LOOKS like a mash-up but BEHAVES
     differently (and better) than any single parent.

Both are about *behaviour*, which we can only know by running the candidate. So
v1 adds a behavioural descriptor = the per-workload timing vector the grounding
harness already produces. Similarity is the mean per-workload min/max ratio --
two approaches with the same wall-clock profile are behaviourally identical no
matter how their source reads, and the LLM cannot fake the timings.
"""

from __future__ import annotations

from .types import Approach, Evidence


def behavioral_similarity(ev1: Evidence, ev2: Evidence) -> float:
    """Similarity in [0, 1] of two timing profiles. 1.0 == identical wall-clock
    behaviour across the shared workloads. Uses the per-workload min/max ratio so
    a single 3x-different workload visibly lowers similarity (cosine would not)."""
    shared = sorted(set(ev1.timings) & set(ev2.timings))
    if not shared:
        return 0.0
    ratios = []
    for w in shared:
        a, b = ev1.timings[w], ev2.timings[w]
        if a <= 0 and b <= 0:
            ratios.append(1.0)
        elif a <= 0 or b <= 0:
            ratios.append(0.0)
        else:
            ratios.append(min(a, b) / max(a, b))
    return sum(ratios) / len(ratios)


def assess_behavioral_novelty(ev: Evidence, priors: list[Evidence],
                              stale_threshold: float) -> tuple[float, bool]:
    """Returns (behavioral_novelty, is_stale). novelty = 1 - max similarity to any
    prior; is_stale == the nearest prior is >= stale_threshold similar (the
    'same behaviour in different clothes' case)."""
    if not priors:
        return 1.0, False
    best = max(behavioral_similarity(ev, p) for p in priors)
    return round(1.0 - best, 4), best >= stale_threshold


def per_workload_winners(approaches: list[Approach],
                         workloads: list[str]) -> dict[str, Approach]:
    """For each workload, the correct approach with the lowest median time. This
    is the behavioural-niche map: who 'owns' each regime."""
    winners: dict[str, Approach] = {}
    for w in workloads:
        best = None
        for a in approaches:
            if a.is_correct and a.evidence and w in a.evidence.timings:
                t = a.evidence.timings[w]
                if best is None or t < best[1]:
                    best = (a, t)
        if best is not None:
            winners[w] = best[0]
    return winners


def describe(ev: Evidence, all_approaches: list[Approach]) -> dict:
    """Build the behaviour record stored on an Approach: its raw per-workload
    times plus the set of workloads it currently WINS against the field."""
    workloads = list(ev.timings.keys())
    winners = per_workload_winners(all_approaches, workloads)
    wins = [w for w, a in winners.items()
            if a.evidence is ev or a.evidence.timings == ev.timings]
    return {"times": dict(ev.timings), "wins": sorted(wins)}


def synthesis_gain(s_ev: Evidence, parents: list[Approach]) -> float | None:
    """How well a SYNTHESIS matches/beats its parents ON THEIR OWN TURF.

    For every workload that a (correct) parent WINS among the parent set, compare
    the synthesis's time to that parent's. gain = min ratio (parent/synth) across
    those workloads: >= 1 means the synthesis is at least as fast as the best
    parent on every regime that parent owned -- i.e. it genuinely COMBINED their
    strengths rather than trading one for another. < 1 means it regressed
    somewhere; that is not a real hybrid.
    """
    relevant = [p for p in parents
                if p.is_correct and p.evidence and p.evidence.timings]
    if not relevant or not s_ev.timings:
        return None
    winners = per_workload_winners(relevant, list(s_ev.timings.keys()))
    gains = []
    for w, p in winners.items():
        pt = p.evidence.timings.get(w)
        st = s_ev.timings.get(w)
        if pt and st:
            gains.append(pt / st)
    return round(min(gains), 3) if gains else None
