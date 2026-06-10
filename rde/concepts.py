"""Conceptual memory -- v1's "richer cumulative memory".

v0 remembered *approaches* (Pareto archive) and *insights* (free text). It could
tell you "approach a4 was incorrect", but not the transferable lesson "indexing a
table by value breaks on negative inputs". v1 distils that structural WHY from the
same evidence + fingerprints the engine already has, so future synthesis builds on
concepts instead of re-deriving them.

Every Lesson here is OBJECTIVE -- correlated from executed evidence and code
structure, never an LLM's say-so. The ledger is bounded (merge-and-strengthen),
and its digest is what gets surfaced to the roles.
"""

from __future__ import annotations

from .config import Config
from .signals import failure_class
from .types import AlgoClass, Approach, Lesson


class ConceptLedger:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.lessons: dict[tuple, Lesson] = {}

    def add(self, lesson: Lesson) -> None:
        existing = self.lessons.get(lesson.key())
        if existing:
            existing.confidence += 1
            for s in lesson.support:
                if s not in existing.support:
                    existing.support.append(s)
            existing.detail = lesson.detail or existing.detail
        else:
            self.lessons[lesson.key()] = lesson
        self._prune()

    def _prune(self) -> None:
        cap = getattr(self.cfg, "memory_max_insights", 40)
        if len(self.lessons) <= cap:
            return
        # Keep the highest-confidence lessons (the most-corroborated concepts).
        kept = sorted(self.lessons.values(),
                      key=lambda L: L.confidence, reverse=True)[:cap]
        self.lessons = {L.key(): L for L in kept}

    def top(self, n: int = 8) -> list[Lesson]:
        return sorted(self.lessons.values(),
                      key=lambda L: L.confidence, reverse=True)[:n]

    def digest(self, n: int = 8) -> str:
        if not self.lessons:
            return "CONCEPTS: (none distilled yet)"
        return "CONCEPTS (structural lessons):\n" + "\n".join(
            f"  - {L.line()}" for L in self.top(n))


# --------------------------------------------------------------------------- #
# Distillation -- turn a round's executed evidence into transferable concepts.
# --------------------------------------------------------------------------- #

def distill(round_approaches: list[Approach],
            all_correct: list[Approach],
            winners: dict[str, Approach],
            domain, cfg: Config) -> list[Lesson]:
    lessons: list[Lesson] = []
    regime = domain.regime if getattr(domain, "regime", None) else (lambda w: w)

    # 1) Failure causes -- prefer a domain-specific structural explanation.
    for a in round_approaches:
        if a.is_correct:
            continue
        expl = domain.explain_failure(a) if getattr(domain, "explain_failure",
                                                    None) else None
        if expl:
            pattern, where, detail = expl
            lessons.append(Lesson(pattern, "hurts", where, detail, support=[a.id]))
        else:
            fc = failure_class(a) or "unknown"
            lessons.append(Lesson(
                f"{a.algo_class.value} as written", "hurts", fc,
                f"{a.id} disqualified ({fc})", support=[a.id]))

    # 2) Who wins which regime (the raw material for synthesis briefs).
    for w, a in winners.items():
        r = regime(w)
        lessons.append(Lesson(
            a.algo_class.value, "helps", r,
            f"{a.algo_class.value} is fastest on {r} workloads (e.g. {w})",
            support=[a.id]))

    # 3) The empirical trap: a correct approach whose asymptotics look good but
    #    whose wall-clock is far worse than the field (e.g. pure-Python
    #    quickselect vs C-backed builtins). Distilled by MEASUREMENT.
    timed = [a for a in all_correct if a.evidence and a.evidence.timings]
    if timed:
        fastest = min(a.evidence.total_time() for a in timed)
        for a in round_approaches:
            if (a.is_correct and a.algo_class == AlgoClass.SELECTION
                    and a.evidence and a.evidence.timings
                    and a.evidence.total_time() > 1.5 * fastest):
                lessons.append(Lesson(
                    "selection / quickselect", "fact", "wall-clock",
                    "better asymptotics but loses on wall-clock to C-backed "
                    "builtins in pure Python", support=[a.id]))
    return lessons


# --------------------------------------------------------------------------- #
# Synthesis brief -- translate "who wins what" into an instruction to combine.
# --------------------------------------------------------------------------- #

def synthesis_opportunity(winners: dict[str, Approach]) -> bool:
    """True when at least two DISTINCT approaches each win a workload -- i.e. the
    strengths are split and a dispatcher could combine them."""
    return len({a.id for a in winners.values()}) >= 2


def synthesis_brief(winners: dict[str, Approach], domain) -> str:
    regime = domain.regime if getattr(domain, "regime", None) else (lambda w: w)
    by_class: dict[str, set] = {}
    for w, a in winners.items():
        by_class.setdefault(a.algo_class.value, set()).add(regime(w))
    parts = [f"{cls} wins {sorted(rs)}" for cls, rs in sorted(by_class.items())]
    return ("Strengths are split across regimes -- " + "; ".join(parts)
            + ". Combine them into ONE adaptive solver that detects the input "
            "regime (e.g. dispatch on k and on input structure) and routes to the "
            "winning strategy, so it wins EVERY regime at once. Do not just return "
            "the best single approach.")


def synthesis_parents(winners: dict[str, Approach]) -> list[str]:
    """Distinct winning approach ids -- the parents a synthesis should combine."""
    seen, out = set(), []
    for a in winners.values():
        if a.id not in seen:
            seen.add(a.id)
            out.append(a.id)
    return out
