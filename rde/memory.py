"""Bounded memory -- a concise, durable record of what has been tried and why.

The spec's "Bounded Memory" mechanism, made concrete. Two problems it solves:

  1. Forgetting.  Naively, each round would have to re-derive lessons already
     learned ("bucket sort breaks on negatives"). Memory persists those as
     compact Insights.

  2. Unbounded growth.  If every approach + full evidence were carried forward,
     the context fed to the LLM roles would grow without limit -- the exact
     "collapse into low-value iteration" the spec warns about. So the approach
     archive is capped to the Pareto frontier (the only approaches still worth
     reasoning about) plus a little recency, and insights are capped to a ring.

Crucially, `digest()` is the ONLY view of history handed to the LLM roles. Its
size is bounded by the caps below, so a 50-iteration run prompts the model with
the same modest amount of context as a 5-iteration run.
"""

from __future__ import annotations

from .concepts import ConceptLedger
from .config import Config
from .types import Approach, Insight


def _dominates(a: Approach, b: Approach) -> bool:
    """Pareto domination over (perf, mem, simplicity): a dominates b if it is
    >= on every axis and strictly > on at least one. Only correct approaches are
    ever compared (incorrect ones have no meaningful scores)."""
    ea, eb = a.evidence, b.evidence
    axes_a = (ea.perf_score, ea.mem_score, ea.simplicity_score)
    axes_b = (eb.perf_score, eb.mem_score, eb.simplicity_score)
    return all(x >= y for x, y in zip(axes_a, axes_b)) and axes_a != axes_b


class Memory:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.approaches: list[Approach] = []   # bounded archive
        self.insights: list[Insight] = []      # bounded ring
        self.lessons = ConceptLedger(cfg)      # v1: conceptual ("why") memory
        self.best: Approach | None = None
        # v4: ids the engine forbids us to evict -- the current (recall, memory)
        # FRONTIER members. The Pareto set _prune_approaches keeps is over a
        # DIFFERENT objective (perf, mem, simplicity); without this a genuine
        # quality-frontier member could be pruned, silently shrinking the frontier
        # and corrupting hypervolume / the recommended point.
        self.protected_ids: set[str] = set()
        # Full count of everything ever seen (archive is pruned; this is not).
        self.total_seen = 0

    def get(self, approach_id: str) -> Approach | None:
        for a in self.approaches:
            if a.id == approach_id:
                return a
        return None

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def add_approach(self, a: Approach) -> None:
        self.total_seen += 1
        self.approaches.append(a)
        if a.is_correct and a.fitness is not None:
            if self.best is None or a.fitness > (self.best.fitness or -1):
                self.best = a
        self._prune_approaches()

    def recompute_best(self) -> None:
        """Re-derive `best` from the retained correct approaches. Call this after
        an approach is downgraded out from under us (e.g. the Adversary confirms
        a break on what was the reigning best), so `best` is never a reference to
        a no-longer-correct approach."""
        best = None
        for a in self.approaches:
            if a.is_correct and a.fitness is not None:
                if best is None or a.fitness > (best.fitness or -1):
                    best = a
        self.best = best

    def add_insight(self, ins: Insight) -> None:
        # De-duplicate identical lessons so the ring is not wasted on repeats.
        if any(i.text == ins.text and i.kind == ins.kind for i in self.insights):
            return
        self.insights.append(ins)
        if len(self.insights) > self.cfg.memory_max_insights:
            # Drop oldest, but never evict the single best 'success' marker.
            self.insights.pop(0)

    # ------------------------------------------------------------------ #
    # Pruning: keep the Pareto frontier of correct approaches, the reigning
    # best, a representative failure per class (lessons live on), and recency.
    # ------------------------------------------------------------------ #
    def _prune_approaches(self) -> None:
        cap = self.cfg.memory_max_approaches
        if len(self.approaches) <= cap:
            return

        correct = [a for a in self.approaches if a.is_correct]
        frontier = [a for a in correct
                    if not any(_dominates(b, a) for b in correct if b is not a)]

        # Membership is tracked by object IDENTITY (id()), never value equality:
        # comparing Approach dataclasses with `in` would invoke __eq__ over their
        # heavy nested evidence/fingerprint structures and is brittle if two
        # approaches ever share field values.
        keep: list[Approach] = []
        keep_ids: set[int] = set()

        def _keep(a: Approach) -> None:
            if id(a) not in keep_ids:
                keep.append(a)
                keep_ids.add(id(a))

        for a in frontier:
            _keep(a)
        if self.best:
            _keep(self.best)

        # v4: protect the (recall, memory) FRONTIER members the engine registered.
        # Kept right after the perf-Pareto set + best (and before failures/recency)
        # so they survive the final `keep[:cap]` truncation: a frontier point whose
        # backing approach is evicted would be silently dropped on the next re-score.
        if self.protected_ids:
            for a in self.approaches:
                if a.id in self.protected_ids:
                    _keep(a)

        # One representative failure per failure-bearing class, so the engine
        # remembers e.g. "BUCKET was tried and was incorrect" even after pruning.
        seen_fail_classes = set()
        for a in self.approaches:
            if not a.is_correct and a.algo_class not in seen_fail_classes:
                _keep(a)
                seen_fail_classes.add(a.algo_class)

        # Protect lineage: never evict an approach that is a PARENT of a retained
        # one, so synthesis_gain / parent resolution never silently break.
        in_use = {pid for a in self.approaches for pid in a.parent_ids}
        for a in self.approaches:
            if a.id in in_use:
                _keep(a)

        # Fill the remainder with the most recent approaches.
        for a in reversed(self.approaches):
            if len(keep) >= cap:
                break
            _keep(a)

        # Preserve original chronological order among the survivors.
        order = {id(a): i for i, a in enumerate(self.approaches)}
        survivors = keep[:cap]
        # v5: a (recall, memory) FRONTIER member is load-bearing -- it backs a live
        # Pareto point + the recommended shippable solution. If the protected set is
        # larger than `cap` (v5's vocabulary search grows the frontier), the
        # `keep[:cap]` truncation alone can still drop one. Re-add any protected
        # member the truncation dropped, even if that means the archive briefly
        # exceeds `cap` (the frontier is small + bounded, so this stays bounded).
        if self.protected_ids:
            surv_ids = {id(a) for a in survivors}
            for a in self.approaches:
                if (a.id in self.protected_ids and id(a) not in surv_ids):
                    survivors.append(a)
                    surv_ids.add(id(a))
        self.approaches = sorted(survivors, key=lambda a: order[id(a)])

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def correct_approaches(self) -> list[Approach]:
        return [a for a in self.approaches if a.is_correct]

    def priors_for_novelty(self) -> list[Approach]:
        """Everything fingerprinted so far is fair game for the novelty check --
        including disqualified ones, so a broken idea is still recognised as
        'already tried' if proposed again."""
        return [a for a in self.approaches if a.fingerprint]

    def digest(self, max_lessons: int = 8) -> str:
        """Compact, bounded textual summary handed to the LLM roles. This is the
        engine's working memory -- deliberately small."""
        lines: list[str] = []
        if self.best:
            ev = self.best.evidence
            lines.append(
                f"BEST SO FAR: {self.best.id} <{self.best.algo_class.value}> "
                f"fitness={self.best.fitness:.3f} "
                f"(perf={ev.perf_score:.2f} mem={ev.mem_score:.2f} "
                f"simpl={ev.simplicity_score:.2f})")
        else:
            lines.append("BEST SO FAR: none correct yet.")

        tried = sorted({a.algo_class.value for a in self.approaches})
        ok = sorted({a.algo_class.value for a in self.correct_approaches()})
        lines.append(f"ALGO CLASSES tried: {tried or ['none']}; "
                     f"proven-correct: {ok or ['none']}")

        # Recent disqualifications, with the concrete reason.
        fails = [a for a in self.approaches if not a.is_correct]
        if fails:
            lines.append("KNOWN-BAD approaches (do not repeat):")
            for a in fails[-5:]:
                why = a.evidence.summary() if a.evidence else "ungrounded"
                lines.append(f"  - {a.id} <{a.algo_class.value}>: {why}")

        # v1: conceptual lessons first -- the transferable "why".
        if self.lessons.lessons:
            lines.append(self.lessons.digest(max_lessons))

        if self.insights:
            lines.append("RECENT EVENTS:")
            for ins in self.insights[-4:]:
                lines.append(f"  - [{ins.kind}] {ins.text}")

        lines.append(f"(memory holds {len(self.approaches)} approaches / "
                     f"{len(self.lessons.lessons)} concepts; "
                     f"{self.total_seen} seen total)")
        return "\n".join(lines)
