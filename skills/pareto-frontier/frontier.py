"""Pareto-frontier tracking + marginal-hypervolume reward.

Extracted from the Recursive Discovery Engine (`rde/frontier.py`), stable since v2.

The idea: when you are *discovering* (not just optimising a scalar), the right
reward is **how much a candidate advances the Pareto frontier** of a 2-objective
trade-off, NOT "is it better than the single best so far". A scalar best collapses
the trade-off and cannot reward a candidate that opens a NEW region of the space
while being worse on one axis. Marginal hypervolume does exactly that:

  * a candidate on/under the current frontier contributes **0** (a recombination is
    automatically unrewarded -- the anti-collapse property, for free);
  * a candidate that reaches a previously-unreachable (quality, cost) combination
    contributes a **positive area**, even if it is worse on one axis.

The two axes are ``quality`` (higher is better, in [0, 1]) and ``cost`` (lower is
better, e.g. memory, latency, money). In RDE: quality = recall@K, cost = peak memory
bytes. Normalising cost by a fixed reference keeps the hypervolume in the unit
square, so the reward lives in [0, 1].

Public API
----------
    FrontierPoint(label, quality, cost, extra={})
    Frontier(cost_ref)
        .gain(point)  -> float    # marginal hypervolume contribution (THE reward)
        .add(point)   -> bool      # insert if non-dominated; prune what it dominates
        .hypervolume() -> float
        .recommended(budget) -> FrontierPoint | None   # ship: best quality <= budget
        .quality_gap_above_budget(budget) -> float     # synthesis opportunity size
        .summary(budget) -> str
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FrontierPoint:
    label: str               # any identifier for the candidate
    quality: float           # higher is better, typically in [0, 1]
    cost: float              # lower is better (memory bytes / latency / $ / ...)
    extra: dict = field(default_factory=dict)   # any secondary metrics you carry


def _dominates(a: FrontierPoint, b: FrontierPoint) -> bool:
    """a Pareto-dominates b: >= on quality, <= on cost, strictly better on one."""
    return (a.quality >= b.quality and a.cost <= b.cost
            and (a.quality > b.quality or a.cost < b.cost))


def non_dominated(points: list[FrontierPoint]) -> list[FrontierPoint]:
    """The Pareto-non-dominated subset, dropping exact (quality, cost) duplicates
    (first seen wins)."""
    out: list[FrontierPoint] = []
    for p in points:
        if any(_dominates(q, p) for q in points if q is not p):
            continue
        if any(o.quality == p.quality and o.cost == p.cost for o in out):
            continue
        out.append(p)
    return out


class Frontier:
    """Maintains the non-dominated (quality, cost) set and scores how much a new
    point advances it. Cost is normalised by a fixed reference, so the hypervolume
    lives in the unit square and the reward is in [0, 1]."""

    def __init__(self, cost_ref: float):
        # cost_ref: a fixed worst-case cost (e.g. the seed/baseline's cost). A point
        # at cost_ref contributes zero "resource saving"; below it, positive.
        self.cost_ref = max(float(cost_ref), 1.0)
        self.points: list[FrontierPoint] = []

    # -- geometry -------------------------------------------------------- #
    def _saving(self, cost: float) -> float:
        """Cost axis mapped to [0,1]: 1.0 == free, 0.0 == at/over the reference."""
        return max(0.0, 1.0 - cost / self.cost_ref)

    def _hv(self, points: list[FrontierPoint]) -> float:
        """Exact 2D hypervolume (area dominated above the origin) of a point set,
        with x = quality in [0,1] and y = cost-saving in [0,1]."""
        nd = non_dominated(points)
        xy = sorted(((p.quality, self._saving(p.cost)) for p in nd),
                    key=lambda t: -t[0])              # quality descending
        hv, prev = 0.0, 0.0
        for x, y in xy:
            if y > prev:                              # staircase step
                hv += x * (y - prev)
                prev = y
        return hv

    def hypervolume(self) -> float:
        return self._hv(self.points)

    # -- the reward ------------------------------------------------------ #
    def gain(self, point: FrontierPoint) -> float:
        """Marginal hypervolume contribution of ``point`` to the CURRENT frontier.
        0.0 if dominated by / equal to the frontier; > 0 if it extends it. This is
        the discovery reward -- use it as the candidate's fitness."""
        before = self._hv(self.points)
        after = self._hv(self.points + [point])
        return round(max(0.0, after - before), 6)

    def add(self, point: FrontierPoint) -> bool:
        """Insert if non-dominated; prune any points it dominates. Returns True iff
        the point became part of the frontier."""
        if any(_dominates(q, point) for q in self.points):
            return False
        if any(q.quality == point.quality and q.cost == point.cost
               for q in self.points):
            return False
        self.points = [q for q in self.points if not _dominates(point, q)]
        self.points.append(point)
        return True

    # -- reporting ------------------------------------------------------- #
    def recommended(self, budget: float) -> "FrontierPoint | None":
        """The point you would SHIP: highest quality within the cost budget;
        tie-break on lower cost. Falls back to the lowest-cost point if none fits."""
        within = [p for p in self.points if p.cost <= budget]
        pool = within or self.points
        if not pool:
            return None
        return max(pool, key=lambda p: (p.quality, -p.cost))

    def quality_gap_above_budget(self, budget: float) -> float:
        """How much more quality is reachable ABOVE the budget than within it -- a
        concrete synthesis opportunity ('reach that quality within the budget')."""
        within = [p.quality for p in self.points if p.cost <= budget]
        above = [p.quality for p in self.points if p.cost > budget]
        return round(max(0.0, max(above, default=0.0) - max(within, default=0.0)), 4)

    def summary(self, budget: float) -> str:
        pts = sorted(self.points, key=lambda p: p.cost)
        body = ", ".join(f"({p.quality:.2f}q,{p.cost:g}c)" for p in pts)
        rec = self.recommended(budget)
        recs = (f"rec=({rec.quality:.2f}q,{rec.cost:g}c,{rec.label})"
                if rec else "rec=none")
        return (f"frontier[{len(self.points)}] hv={self.hypervolume():.3f} "
                f"{recs} :: {body}")


if __name__ == "__main__":   # smoke test
    f = Frontier(cost_ref=1000)
    f.add(FrontierPoint("seed", 1.0, 1000))           # at the cost ref -> no saving
    assert f.gain(FrontierPoint("cheap", 0.8, 100)) > 0     # opens a low-cost region
    f.add(FrontierPoint("cheap", 0.8, 100))
    assert f.gain(FrontierPoint("dom", 0.5, 500)) == 0.0    # dominated -> 0 reward
    assert f.gain(FrontierPoint("dup", 0.8, 100)) == 0.0    # duplicate -> 0 reward
    assert f.recommended(budget=500).label == "cheap"
    assert f.quality_gap_above_budget(500) > 0             # quality sits above budget
    print("SMOKE_OK")
