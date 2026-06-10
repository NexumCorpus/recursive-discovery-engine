"""Pareto-frontier tracking + hypervolume reward -- the heart of v2.

v0/v1 rewarded "beat the best single approach" via a scalar fitness. That is the
wrong objective for *discovery*: it collapses a multi-objective space to one
number and cannot reward a candidate that opens a NEW region of the
quality/resource trade-off while being worse on one axis.

v2 instead tracks the empirical Pareto frontier over (quality, resource) and
rewards **advancing it**. We use 2D: quality = recall@K (higher better) and
resource = peak memory (lower better) -- the binding axis for "heavy hitters
under a memory budget". Throughput is recorded as a secondary metric (extending
to 3D is mechanical; 2D keeps the hypervolume exact and the demo verifiable).

The reward is the **marginal hypervolume contribution** of a candidate's point:
  * a candidate on/under the existing frontier contributes 0  (a recombination is
    automatically unrewarded -- the anti-collapse property, for free);
  * a candidate that reaches a previously-unreachable (recall, memory) combo
    contributes a positive area, even if it is worse on one axis.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrontierPoint:
    approach_id: str
    quality: float       # recall@K in [0,1], higher is better
    memory: int          # peak bytes, lower is better
    time: float = 0.0    # total seconds (secondary axis, recorded not optimised)


def _dominates(a: FrontierPoint, b: FrontierPoint) -> bool:
    """a Pareto-dominates b: >= on quality, <= on memory, and strictly better on
    at least one."""
    return (a.quality >= b.quality and a.memory <= b.memory
            and (a.quality > b.quality or a.memory < b.memory))


def non_dominated(points: list[FrontierPoint]) -> list[FrontierPoint]:
    out: list[FrontierPoint] = []
    for p in points:
        if any(_dominates(q, p) for q in points if q is not p):
            continue
        # Drop exact duplicates (same quality+memory) keeping the first seen.
        if any(o.quality == p.quality and o.memory == p.memory for o in out):
            continue
        out.append(p)
    return out


class Frontier:
    """Maintains the non-dominated (quality, memory) set and scores how much a
    new point advances it. Memory is normalised by a fixed reference so the
    hypervolume lives in the unit square and the reward is in [0, 1]."""

    def __init__(self, memory_ref: float):
        # memory_ref: a fixed worst-case memory (set from the seed). A point at
        # memory_ref contributes zero "resource saving"; below it, positive.
        self.memory_ref = max(float(memory_ref), 1.0)
        self.points: list[FrontierPoint] = []

    # -- geometry -------------------------------------------------------- #
    def _b(self, memory: float) -> float:
        """Resource axis mapped to [0,1]: 1.0 == free, 0.0 == at/over the ref."""
        return max(0.0, 1.0 - memory / self.memory_ref)

    def _hv(self, points: list[FrontierPoint]) -> float:
        """Exact 2D hypervolume (area dominated above the origin) of a point set,
        with x=quality in [0,1] and y=resource-saving in [0,1]."""
        nd = non_dominated(points)
        xy = sorted(((p.quality, self._b(p.memory)) for p in nd),
                    key=lambda t: -t[0])              # quality descending
        hv = 0.0
        prev_b = 0.0
        for x, b in xy:
            if b > prev_b:                            # staircase step
                hv += x * (b - prev_b)
                prev_b = b
        return hv

    def hypervolume(self) -> float:
        return self._hv(self.points)

    # -- the reward ------------------------------------------------------ #
    def gain(self, point: FrontierPoint) -> float:
        """Marginal hypervolume contribution of `point` to the CURRENT frontier.
        0.0 if it is dominated by / equal to the frontier; > 0 if it extends it.
        This is the v2 discovery reward."""
        before = self._hv(self.points)
        after = self._hv(self.points + [point])
        return round(max(0.0, after - before), 6)

    def add(self, point: FrontierPoint) -> bool:
        """Insert if non-dominated; prune any points it dominates. Returns True if
        the point became part of the frontier."""
        if any(_dominates(q, point) for q in self.points):
            return False
        if any(q.quality == point.quality and q.memory == point.memory
               for q in self.points):
            return False
        self.points = [q for q in self.points if not _dominates(point, q)]
        self.points.append(point)
        return True

    # -- reporting ------------------------------------------------------- #
    def recommended(self, budget: float) -> FrontierPoint | None:
        """The point a user would ship: highest recall within the memory budget;
        tie-break on lower memory. Falls back to the lowest-memory point if none
        fits the budget."""
        within = [p for p in self.points if p.memory <= budget]
        pool = within or self.points
        if not pool:
            return None
        return max(pool, key=lambda p: (p.quality, -p.memory))

    def quality_gap_above_budget(self, budget: float) -> float:
        """How much more recall is reachable ABOVE the budget than within it -- a
        concrete synthesis opportunity: 'get that recall within the budget'."""
        within = [p.quality for p in self.points if p.memory <= budget]
        above = [p.quality for p in self.points if p.memory > budget]
        best_within = max(within, default=0.0)
        best_above = max(above, default=0.0)
        return round(max(0.0, best_above - best_within), 4)

    def summary(self, budget: float) -> str:
        pts = sorted(self.points, key=lambda p: p.memory)
        body = ", ".join(f"({p.quality:.2f}r,{p.memory // 1024}KB)" for p in pts)
        rec = self.recommended(budget)
        recs = (f"rec=({rec.quality:.2f}r,{rec.memory // 1024}KB,{rec.approach_id})"
                if rec else "rec=none")
        return (f"frontier[{len(self.points)}] hv={self.hypervolume():.3f} "
                f"{recs} :: {body}")
