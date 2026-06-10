"""v6: the machinery for *aggressively pursuing proposal QUALITY* -- causal
guidance, Builder/Adversary self-play, explicit utility labelling, and a
cross-run utility history.

v5 proved proposals can be authored and grounded honestly; its honest limitation
was that they only MATCHED human baselines. v6 adds the three signals that make a
proposal *earn its keep*:

  * CausalGuidance -- the actual streams a policy fails on and WHICH true heavy
    hitters it missed, fed to the proposer as the PRIMARY steering signal (it
    replaces v5's keyword tags as the main driver of what to propose next).
  * UtilityFlag    -- every proposed op is labelled beats_human / matches_human /
    below_human against the HUMAN Pareto frontier on the hardened suite.
  * SelfPlayLedger -- the Builder authors ops to beat the Adversary's newest
    generators; the win rate + improvement margin are first-class metrics, and the
    Adversary answers back. (Robustness, not overfitting: self-play scores are
    DESCRIPTIVE; the frontier + the full hardened suite remain the arbiters of
    admission, so the engine cannot "win" by gaming its own attacks.)

This module is pure data + bookkeeping; the measurement lives in the engine (which
owns the harness) and the proposers live in the roles / domain.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from pathlib import Path


class UtilityFlag(str, enum.Enum):
    """How a proposed op's best policy compares to the HUMAN Pareto frontier on the
    hardened suite, at the same (or less) memory."""
    BEATS_HUMAN = "beats_human"        # strictly better recall at <= the memory
    MATCHES_HUMAN = "matches_human"    # within eps of the best human at that memory
    BELOW_HUMAN = "below_human"        # worse than the best human at that memory
    UNKNOWN = "unknown"                # never reached a scorable policy

    @property
    def rank(self) -> int:
        return {"beats_human": 2, "matches_human": 1,
                "below_human": 0, "unknown": -1}[self.value]


@dataclass
class CausalGuidance:
    """The PRIMARY v6 steering signal: a concrete, grounded failure of the current
    best policy on ONE specific workload -- the true heavy hitters it MISSED. This
    is what the proposer should target, not a keyword."""
    workload: str = ""            # the failing workload's name
    dist: str = ""                # its generator dist (so the proposer knows the shape)
    recall: float = 1.0           # the policy's recall there
    expected: list = field(default_factory=list)   # oracle top-k
    got: list = field(default_factory=list)        # policy top-k
    missed: list = field(default_factory=list)     # true top-k items NOT returned
    hypothesis: str = ""          # a structural guess at the failure mode
    # weakness TAGS (kept as a SECONDARY signal, derived from the dist/hypothesis).
    tags: set = field(default_factory=set)

    @property
    def has_signal(self) -> bool:
        return bool(self.workload) and self.recall < 0.999

    def brief(self) -> str:
        if not self.has_signal:
            return "no concrete failure to target (best policy is near-perfect)."
        # Factual: these are oracle-top-k items the policy did NOT return -- on
        # low-separation streams some may be tie-boundary items, not recoverable
        # misses, so we do not assert they are 'true heavy hitters'.
        return (f"on workload '{self.workload}' (dist={self.dist}) the best policy "
                f"scored recall={self.recall:.2f}; oracle-top-k items it did not "
                f"return: {self.missed[:8]} (oracle top-k={self.expected[:8]}, "
                f"policy returned {self.got[:8]}). {self.hypothesis}")


@dataclass
class SelfPlayMatch:
    """One Builder-op vs Adversary-generator matchup, scored on that generator. v7:
    the baseline is the BEST HUMAN policy on this generator AT THE OP'S CAPACITY (not
    a fixed incumbent default) -- so a 'win' means the proposed op beats the strongest
    human a person would ship at the same memory budget, on the Adversary's specific
    attack. NB this is a SPECIALIST / robustness signal on individual attacks, which
    is DISTINCT from the `beats_human` utility flag (Pareto-frontier dominance on the
    main suite); an op can be a better generalist (utility=beats_human) yet not beat
    the best human on every targeted attack. Self-play is DESCRIPTIVE, never gating."""
    iteration: int
    op_name: str
    target_generator: str
    op_recall: float
    baseline_recall: float        # v7: best human at the op's capacity on this gen

    @property
    def won(self) -> bool:
        return self.op_recall > self.baseline_recall + 1e-9

    @property
    def margin(self) -> float:
        return round(self.op_recall - self.baseline_recall, 4)


@dataclass
class SelfPlayLedger:
    """First-class self-play record: which proposed ops beat the baseline on which
    Adversary generators, and by how much."""
    matches: list = field(default_factory=list)

    def add(self, m: SelfPlayMatch) -> None:
        self.matches.append(m)

    def win_rate(self) -> float:
        if not self.matches:
            return 0.0
        return round(sum(1 for m in self.matches if m.won) / len(self.matches), 3)

    def mean_margin(self) -> float:
        if not self.matches:
            return 0.0
        return round(sum(m.margin for m in self.matches) / len(self.matches), 4)

    def best_margin(self):
        return max(self.matches, key=lambda m: m.margin) if self.matches else None

    def summary(self) -> dict:
        bm = self.best_margin()
        return {
            "matches": len(self.matches),
            "win_rate": self.win_rate(),
            "mean_margin": self.mean_margin(),
            "best": (f"{bm.op_name} beat the best human on "
                     f"{bm.target_generator} by {bm.margin:+.2f}" if bm else None),
        }


# --------------------------------------------------------------------------- #
# Cross-run utility history -- is the system's proposal quality TRENDING up?
# --------------------------------------------------------------------------- #

def append_utility_history(path: str | Path, records: list) -> None:
    """Append one JSON line per proposed op admitted this run, so multiple runs
    accumulate an auditable record of whether system-proposed primitives are
    getting better (more beats_human) over time."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def summarize_utility_history(path: str | Path) -> dict:
    """Aggregate the cross-run history into a trend: counts per utility flag, the
    beats_human rate, and the rate over the most recent vs earliest runs."""
    p = Path(path)
    if not p.exists():
        return {"runs": 0, "ops": 0}
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return {"runs": 0, "ops": 0}
    # v10 P3: records are versioned by baseline tier; the LEGACY trend aggregates
    # the STATIC tier only (pre-v10 records carry no `baseline` field and are
    # static by definition), so adaptive-tier records never double-count an op.
    rows = [r for r in rows if r.get("baseline", "static") == "static"]
    if not rows:
        return {"runs": 0, "ops": 0}
    # Sort run ids by their NUMERIC suffix (run1, run2, ..., run10, run11) -- a plain
    # string sort orders run10 before run2 and can INVERT the early/late trend.
    def _run_key(s):
        digits = "".join(c for c in str(s) if c.isdigit())
        return (int(digits) if digits else 0, str(s))
    runs = sorted({r.get("run") for r in rows if r.get("run")}, key=_run_key)
    flags = [r.get("utility", "unknown") for r in rows]

    def rate(rs):
        n = len(rs)
        return round(sum(1 for r in rs if r.get("utility") == "beats_human")
                     / n, 3) if n else 0.0

    base = {
        "runs": len(runs),
        "ops": len(rows),
        "beats_human": flags.count("beats_human"),
        "matches_human": flags.count("matches_human"),
        "below_human": flags.count("below_human"),
        "beats_human_rate": rate(rows),
    }
    # A trend needs at least two runs to compare; before that it is undefined (the
    # old code split [run1] into early=[run1], late=[] and always reported 'down').
    if len(runs) < 2:
        base.update({"beats_rate_early": None, "beats_rate_late": None,
                     "trend": "n/a"})
        return base
    half = len(runs) // 2
    early_runs, late_runs = set(runs[:half]), set(runs[half:])
    early = [r for r in rows if r.get("run") in early_runs]
    late = [r for r in rows if r.get("run") in late_runs]
    base.update({
        "beats_rate_early": rate(early),
        "beats_rate_late": rate(late),
        "trend": ("up" if rate(late) > rate(early) + 1e-9
                  else "flat" if abs(rate(late) - rate(early)) <= 1e-9
                  else "down"),
    })
    return base
