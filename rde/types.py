"""Core data types for the Recursive Discovery Engine.

Everything that flows between the three roles is a typed record, not free text.
That is deliberate: the spec's "novelty / recombination check" and "drift
awareness" are impossible over a flat transcript -- you cannot ask "is this just
a variation of attempt #3?" unless attempts are first-class objects with
*lineage* and *evidence* attached. So these dataclasses are the backbone that
makes the protective mechanisms implementable.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #

class Action(str, enum.Enum):
    """The decisions the Synthesizer -- and ONLY the Synthesizer -- may take."""
    CONTINUE = "continue"        # keep building in the current direction
    SYNTHESIZE = "synthesize"    # combine/learn-from prior work into a new approach
    EXPLORE = "explore"          # v1: deliberately seek a behaviourally novel niche
    RESET = "reset"              # force a direction change (escape local patching)
    ACCEPT = "accept"            # current best is good enough; stop with a winner
    TERMINATE = "terminate"      # stop: further progress unlikely w/o new capability


class AlgoClass(str, enum.Enum):
    """Coarse algorithmic family. The recombination check leans on this: two
    approaches in the same class with near-identical fingerprints are variations,
    not discoveries."""
    FULL_SORT = "full-sort"      # sort everything, slice top-k       O(n log n)
    HEAP = "heap"                # bounded heap of size k             O(n log k)
    SELECTION = "selection"      # quickselect / partition-based      O(n) avg
    BUCKET = "bucket"            # counting / bucketing               O(n) bounded
    HYBRID = "hybrid"            # v1/v2: adaptive dispatch / budget-switch
    # v2 streaming heavy-hitters families:
    EXACT = "exact"              # full counter over all distinct items
    MISRA_GRIES = "misra-gries"  # bounded deterministic counters
    COUNT_MIN = "count-min"      # probabilistic sketch + candidate set
    SAMPLE = "sample"            # sub-sample then count
    # v3: a program assembled by compositional synthesis or AST mutation -- its
    # core logic was NOT a single library strategy.
    INVENTED = "invented"
    # v7 cache-eviction families:
    RECENCY = "recency"          # LRU / MRU -- evict by access time
    FREQUENCY = "frequency"      # LFU -- evict by access count
    INSERTION = "insertion"      # FIFO -- evict by insertion order
    # v10 P5: BLENDED = a STATIC score mixing recency + frequency (LRFU-like);
    # ADAPTIVE is reserved for policies whose active score READS the aux state
    # (regime detection during the trace). Conflating them hid which class a
    # policy truly was.
    BLENDED = "blended"
    ADAPTIVE = "adaptive"        # reads aux: switches behaviour mid-trace
    OTHER = "other"              # unclassified


class Origin(str, enum.Enum):
    SEED = "seed"                # the engine's starting baseline(s)
    BUILDER = "builder"          # a fresh proposal
    SYNTHESIS = "synthesis"      # explicitly recombined from prior approaches


# --------------------------------------------------------------------------- #
# Evidence -- the currency of the engine
# --------------------------------------------------------------------------- #

@dataclass
class Evidence:
    """The result of actually RUNNING an approach. No field here is ever filled
    by an LLM's assertion; every value is computed by the grounding harness in an
    isolated subprocess. `grounded=True` is the proof-of-execution stamp -- the
    engine refuses to advance any approach whose evidence is not grounded."""
    grounded: bool = False
    correct: bool = False
    # Each failure: {"case": <name>, "input_k": k, "expected": ..., "got": ...}
    failures: list[dict] = field(default_factory=list)
    mutated_input: bool = False          # did solve() mutate the caller's list?
    # workload-name -> seconds (median of repeats)
    timings: dict[str, float] = field(default_factory=dict)
    # workload-name -> sample stddev of the repeats (v1: noise-aware progress)
    timing_stddev: dict[str, float] = field(default_factory=dict)
    peak_memory_bytes: int = 0
    # v2: continuous quality (e.g. recall@K). 1.0 for an exact/closed-oracle
    # domain; in [0,1] for approximate domains. quality_by_workload is per-stream.
    quality: float = 1.0
    quality_by_workload: dict[str, float] = field(default_factory=dict)
    loc: int = 0                          # non-blank source lines
    ast_nodes: int = 0                    # AST node count (complexity proxy)
    # Normalised sub-scores in [0, 1], 1 == best, relative to a fixed baseline.
    perf_score: float = 0.0
    mem_score: float = 0.0
    simplicity_score: float = 0.0
    error: Optional[str] = None          # import/exec/timeout error, if any
    traceback: Optional[str] = None

    @property
    def noise(self) -> float:
        """Mean coefficient of variation (stddev/median) across workloads -- the
        benchmark's relative measurement noise. Used so a 'speedup' inside the
        noise band is not mistaken for real progress."""
        cvs = [self.timing_stddev[w] / self.timings[w]
               for w in self.timings
               if self.timings.get(w, 0) > 0 and w in self.timing_stddev]
        return sum(cvs) / len(cvs) if cvs else 0.0

    def total_time(self) -> float:
        return sum(self.timings.values())

    def summary(self) -> str:
        if self.error:
            return f"ERROR: {self.error}"
        if not self.correct:
            n = len(self.failures)
            mut = " +mutates-input" if self.mutated_input else ""
            return f"INCORRECT ({n} failing case(s){mut})"
        t = sum(self.timings.values())
        return (f"ok | recall={self.quality:.2f} mem={self.peak_memory_bytes // 1024}KB "
                f"| perf={self.perf_score:.2f} simpl={self.simplicity_score:.2f} "
                f"| wallt={t*1e3:.1f}ms (noise={self.noise*100:.0f}%)")


# --------------------------------------------------------------------------- #
# Approach -- a candidate solution with full lineage
# --------------------------------------------------------------------------- #

@dataclass
class Approach:
    id: str
    code: str                            # python source defining solve(items, k)
    rationale: str
    algo_class: AlgoClass = AlgoClass.OTHER
    origin: Origin = Origin.BUILDER
    parent_ids: list[str] = field(default_factory=list)   # lineage
    created_iter: int = 0
    evidence: Optional[Evidence] = None
    fitness: Optional[float] = None      # composite; None == disqualified
    novelty: Optional[float] = None      # 1 - max STRUCTURAL similarity to priors
    is_recombination: bool = False       # flagged by the structural novelty check
    fingerprint: dict = field(default_factory=dict)  # structural features
    # v1 -------------------------------------------------------------------- #
    behavior: dict = field(default_factory=dict)      # per-workload time vector + wins
    behavioral_novelty: Optional[float] = None        # 1 - max BEHAVIOURAL similarity
    synthesis_gain: Optional[float] = None            # for SYNTHESIS: vs its parents
    synthesis_brief: str = ""                          # what it was asked to combine
    frontier_gain: Optional[float] = None             # v2: marginal Pareto HV added
    # v5: the vocabulary ops this policy was composed from (victim/inherit/pre op
    # names). Lets the engine flag which PROPOSED ops reached the frontier.
    ops_used: list[str] = field(default_factory=list)

    @property
    def is_correct(self) -> bool:
        return bool(self.evidence and self.evidence.grounded and self.evidence.correct)

    def short(self) -> str:
        fit = f"{self.fitness:.3f}" if self.fitness is not None else "DSQ"
        nov = f"{self.novelty:.2f}" if self.novelty is not None else "?"
        bnov = f"/b{self.behavioral_novelty:.2f}" if \
            self.behavioral_novelty is not None else ""
        rec = " [recomb]" if self.is_recombination else ""
        sg = f" sgain={self.synthesis_gain:.2f}" if \
            self.synthesis_gain is not None else ""
        fg = f" fgain={self.frontier_gain:.3f}" if \
            self.frontier_gain is not None else ""
        ev = self.evidence.summary() if self.evidence else "ungrounded"
        return (f"{self.id} <{self.algo_class.value}> fit={fit} "
                f"nov={nov}{bnov}{rec}{sg}{fg} :: {ev}")


# --------------------------------------------------------------------------- #
# Adversary output
# --------------------------------------------------------------------------- #

@dataclass
class AdversaryFinding:
    """A single attack the Adversary mounted. `broke` is only ever True when the
    grounding harness CONFIRMED a discrepancy against the oracle -- the Adversary
    cannot 'claim' a break; the executed evidence decides."""
    hypothesis: str
    broke: bool = False
    grounded: bool = False
    counterexample: Optional[dict] = None   # {"items": [...], "k": k, "expected":, "got":}
    notes: str = ""


@dataclass
class AdversaryReport:
    approach_id: str
    findings: list[AdversaryFinding] = field(default_factory=list)

    @property
    def broke_it(self) -> bool:
        return any(f.broke for f in self.findings)

    def confirmed_breaks(self) -> list[AdversaryFinding]:
        return [f for f in self.findings if f.broke and f.grounded]


# --------------------------------------------------------------------------- #
# Synthesizer output
# --------------------------------------------------------------------------- #

@dataclass
class Directive:
    """The Synthesizer's decision for the next round. This is the 'Command Packet'
    surfaced at human-override checkpoints."""
    action: Action
    reason: str
    # For SYNTHESIZE/RESET: a concrete instruction for the Builder.
    instruction: str = ""
    target_class: Optional[AlgoClass] = None   # RESET/EXPLORE -> aim at this family
    parent_ids: list[str] = field(default_factory=list)
    # v1: a concrete brief for genuine synthesis -- which prior approaches to
    # combine and along which axis (e.g. "heap wins small-k, sort wins large-k;
    # dispatch on k"). Empty unless action == SYNTHESIZE.
    synthesis_brief: str = ""
    # For EXPLORE: the behavioural niche (workload) to try to win.
    explore_target: str = ""
    # Explicit novelty bookkeeping the spec demands ("meaningful novelty vs
    # primarily recombination").
    novelty_assessment: str = ""
    drift_flags: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Memory ledger entry
# --------------------------------------------------------------------------- #

@dataclass
class Insight:
    """A compressed, durable lesson. Bounded memory keeps these instead of the
    full transcript so context does not grow without limit."""
    iteration: int
    kind: str            # "lesson" | "failure" | "success" | "drift" | "synthesis"
    text: str
    related: list[str] = field(default_factory=list)   # approach ids
    ts: float = field(default_factory=lambda: time.time())


@dataclass
class Lesson:
    """v1 conceptual memory: a STRUCTURAL 'why', distilled objectively from
    evidence + fingerprints, not an LLM assertion. E.g. pattern='value-indexed
    count table', polarity='hurts', regime='negative/float inputs'. Confidence is
    the number of independent observations supporting it."""
    pattern: str                 # the structural / behavioural feature
    polarity: str                # "helps" | "hurts" | "fact"
    regime: str                  # where it applies (workload / input property)
    detail: str                  # human-readable explanation
    support: list[str] = field(default_factory=list)   # approach ids
    confidence: int = 1
    ts: float = field(default_factory=lambda: time.time())

    def key(self) -> tuple:
        return (self.pattern, self.polarity, self.regime)

    def line(self) -> str:
        c = f" (x{self.confidence})" if self.confidence > 1 else ""
        return f"[{self.polarity}] {self.pattern} @ {self.regime}: {self.detail}{c}"


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses/enums into JSON-serialisable structures
    for run logging."""
    if isinstance(obj, enum.Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj
