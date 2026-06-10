"""Objective progress signals -- the antidote to a meta-role rationalising its
own collapse.

The single most important design decision in the engine: the Synthesizer does
NOT get to *decide* whether the process is drifting. The engine *computes* the
drift/stall/recombination signals from the run history, deterministically, and
hands them to the Synthesizer as facts. The Synthesizer reasons about what to do
about them, but it cannot make a detected stall disappear by ignoring it.

This is what stops the classic failure mode where an LLM, asked "are we making
progress?", cheerfully says "yes!" while producing the third cosmetic variation
of the same idea.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .types import Approach


@dataclass
class Signals:
    # Best composite fitness reached so far, and the gain over the stall window.
    best_fitness: float | None = None
    fitness_gain_recent: float = 0.0
    stalled: bool = False                 # no meaningful gain over stall_window

    # Recombination pressure (local patching disguised as progress).
    consecutive_recombinations: int = 0
    patching: bool = False                # >= patching_limit recombinations in a row

    # Recurring failure classes (a shared blind spot the engine keeps hitting).
    failure_class_counts: dict[str, int] = field(default_factory=dict)
    recurring_failures: list[str] = field(default_factory=list)

    # Exploration breadth -- how much of the algorithm space has been tried.
    classes_tried: list[str] = field(default_factory=list)
    classes_correct: list[str] = field(default_factory=list)

    # Novelty of the most recent round's proposals (mean).
    last_round_mean_novelty: float | None = None

    # --- v1 additions (populated by the engine from measured behaviour) ---- #
    # Mean BEHAVIOURAL novelty of the last round (1 - max timing-profile sim).
    last_round_mean_behavioral_novelty: float | None = None
    # Recombination is the DOMINANT recent pattern: many recent rounds were
    # structurally-recombinant OR behaviourally-stale. The strongest "we are
    # local-patching, not discovering" signal -- and it is measured, not judged.
    recombination_dominant: bool = False
    # Distinct approaches win distinct workloads -> a dispatcher could combine
    # them: a concrete synthesis opportunity.
    synthesis_available: bool = False
    # For the last round's SYNTHESIS attempt: did it beat its parents on their
    # own turf? (>= cfg.synthesis_min_gain == a genuine hybrid.)
    last_round_synthesis_gain: float | None = None
    # Benchmark measurement noise of the current best (mean CV).
    noise_floor: float = 0.0
    # How many SYNTHESIZE rounds have been spent so far (convergence cap).
    synthesis_attempts: int = 0

    # --- v2 frontier signals (populated by the engine) ------------------- #
    frontier_size: int = 0
    frontier_hv: float = 0.0
    frontier_advanced: bool = False          # last round added/extended a point
    frontier_gap: float = 0.0                # recall reachable above budget only
    metric_hardened: bool = False            # adversary added a workload this round
    # The most recent best-fitness improvement is within the noise band -> it may
    # not be real progress at all.
    last_improvement_within_noise: bool = False

    def headline(self) -> str:
        bits = []
        if self.best_fitness is not None:
            bits.append(f"best_fit={self.best_fitness:.3f}")
        bits.append(f"recent_gain={self.fitness_gain_recent:+.3f}")
        if self.last_improvement_within_noise:
            bits.append("GAIN<=NOISE")
        if self.stalled:
            bits.append("STALLED")
        if self.patching:
            bits.append(f"PATCHING(x{self.consecutive_recombinations})")
        if self.recombination_dominant:
            bits.append("RECOMB-DOMINANT")
        if self.synthesis_available:
            bits.append("SYNTH-AVAIL")
        if self.last_round_synthesis_gain is not None:
            bits.append(f"sgain={self.last_round_synthesis_gain:.2f}")
        if self.recurring_failures:
            bits.append("RECURRING:" + ",".join(self.recurring_failures))
        if self.frontier_size:
            bits.append(f"frontier[{self.frontier_size}] hv={self.frontier_hv:.3f}"
                        + (" ADVANCED" if self.frontier_advanced else ""))
            if self.frontier_gap > 0:
                bits.append(f"gap={self.frontier_gap:.2f}")
            if self.metric_hardened:
                bits.append("METRIC-HARDENED")
        bits.append("classes=" + (",".join(self.classes_tried) or "none"))
        return " | ".join(bits)

    def has_drift(self) -> bool:
        """Any condition the Synthesizer is REQUIRED to address explicitly."""
        return bool(self.stalled or self.patching or self.recurring_failures
                    or self.recombination_dominant)


def failure_class(approach: Approach) -> str | None:
    """Bucket *why* an approach was disqualified, so we can notice the same kind
    of mistake recurring. Returns None for a correct approach."""
    ev = approach.evidence
    if ev is None:
        return "ungrounded"
    if ev.error:
        if ev.error.startswith("static-safety"):
            return "unsafe-code"
        if "timeout" in ev.error:
            return "timeout"
        if "import" in ev.error or "syntax" in ev.error or "no top-level" in ev.error:
            return "wont-load"
        return "runtime-error"
    if ev.mutated_input:
        return "mutates-input"
    if not ev.correct:
        return "incorrect-output"
    return None


def compute(history_best_fitness: list[float],
            recombination_streak: int,
            all_approaches: list[Approach],
            last_round: list[Approach],
            cfg: Config) -> Signals:
    """Fold the run history into a Signals snapshot.

    `history_best_fitness` is the best-so-far fitness recorded at the end of each
    iteration (monotonic by construction). `recombination_streak` is the current
    run of consecutive rounds whose proposals were all recombinations.
    """
    s = Signals()

    # --- fitness / stall -------------------------------------------------- #
    if history_best_fitness:
        s.best_fitness = history_best_fitness[-1]
        window = cfg.stall_window
        if len(history_best_fitness) > window:
            past = history_best_fitness[-(window + 1)]
            s.fitness_gain_recent = history_best_fitness[-1] - past
            s.stalled = s.fitness_gain_recent < cfg.stall_epsilon
        else:
            # Not enough history to call a stall yet.
            s.fitness_gain_recent = history_best_fitness[-1] - history_best_fitness[0]

    # --- recombination / patching ----------------------------------------- #
    s.consecutive_recombinations = recombination_streak
    s.patching = recombination_streak >= cfg.patching_limit

    # --- recurring failure classes ---------------------------------------- #
    for a in all_approaches:
        fc = failure_class(a)
        if fc:
            s.failure_class_counts[fc] = s.failure_class_counts.get(fc, 0) + 1
    s.recurring_failures = sorted(
        fc for fc, n in s.failure_class_counts.items()
        if n >= cfg.recurring_failure_limit)

    # --- breadth ----------------------------------------------------------- #
    seen, seen_ok = [], []
    for a in all_approaches:
        c = a.algo_class.value
        if c not in seen:
            seen.append(c)
        if a.is_correct and c not in seen_ok:
            seen_ok.append(c)
    s.classes_tried, s.classes_correct = seen, seen_ok

    # --- novelty of the latest round -------------------------------------- #
    novs = [a.novelty for a in last_round if a.novelty is not None]
    if novs:
        s.last_round_mean_novelty = round(sum(novs) / len(novs), 4)

    return s
