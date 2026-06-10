"""The orchestration loop -- where the three roles, the grounding harness, the
bounded memory, and the protective mechanisms are wired into the spec's Core Loop.

This module is deliberately backend-agnostic: it is handed a `Roles` trio and a
`Domain` and runs the identical loop whether the roles are the deterministic
reference policy or live LLMs. The loop's shape IS the spec:

    seed -> [ propose -> GROUND -> score -> novelty-check -> ADVERSARY ->
             remember -> compute signals -> SYNTHESIZE decision ->
             human checkpoint -> act ] * N  -> accept / terminate

Every arrow that could let an unproven claim through is intercepted: code is only
ever believed after the grounding subprocess runs it; the Adversary's breaks only
count when confirmed; the Synthesizer's decision is checked against signals it
cannot fake; and a human can intervene at every synthesis checkpoint.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import behavior
from . import concepts
from . import signals as sig
from .config import Config
from .domain import Domain
from .fingerprint import assess_novelty, fingerprint
from .frontier import Frontier, FrontierPoint
from .grounding import (GroundingSpec, all_safety_violations, baseline_from,
                        diagnose, fitness, ground, measure_recall_on,
                        probe_generator, score)
from .memory import Memory
from .roles import Roles, RoundContext, untried_classes
from .selfplay import (CausalGuidance, SelfPlayLedger, SelfPlayMatch, UtilityFlag,
                       append_utility_history, summarize_utility_history)
from .surrogate import surrogate_quality
from .types import (Action, AdversaryFinding, AdversaryReport, Approach, Evidence,
                    Directive, Insight, Lesson, Origin, to_jsonable)


# Hook signature: (state_dict, proposed_directive) -> directive to actually use.
CheckpointHook = Callable[[dict, Directive], Directive]


def report_recalls(report: AdversaryReport) -> dict:
    """{workload_name: recall} extracted from a frontier Adversary report."""
    out = {}
    for f in report.findings:
        ce = f.counterexample or {}
        if "recall" in ce:
            out[f.hypothesis] = ce["recall"]
    return out


def report_workload_spec(report: AdversaryReport, name: str):
    for f in report.findings:
        ce = f.counterexample or {}
        if f.hypothesis == name and "workload" in ce:
            return ce["workload"]
    return None


@dataclass
class IterationRecord:
    iteration: int
    proposals: list[Approach]
    adversary_reports: list[AdversaryReport]
    signals: sig.Signals
    directive: Directive
    best_id: Optional[str]
    best_fitness: Optional[float]


@dataclass
class RunResult:
    domain: str
    backend: str
    status: str                       # accepted | terminated | exhausted
    seed: Approach
    best: Optional[Approach]
    iterations: list[IterationRecord] = field(default_factory=list)
    insights: list[Insight] = field(default_factory=list)
    lessons: list = field(default_factory=list)   # v1: conceptual ledger snapshot
    improvement: dict = field(default_factory=dict)
    vocab_provenance: dict = field(default_factory=dict)   # v5: human vs proposed
    v6_summary: dict = field(default_factory=dict)          # v6: utility + self-play
    usage: str = ""
    run_dir: str = ""


class Engine:
    def __init__(self, cfg: Config, domain: Domain, roles: Roles, *,
                 llm=None, on_checkpoint: Optional[CheckpointHook] = None,
                 logger: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.domain = domain
        self.roles = roles
        self.llm = llm
        self.on_checkpoint = on_checkpoint
        self.log = logger or (lambda m: None)
        self.memory = Memory(cfg)

        # Per-run COPY of the trusted suite, so adversary-discovered cases can be
        # appended (the harness's eyes grow) without mutating the domain globally.
        base = domain.grounding_spec
        self.spec = GroundingSpec(
            oracle_src=base.oracle_src,
            workload_gen_src=base.workload_gen_src,
            criteria_cases=list(base.criteria_cases),
            workloads=list(base.workloads),     # copy: adversary may append to it
            mem_workload=base.mem_workload,
            verdict_src=base.verdict_src,       # MUST carry the recall verdict
            driver_src=base.driver_src,         # v8: MUST carry the trusted driver
            driver_funcs=base.driver_funcs,     # v9: + its required-symbol contract
            driver_literal_ints=base.driver_literal_ints)

        self.baseline: dict = {}
        self.history_best: list[float] = []      # best-so-far fitness per round
        self.recomb_streak = 0
        # v1 state
        self.winners: dict = {}                  # workload -> Approach (prior round)
        self.recomb_stale_history: list[bool] = []   # per-round non-progress flag
        self.prev_best_time: Optional[float] = None  # best total time, prev round
        self.synthesis_attempts = 0              # SYNTHESIZE rounds spent (cap)
        self.workload_names = [w["name"] for w in self.spec.workloads]

        # v2 frontier state (only used when the domain opts in).
        self.uses_frontier = bool(getattr(domain, "uses_frontier", False))
        self.frontier: Frontier | None = None
        self.budget: float = 0.0                 # memory budget for "recommended"
        self.adv_workloads = list(getattr(domain, "adversary_workloads", []))
        self._metric_hardened_this_round = False
        self._round_new_workloads: list = []     # v4: suite additions this round

        # v5 OPEN VOCABULARY: the live registry the Builder/Adversary extend. Only
        # built when enabled AND the domain provides the hooks (topk has none, so it
        # transparently falls back to v1-v4 behaviour).
        self.open_vocab = bool(getattr(cfg, "open_vocabulary", False)
                               and getattr(domain, "seed_vocabulary", None)
                               and getattr(domain, "assemble_from_vocab", None))
        self.vocab = domain.seed_vocabulary() if self.open_vocab else None
        self._proposed_gen_specs: list = []      # admitted proposed-generator specs
        self._rejected_op_names: set = set()     # v9: rejected proposals, not retried
        # v10 P1: explicit incompleteness flags (never silently renormalise).
        self._utility_incomplete: dict = {}      # op_name -> [missing workload names]
        self._last_contract_error: str | None = None   # v10 P4: LLM feedback
        self._human_incomplete: list = []        # [{"label","missing"}] for humans
        self._n_proposed_ops = 0                 # admitted-this-run counters (caps)
        self._n_proposed_gens = 0
        # v6: proposal-QUALITY machinery -- causal guidance, self-play, utility.
        self.selfplay = SelfPlayLedger()
        self._causal = CausalGuidance()          # latest diagnosed failure
        self._human_frontier: list | None = None  # (recall, mem) Pareto, cached
        self._human_frontier_suite = -1          # suite size the cache was built on
        self._adaptive_frontier: list | None = None   # v10: + ARC/LeCaR references
        self._adaptive_frontier_suite = -1
        self.v6_summary: dict = {}               # surfaced in the report
        # v6: best policy CODE per proposed op, recorded as policies are processed so
        # utility can be computed even after bounded memory evicts the Approach.
        # v9 review: keep the best policy PER (op, capacity) -- selection between an
        # op's own capacity variants happens at finalize on the WEIGHTED BENCHMARK
        # (the judgment metric), not on the live-suite mean (which can rank them
        # differently). op_name -> {cap: (live_quality, mem, code)}
        self._op_policy_best: dict = {}

    # ------------------------------------------------------------------ #
    # Evaluation: the single choke point through which every approach must
    # pass before any role is allowed to reason about it.
    # ------------------------------------------------------------------ #
    def _evaluate(self, a: Approach, priors: list[Approach]) -> Approach:
        ev = ground(a.code, self.spec, self.cfg)
        self._apply_cost_axis(ev, a.code)        # v8: exact declared-capacity cost
        # v10 P4: remember a CONTRACT rejection verbatim so the next LLM Builder
        # prompt can correct the interface instead of repeating it (the v9 LLM run
        # wasted 3 of 4 rounds on the same gate failure). Cleared on the next
        # candidate that passes the gate.
        err = ev.error or ""
        if ("driver-contract" in err or "defines neither" in err
                or "no top-level solve" in err):
            self._last_contract_error = err
        elif ev.grounded:
            self._last_contract_error = None
        if self.baseline:
            score(ev, self.baseline)
        a.evidence = ev
        # In the frontier domain, fitness == marginal Pareto contribution and is
        # assigned in _admit_frontier (after the Adversary may downgrade quality).
        a.fitness = None if self.uses_frontier else fitness(ev, self.cfg)
        a.fingerprint = fingerprint(a.code, a.algo_class)
        a.novelty, a.is_recombination, _ = assess_novelty(
            a, priors, self.cfg.recombination_similarity)

        # v1: BEHAVIOURAL novelty -- measured, not asserted. A candidate whose
        # timing profile matches a prior is "the same thing in disguise" even if
        # its source is novel (sophisticated local patching).
        a.behavior = {"times": dict(ev.timings), "wins": [], "stale": False}
        if ev.correct and ev.timings:
            prior_evs = [p.evidence for p in priors
                         if p.evidence and p.evidence.timings and p.is_correct]
            bnov, stale = behavior.assess_behavioral_novelty(
                ev, prior_evs, self.cfg.behavioral_similarity)
            a.behavioral_novelty = bnov
            a.behavior["stale"] = stale

        # v1: for a SYNTHESIS, measure whether it actually beat its parents on
        # their own turf. (Time-based; meaningless in the frontier domain, where
        # frontier_gain is the synthesis reward -- so skip it there.)
        if (not self.uses_frontier and a.origin == Origin.SYNTHESIS
                and ev.correct and a.parent_ids):
            parents = [p for p in (self.memory.get(i) for i in a.parent_ids) if p]
            a.synthesis_gain = behavior.synthesis_gain(ev, parents)
        return a

    # ------------------------------------------------------------------ #
    # v2 frontier admission + adversarial metric attack
    # ------------------------------------------------------------------ #
    def _suite_quality(self, a: Approach) -> float:
        """The FRONTIER's quality axis. v10 P4: for a domain with a declared
        benchmark_suite this is the WEIGHTED BENCHMARK CR -- the same metric the
        utility verdict and the recommended point use, so the loop optimises what
        the verdict judges (v9's loop optimised the live-suite mean while the
        verdict judged the benchmark; the run's best and the headline could
        disagree). The adversary's hardened workloads remain PRESSURE/diagnosis
        (exposure, generator credit, causal guidance) but are not the quality
        denominator. A benchmark workload somehow still unmeasured after
        _ensure_full_coverage scores 0.0 -- penalised, never renormalised (P1).
        Legacy domains keep the live-suite mean."""
        qbw = a.evidence.quality_by_workload
        bs = getattr(self.domain, "benchmark_suite", None)
        if bs:
            rec = {w["name"]: qbw.get(w["name"], 0.0) for w in bs}
            return self._weighted_quality(rec, bs)
        main = [w["name"] for w in self.spec.workloads]
        qs = [qbw[w] for w in main if w in qbw]
        return round(sum(qs) / len(qs), 4) if qs else 0.0

    def _ensure_full_coverage(self, a: Approach) -> None:
        """v4 comparability: guarantee `a` carries a recall measurement for EVERY
        workload in the live suite before it is scored -- including workloads a
        SIBLING candidate's adversary pass added this round. The deterministic
        Adversary throws an identical workload set at every candidate, but the LLM
        Adversary composes PER-CANDIDATE attacks, so a candidate can be missing a
        sibling's hardening workload. Filling the gap here makes _admit_frontier's
        mean a true full-suite mean for BOTH backends (no-op when nothing missing,
        so the deterministic path pays nothing).

        v10 P4: also guarantees coverage of the declared benchmark_suite (the
        frontier's quality axis)."""
        if not (a.is_correct and a.evidence):
            return
        have = a.evidence.quality_by_workload or {}
        missing = [w for w in self.spec.workloads if w["name"] not in have]
        for w in (getattr(self.domain, "benchmark_suite", None) or []):
            if w["name"] not in have and all(m["name"] != w["name"]
                                             for m in missing):
                missing.append(w)
        if missing:
            # v5: measure against the LIVE spec (carries proposed-generator source)
            # so a `prop:<name>` workload actually generates.
            rec = measure_recall_on(a.code, missing, self.spec, self.cfg)
            a.evidence.quality_by_workload.update(rec)

    def _admit_frontier(self, a: Approach) -> None:
        """Score a candidate by its marginal Pareto-hypervolume contribution and
        fold it into the frontier. This is the v2 reward: 0 for a dominated/
        recombinant point, > 0 for one that reaches a new (recall, memory) region."""
        if not (self.frontier and a.is_correct and a.evidence):
            a.fitness = a.fitness or 0.0
            return
        # v4 consistency: score this candidate over the LIVE suite (which the
        # Adversary may have hardened this round, and on which the frontier members
        # were just re-scored), with full coverage forced first so the candidate and
        # the incumbents it competes against are measured on EXACTLY the same metric
        # -- closing v3's (and the LLM path's per-candidate) comparability gap.
        self._ensure_full_coverage(a)
        a.evidence.quality = self._suite_quality(a)
        point = FrontierPoint(a.id, a.evidence.quality,
                              a.evidence.peak_memory_bytes, a.evidence.total_time())
        gain = self.frontier.gain(point)
        a.frontier_gain = gain
        a.fitness = gain
        self.frontier.add(point)

    def _rescore_frontier(self, new_wls: list) -> None:
        """v4: when the Adversary HARDENS the suite with a new attack/generator,
        re-measure every existing frontier member's recall on the new workloads,
        recompute its quality over the FULL current suite, and rebuild the
        frontier. This keeps the frontier HONEST -- all members are compared on the
        same (growing) metric, fixing v3's cross-round comparability gap."""
        if not (self.uses_frontier and self.frontier and new_wls):
            return
        rebuilt: list[FrontierPoint] = []
        for p in self.frontier.points:
            a = self.memory.get(p.approach_id)
            if a and a.evidence:
                rec = measure_recall_on(a.code, new_wls, self.spec, self.cfg)
                a.evidence.quality_by_workload.update(rec)
                a.evidence.quality = self._suite_quality(a)
                if a.is_correct:
                    rebuilt.append(FrontierPoint(
                        a.id, a.evidence.quality, a.evidence.peak_memory_bytes,
                        a.evidence.total_time()))
            else:
                # The backing approach was evicted from bounded memory: KEEP the
                # existing point rather than silently dropping a genuine Pareto
                # member. (With protected_ids registered this should not happen --
                # this is belt-and-suspenders so the frontier never shrinks.)
                rebuilt.append(p)
        self.frontier.points = []
        for p in rebuilt:
            self.frontier.add(p)
        self.log(f"    frontier RE-SCORED on {len(new_wls)} new workload(s) -> "
                 f"{self.frontier.summary(self.budget)}")

    def _adversary_metric_attack(self, a: Approach, ctx: RoundContext,
                                 reports: list[AdversaryReport]) -> None:
        """v2 Adversary: attack the METRIC. Measure the candidate's recall on
        hostile workloads; if recall COLLAPSES (looked good on easy streams, fails
        a hard one), expose it -- downgrade its quality AND permanently harden the
        evaluation suite with that workload (so future candidates face it too)."""
        if not a.is_correct or not a.evidence:
            return
        report = self.roles.adversary.attack(a, ctx)   # proposes hostile workloads
        reports.append(report)
        recalls = report_recalls(report)               # {workload_name: recall}
        if not recalls:
            return
        # Use the candidate's BEST-CASE base recall, computed BEFORE folding in the
        # adversary measurements: gaming/overfitting means it CAN look good on some
        # easy stream but COLLAPSES on a hostile one. (Mean would hide this for
        # methods that are uniformly mediocre.)
        base_q = a.evidence.quality_by_workload
        base_vals = [base_q[w["name"]]
                     for w in self.domain.grounding_spec.workloads
                     if w["name"] in base_q]
        base = max(base_vals) if base_vals else a.evidence.quality
        # v4 consistency: fold EVERY measured adversary recall into the candidate's
        # record REGARDLESS of whether it was exposed. Admission (_admit_frontier)
        # recomputes quality over the LIVE suite, so each candidate must already
        # carry its measurements on every adversary workload the round produced --
        # otherwise an un-exposed candidate would be scored on a different (easier)
        # suite than the re-scored frontier members it is ranked against.
        a.evidence.quality_by_workload.update(recalls)
        exposed = []
        for wl_name, r in recalls.items():
            if (base - r) > self.cfg.metric_collapse_margin \
                    and r < self.cfg.metric_collapse_floor:
                exposed.append((wl_name, r))
        # v5: credit a PROPOSED generator that collapsed this candidate (provenance:
        # "the Adversary's invented generator did real work").
        if self.open_vocab and self.vocab:
            for wl_name, _ in exposed:
                g = self.vocab.generators.get(wl_name)
                if g and g.origin.is_proposed:
                    g.exposed_a_candidate = True
        if not exposed:
            return
        # Harden the suite: add the exposing workloads to the eval set so EVERY
        # future candidate is measured on them (the metric gets harder over time).
        hardened = []
        existing = {w["name"] for w in self.spec.workloads}
        for wl_name, _ in exposed:
            spec_wl = report_workload_spec(report, wl_name)
            if spec_wl and wl_name not in existing:
                self.spec.workloads.append(spec_wl)
                self.workload_names.append(wl_name)
                self._round_new_workloads.append(spec_wl)   # -> triggers re-score
                hardened.append(wl_name)
        # Downgrade the exposed candidate immediately: recompute its mean recall
        # over the live suite (the recalls were already folded in above). This gives
        # the log/insight an accurate quality now; _admit_frontier recomputes again
        # at admission in case a LATER candidate hardens the suite further.
        main = [w["name"] for w in self.spec.workloads]
        qs = [a.evidence.quality_by_workload[w] for w in main
              if w in a.evidence.quality_by_workload]
        a.evidence.quality = round(sum(qs) / len(qs), 4) if qs else 0.0
        a.evidence.failures.append({"case": "adversary-metric",
                                    "exposed_on": [w for w, _ in exposed]})
        self.memory.add_insight(Insight(
            iteration=ctx.iteration, kind="failure",
            text=(f"Adversary exposed {a.id} <{a.algo_class.value}> gaming the "
                  f"metric: recall collapsed on {[w for w, _ in exposed]}"
                  + (f"; suite hardened with {hardened}" if hardened else "")),
            related=[a.id]))
        self._metric_hardened_this_round = self._metric_hardened_this_round or bool(
            hardened)
        self.log(f"    adversary EXPOSED {a.id}: recall collapse on "
                 f"{[w for w, _ in exposed]} -> quality downgraded"
                 + (f", suite hardened {hardened}" if hardened else ""))

    # ------------------------------------------------------------------ #
    # v5 OPEN VOCABULARY: propose -> AST-screen -> ground -> admit
    # ------------------------------------------------------------------ #
    def _guidance(self) -> dict:
        """What the proposers should target. v6: the PRIMARY signal is CAUSAL --
        the concrete workload the current best policy fails on + the true heavy
        hitters it missed. Keyword `weak` tags + the frontier gap are SECONDARY."""
        g: dict = {"frontier_gap": 0.0, "weak": set()}
        if self.uses_frontier and self.frontier:
            g["frontier_gap"] = round(
                self.frontier.quality_gap_above_budget(self.budget), 3)
        text = " ".join(i.text for i in self.memory.insights).lower()
        try:
            text += " " + self.memory.lessons.digest(20).lower()
        except Exception:
            pass
        tagmap = {
            "late": ("late", "param_l", "freq_late", "front"),
            "overflow": ("overflow", "skew", "heavy", "_h400", "_h120"),
            "collision": ("collision", "spike", "mixture", "decoy", "comeback"),
        }
        for tag, needles in tagmap.items():
            if any(nd in text for nd in needles):
                g["weak"].add(tag)
        # v6: the diagnosed failure (built lazily before op proposal) + the
        # Adversary's most-recent generators (the self-play targets).
        g["contract_error"] = self._last_contract_error
        g["causal"] = self._causal
        g["recent_generators"] = [s["name"] for s in self._proposed_gen_specs[-3:]]
        return g

    # ------------------------------------------------------------------ #
    # v6 PROPOSAL QUALITY: causal guidance, self-play, utility flags
    # ------------------------------------------------------------------ #
    def _build_causal_guidance(self) -> CausalGuidance:
        """Diagnose the CURRENT best policy's worst workload -- the concrete failure
        the proposer should target (which true heavy hitters it missed). This is the
        v6 primary steering signal; it replaces keyword guessing."""
        g = CausalGuidance()
        best = self.memory.best
        if not (best and best.is_correct and best.evidence):
            return g
        qbw = best.evidence.quality_by_workload or {}
        suite = [w for w in self.spec.workloads if w["name"] in qbw]
        if not suite:
            return g
        worst = min(suite, key=lambda w: qbw.get(w["name"], 1.0))
        r = qbw.get(worst["name"], 1.0)
        if r >= 0.999:
            return g                              # the best policy fails nowhere
        diag = diagnose(best.code, worst, self.spec, self.cfg)
        g.workload = worst["name"]
        g.dist = worst.get("dist", "")
        g.recall = round(r, 3)
        g.expected = diag.get("expected", [])
        g.got = diag.get("got", [])
        g.missed = diag.get("missed", [])
        # A FACTUAL, dist-conditional hypothesis (the old text asserted "transient
        # inflation" universally, which is wrong on late/tie-heavy streams).
        dist = g.dist.lower()
        if any(t in dist for t in ("decoy", "spike", "comeback", "mixture",
                                   "prop:", "collision")):
            g.hypothesis = ("bursty/decoy stream: transient items are likely "
                            "inflated into the top-k -- favour threshold-pruning / "
                            "no-inflation ops.")
        elif any(t in dist for t in ("param", "late", "freq_late", "front")):
            g.hypothesis = ("late-arrival / many-heavy stream: genuine hitters may "
                            "be under-counted or beyond capacity, and SOME of this "
                            "divergence is a tie-break boundary effect, not a "
                            "recoverable miss.")
        else:
            g.hypothesis = ("the policy's top-k diverges from the oracle's here "
                            "(may include tie-boundary items).")
        return g

    def _utility_workloads(self) -> list:
        """v8: a STABLE, comprehensive suite for the UTILITY flag -- the domain's
        declared natural workloads + its STATIC adversary workloads. Measuring
        beats/matches/below-human here (rather than on the run-variable hardened
        suite) makes the verdict REPRODUCIBLE: it no longer depends on which attacks
        the Adversary happened to admit this run -- a fragility the v8 exact-cost
        axis exposed (the run-variable count was not reproducible).

        v9: when the domain declares a STANDARDISED WEIGHTED benchmark_suite, THAT
        is the utility suite (each workload carries a `weight`; the verdict is the
        weighted mean). Otherwise the v8 fallback: natural + static adversary
        workloads at equal weight."""
        bs = getattr(self.domain, "benchmark_suite", None)
        if bs:
            return [dict(w) for w in bs]
        base = list(self.domain.grounding_spec.workloads)
        seen = {w["name"] for w in base}
        for w in (self.domain.adversary_workloads or []):
            if isinstance(w, dict) and w.get("name") not in seen:
                base.append(w)
                seen.add(w["name"])
        return base

    def _measure_suite_complete(self, code: str, suite: list, spec=None) -> tuple:
        """v10 P1 COMPLETENESS GATE: a weighted benchmark score is valid only with a
        measurement for EVERY workload in the declared suite. On a missing
        measurement: retry the whole suite once; if still incomplete, score the
        missing workloads 0.0 (penalised) and return their names as an explicit
        `incomplete` flag -- NEVER silently renormalise over the measured subset.
        Returns (rec_with_full_coverage, missing_names)."""
        spec = spec if spec is not None else self.spec
        names = [w["name"] for w in suite]
        rec = measure_recall_on(code, suite, spec, self.cfg)
        missing = [n for n in names if n not in rec]
        if missing:
            retry = measure_recall_on(code, suite, spec, self.cfg)
            merged = dict(rec)
            merged.update(retry)
            rec = merged
            missing = [n for n in names if n not in rec]
        for n in missing:
            rec[n] = 0.0
        return rec, missing

    @staticmethod
    def _weighted_quality(rec: dict, suite: list) -> float:
        """Weighted mean quality over a suite ({name: quality} vs declared weights;
        a workload that was not measured contributes neither value nor weight)."""
        num = den = 0.0
        for w in suite:
            nm = w.get("name")
            if nm in rec:
                wt = float(w.get("weight", 1.0))
                num += rec[nm] * wt
                den += wt
        return round(num / den, 4) if den else 0.0

    def _human_frontier_points(self) -> list:
        """The HUMAN-only Pareto frontier (best recall per memory) on the CURRENT
        suite, the baseline the utility flag is measured against. Cached per suite
        size (recomputed only when the suite hardens)."""
        if not getattr(self.domain, "human_baseline_policies", None):
            return []
        suite = self._utility_workloads()           # v8: stable, reproducible suite
        if (self._human_frontier is not None
                and self._human_frontier_suite == len(suite)):
            return self._human_frontier
        pts = []
        for label, code in self.domain.human_baseline_policies(self.vocab):
            # v10 P1: the completeness gate applies SYMMETRICALLY to humans -- an
            # incomplete human measurement (after one retry) is penalised the same
            # way AND flagged at run level, because a silently weakened human
            # baseline corrupts the verdict in the candidates' favour.
            rec, missing = self._measure_suite_complete(code, suite)
            if missing:
                self._human_incomplete.append(
                    {"label": label, "missing": missing})
                self.log(f"    !! human baseline '{label}' INCOMPLETE after retry: "
                         f"{missing} scored 0.0 -- verdict flagged")
            q = self._weighted_quality(rec, suite)      # v9: weighted benchmark mean
            ev = ground(code, self._op_probe_spec(), self.cfg)
            self._apply_cost_axis(ev, code)      # v8: same exact cost basis as candidates
            mem = ev.peak_memory_bytes if ev.correct else 0
            if mem:
                pts.append((q, mem))
        # Pareto front: keep points not dominated (>= recall at <= memory) by another.
        front = [p for p in pts
                 if not any(o is not p and o[0] >= p[0] and o[1] <= p[1]
                            and o != p for o in pts)]
        self._human_frontier = front
        self._human_frontier_suite = len(suite)
        return front

    def _adaptive_frontier_points(self) -> list:
        """v10 P3: the ADAPTIVE-tier human frontier = static human points + the
        trusted reference policies (ARC / LeCaR), each measured through the normal
        harness (driver_src swapped in) under the same completeness gate, with cost
        from the domain's declared ghost-charging rule. Cached like the static
        frontier. Falls back to the static frontier when the domain declares no
        adaptive baselines."""
        hook = getattr(self.domain, "adaptive_baselines", None)
        if not hook:
            return self._human_frontier_points()
        suite = self._utility_workloads()
        if (self._adaptive_frontier is not None
                and self._adaptive_frontier_suite == len(suite)):
            return self._adaptive_frontier
        base_k = self.spec.workloads[0]["k"] if self.spec.workloads else 10
        pts = list(self._human_frontier_points())
        for ref in hook():
            spec_ref = dataclasses.replace(self.spec, driver_src=ref["driver_src"])
            rec, missing = self._measure_suite_complete(ref["stub"], suite,
                                                        spec=spec_ref)
            if missing:
                self._human_incomplete.append(
                    {"label": ref["label"], "missing": missing})
                self.log(f"    !! adaptive baseline '{ref['label']}' INCOMPLETE "
                         f"after retry: {missing} scored 0.0 -- verdict flagged")
            q = self._weighted_quality(rec, suite)
            mem = int(round(ref["cost_units"] * base_k * self._BYTES_PER_ENTRY))
            pts.append((q, mem))
        front = [p for p in pts
                 if not any(o is not p and o[0] >= p[0] and o[1] <= p[1]
                            and o != p for o in pts)]
        self._adaptive_frontier = front
        self._adaptive_frontier_suite = len(suite)
        return front

    def _classify_vs(self, front: list, recall: float, memory: int) -> tuple:
        """Grade (recall, memory) against a (quality, cost) frontier at <= memory."""
        reachable = [q for (q, m) in front if m <= memory]
        if reachable:
            human = max(reachable)              # best human at <= this memory
        elif front:
            # No human reaches this (low) memory -- compare to the CHEAPEST human
            # point's recall (the nearest real comparator), NOT a -1.0 sentinel that
            # would inflate the margin into a meaningless recall+1.
            human = min(front, key=lambda p: p[1])[0]
        else:
            human = 0.0
        margin = round(recall - human, 4)
        eps = self.cfg.utility_eps
        if margin > eps:
            return (UtilityFlag.BEATS_HUMAN, margin)
        if margin >= -eps:
            return (UtilityFlag.MATCHES_HUMAN, margin)
        return (UtilityFlag.BELOW_HUMAN, margin)

    def _classify_utility(self, recall: float, memory: int) -> tuple:
        """STATIC tier (v6 semantics preserved for history continuity):
        beats/matches/below vs the static-classic human frontier."""
        return self._classify_vs(self._human_frontier_points(), recall, memory)

    def _classify_utility_adaptive(self, recall: float, memory: int) -> tuple:
        """v10 P3 ADAPTIVE tier -- the headline: vs the best human INCLUDING the
        ARC/LeCaR references at <= equal declared cost."""
        return self._classify_vs(self._adaptive_frontier_points(), recall, memory)

    @staticmethod
    def _policy_cap(code: str) -> int:
        # Read the multiplier robustly: the `# cmul=N`/`# mmul=N` marker, OR the v8
        # `CMUL = N` assignment (which SURVIVES ast.unparse when mutation strips the
        # comment; the optional `: <ann>` arm covers the gate-legal annotated form
        # `CMUL: int = N`), OR the legacy `max(N * k` source idiom.
        m = (re.search(r"#\s*[cm]mul=(\d+)", code)
             or re.search(r"CMUL\s*(?::[^=\n]*)?=\s*(\d+)", code)
             or re.search(r"max\((\d+) ?\* ?k", code))
        return int(m.group(1)) if m else 16

    # v8: when a domain declares a trusted driver, the cache cost axis is the EXACT
    # declared capacity (CMUL * k * a fixed per-entry size), a deterministic integer
    # function of the policy -- no tracemalloc measurement noise, so two policies at
    # the same CMUL are exactly cost-comparable. ~160 B/entry matches the v7
    # tracemalloc scale (cmul 2 -> ~3KB, cmul 4 -> ~6KB) so budgets carry over.
    _BYTES_PER_ENTRY = 160

    def _policy_cost(self, code: str) -> int:
        base_k = self.spec.workloads[0]["k"] if self.spec.workloads else 10
        return self._policy_cap(code) * base_k * self._BYTES_PER_ENTRY

    # v10 P2 backstop: a candidate whose MEASURED tracemalloc peak exceeds this
    # multiple of its DECLARED cost is rejected. The primary statelessness
    # mechanism is the AST module-shape screen; this catches anything that slips
    # past it. Calibration: honest template policies measure 1.1-1.4x declared on
    # the largest workload (n=6000); 8x is far above any honest policy and far
    # below a meaningful stash (a 6000-key shadow dict measures >> 8x).
    _COST_XCHECK_MULT = 8

    def _apply_cost_axis(self, ev, code: str) -> None:
        """For trusted-driver domains, replace the (now-deterministic) tracemalloc
        peak with the EXACT declared-capacity cost, so the frontier, the budget, and
        the utility flag all use one clean, noise-free cost basis. Before replacing,
        cross-check the measured peak against the declared cost (backstop above)."""
        if getattr(self.spec, "driver_src", ""):
            declared = self._policy_cost(code)
            measured = ev.peak_memory_bytes
            if (ev.correct and measured
                    and measured > self._COST_XCHECK_MULT * declared):
                ev.correct = False
                ev.error = (((ev.error + " | ") if ev.error else "")
                            + f"cost-cross-check: measured peak {measured}B > "
                              f"{self._COST_XCHECK_MULT}x declared {declared}B "
                              f"(undeclared state?)")
            ev.peak_memory_bytes = declared

    def _record_op_policies(self, processed: list) -> None:
        """v6: stash the best policy CODE per proposed op as it is processed, so its
        utility can be computed at the end even after bounded memory evicts the
        Approach (non-frontier proposed-op policies are routinely pruned)."""
        if not (self.open_vocab and self.vocab):
            return
        for a in processed:
            if not (a.is_correct and a.evidence and getattr(a, "ops_used", None)):
                continue
            for nm in a.ops_used:
                e = self.vocab.op_entry(nm)
                if not (e and e.origin.is_proposed):
                    continue
                cap = self._policy_cap(a.code)
                slot = self._op_policy_best.setdefault(nm, {})
                cur = slot.get(cap)
                q = a.evidence.quality
                if cur is None or q > cur[0]:
                    slot[cap] = (q, a.evidence.peak_memory_bytes, a.code)

    def _compose_uncovered_ops(self) -> None:
        """v8: ensure every admitted proposed op has a grounded policy in
        `_op_policy_best`. Composes a policy (via the domain grammar) for any op that
        the Builder never composed -- closing the propose->compose loop so an invented
        op reaches the frontier-measured world instead of reading 'unknown'."""
        assembler = getattr(self.domain, "assemble_from_vocab", None)
        if not assembler:
            return
        covered = set(self._op_policy_best)
        uncovered = {e.name for e in self.vocab.proposed_ops()} - covered
        if not uncovered:
            return
        for (_label, code, ops_used) in assembler(self.vocab):
            target = [o for o in ops_used if o in uncovered]
            if not target:
                continue
            # v10 Phase-0 fix: record EVERY capacity variant of an uncovered op
            # (the old early-discard kept only the first, so a finalize-composed op
            # was judged c2-only -- and a load-induced grounding timeout silently
            # flipped which capacity represented it, the observed 6<->7 instability).
            # A failed grounding is RETRIED once; a variant still failing after the
            # retry is recorded in the op's incomplete flag instead of silently
            # changing the representative.
            ev = ground(code, self._op_probe_spec(), self.cfg)
            if not ev.correct:
                ev = ground(code, self._op_probe_spec(), self.cfg)     # one retry
            self._apply_cost_axis(ev, code)
            cap = self._policy_cap(code)
            if not ev.correct:
                for o in target:
                    self._utility_incomplete.setdefault(o, []).append(
                        f"variant-c{cap}-unground")
                continue
            for o in target:
                self._op_policy_best.setdefault(o, {})[cap] = (
                    0.0, ev.peak_memory_bytes, code)
        # only NOW mark ops covered (after every variant had its chance).
        # ops with no surviving variant at all stay uncovered -> 'unknown' would
        # surface, which the e2e tests treat as a failure -- loudness is the point.

    def _finalize_v6(self) -> None:
        """End-of-run: label every proposed op with its utility vs the human
        frontier, run the self-play tournament (proposed ops vs the human default on
        the Adversary's generators), append the cross-run history, and summarise."""
        if not (self.open_vocab and self.vocab and self.frontier):
            return
        # v8: CLOSE THE PROPOSE->COMPOSE LOOP. The deterministic Builder composes
        # admitted ops via the grammar (assemble_from_vocab), but the LLM Builder
        # writes FREE-FORM code and does not -- so an LLM-invented op would never get
        # a grounded policy and would read 'unknown'. Here the ENGINE composes a
        # policy for every admitted proposed op that lacks one, so EVERY invention --
        # whoever proposed it -- earns an honest utility flag, not silence.
        self._compose_uncovered_ops()
        # 1) Utility per proposed op: RE-MEASURE its best policy on the FINAL
        #    hardened suite (the recorded quality may be stale, and the Approach may
        #    have been evicted), then classify vs the human frontier.
        util_suite = self._utility_workloads()     # v8: stable suite (same as humans)
        best_policy: dict = {}     # op_name -> (recall, memory, code)
        for nm, by_cap in self._op_policy_best.items():
            # v9: measure EACH capacity variant on the weighted benchmark and keep
            # the best -- so the policy that REPRESENTS the op is selected by the
            # same metric it is judged on (the live-suite mean can rank variants
            # differently, mislabelling an op whose other variant beats).
            # v10 P1: each variant's suite must be COMPLETE (retry once, then
            # penalise missing as 0.0 + flag) -- so the chosen representative can
            # never flip because a measurement quietly went missing.
            best = None
            best_missing: list = []
            best_rec: dict = {}
            for _cap, (_q_old, mem, code) in sorted(by_cap.items()):
                rec, missing = self._measure_suite_complete(code, util_suite)
                q = self._weighted_quality(rec, util_suite)
                if best is None or q > best[0]:
                    best = (q, mem, code)
                    best_missing = missing
                    best_rec = rec
            if best is None:
                continue
            # v10 P4 COHERENCE: a finalize-graded policy must not be invisible to
            # the frontier (the INCOHERENT-RESULT assertion caught exactly this
            # seam on its first test run: catch-all-composed variants were graded
            # by the verdict but never offered to the frontier). Route any
            # frontier-EXTENDING policy through the SAME admission machinery --
            # full coverage + marginal-hypervolume reward, no weakened gates
            # (a dominated point would earn zero gain and change nothing).
            if (self.frontier is not None
                    and self.frontier.gain(FrontierPoint(
                        f"final_{nm}", best[0], best[1], 0.0)) > 0):
                fa = Approach(
                    id=f"final_{nm}", code=best[2],
                    rationale=(f"finalize composition of admitted op '{nm}' "
                               "(catch-all path, admitted through the standard "
                               "frontier machinery)"),
                    algo_class=self.domain.classify(best[2]),
                    origin=Origin.BUILDER, created_iter=0, ops_used=[nm])
                fa.evidence = Evidence(grounded=True, correct=True)
                fa.evidence.quality_by_workload = dict(best_rec)
                fa.evidence.peak_memory_bytes = best[1]
                self._admit_frontier(fa)
                if fa.frontier_gain and fa.frontier_gain > 0:
                    self.memory.protected_ids = {p.approach_id
                                                 for p in self.frontier.points}
                    self.memory.add_approach(fa)
                    e0 = self.vocab.op_entry(nm)
                    if e0:
                        e0.used_on_frontier = True
            if best_missing:
                self._utility_incomplete[nm] = best_missing
                self.log(f"    !! op '{nm}' utility measurement INCOMPLETE after "
                         f"retry: {best_missing} scored 0.0")
            best_policy[nm] = best
            q, mem, code = best
            flag, margin = self._classify_utility(q, mem)
            aflag, amargin = self._classify_utility_adaptive(q, mem)
            e = self.vocab.op_entry(nm)
            if e:
                e.utility_adaptive = aflag.value
                e.utility_adaptive_margin = amargin
                e.utility = flag.value
                e.utility_margin = margin

        # 2) Self-play tournament (v7): each proposed op's best policy vs the BEST
        #    HUMAN policy ON EACH GENERATOR -- not a fixed incumbent default. A win
        #    now means "beats the STRONGEST human on that attack", matching the
        #    win_rate's implied semantics (v6's fixed-vhalf baseline overstated wins).
        net_positive = False
        if self.cfg.self_play and getattr(self.domain,
                                          "human_baseline_policies", None):
            gens = list(self._proposed_gen_specs)
            # Group human policies by capacity so the comparison is at the SAME
            # memory budget (a bigger-cache human is not a fair baseline for a
            # smaller-cache proposed op).
            human_by_cap: dict = {}
            for (_l, hc) in self.domain.human_baseline_policies(self.vocab):
                human_by_cap.setdefault(self._policy_cap(hc), []).append(hc)
            best_human_on: dict = {}            # (generator, cap) -> best human recall
            for nm, (_q, _mem, code) in best_policy.items():
                cap = self._policy_cap(code)
                # Fair baseline: humans at the SAME cap; else the nearest human cap
                # NOT EXCEEDING the op's cap (never a bigger-cache human, which would
                # make the matchup unfairly hard). Only if none exists do we widen.
                pool = human_by_cap.get(cap)
                if not pool:
                    lower = [c for c in human_by_cap if c <= cap]
                    if lower:
                        pool = human_by_cap[max(lower)]
                    else:
                        pool = human_by_cap[min(human_by_cap)] if human_by_cap else []
                for g in gens:
                    key = (g["name"], cap)
                    if key not in best_human_on:
                        best = 0.0
                        for hc in pool:
                            r = measure_recall_on(hc, [g], self.spec,
                                                  self.cfg).get(g["name"])
                            if r is not None and r > best:
                                best = r
                        best_human_on[key] = round(best, 3)
                    op_r = measure_recall_on(code, [g], self.spec,
                                             self.cfg).get(g["name"])
                    if op_r is None:
                        continue
                    self.selfplay.add(SelfPlayMatch(
                        iteration=0, op_name=nm, target_generator=g["name"],
                        op_recall=round(op_r, 3),
                        baseline_recall=best_human_on[key]))
            # v7: is self-play producing NET-POSITIVE FRONTIER MOVEMENT (an op that
            # wins self-play AND reached the frontier), not just robustness?
            sp_winners = {m.op_name for m in self.selfplay.matches if m.won}
            on_frontier = {e.name for e in self.vocab.proposed_ops()
                           if e.used_on_frontier}
            net_positive = bool(sp_winners & on_frontier)

        # 3) Cross-run utility history + summary. The run id is content-independent
        #    (timestamp + pid) so CONCURRENT runs on the shared history file never
        #    collide -- no racy read-then-write of the run count.
        prov = self.vocab.provenance_summary()
        run_id = f"run-{time.strftime('%Y%m%d_%H%M%S')}-{os.getpid()}"
        records = []
        for nm in best_policy:
            e = self.vocab.op_entry(nm)
            records.append({"run": run_id, "op": nm,
                            "origin": e.origin.value if e else "?",
                            "utility": e.utility if e else "unknown",
                            "margin": e.utility_margin if e else 0.0,
                            # v10 P1: incompleteness is explicit in the history.
                            "incomplete": self._utility_incomplete.get(nm, []),
                            # v10 P3: records are VERSIONED by baseline tier.
                            "baseline": "static"})
            records.append({"run": run_id, "op": nm,
                            "origin": e.origin.value if e else "?",
                            "utility": e.utility_adaptive if e else "unknown",
                            "margin": e.utility_adaptive_margin if e else 0.0,
                            "incomplete": self._utility_incomplete.get(nm, []),
                            "baseline": "adaptive"})
        if records:
            try:
                append_utility_history(self.cfg.utility_history_path, records)
            except Exception:                              # noqa: BLE001
                pass
        # v10 P4 CONSISTENCY ASSERTION: if any op grades beats_*_human, the final
        # frontier must contain a policy whose weighted benchmark CR is >= that
        # op's best policy CR. With the frontier on the benchmark-CR axis this is
        # vacuously true; if it ever fires, something upstream is broken -- fail
        # LOUDLY with both numbers, never adjust admission to make it pass.
        incoherent = None
        beating = [nm for nm in best_policy
                   if (self.vocab.op_entry(nm) is not None
                       and (self.vocab.op_entry(nm).utility == "beats_human"
                            or self.vocab.op_entry(nm).utility_adaptive
                            == "beats_human"))]
        if beating and self.frontier and self.frontier.points:
            op_max = max(best_policy[nm][0] for nm in beating)
            fr_max = max(p.quality for p in self.frontier.points)
            if fr_max + 1e-9 < op_max:
                incoherent = {"frontier_max_cr": round(fr_max, 4),
                              "beating_op_max_cr": round(op_max, 4)}
                self.log(f"!! INCOHERENT-RESULT: a beats_human op scores CR "
                         f"{op_max:.4f} but the frontier's best is {fr_max:.4f} "
                         f"-- investigate; do NOT adjust admission to mask this.")

        self.v6_summary = {
            "incoherent": incoherent,
            "utility": {"beats_human": prov["beats_human"],
                        "matches_human": prov["matches_human"],
                        "below_human": prov["below_human"]},
            # v10 P3: the HEADLINE tier -- vs best human incl. ARC/LeCaR.
            "utility_adaptive": {
                "beats_human": prov["beats_adaptive_human"],
                "matches_human": prov["matches_adaptive_human"],
                "below_human": prov["below_adaptive_human"]},
            "op_utility_adaptive": prov["proposed_op_utility_adaptive"],
            "op_utility": prov["proposed_op_utility"],
            "self_play": self.selfplay.summary(),
            "self_play_net_positive": net_positive,
            "history": summarize_utility_history(self.cfg.utility_history_path),
            # v10 P1: explicit completeness flags (empty == fully measured).
            "incomplete": dict(self._utility_incomplete),
            "human_baseline_incomplete": list(self._human_incomplete),
        }

    def _op_probe_spec(self) -> GroundingSpec:
        """A cheap CORRECTNESS-focused spec for grounding an op probe: the live
        machinery (oracle / gen / verdict + trusted criteria) but a SINGLE small
        overflow-inducing workload. So op admission turns on well-formedness, not on
        whether the full hardened-suite benchmark beats the timeout (which made a
        correct-but-slow op's admission nondeterministic)."""
        return GroundingSpec(
            oracle_src=self.spec.oracle_src,
            workload_gen_src=self.spec.workload_gen_src,
            criteria_cases=self.spec.criteria_cases,
            workloads=[{"name": "op_probe", "n": 4000, "k": 10, "dist": "zipf"}],
            mem_workload=self.spec.mem_workload,
            verdict_src=self.spec.verdict_src,
            driver_src=self.spec.driver_src,            # v8: carry the trusted driver
            driver_funcs=self.spec.driver_funcs,
            driver_literal_ints=self.spec.driver_literal_ints)

    def _surrogate_workload(self) -> dict | None:
        """The workload to probe in the surrogate: the one where the human default is
        WEAKEST -- i.e. the regime with the most HEADROOM, where a high-value op is
        meant to shine. Probing the default-FRIENDLY workload would wrongly reject
        aging/forgetting ops that target the hard regimes (the v7 review's finding).
        Computed once per run."""
        if getattr(self, "_surrogate_wl", "unset") != "unset":
            return self._surrogate_wl
        self._surrogate_wl = None
        if not self.spec.workloads:
            return None
        ref = getattr(self.domain, "human_default_policy", None)
        ref_code = None
        try:
            ref_code = ref(self.vocab, 2) if ref else None
        except Exception:                            # noqa: BLE001
            ref_code = None
        worst, worst_q = self.spec.workloads[0], 2.0
        if ref_code:
            for w in self.spec.workloads:
                small = {**w, "name": "sg_probe",
                         "n": min(int(w.get("n", 2000)), self.cfg.surrogate_n)}
                q = surrogate_quality(ref_code, self.spec, small)
                if q is not None and q < worst_q:
                    worst, worst_q = w, q
        wl = {**worst, "name": "surrogate",
              "n": min(int(worst.get("n", 2000)), self.cfg.surrogate_n)}
        # cache the chosen workload + the reference quality on it together.
        self._surrogate_wl = wl
        self._surrogate_ref_q = (surrogate_quality(ref_code, self.spec, wl)
                                 if ref_code else None)
        return wl

    def _surrogate_ok(self, probe_code: str) -> bool:
        """v7 cheap pre-filter via the grounded-code-execution SKILL: estimate the
        op's value on the highest-headroom workload and skip ops clearly below the
        human default THERE. Lenient by design (only the obviously-bad); never blocks
        if the surrogate (or a human reference) is unavailable."""
        if not getattr(self.cfg, "surrogate_prefilter", False):
            return True
        wl = self._surrogate_workload()
        if wl is None:
            return True
        op_q = surrogate_quality(probe_code, self.spec, wl)
        if op_q is None:
            return True                              # skill unavailable -> don't block
        ref_q = getattr(self, "_surrogate_ref_q", None)
        if ref_q is None:
            return True
        return op_q >= ref_q - self.cfg.surrogate_margin

    def _log_reject_op(self, ctx: RoundContext, p, why: str) -> None:
        self._rejected_op_names.add(p.name)     # memo: do not re-screen next round
        self.log(f"    vocab- op '{p.name}' <{p.kind}> REJECTED: {why}")
        self.memory.add_insight(Insight(
            iteration=ctx.iteration, kind="failure",
            text=f"proposed {p.kind} op '{p.name}' rejected: {why[:120]}"))

    def _extend_op_vocabulary(self, ctx: RoundContext) -> None:
        """Builder proposes new primitive OPS as source. Each is screened (AST) +
        grounded (a probe policy USING it must run correct) before admission. This
        is the v5 Empirical Grounding Rule for the vocabulary itself: nothing joins
        on the strength of a proposal alone."""
        if not self.open_vocab or self._n_proposed_ops >= self.cfg.proposed_ops_cap:
            return
        # v6: diagnose the best policy's worst workload FIRST, so the proposer is
        # steered by the concrete failure (the missed heavy hitters), not keywords.
        self._causal = self._build_causal_guidance()
        ctx.guidance = self._guidance()
        if self._causal.has_signal:
            self.log(f"    causal: {self._causal.brief()[:160]}")
        proposer = getattr(self.roles.builder, "propose_operations", None)
        if not proposer:
            return
        proposals = (proposer(ctx) or [])[:self.cfg.max_proposed_ops_per_round]
        for p in proposals:
            if self._n_proposed_ops >= self.cfg.proposed_ops_cap:
                break
            # v9 review: a deterministic proposer re-emits unmatched templates every
            # round; without a memo each REJECTED one would be re-screened and
            # re-grounded (2 subprocesses) every SYNTHESIZE *and now* EXPLORE round.
            if p.name in self._rejected_op_names:
                continue
            # Reject a name that exists in ANY slot, not just this one -- a cross-slot
            # collision (e.g. an LLM op named 'min' in the inherit slot) would corrupt
            # name-keyed provenance.
            if self.vocab.name_in_any_slot(p.name):
                self._log_reject_op(ctx, p, "name collides with an existing op")
                continue
            try:
                probe = self.domain.op_probe(self.vocab, p)
            except Exception as e:                      # noqa: BLE001
                self._log_reject_op(ctx, p, f"probe build failed: {e}")
                continue
            viol = all_safety_violations(probe)
            if viol:
                self._log_reject_op(ctx, p, "AST screen: " + ", ".join(viol))
                continue
            try:
                compile(probe, "<op>", "exec")
            except SyntaxError as e:
                self._log_reject_op(ctx, p, f"does not compile: {e}")
                continue
            # v7 SURROGATE PRE-FILTER: a cheap (one tiny workload, via the
            # grounded-code-execution skill) value estimate BEFORE full grounding;
            # skip ops clearly below the human default. Lenient -- the frontier +
            # utility do the real selection; this just trims obvious chaff as volume
            # grows. (Strict technical gates still apply to whatever passes.)
            if not self._surrogate_ok(probe):
                self._log_reject_op(ctx, p, "surrogate pre-filter: clearly below "
                                    "the human default on a cheap probe")
                continue
            # Ground on a CORRECTNESS-focused spec (one small workload) so the op is
            # admitted on well-formedness, not on whether the full hardened-suite
            # benchmark happens to finish inside the timeout (which would make
            # admission of a correct-but-slow op nondeterministic).
            ev = ground(probe, self._op_probe_spec(), self.cfg)
            if not ev.correct:
                self._log_reject_op(ctx, p, "grounding: "
                                    + (ev.error or "incorrect / unbounded"))
                continue
            self.domain.admit_op(self.vocab, p, ctx.iteration)
            self._n_proposed_ops += 1
            self.memory.add_insight(Insight(
                iteration=ctx.iteration, kind="lesson",
                text=(f"Builder proposed + grounded new {p.kind} op '{p.name}' "
                      f"({p.origin.value}): {p.note[:80]}")))
            self.log(f"    vocab+ op '{p.name}' <{p.kind}> ADMITTED "
                     f"({p.origin.value})")

    def _log_reject_gen(self, ctx: RoundContext, p, why: str) -> None:
        self.log(f"    vocab- generator '{p.name}' REJECTED: {why}")
        self.memory.add_insight(Insight(
            iteration=ctx.iteration, kind="failure",
            text=f"proposed generator '{p.name}' rejected: {why[:120]}"))

    def _extend_generator_vocabulary(self, ctx: RoundContext) -> None:
        """Adversary proposes new GENERATORS as source. Each is screened (AST) +
        grounded (must produce a runnable stream a trusted probe solver can score)
        before it is appended to the live spec and thrown at candidates."""
        if (not self.open_vocab
                or self._n_proposed_gens >= self.cfg.proposed_generators_cap):
            return
        proposer = getattr(self.roles.adversary, "propose_generators", None)
        if not proposer:
            return
        proposals = (proposer(ctx)
                     or [])[:self.cfg.max_proposed_generators_per_round]
        k = self.spec.workloads[0]["k"] if self.spec.workloads else 10
        # v9 review: include EVERY name the run can ever measure under -- the static
        # adversary arsenal, the param/generator attacks, and the declared benchmark
        # -- not just what has already hardened into spec.workloads. An LLM generator
        # named e.g. 'adv_loop' or 'bench_scan' must not shadow a declared workload.
        existing_names = {w["name"] for w in self.spec.workloads} \
            | {g["name"] for g in self._proposed_gen_specs} \
            | {w["name"] for w in (self.domain.adversary_workloads or [])
               if isinstance(w, dict)} \
            | {w["name"] for w in (getattr(self.domain, "benchmark_suite", None)
                                   or [])} \
            | {a["name"] for a in (self.domain.adversary_param_attacks(k)
                                   if self.domain.adversary_param_attacks else [])} \
            | {a["name"] for a in (self.domain.adversary_generators(k)
                                   if self.domain.adversary_generators else [])}
        for p in proposals:
            if self._n_proposed_gens >= self.cfg.proposed_generators_cap:
                break
            if self.vocab.has_gen(p.name) or p.name in existing_names:
                # v5: a proposed name must not shadow an arsenal/param/existing
                # workload, or its exposure would be miscredited.
                self._log_reject_gen(ctx, p, "name collides with an existing "
                                     "workload/generator")
                continue
            try:
                block = self.domain.generator_block(p)
            except Exception as e:                      # noqa: BLE001
                self._log_reject_gen(ctx, p, f"block build failed: {e}")
                continue
            viol = all_safety_violations(block)
            if viol:
                self._log_reject_gen(ctx, p, "AST screen: " + ", ".join(viol))
                continue
            try:
                compile(block, "<gen>", "exec")
            except SyntaxError as e:
                self._log_reject_gen(ctx, p, f"does not compile: {e}")
                continue
            # Ground: append to a TRIAL spec and PROBE the generator at production n.
            # It EARNS admission only if it produces a well-formed, BOUNDED, runnable
            # stream -- a generator that raises, returns a non-stream, or allocates
            # super-linearly is rejected here, never reaching the live suite.
            trial = GroundingSpec(
                oracle_src=self.spec.oracle_src,
                workload_gen_src=self.spec.workload_gen_src + "\n" + block,
                criteria_cases=list(self.spec.criteria_cases),
                workloads=self.spec.workloads, mem_workload=self.spec.mem_workload,
                verdict_src=self.spec.verdict_src,
                driver_src=self.spec.driver_src,        # v8: carry the trusted driver
                driver_funcs=self.spec.driver_funcs,
                driver_literal_ints=self.spec.driver_literal_ints)
            probe_spec = self.domain.generator_probe_spec(p.name, k)
            ran_ok, slen, err = probe_generator(self.domain.probe_solver,
                                                probe_spec, trial, self.cfg)
            if not ran_ok:
                self._log_reject_gen(ctx, p, f"no runnable stream ({err})")
                continue
            if slen > 4 * probe_spec["n"]:
                self._log_reject_gen(ctx, p, f"unbounded stream: len={slen} "
                                     f">> n={probe_spec['n']}")
                continue
            # Admit: make the block permanent in the LIVE spec, register the entry,
            # and add a hostile workload spec the Adversary will throw this round on.
            self.spec.workload_gen_src = self.spec.workload_gen_src + "\n" + block
            entry = self.domain.admit_generator(self.vocab, p, ctx.iteration, k)
            self._proposed_gen_specs.append(
                {"name": p.name, "n": 40000, "k": k, "dist": entry.dist})
            self._n_proposed_gens += 1
            self.memory.add_insight(Insight(
                iteration=ctx.iteration, kind="lesson",
                text=(f"Adversary proposed + grounded new generator '{p.name}' "
                      f"({p.origin.value}): {p.note[:80]}")))
            self.log(f"    vocab+ generator '{p.name}' ADMITTED "
                     f"({p.origin.value})")

    def _mark_vocab_provenance(self) -> None:
        """Flag proposed ops whose policy reached the frontier (so the report can
        credit a system-proposed op that genuinely advanced the discovery)."""
        if not (self.open_vocab and self.frontier and self.vocab):
            return
        used = set()
        for pt in self.frontier.points:
            a = self.memory.get(pt.approach_id)
            if a and getattr(a, "ops_used", None):
                used.update(a.ops_used)
        for name in used:
            e = self.vocab.op_entry(name)
            if e and e.origin.is_proposed:
                e.used_on_frontier = True

    def _frontier_winners(self, correct_all: list[Approach]) -> dict:
        """The two ENDS of the frontier to combine: the highest-recall approach
        and the lowest-memory approach. Synthesis combines them into a bounded
        high-recall hybrid."""
        pts = [a for a in correct_all if a.is_correct and a.evidence]
        if not pts:
            return {}
        hi = max(pts, key=lambda a: a.evidence.quality)
        lo = min(pts, key=lambda a: a.evidence.peak_memory_bytes)
        w = {"high-recall": hi}
        if lo is not hi:
            w["low-memory"] = lo
        return w

    def _distill_frontier(self, processed: list[Approach],
                          correct_all: list[Approach]) -> None:
        """Conceptual lessons for the frontier domain: failure/gaming causes plus
        each frontier point as a transferable (recall @ memory) trade-off fact."""
        for a in processed:
            exposed = bool(a.evidence and any(
                f.get("case") == "adversary-metric" for f in a.evidence.failures))
            if (not a.is_correct) or exposed:
                expl = (self.domain.explain_failure(a)
                        if self.domain.explain_failure else None)
                if expl:
                    p, r, d = expl
                    self.memory.lessons.add(Lesson(p, "hurts", r, d, support=[a.id]))
        if self.frontier:
            for pt in self.frontier.points:
                ap = self.memory.get(pt.approach_id)
                cls = ap.algo_class.value if ap else "approach"
                self.memory.lessons.add(Lesson(
                    cls, "fact", f"{pt.memory // 1024}KB",
                    f"{cls} reaches recall {pt.quality:.2f} at "
                    f"{pt.memory // 1024}KB", support=[pt.approach_id]))

    def _augment_signals(self, s: sig.Signals, processed: list[Approach]) -> None:
        """Fill the v1 measured signals the Synthesizer must reckon with."""
        s.synthesis_available = concepts.synthesis_opportunity(self.winners)

        bnovs = [a.behavioral_novelty for a in processed
                 if a.behavioral_novelty is not None]
        s.last_round_mean_behavioral_novelty = (
            round(sum(bnovs) / len(bnovs), 4) if bnovs else None)

        sgs = [a.synthesis_gain for a in processed if a.synthesis_gain is not None]
        s.last_round_synthesis_gain = max(sgs) if sgs else None

        s.synthesis_attempts = self.synthesis_attempts

        # Recombination is dominant if enough of a FULL recent window were
        # structurally-recombinant OR behaviourally stale. Requiring a full
        # window avoids firing on one or two early rounds (insufficient history).
        if len(self.recomb_stale_history) >= self.cfg.recomb_window:
            window = self.recomb_stale_history[-self.cfg.recomb_window:]
            s.recombination_dominant = sum(window) >= self.cfg.recomb_dominant_limit

        # Noise floor + noise-aware progress: is the latest 'best' a real speedup?
        if self.memory.best and self.memory.best.evidence:
            be = self.memory.best.evidence
            s.noise_floor = round(be.noise, 4)
            cur = be.total_time()
            if self.prev_best_time is not None and cur < self.prev_best_time:
                delta = self.prev_best_time - cur
                band = self.cfg.noise_sigmas * be.noise * cur
                s.last_improvement_within_noise = delta <= band
            self.prev_best_time = cur

        # v2 frontier signals (override synthesis_available with the budget gap:
        # a synthesis opportunity is "high recall reachable only above the budget"
        # -- which closes once a bounded-memory hybrid achieves it).
        if self.uses_frontier and self.frontier:
            s.frontier_size = len(self.frontier.points)
            s.frontier_hv = round(self.frontier.hypervolume(), 4)
            s.frontier_advanced = any(
                (a.frontier_gain or 0) > self.cfg.frontier_gain_eps
                for a in processed)
            s.frontier_gap = self.frontier.quality_gap_above_budget(self.budget)
            s.synthesis_available = s.frontier_gap > self.cfg.frontier_gap_threshold
            s.metric_hardened = getattr(self, "_metric_hardened_this_round", False)

    def _adversary_pass(self, a: Approach, ctx: RoundContext,
                        reports: list[AdversaryReport]) -> None:
        """Attack a grounded-correct approach. A CONFIRMED, novel counterexample
        downgrades it AND is folded into the trusted suite for all future rounds."""
        if not a.is_correct:
            return
        report = self.roles.adversary.attack(a, ctx)
        reports.append(report)
        breaks = report.confirmed_breaks()
        if not breaks:
            return
        # The approach passed the suite but the Adversary broke it: empirical
        # evidence wins. Downgrade and harden the suite.
        a.evidence.correct = False
        a.fitness = None
        for f in breaks:
            ce = f.counterexample or {}
            a.evidence.failures.append({"case": "adversary:" + f.hypothesis[:40],
                                        "k": ce.get("k"), "got": ce.get("got"),
                                        "expected": ce.get("expected")})
            # Add as a permanent criteria case (dedup by content).
            case = (f"adv_i{ctx.iteration}_{len(self.spec.criteria_cases)}",
                    ce.get("items", []), ce.get("k", 0))
            if case[1] is not None:
                self.spec.criteria_cases.append(case)
        # If the downgraded approach was (or becomes) the reigning best, re-derive
        # it so `memory.best` is never a no-longer-correct reference.
        self.memory.recompute_best()
        self.memory.add_insight(Insight(
            iteration=ctx.iteration, kind="failure",
            text=(f"Adversary broke {a.id} <{a.algo_class.value}> that passed the "
                  f"base suite; {len(breaks)} case(s) added to the trusted suite."),
            related=[a.id]))
        self.log(f"    adversary BROKE {a.id}: {len(breaks)} confirmed "
                 f"counterexample(s) -> suite hardened")

    # ------------------------------------------------------------------ #
    # Seeding
    # ------------------------------------------------------------------ #
    def _seed(self) -> Approach:
        name = self.domain.seed_strategy
        s = self.domain.strategies[name]
        seed = Approach(id=f"seed_{name}", code=s.code, rationale=s.rationale,
                        algo_class=s.algo_class, origin=Origin.SEED,
                        created_iter=0)
        # Ground the seed FIRST to establish the baseline everyone is scored on.
        ev = ground(seed.code, self.spec, self.cfg)
        self._apply_cost_axis(ev, seed.code)     # v8: exact declared-capacity cost
        if not (ev.grounded and ev.correct):
            raise RuntimeError(f"Seed '{name}' is not correct/grounded: "
                               f"{ev.summary()} -- cannot establish a baseline.")
        self.baseline = baseline_from(ev)
        score(ev, self.baseline)
        seed.evidence = ev
        seed.fingerprint = fingerprint(seed.code, seed.algo_class)
        seed.novelty = 1.0

        if self.uses_frontier:
            # The seed fixes the frontier's memory reference and the memory budget
            # for the "recommended" point. v10 P5: for a driver domain with a
            # declared capacity range, the reference comes from the MAX legal
            # capacity (1.2 x max-genome cost) -- the old 1.2 x seed_mem made every
            # max-capacity point contribute ZERO hypervolume gain (no resource
            # headroom above the reference; confirmed in run artifacts as
            # frontier-extending points scoring fit=0.000).
            seed_mem = max(ev.peak_memory_bytes, 1)
            units = getattr(self.domain, "max_policy_cost_units", 0)
            if getattr(self.spec, "driver_src", "") and units:
                base_k = self.spec.workloads[0]["k"] if self.spec.workloads else 10
                mem_ref = 1.2 * units * base_k * self._BYTES_PER_ENTRY
            else:
                mem_ref = seed_mem * 1.2
            self.frontier = Frontier(memory_ref=mem_ref)
            self.budget = self.domain.memory_budget_frac * seed_mem
            self._admit_frontier(seed)          # sets seed.fitness = its HV
            self.memory.protected_ids = {p.approach_id
                                         for p in self.frontier.points}
            self.memory.add_approach(seed)
            self.memory.best = seed
            self.history_best.append(self.frontier.hypervolume())
        else:
            seed.fitness = fitness(ev, self.cfg)
            self.memory.add_approach(seed)
            self.history_best.append(seed.fitness or 0.0)
        self.memory.add_insight(Insight(
            iteration=0, kind="lesson",
            text=f"Seed baseline '{name}' established: {ev.summary()}.",
            related=[seed.id]))
        self.log(f"[seed] {seed.short()}")
        return seed

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> RunResult:
        t0 = time.time()
        seed = self._seed()
        result = RunResult(domain=self.domain.name, backend=self.cfg.backend,
                           status="exhausted", seed=seed, best=self.memory.best)

        directive: Optional[Directive] = None
        for it in range(1, self.cfg.max_iterations + 1):
            self.log(f"\n=== iteration {it}/{self.cfg.max_iterations} ===")
            ctx = RoundContext(cfg=self.cfg, domain=self.domain,
                               memory=self.memory, iteration=it,
                               directive=directive, winners=dict(self.winners),
                               llm=self.llm, vocab=self.vocab,
                               spec=self.spec, guidance=self._guidance(),
                               proposed_generators=list(self._proposed_gen_specs))

            # --- v5: BEFORE synthesising, let the Builder EXTEND the op vocabulary
            #     (propose new ops as source; the engine screens + grounds each). The
            #     newly admitted ops become available to assemble_from_vocab below.
            #     v9: ALSO on EXPLORE rounds -- exploration is exactly when new
            #     primitives are wanted, and tying proposals solely to synthesis
            #     scheduling made them hostage to directive dynamics (a run that
            #     never synthesised proposed zero ops). Volume stays capped by
            #     max_proposed_ops_per_round / proposed_ops_cap, and every admitted
            #     op gets an honest utility flag via _compose_uncovered_ops even if
            #     no synthesis round ever composes it. ------------------------- -#
            new_ops: set = set()
            if (self.open_vocab and directive
                    and directive.action in (Action.SYNTHESIZE, Action.EXPLORE)):
                before = ({e.name for e in self.vocab.proposed_ops()}
                          if self.vocab else set())
                self._extend_op_vocabulary(ctx)
                new_ops = (({e.name for e in self.vocab.proposed_ops()} - before)
                           if self.vocab else set())

            # --- Builder (may PROPOSE, SYNTHESIZE, or EXPLORE per directive) - #
            if directive and directive.action == Action.SYNTHESIZE:
                self.synthesis_attempts += 1
            proposals = self.roles.builder.propose(ctx)

            # v10 P4: compose newly admitted ops into FRONTIER CANDIDATES in the
            # round they are admitted, so an invention reaches the frontier while
            # the run is alive (v9 composed LLM inventions only at finalize -- the
            # loop accepted seed LRU as best while the verdict crowned the ops).
            # Finalize-time composition remains as a catch-all only.
            if new_ops and self.domain.assemble_from_vocab:
                seen_code = {a.code for a in proposals}
                added = 0
                for (lbl, code, ops_used) in self.domain.assemble_from_vocab(
                        self.vocab):
                    if added >= 6 or code in seen_code:
                        continue
                    if not any(o in new_ops for o in ops_used):
                        continue
                    proposals.append(Approach(
                        id=f"i{ctx.iteration}.v{added}_{lbl}", code=code,
                        rationale=("in-round composition of newly admitted op(s) "
                                   + ", ".join(o for o in ops_used
                                               if o in new_ops)),
                        algo_class=self.domain.classify(code),
                        origin=Origin.BUILDER, created_iter=ctx.iteration,
                        ops_used=list(ops_used)))
                    seen_code.add(code)
                    added += 1

            # --- Ground + score + novelty (Empirical Grounding Rule) -------- #
            processed: list[Approach] = []
            for a in proposals:
                priors = self.memory.priors_for_novelty() + processed
                self._evaluate(a, priors)
                processed.append(a)
                self.log(f"  proposed {a.short()}")

            # --- Adversary ------------------------------------------------- #
            ctx.current_round = processed
            reports: list[AdversaryReport] = []
            self._metric_hardened_this_round = False
            self._round_new_workloads = []
            # v5: let the Adversary EXTEND the generator vocabulary (propose new
            # generators as source; the engine screens + grounds each). Admitted
            # generators join ctx.proposed_generators so the attacks below throw
            # them at every candidate this round.
            if self.open_vocab and self.uses_frontier:
                self._extend_generator_vocabulary(ctx)
                ctx.proposed_generators = list(self._proposed_gen_specs)
            for a in processed:
                if self.uses_frontier:
                    self._adversary_metric_attack(a, ctx, reports)
                else:
                    self._adversary_pass(a, ctx, reports)

            # --- v4: re-score the existing frontier on any new attacks, THEN
            #     admit this round's candidates (all on the same current suite) -#
            if self.uses_frontier:
                if self._round_new_workloads:
                    self._rescore_frontier(self._round_new_workloads)
                for a in processed:
                    self._admit_frontier(a)
                # v4: forbid bounded-memory from evicting a current frontier member
                # (its backing approach must stay resolvable for the next re-score
                # and for `recommended`). Registered BEFORE add_approach prunes.
                self.memory.protected_ids = {p.approach_id
                                             for p in self.frontier.points}

            # --- Remember + record insights --------------------------------- #
            for a in processed:
                self.memory.add_approach(a)
                self._record_insight(it, a)

            # v5: flag which PROPOSED ops reached the frontier (provenance). MUST
            # run AFTER add_approach -- frontier points reference approaches that
            # are only resolvable in memory once added.
            if self.uses_frontier:
                self._mark_vocab_provenance()
                self._record_op_policies(processed)   # v6: survive memory eviction

            # --- Niches / winners + conceptual lessons ---------------------- #
            correct_all = self.memory.correct_approaches()
            if self.uses_frontier:
                self.winners = self._frontier_winners(correct_all)
                rec = self.frontier.recommended(self.budget) if self.frontier else None
                if rec:
                    chosen = self.memory.get(rec.approach_id)
                    if chosen:
                        self.memory.best = chosen   # recommended shippable point
                self._distill_frontier(processed, correct_all)
            else:
                self.winners = behavior.per_workload_winners(
                    correct_all, self.workload_names)
                for a in correct_all:
                    a.behavior["wins"] = [w for w, win in self.winners.items()
                                          if win is a]
                for lesson in concepts.distill(processed, correct_all,
                                               self.winners, self.domain, self.cfg):
                    self.memory.lessons.add(lesson)
            ctx.winners = dict(self.winners)

            # --- Streaks: structural recombination (patching) AND the v1
            #     non-progress flag (recombinant OR behaviourally stale) ------ #
            round_is_recomb = bool(processed) and all(
                a.is_recombination for a in processed)
            self.recomb_streak = self.recomb_streak + 1 if round_is_recomb else 0
            if self.uses_frontier:
                round_improved = any((a.frontier_gain or 0)
                                     > self.cfg.frontier_gain_eps for a in processed)
            else:
                round_improved = any(self.memory.best is a for a in processed)
            correct_props = [a for a in processed if a.is_correct]
            round_stale = (not round_improved and bool(correct_props) and all(
                a.is_recombination or a.behavior.get("stale") for a in correct_props))
            self.recomb_stale_history.append(round_stale)

            # --- Compute OBJECTIVE signals (Synthesizer can't fake these) --- #
            s = sig.compute(self.history_best, self.recomb_streak,
                            self.memory.approaches, processed, self.cfg)
            self._augment_signals(s, processed)
            ctx.signals = s
            self.log(f"  signals: {s.headline()}")

            # --- Synthesizer decision --------------------------------------- #
            directive = self.roles.synthesizer.decide(ctx)

            # --- Human override checkpoint ---------------------------------- #
            if self.on_checkpoint is not None:
                state = self._checkpoint_state(it, processed, s)
                directive = self.on_checkpoint(state, directive)
            self.log(f"  SYNTHESIZER -> {directive.action.value.upper()}: "
                     f"{directive.reason}")
            if directive.drift_flags:
                self.log(f"    drift addressed: {directive.drift_flags}")
                self.memory.add_insight(Insight(
                    iteration=it, kind="drift",
                    text=f"Drift {directive.drift_flags} -> "
                         f"{directive.action.value}: {directive.reason[:160]}"))

            # --- Record the round ------------------------------------------- #
            result.iterations.append(IterationRecord(
                iteration=it, proposals=processed, adversary_reports=reports,
                signals=s, directive=directive,
                best_id=self.memory.best.id if self.memory.best else None,
                best_fitness=self.memory.best.fitness if self.memory.best else None))
            if self.uses_frontier and self.frontier:
                self.history_best.append(self.frontier.hypervolume())
            else:
                self.history_best.append(
                    self.memory.best.fitness if self.memory.best else 0.0)

            # --- Act on the (possibly overridden) directive ----------------- #
            if directive.action == Action.ACCEPT:
                result.status = "accepted"
                break
            if directive.action == Action.TERMINATE:
                result.status = "terminated"
                break
            # CONTINUE / SYNTHESIZE / RESET -> carry directive into next round.

        # --- Finalise ------------------------------------------------------- #
        result.best = self.memory.best
        result.insights = list(self.memory.insights)
        result.lessons = self.memory.lessons.top(20)
        result.improvement = self._improvement(seed, self.memory.best)
        # v6: label proposed-op utility, run the self-play tournament, append the
        # cross-run history -- BEFORE reading provenance (which now carries utility).
        self._finalize_v6()
        if self.open_vocab and self.vocab:
            result.vocab_provenance = self.vocab.provenance_summary()
            result.v6_summary = self.v6_summary
        result.usage = self.llm.usage.summary() if self.llm else "deterministic"
        result.run_dir = self._persist(result, t0)
        return result

    # ------------------------------------------------------------------ #
    # Insight bookkeeping
    # ------------------------------------------------------------------ #
    def _record_insight(self, it: int, a: Approach) -> None:
        if a.is_correct:
            improved = (self.memory.best is a)
            kind = "success" if improved else "lesson"
            txt = (f"{a.id} <{a.algo_class.value}> correct, fitness={a.fitness:.3f}"
                   + (" (new best)" if improved else ""))
            if a.is_recombination:
                txt += " [flagged recombination]"
            self.memory.add_insight(Insight(it, kind, txt, related=[a.id]))
        else:
            fc = sig.failure_class(a) or "unknown"
            self.memory.add_insight(Insight(
                it, "failure",
                f"{a.id} <{a.algo_class.value}> DISQUALIFIED ({fc}): "
                f"{a.evidence.summary() if a.evidence else 'ungrounded'}",
                related=[a.id]))

    # ------------------------------------------------------------------ #
    # Human checkpoint state
    # ------------------------------------------------------------------ #
    def _checkpoint_state(self, it: int, proposals: list[Approach],
                          s: sig.Signals) -> dict:
        return {
            "iteration": it,
            "signals": s.headline(),
            "untried": [c.value for c in untried_classes(s, self.domain)],
            "proposals": [a.short() for a in proposals],
            "best": self.memory.best.short() if self.memory.best else None,
            "memory_digest": self.memory.digest(),
        }

    # ------------------------------------------------------------------ #
    # Reporting / persistence
    # ------------------------------------------------------------------ #
    def _improvement(self, seed: Approach, best: Optional[Approach]) -> dict:
        if self.uses_frontier and self.frontier:
            rec = self.frontier.recommended(self.budget)
            out = {
                "mode": "frontier",
                "seed": (f"recall={seed.evidence.quality:.2f} "
                         f"mem={seed.evidence.peak_memory_bytes // 1024}KB"),
                "budget_KB": int(self.budget // 1024),
                "recommended": (f"{rec.approach_id}: recall={rec.quality:.2f} "
                                f"mem={rec.memory // 1024}KB" if rec else None),
                "frontier_hypervolume": round(self.frontier.hypervolume(), 4),
                "frontier_points": [
                    {"id": p.approach_id, "recall": round(p.quality, 3),
                     "mem_KB": p.memory // 1024}
                    for p in sorted(self.frontier.points, key=lambda p: p.memory)],
            }
            # v10 P4: grade the RECOMMENDED ARTIFACT itself against the best human
            # at <= its memory, on BOTH tiers, printed next to the headline -- so
            # best_solution.py and the headline can never silently disagree again.
            if rec is not None and self.open_vocab and self.vocab:
                try:
                    sflag, smargin = self._classify_utility(rec.quality, rec.memory)
                    aflag, amargin = self._classify_utility_adaptive(
                        rec.quality, rec.memory)
                    out["recommended_grade"] = {
                        "static": sflag.value, "static_margin": smargin,
                        "adaptive": aflag.value, "adaptive_margin": amargin}
                except Exception:                          # noqa: BLE001
                    pass
            return out
        if not best or not best.evidence or best.fitness is None:
            return {"seed_fitness": seed.fitness, "best_fitness": None}
        seed_t = sum(seed.evidence.timings.values()) or 1e-9
        best_t = sum(best.evidence.timings.values()) or 1e-9
        return {
            "seed_id": seed.id,
            "best_id": best.id,
            "seed_fitness": round(seed.fitness, 4),
            "best_fitness": round(best.fitness, 4),
            "fitness_multiple": round((best.fitness or 0) / (seed.fitness or 1e-9), 2),
            "wallclock_speedup_x": round(seed_t / best_t, 1),
            "seed_class": seed.algo_class.value,
            "best_class": best.algo_class.value,
        }

    def _persist(self, result: RunResult, t0: float) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(__file__).resolve().parent.parent / "runs" / \
            f"{ts}_{self.domain.name}_{self.cfg.backend}"
        run_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "domain": result.domain, "backend": result.backend,
            "profile": self.cfg.profile, "models": self.cfg.models,
            "status": result.status, "elapsed_s": round(time.time() - t0, 1),
            "iterations_run": len(result.iterations),
            "improvement": result.improvement,
            "usage": result.usage,
            "seed": to_jsonable(result.seed),
            "best": to_jsonable(result.best) if result.best else None,
            "insights": [to_jsonable(i) for i in result.insights],
        }
        summary["concepts"] = [to_jsonable(L) for L in result.lessons]
        if result.vocab_provenance:
            summary["vocab_provenance"] = result.vocab_provenance
        if result.v6_summary:
            summary["v6_summary"] = result.v6_summary
        (run_dir / "result.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")

        # v10 P5 CLAIMS PINNING: a machine-readable claims.json per run, so the
        # README's verified-result numbers can be diffed against a NAMED run by
        # scripts/check_claims.py (a README that drifts from what the code
        # produces is exactly the declared-vs-actual gap v10 closes).
        v6 = result.v6_summary or {}
        imp = result.improvement or {}
        claims = {
            "run_id": run_dir.name,
            "domain": result.domain, "backend": result.backend,
            "status": result.status,
            "utility_adaptive": v6.get("utility_adaptive"),
            "utility_static": v6.get("utility"),
            "op_utility_adaptive": v6.get("op_utility_adaptive"),
            "op_utility_static": v6.get("op_utility"),
            "incomplete": v6.get("incomplete"),
            "human_baseline_incomplete": v6.get("human_baseline_incomplete"),
            "incoherent": v6.get("incoherent"),
            "recommended": imp.get("recommended"),
            "recommended_grade": imp.get("recommended_grade"),
            "frontier_hypervolume": imp.get("frontier_hypervolume"),
            "frontier_points": imp.get("frontier_points"),
        }
        (run_dir / "claims.json").write_text(
            json.dumps(claims, indent=2), encoding="utf-8")

        with (run_dir / "iterations.jsonl").open("w", encoding="utf-8") as fh:
            for rec in result.iterations:
                fh.write(json.dumps({
                    "iteration": rec.iteration,
                    "signals": to_jsonable(rec.signals),
                    "directive": to_jsonable(rec.directive),
                    "proposals": [to_jsonable(a) for a in rec.proposals],
                    "adversary": [to_jsonable(r) for r in rec.adversary_reports],
                }) + "\n")

        (run_dir / "report.txt").write_text(
            format_report(result), encoding="utf-8")
        if result.best:
            (run_dir / "best_solution.py").write_text(
                result.best.code.strip() + "\n", encoding="utf-8")
        return str(run_dir)


# --------------------------------------------------------------------------- #
# Human-readable report
# --------------------------------------------------------------------------- #

def format_report(r: RunResult) -> str:
    L = []
    L.append("=" * 70)
    L.append(f"RECURSIVE DISCOVERY ENGINE -- run report ({r.domain}, {r.backend})")
    L.append("=" * 70)
    L.append(f"status: {r.status.upper()}   iterations: {len(r.iterations)}")
    L.append("")
    L.append(f"SEED:  {r.seed.short()}")
    L.append(f"BEST:  {r.best.short() if r.best else 'none'}")
    imp = r.improvement
    if imp.get("mode") == "frontier":
        L.append("")
        L.append("PARETO FRONTIER (recall x memory) -- the v2 result:")
        for p in imp["frontier_points"]:
            L.append(f"    {p['id']:24} recall={p['recall']:.2f}  {p['mem_KB']}KB")
        L.append(f"  hypervolume={imp['frontier_hypervolume']}  "
                 f"budget={imp['budget_KB']}KB")
        L.append(f"  seed: {imp['seed']}")
        L.append(f"  RECOMMENDED (best recall within budget): {imp['recommended']}")
        rg = imp.get("recommended_grade")
        if rg:
            L.append(f"  RECOMMENDED ARTIFACT GRADE vs best human at <= its memory: "
                     f"adaptive tier: {rg['adaptive']} ({rg['adaptive_margin']:+}) | "
                     f"static tier: {rg['static']} ({rg['static_margin']:+})")
        prov = r.vocab_provenance
        if prov:
            L.append("")
            L.append("VOCABULARY PROVENANCE (v5: human-designed vs system-proposed):")
            L.append(f"  ops: {prov['human_ops']} human, "
                     f"{prov['proposed_ops']} proposed "
                     f"({prov['proposed_ops_used']} on the frontier; see PROPOSAL "
                     f"QUALITY below for beats/matches/below-human)")
            if prov["proposed_op_names"]:
                L.append(f"      proposed ops: {prov['proposed_op_names']}")
            L.append(f"  generator TYPES: {prov['human_generators']} human "
                     f"(+ param/arsenal streams), {prov['proposed_generators']} "
                     f"proposed ({prov['proposed_generators_exposing']} exposed a "
                     f"candidate)")
            if prov["proposed_generator_names"]:
                L.append(f"      proposed generators: "
                         f"{prov['proposed_generator_names']}")
        v6 = r.v6_summary
        if v6:
            u = v6.get("utility", {})
            L.append("")
            if v6.get("incoherent"):
                L.append("  !! INCOHERENT-RESULT: frontier max CR "
                         f"{v6['incoherent']['frontier_max_cr']} < beating-op CR "
                         f"{v6['incoherent']['beating_op_max_cr']} -- upstream bug; "
                         "investigate before trusting this run")
            L.append("PROPOSAL QUALITY:")
            ua = v6.get("utility_adaptive") or {}
            if ua:
                L.append(f"  HEADLINE -- vs best ADAPTIVE human (incl. ARC/LeCaR): "
                         f"{ua.get('beats_human', 0)} beats, "
                         f"{ua.get('matches_human', 0)} matches, "
                         f"{ua.get('below_human', 0)} below")
                for nm, fl in (v6.get("op_utility_adaptive") or {}).items():
                    L.append(f"      {nm}: {fl}")
            L.append(f"  static tier (v6-compatible, vs the classic statics): "
                     f"{u.get('beats_human', 0)} beats_human, "
                     f"{u.get('matches_human', 0)} matches_human, "
                     f"{u.get('below_human', 0)} below_human")
            inc = v6.get("incomplete") or {}
            for nm, fl in (v6.get("op_utility") or {}).items():
                tail = (f"   [INCOMPLETE: {inc[nm]} scored 0.0]"
                        if inc.get(nm) else "")
                L.append(f"      {nm}: {fl}{tail}")
            for h in (v6.get("human_baseline_incomplete") or []):
                L.append(f"      !! HUMAN BASELINE '{h['label']}' INCOMPLETE: "
                         f"{h['missing']} scored 0.0 -- verdict flagged")
            sp = v6.get("self_play", {})
            if sp.get("matches"):
                netp = v6.get("self_play_net_positive")
                L.append(f"  self-play (proposed ops vs the BEST HUMAN policy per "
                         f"generator): {sp.get('matches')} matches, "
                         f"win_rate={sp.get('win_rate')}, "
                         f"mean_margin={sp.get('mean_margin', 0):+}; "
                         f"net-positive frontier movement="
                         f"{'YES' if netp else 'no'}")
                if sp.get("best"):
                    L.append(f"      best: {sp['best']}")
            h = v6.get("history", {})
            if h.get("runs"):
                trend = h.get("trend", "n/a")
                if trend == "n/a" or h.get("beats_rate_early") is None:
                    tail = "(trend n/a until >=2 runs)"
                else:
                    tail = (f"(early={h['beats_rate_early']} -> "
                            f"late={h['beats_rate_late']}, {trend})")
                L.append(f"  cross-run trend ({h['runs']} runs, {h['ops']} ops): "
                         f"beats_human_rate={h.get('beats_human_rate')} {tail}")
    elif imp.get("best_fitness") is not None:
        L.append("")
        L.append(f"IMPROVEMENT over seed: "
                 f"fitness x{imp['fitness_multiple']} "
                 f"({imp['seed_fitness']} -> {imp['best_fitness']}), "
                 f"wall-clock x{imp['wallclock_speedup_x']} faster "
                 f"({imp['seed_class']} -> {imp['best_class']})")
    L.append("")
    L.append("PER-ITERATION DECISIONS:")
    for rec in r.iterations:
        L.append(f"  it{rec.iteration}: {rec.directive.action.value.upper():9} "
                 f"| {rec.signals.headline()}")
        L.append(f"          why: {rec.directive.reason}")
        if rec.directive.novelty_assessment:
            L.append(f"          novelty: {rec.directive.novelty_assessment}")
    if r.lessons:
        L.append("")
        L.append("CONCEPTS DISTILLED (transferable, structural):")
        for lesson in r.lessons[:12]:
            L.append(f"  - {lesson.line()}")
    L.append("")
    L.append("EVENTS (bounded memory):")
    for ins in r.insights[-10:]:
        L.append(f"  [{ins.kind}] (it{ins.iteration}) {ins.text}")
    L.append("")
    L.append(f"cost: {r.usage}")
    L.append(f"artifacts: {r.run_dir}")
    if r.best:
        L.append("")
        L.append("BEST SOLUTION:")
        L.append("-" * 40)
        L.append(r.best.code.strip())
    return "\n".join(L)
