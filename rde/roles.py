"""The three roles -- Builder, Adversary, Synthesizer -- in two flavours each.

  * Deterministic*  : no LLM. The Builder draws from a curated strategy library,
    the Adversary throws a fixed arsenal, the Synthesizer applies the protective
    mechanisms as an explicit, auditable policy. This flavour is the REFERENCE
    implementation of the engine's control logic: it runs with zero API cost, it
    is what the test-suite exercises, and it is the behaviour the LLM flavour is
    held to.

  * LLM*            : each role is a prompt against a (deliberately different)
    model on OpenRouter. The model proposes; the engine still *grounds* every
    claim and still *computes* the drift/novelty signals itself. The LLM never
    gets to assert correctness or wave away a detected stall.

The split keeps the loop in engine.py identical for both backends -- it just
receives a different trio of role objects.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Protocol

from .concepts import synthesis_brief, synthesis_parents
from .config import Config
from .domain import Domain
from .grounding import check_cases, measure_recall_on
from .llm import LLMClient, LLMError, extract_json
from .memory import Memory
from .fingerprint import extract_code
from .mutate import mutate
from .signals import Signals
from .types import (Action, AdversaryFinding, AdversaryReport, AlgoClass,
                    Approach, Directive, Origin)
from .vocab import GenProposal, OpProposal, Provenance


# --------------------------------------------------------------------------- #
# Shared round context (the only thing roles ever see)
# --------------------------------------------------------------------------- #

@dataclass
class RoundContext:
    cfg: Config
    domain: Domain
    memory: Memory
    iteration: int
    directive: Optional[Directive] = None   # last round's decision -> steers Builder
    signals: Optional[Signals] = None       # computed facts -> for Synthesizer
    current_round: list[Approach] = field(default_factory=list)  # for Synthesizer
    # v1: per-workload winners (workload -> Approach), the raw material for
    # synthesis briefs and the behavioural-niche map.
    winners: dict = field(default_factory=dict)
    llm: Optional[LLMClient] = None
    # v5: open-vocabulary state, all owned by the engine and exposed read-only to
    # the roles. `spec` is the LIVE grounding spec (includes proposed-generator
    # source) -- roles MUST measure against it, not domain.grounding_spec, or a
    # `prop:<name>` workload will not generate. `vocab` is the live op registry;
    # `proposed_generators` are the admitted proposed-generator workload specs the
    # Adversary should also throw; `guidance` steers what to propose next.
    vocab: object = None
    spec: object = None
    proposed_generators: list = field(default_factory=list)
    guidance: dict = field(default_factory=dict)

    def live_spec(self):
        """The spec roles should measure against -- live spec if the engine set
        one (v5), else the domain's static spec (v1-v4 backward compatibility)."""
        return self.spec if self.spec is not None else self.domain.grounding_spec


# Default algorithmic families worth exploring (top-K). A domain may override via
# `domain.meaningful_classes` (e.g. the streaming families in heavy_hitters).
MEANINGFUL_CLASSES = [AlgoClass.FULL_SORT, AlgoClass.HEAP,
                      AlgoClass.SELECTION, AlgoClass.BUCKET]

# Classes that count as "a synthesised / invented result" -- v1 adaptive hybrids
# and v3 compositionally-invented programs. Reaching one means synthesis succeeded.
_SYNTH_CLASSES = (AlgoClass.HYBRID, AlgoClass.INVENTED)


def untried_classes(signals: Signals, domain=None) -> list[AlgoClass]:
    classes = (domain.meaningful_classes if domain
               and getattr(domain, "meaningful_classes", None)
               else MEANINGFUL_CLASSES)
    tried = set(signals.classes_tried)
    return [c for c in classes if c.value not in tried]


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #

class Builder(Protocol):
    def propose(self, ctx: RoundContext) -> list[Approach]: ...


class Adversary(Protocol):
    def attack(self, approach: Approach, ctx: RoundContext) -> AdversaryReport: ...


class Synthesizer(Protocol):
    def decide(self, ctx: RoundContext) -> Directive: ...


# =========================================================================== #
# Deterministic roles
# =========================================================================== #

class DetBuilder:
    """Reference Builder. Dispatches on the Synthesizer's directive:
      * SYNTHESIZE -> compose a genuine adaptive HYBRID from the regime winners
        (via the domain's `compose` hook) -- not a pick of the best, a combination;
      * EXPLORE    -> reach for an untried niche;
      * else       -> walk the curated class repertoire.
    When the distinct ideas run out it emits cosmetic variants of the best -- the
    'local patching' the Synthesizer is built to detect and stop."""

    def __init__(self):
        self.used: set[str] = set()
        self._variant_n = 0
        self._used_comps: set[str] = set()       # v3: grammar compositions tried

    def propose(self, ctx: RoundContext) -> list[Approach]:
        d = ctx.directive
        if d and d.action == Action.SYNTHESIZE:
            syn = self._synthesize(ctx)
            if syn:
                return syn
            # Nothing new to synthesise -> fall through to exploration.
        if d and d.action == Action.EXPLORE:
            return self._explore(ctx)
        return self._propose_class(ctx)

    def _resolve_parents(self, ctx: RoundContext) -> list[Approach]:
        d = ctx.directive
        ids = (d.parent_ids if d and d.parent_ids
               else [a.id for a in ctx.winners.values()])
        parents = [p for p in (ctx.memory.get(i) for i in ids) if p]
        return parents or list(ctx.winners.values())

    # -- v5 open-vocabulary op proposal (Builder authors new primitive ops) --- #
    def propose_operations(self, ctx: RoundContext) -> list:
        """Deterministic op-proposer: delegate to the domain's guided meta-proposer.
        The engine AST-screens + grounds each before admitting it to the vocabulary
        -- this method only AUTHORS candidates, it does not get to admit them."""
        dom = ctx.domain
        if not (ctx.vocab is not None and getattr(dom, "propose_operations", None)):
            return []
        rng = random.Random(ctx.iteration * 911 + 7)
        return dom.propose_operations(ctx.vocab, ctx.guidance or {}, rng)

    # -- synthesis: v5 vocabulary, v3 grammar, then the v2 budget-hybrid ------ #
    def _synthesize(self, ctx: RoundContext) -> list[Approach]:
        dom = ctx.domain
        d = ctx.directive
        parents = self._resolve_parents(ctx)

        # v5: compose over the LIVE vocabulary (which may now include PROPOSED ops
        # the engine admitted this round). Proposed-op policies lead, so they are
        # grounded in the same round the op was admitted; the frontier filters them.
        if (ctx.vocab is not None and getattr(dom, "assemble_from_vocab", None)):
            assembled = [(lbl, code, ops)
                         for (lbl, code, ops) in dom.assemble_from_vocab(ctx.vocab)
                         if lbl not in self._used_comps]
            if assembled:
                # Cover EVERY proposed-op policy this round (so a freshly-admitted op
                # is always grounded), plus a batch of the human genome.
                prop_names = ctx.vocab.proposed_op_names()
                n_prop = sum(1 for (_l, _c, ops) in assembled
                             if any(o in prop_names for o in ops))
                batch_n = max(getattr(ctx.cfg, "synthesis_batch", 3), 1) + n_prop
                batch = assembled[:batch_n]
                out = []
                for j, (label, code, ops) in enumerate(batch):
                    self._used_comps.add(label)
                    out.append(Approach(
                        id=f"i{ctx.iteration}.{j}_{label}", code=code,
                        rationale=f"vocabulary synthesis ({label}; ops={ops})",
                        algo_class=AlgoClass.INVENTED, origin=Origin.SYNTHESIS,
                        parent_ids=[p.id for p in parents],
                        created_iter=ctx.iteration, ops_used=list(ops),
                        synthesis_brief=(d.synthesis_brief if d else "")))
                return out

        # v3: assemble GENUINELY NEW programs from the primitive grammar (e.g.
        # Space-Saving), walking the search space across synthesis rounds.
        if dom.synthesize_programs:
            progs = [(label, code)
                     for (label, code) in dom.synthesize_programs(parents)
                     if label not in self._used_comps]
            if progs:
                batch = progs[:max(getattr(ctx.cfg, "synthesis_batch", 3), 1)]
                out = []
                for j, (label, code) in enumerate(batch):
                    self._used_comps.add(label)
                    out.append(Approach(
                        id=f"i{ctx.iteration}.{j}_{label}", code=code,
                        rationale=f"compositional synthesis ({label})",
                        algo_class=AlgoClass.INVENTED, origin=Origin.SYNTHESIS,
                        parent_ids=[p.id for p in parents],
                        created_iter=ctx.iteration,
                        synthesis_brief=(d.synthesis_brief if d else "")))
                return out

        # Fallback: the v2 single budget-switch hybrid, proposed once.
        if dom.compose and "__hybrid__" not in self._used_comps:
            code = dom.compose(parents)
            if code:
                self._used_comps.add("__hybrid__")
                return [Approach(
                    id=f"i{ctx.iteration}.0_hybrid", code=code,
                    rationale="adaptive budget-switch hybrid from "
                              + ", ".join(p.id for p in parents),
                    algo_class=dom.classify(code), origin=Origin.SYNTHESIS,
                    parent_ids=[p.id for p in parents],
                    created_iter=ctx.iteration,
                    synthesis_brief=(d.synthesis_brief if d else ""))]
        return []

    # -- exploration: v3 AST-mutate a frontier member; else try a new class --- #
    def _explore(self, ctx: RoundContext) -> list[Approach]:
        dom = ctx.domain
        base = ctx.memory.best
        if base and base.code:
            rng = random.Random(ctx.iteration * 131 + self._variant_n)
            self._variant_n += 1
            res = mutate(base.code, rng)
            if res:
                code, desc = res
                cls = dom.classify(code) if dom.classify else base.algo_class
                return [Approach(
                    id=f"i{ctx.iteration}.0_mut{self._variant_n}", code=code,
                    rationale=f"AST mutation of {base.id} ({desc})",
                    algo_class=cls, origin=Origin.BUILDER,
                    parent_ids=[base.id], created_iter=ctx.iteration)]
        pool = [n for n in dom.builder_priority if n not in self.used]
        if pool:
            name = pool.pop(0)
            self.used.add(name)
            s = dom.strategies[name]
            return [Approach(
                id=f"i{ctx.iteration}.0_{name}", code=s.code,
                rationale="explore: " + s.rationale, algo_class=s.algo_class,
                origin=Origin.BUILDER, parent_ids=[],
                created_iter=ctx.iteration)]
        return [self._cosmetic_variant(ctx, 0,
                [ctx.memory.best.id] if ctx.memory.best else [])]

    def _propose_class(self, ctx: RoundContext) -> list[Approach]:
        dom = ctx.domain
        directive = ctx.directive
        target = directive.target_class if directive else None
        synth = bool(directive and directive.action == Action.SYNTHESIZE)

        if target:
            pool = [n for n in dom.strategies_of_class(target) if n not in self.used]
            if not pool:    # nothing new in that class -> fall back to priority
                pool = [n for n in dom.builder_priority if n not in self.used]
        else:
            pool = [n for n in dom.builder_priority if n not in self.used]

        parents = [ctx.memory.best.id] if ctx.memory.best else []
        out: list[Approach] = []
        for j in range(ctx.cfg.builder_fanout):
            if pool:
                name = pool.pop(0)
                self.used.add(name)
                s = dom.strategies[name]
                out.append(Approach(
                    id=f"i{ctx.iteration}.{j}_{name}",
                    code=s.code, rationale=s.rationale, algo_class=s.algo_class,
                    origin=Origin.SYNTHESIS if synth else Origin.BUILDER,
                    parent_ids=parents, created_iter=ctx.iteration))
            else:
                # Repertoire exhausted -> cosmetic variant (a recombination).
                out.append(self._cosmetic_variant(ctx, j, parents))
        return out

    def _cosmetic_variant(self, ctx: RoundContext, j: int,
                          parents: list[str]) -> Approach:
        self._variant_n += 1
        base = ctx.memory.best
        base_code = base.code if base else ctx.domain.strategies[
            ctx.domain.seed_strategy].code
        base_class = base.algo_class if base else AlgoClass.OTHER
        # A pure-comment edit: structurally identical -> similarity ~1.0 ->
        # guaranteed to trip the recombination check. Behaviour is unchanged.
        code = base_code.rstrip() + f"\n# cosmetic variant {self._variant_n}\n"
        return Approach(
            id=f"i{ctx.iteration}.{j}_variant{self._variant_n}",
            code=code,
            rationale=f"Cosmetic variation #{self._variant_n} of "
                      f"{base.id if base else 'seed'} (no algorithmic change).",
            algo_class=base_class, origin=Origin.SYNTHESIS,
            parent_ids=parents, created_iter=ctx.iteration)


class DetAdversary:
    """Throws the domain's fixed arsenal at a candidate. For closed-oracle domains
    it reports only correctness breaks the harness CONFIRMS; for frontier domains
    it attacks the METRIC -- measuring recall on hostile workloads (the engine
    decides what counts as a collapse)."""

    def propose_generators(self, ctx: RoundContext) -> list:
        """v5 deterministic generator-proposer: delegate to the domain's meta-
        proposer. The engine AST-screens + grounds each (must yield a runnable
        stream) before it joins the suite -- this only AUTHORS candidates."""
        dom = ctx.domain
        if not (ctx.vocab is not None and getattr(dom, "propose_generators", None)):
            return []
        rng = random.Random(ctx.iteration * 523 + 11)
        return dom.propose_generators(ctx.vocab, ctx.guidance or {}, rng)

    def attack(self, approach: Approach, ctx: RoundContext) -> AdversaryReport:
        report = AdversaryReport(approach_id=approach.id)
        if getattr(ctx.domain, "uses_frontier", False):
            wls = list(ctx.domain.adversary_workloads)
            wl0 = ctx.domain.grounding_spec.workloads
            k = wl0[0]["k"] if wl0 else 10
            # v3: parameterised hostile streams.
            if getattr(ctx.domain, "adversary_param_attacks", None):
                wls = wls + ctx.domain.adversary_param_attacks(k)
            # v4: GENERATOR-LEVEL -- new attack TYPES via the generator DSL.
            if getattr(ctx.domain, "adversary_generators", None):
                wls = wls + ctx.domain.adversary_generators(k)
            # v5: PROPOSED generators (Adversary-authored, already grounded).
            wls = wls + list(ctx.proposed_generators)
            recalls = measure_recall_on(approach.code, wls,
                                        ctx.live_spec(), ctx.cfg)
            for wl in wls:
                r = recalls.get(wl["name"])
                if r is None:
                    continue
                report.findings.append(AdversaryFinding(
                    hypothesis=wl["name"],
                    broke=r < ctx.cfg.metric_collapse_floor,
                    grounded=True,
                    counterexample={"workload": wl, "recall": r}))
            return report

        cases = ctx.domain.adversary_cases
        res = check_cases(approach.code, cases, ctx.domain.grounding_spec, ctx.cfg)
        case_results = res.get("cases", {})
        for name, items, k in cases:
            rec = case_results.get(name)
            if rec is None:
                continue
            broke = not rec.get("ok", False)
            report.findings.append(AdversaryFinding(
                hypothesis=f"case {name}",
                broke=broke,
                grounded=res.get("import_ok", False),
                counterexample=({"items": list(items), "k": k,
                                 "expected": rec.get("expected"),
                                 "got": rec.get("got"),
                                 "error": rec.get("error")} if broke else None),
                notes=("mutated input" if rec.get("mutated") else "")))
        return report


class DetSynthesizer:
    """The reference control policy (v1). Consumes the engine-computed Signals
    (which it CANNOT alter) and chooses the next Action by an explicit, auditable
    priority order. v1 additions: it actively PUSHES synthesis when distinct
    approaches own distinct regimes, and PUSHES exploration when recent work is
    behaviourally stale -- rather than passively accepting a single best."""

    def decide(self, ctx: RoundContext) -> Directive:
        s = ctx.signals
        mem = ctx.memory
        cfg = ctx.cfg
        untried = untried_classes(s, ctx.domain)
        drift_flags = self._drift_flags(s)
        nov = self._novelty_note(s, ctx.current_round)
        winners = ctx.winners or {}
        # Synthesis is "achieved" once the best is a correct adaptive hybrid --
        # whether the engine directed it OR the Builder produced it spontaneously
        # (an LLM Builder often jumps straight to a hybrid). The synthesis_gain
        # metric grades a *directed* attempt; it is not a gate on recognising a
        # winning hybrid, else the engine loops re-synthesising what it already has.
        have_hybrid = bool(mem.best and mem.best.algo_class in _SYNTH_CLASSES
                           and mem.best.is_correct)
        # A synthesis was already attempted but did NOT become the best -> do not
        # loop on the same synthesis forever. Also treat "we just asked to
        # synthesize but nothing composed came out" as a failed attempt, else
        # SYNTHESIZE re-issues endlessly.
        last_was_synth = bool(ctx.directive
                              and ctx.directive.action == Action.SYNTHESIZE)
        round_made_hybrid = any(a.algo_class in _SYNTH_CLASSES
                                for a in ctx.current_round)
        synthesis_failed = (
            (any(a.algo_class in _SYNTH_CLASSES for a in mem.approaches)
             and not have_hybrid)
            or (last_was_synth and not round_made_hybrid))
        synth_ready = (s.synthesis_available and not have_hybrid
                       and not synthesis_failed
                       and s.synthesis_attempts < cfg.max_synthesis_attempts)

        def synth(reason: str) -> Directive:
            return Directive(
                Action.SYNTHESIZE, reason=reason,
                instruction="Combine the regime winners into ONE adaptive solver.",
                synthesis_brief=(synthesis_brief(winners, ctx.domain)
                                 if winners else ""),
                parent_ids=(synthesis_parents(winners) if winners
                            else ([mem.best.id] if mem.best else [])),
                novelty_assessment=nov, drift_flags=drift_flags)

        def explore(reason: str) -> Directive:
            return Directive(
                Action.EXPLORE, reason=reason,
                instruction="Produce a behaviourally DIFFERENT candidate.",
                target_class=untried[0] if untried else None,
                explore_target=self._weak_workload(winners),
                novelty_assessment=nov, drift_flags=drift_flags)

        # 1) Breadth first: explore unseen algorithmic classes while any remain
        #    (and we are not already in a patching/recombination spiral).
        if untried and not s.patching and not s.recombination_dominant:
            return Directive(
                Action.CONTINUE,
                reason=f"Healthy exploration; classes still unseen: "
                       f"{[c.value for c in untried]}.",
                instruction=f"Try a {untried[0].value} approach next.",
                target_class=untried[0],
                novelty_assessment=nov, drift_flags=drift_flags)

        # 2) Local patching / recombination dominant -> ESCAPE, preferring real
        #    synthesis over yet another variation (the v1 anti-collapse upgrade).
        if s.patching or s.recombination_dominant:
            if synth_ready:
                return synth("Recent work is recombination / behaviourally stale, "
                             "but strengths are split across regimes -> synthesise "
                             "an adaptive hybrid instead of patching.")
            if untried:
                return explore("Local patching detected; deliberately exploring an "
                               "unseen class rather than varying the best.")
            if mem.best and mem.best.is_correct:
                return Directive(Action.ACCEPT,
                    reason="Only recombinations remain; accepting the empirical best.",
                    novelty_assessment=nov, drift_flags=drift_flags)
            return Directive(Action.TERMINATE,
                reason="Local patching and nothing new to try.",
                novelty_assessment=nov, drift_flags=drift_flags)

        # 3) Recurring failure class with no fresh success.
        produced_correct = any(a.is_correct for a in ctx.current_round)
        if s.recurring_failures and not produced_correct and untried:
            return explore(f"Recurring failure {s.recurring_failures} with no new "
                           f"correct approach; redirecting to a new niche.")

        # 4) Synthesis opportunity not yet realised -> SYNTHESIZE (the v1 push).
        if synth_ready:
            return synth("Distinct approaches win distinct regimes -> synthesise an "
                         "adaptive hybrid to win all regimes at once.")

        # 5) Recent round behaviourally stale + niches remain -> EXPLORE.
        if (s.last_round_mean_behavioral_novelty is not None
                and s.last_round_mean_behavioral_novelty
                < cfg.behavioral_novelty_floor and untried):
            return explore("Recent proposals are behaviourally stale; exploring a "
                           "genuinely different niche.")

        # 6) Stall -> accept the winner or stop.
        if s.stalled:
            if mem.best and mem.best.is_correct:
                return Directive(Action.ACCEPT,
                    reason="Progress has stalled; accepting the empirical winner.",
                    novelty_assessment=nov, drift_flags=drift_flags)
            return Directive(Action.TERMINATE,
                reason="Stalled with no correct approach.",
                novelty_assessment=nov, drift_flags=drift_flags)

        # 7) Nothing left -> accept the grounded best (incl. a successful hybrid).
        if mem.best and mem.best.is_correct:
            return Directive(Action.ACCEPT,
                reason="All classes explored and (where possible) synthesised; "
                       "accepting the grounded, correct best.",
                novelty_assessment=nov, drift_flags=drift_flags)
        return Directive(Action.TERMINATE,
            reason="No correct approach found and the search space is exhausted.",
            novelty_assessment=nov, drift_flags=drift_flags)

    @staticmethod
    def _weak_workload(winners: dict) -> str:
        """A workload with no current winner, if any (an empty niche to aim at)."""
        return ""   # the engine/Builder will pick; kept simple for the det path

    @staticmethod
    def _drift_flags(s: Signals) -> list[str]:
        flags = []
        if s.stalled:
            flags.append("stalled")
        if s.patching:
            flags.append(f"patching:{s.consecutive_recombinations}")
        if s.recombination_dominant:
            flags.append("recombination-dominant")
        flags += [f"recurring:{fc}" for fc in s.recurring_failures]
        return flags

    @staticmethod
    def _novelty_note(s: Signals, current: list[Approach]) -> str:
        if s.last_round_mean_novelty is None:
            note = "No fresh proposals to assess this round."
        else:
            note = (f"Structural novelty={s.last_round_mean_novelty:.2f}, "
                    f"behavioural novelty="
                    f"{s.last_round_mean_behavioral_novelty if s.last_round_mean_behavioral_novelty is not None else '?'}.")
        recs = [a.id for a in current if a.is_recombination]
        note += (f" Recombination(s): {recs}." if recs
                 else " No structural recombination.")
        if s.recombination_dominant:
            note += " RECOMBINATION IS THE DOMINANT RECENT PATTERN."
        if s.last_round_synthesis_gain is not None:
            note += f" Last synthesis gain={s.last_round_synthesis_gain:.2f}."
        return note


# =========================================================================== #
# LLM roles
# =========================================================================== #

_BUILDER_SYS = (
    "You are the BUILDER in a recursive discovery engine. You write a single, "
    "correct, fast Python implementation of a function. You value genuinely "
    "novel algorithmic ideas over variations of what already exists. "
    "Respond with ONLY one fenced python code block -- no prose, no explanation.")

# v1: a distinct system prompt for genuine synthesis -- the model is told to
# COMBINE and REFRAME, and explicitly forbidden from returning a parent.
_SYNTH_BUILDER_SYS = (
    "You are the BUILDER acting as a SYNTHESIZER of code. You are given several "
    "prior implementations, each with its MEASURED per-workload performance and "
    "the workloads it WINS. Your job is to produce ONE genuinely HYBRID solver "
    "that COMBINES their strengths -- typically by detecting the input regime "
    "(e.g. the size of k, whether the input is already ordered, the value range) "
    "and dispatching to the strategy that wins that regime. It MUST match or beat "
    "EACH parent on that parent's best workload. Do NOT return any parent "
    "unchanged, and do NOT merely pick the best one. You MAY also INVENT a new "
    "structure -- a different counter / eviction / summary discipline (e.g. "
    "replace-the-minimum, decay, hashed buckets) -- not only dispatch between the "
    "parents. Respond with ONLY one fenced python code block.")

_ADVERSARY_SYS = (
    "You are the ADVERSARY in a recursive discovery engine. Your sole job is to "
    "find INPUTS on which a candidate function disagrees with its specification. "
    "You do NOT get to claim success -- a separate harness will execute your "
    "inputs against a trusted oracle and decide. Propose the trickiest inputs you "
    "can. Respond with ONLY a JSON array of objects, each "
    '{"items": [...], "k": <int>, "hypothesis": "<why this might break it>"}.')

_SYNTH_SYS = (
    "You are the SYNTHESIZER in a recursive discovery engine -- the only role "
    "allowed to change direction, declare convergence, or terminate. You are "
    "given OBJECTIVE, PRE-COMPUTED signals you CANNOT override and MUST address: "
    "stall, recurring failures, whether recombination is the dominant recent "
    "pattern (structural OR behavioural), whether a SYNTHESIS opportunity exists "
    "(distinct approaches winning distinct regimes), and the measured gain of any "
    "recent synthesis. Prefer SYNTHESIZE when strengths are split; prefer EXPLORE "
    "when work is behaviourally stale but no synthesis is available; only ACCEPT a "
    "grounded, correct winner. Respond with ONLY a JSON object: "
    '{"action": one of ["continue","synthesize","explore","reset","accept",'
    '"terminate"], "reason": "...", "instruction": "guidance for the Builder", '
    '"target_class": one of ["full-sort","heap","selection","bucket"] or null, '
    '"novelty_assessment": "genuine novelty vs recombination?", '
    '"drift_flags": ["..."]}')


def _parent_brief(p: Approach) -> str:
    ev = p.evidence
    times = ({w: round(t * 1e3, 2) for w, t in ev.timings.items()} if ev else {})
    wins = p.behavior.get("wins", []) if p.behavior else []
    quality = (f"  recall={ev.quality:.2f} mem={ev.peak_memory_bytes // 1024}KB"
               if ev else "")
    return (f"# {p.id} <{p.algo_class.value}>{quality}  wins={wins}  ms={times}\n"
            f"```python\n{p.code.strip()}\n```")


# v5: prompts + coercion for OPEN-VOCABULARY proposals (new ops / generators). The
# model authors a small source fragment; the ENGINE AST-screens + grounds it before
# it can join the vocabulary -- so a malformed or unsafe fragment is simply dropped.
_OP_PROPOSER_SYS = (
    "You are the BUILDER extending the primitive vocabulary of a streaming "
    "heavy-hitters engine. A policy keeps a bounded dict `counts` (item -> count); "
    "on overflow it may PRE-process, then chooses a victim key `vk`, pops it (its "
    "count is `vv`), and admits the new item `x` with an INHERITED count. Propose "
    "ONE genuinely new primitive OPERATION expected to improve recall on the listed "
    "weaknesses. Allowed kinds: 'inherit' = a Python EXPRESSION over `vv` that is "
    ">= 1 (e.g. max(1, vv*3//4)); 'victim' = statements that end by assigning `vk` "
    "a key of `counts`; 'pre' = statements that may delete keys from `counts`. Use "
    "ONLY `counts`, `x`, `vv`, `vk`, plain builtins and simple loops -- no imports, "
    "no I/O. Respond with ONLY a JSON object: "
    '{"kind":"inherit"|"victim"|"pre","name":"<short_identifier>",'
    '"fragment":"<expr>" or ["<line>", ...],"rationale":"<why>"}.')

_GEN_PROPOSER_SYS = (
    "You are the ADVERSARY inventing a NEW stream generator to attack a streaming "
    "heavy-hitters summary. Write the BODY of a function gen(n, rng) returning a "
    "list of `n` integers (rng is a seeded random.Random). Design a STRUCTURALLY "
    "new hostile pattern (e.g. heavy hitters that vanish then return, a drifting "
    "heavy set, decoy spikes) that collapses recall@k while a perfect counter still "
    "succeeds. Use ONLY `n`, `rng`, ints, lists and loops -- no imports, no I/O; the "
    "body MUST `return` a list. Respond with ONLY a JSON object: "
    '{"name":"<short_identifier>","body":"<python statements that return a list>",'
    '"rationale":"<why it is hard>"}.')


def _coerce_op_proposal(d, vocab, valid_kinds=("victim", "inherit", "pre"),
                        str_kinds=("inherit",)):
    """Validate a model's op JSON into an OpProposal (shape only; the engine still
    AST-screens + grounds it). `valid_kinds` / `str_kinds` are domain-provided so the
    same coercer serves heavy-hitters (victim/inherit/pre) and cache (score/update).
    Returns None on anything ill-formed."""
    if not isinstance(d, dict):
        return None
    kind = str(d.get("kind", "")).strip()
    name = str(d.get("name", "")).strip()
    frag = d.get("fragment")
    if kind not in valid_kinds or not name.isidentifier():
        return None
    if vocab.has_op(kind, name):
        return None
    if kind in str_kinds:
        if not isinstance(frag, str) or not frag.strip():
            return None
        frag = frag.strip()
    else:
        if isinstance(frag, str):
            frag = frag.splitlines()
        if not (isinstance(frag, list) and frag
                and all(isinstance(ln, str) for ln in frag)):
            return None
    return OpProposal(kind, name, frag, Provenance.PROPOSED_LLM,
                      str(d.get("rationale", ""))[:200])


def _coerce_gen_proposal(d, vocab):
    if not isinstance(d, dict):
        return None
    name = str(d.get("name", "")).strip()
    body = d.get("body")
    if not name.isidentifier() or vocab.has_gen(name):
        return None
    if not isinstance(body, str) or "return" not in body:
        return None
    return GenProposal(name, body, Provenance.PROPOSED_LLM,
                       str(d.get("rationale", ""))[:200])


class LLMBuilder:
    def propose(self, ctx: RoundContext) -> list[Approach]:
        assert ctx.llm is not None
        d = ctx.directive
        if d and d.action == Action.SYNTHESIZE:
            return self._synthesize(ctx)
        if d and d.action == Action.EXPLORE:
            return self._explore(ctx)
        return self._propose(ctx)

    def _propose(self, ctx: RoundContext) -> list[Approach]:
        dom = ctx.domain
        model = ctx.cfg.models["builder"]
        d = ctx.directive
        steer = ""
        if d and d.instruction:
            steer += f"\nDIRECTIVE: {d.instruction}"
        if d and d.target_class:
            steer += f"\nUse a {d.target_class.value} style approach."
        if ctx.signals and ctx.signals.classes_tried:
            steer += ("\nAlready-tried classes (prefer something different): "
                      f"{ctx.signals.classes_tried}")
        # v10 P4: feed a previous CONTRACT rejection back verbatim, so the model
        # fixes the interface instead of repeating it for the rest of the run.
        cerr = (ctx.guidance or {}).get("contract_error")
        if cerr:
            steer += (f"\nYOUR PREVIOUS CANDIDATE WAS REJECTED BEFORE EXECUTION: "
                      f"{cerr}\nFollow the CONTRACT exactly -- define the required "
                      f"symbols at module top level and nothing else there.")
        user = (f"PROBLEM: {dom.problem_desc}\n\nCONTRACT:\n{dom.contract}\n\n"
                f"WHAT HAS BEEN TRIED:\n{ctx.memory.digest()}\n{steer}\n\n"
                f"Write ONE implementation now.")
        return self._emit(ctx, _BUILDER_SYS, user, Origin.BUILDER,
                          parents=[], label="llm")

    def _synthesize(self, ctx: RoundContext) -> list[Approach]:
        dom = ctx.domain
        d = ctx.directive
        ids = (d.parent_ids if d and d.parent_ids
               else [a.id for a in ctx.winners.values()])
        parents = [p for p in (ctx.memory.get(i) for i in ids) if p] or \
            list(ctx.winners.values())
        briefs = "\n\n".join(_parent_brief(p) for p in parents)
        user = (f"PROBLEM: {dom.problem_desc}\n\nCONTRACT:\n{dom.contract}\n\n"
                f"SYNTHESIS BRIEF: {d.synthesis_brief if d else ''}\n\n"
                f"PARENTS TO COMBINE (with measured behaviour):\n{briefs}\n\n"
                f"CONCEPTS LEARNED:\n{ctx.memory.lessons.digest()}\n\n"
                f"Now write ONE adaptive hybrid that wins EVERY regime.")
        return self._emit(ctx, _SYNTH_BUILDER_SYS, user, Origin.SYNTHESIS,
                          parents=[p.id for p in parents], label="hybrid",
                          temp=0.2, brief=(d.synthesis_brief if d else ""))

    def _explore(self, ctx: RoundContext) -> list[Approach]:
        dom = ctx.domain
        d = ctx.directive
        elites = list(ctx.winners.values()) or (
            [ctx.memory.best] if ctx.memory.best else [])
        profiles = "\n".join(_parent_brief(p) for p in elites[:4])
        target = (f" Aim to WIN the '{d.explore_target}' workload."
                  if d and d.explore_target else "")
        user = (f"PROBLEM: {dom.problem_desc}\n\nCONTRACT:\n{dom.contract}\n\n"
                f"EXISTING approaches and their measured behaviour:\n{profiles}\n\n"
                f"CONCEPTS:\n{ctx.memory.lessons.digest()}\n\n"
                f"Produce ONE solver that BEHAVES DIFFERENTLY from all of the "
                f"above (a different performance profile), not just different "
                f"source.{target} Be inventive; still satisfy the contract.")
        # Higher temperature: exploration wants diversity, not the safe answer.
        return self._emit(ctx, _BUILDER_SYS, user, Origin.BUILDER,
                          parents=[], label="explore", temp=0.9)

    def _emit(self, ctx: RoundContext, sys: str, user: str, origin: Origin,
              parents: list[str], label: str, temp: float = 0.4,
              brief: str = "") -> list[Approach]:
        model = ctx.cfg.models["builder"]
        out: list[Approach] = []
        for j in range(ctx.cfg.builder_fanout):
            try:
                reply = ctx.llm.chat(model, sys, user,
                                     temperature=min(temp + 0.2 * j, 1.0))
                code = extract_code(reply)
            except LLMError as e:
                code = f"# builder error: {e}\n"
            cls = ctx.domain.classify(code)
            out.append(Approach(
                id=f"i{ctx.iteration}.{j}_{label}", code=code,
                rationale=f"LLM Builder ({model}) {label} (classified {cls.value}).",
                algo_class=cls, origin=origin, parent_ids=parents,
                created_iter=ctx.iteration, synthesis_brief=brief))
        return out

    # -- v5: the LLM Builder authors a NEW primitive op as source ------------ #
    def propose_operations(self, ctx: RoundContext) -> list:
        if ctx.vocab is None or ctx.llm is None:
            return []
        dom = ctx.domain
        model = ctx.cfg.models["builder"]
        weak = ", ".join(ctx.guidance.get("weak", [])) or "general recall"
        gap = ctx.guidance.get("frontier_gap", 0.0)
        causal = ctx.guidance.get("causal")
        recent = ctx.guidance.get("recent_generators", [])
        # v7: the op vocabulary KINDS + the proposer system prompt are DOMAIN-PROVIDED
        # so the LLM proposes ops that fit THIS domain (heavy-hitters: victim/inherit/
        # pre; cache: score/update), not a hardcoded set.
        kinds = getattr(dom, "op_kinds", ()) or ("victim", "inherit", "pre")
        str_kinds = getattr(dom, "op_str_kinds", ()) or ("inherit",)
        sys_prompt = getattr(dom, "op_proposer_sys", "") or _OP_PROPOSER_SYS
        existing = "; ".join(f"{k}: {ctx.vocab.op_names(k)}" for k in kinds)
        causal_line = (f"CONCRETE FAILURE TO FIX: {causal.brief()}\n"
                       if causal is not None and causal.has_signal else "")
        target_line = (f"SELF-PLAY: author an op so a policy using it BEATS the best "
                       f"human on the Adversary's recent generators {recent}.\n"
                       if recent else "")
        user = (f"{causal_line}{target_line}"
                f"(secondary) weakness tags: {weak}; frontier_gap={gap:.2f}.\n"
                f"EXISTING ops -- {existing}.\n"
                f"Propose ONE NEW operation (not already listed) that targets the "
                f"concrete failure above.")
        try:
            data = extract_json(ctx.llm.chat(model, sys_prompt, user,
                                             temperature=0.7))
        except (LLMError, ValueError):
            return []
        out = []
        for d in (data if isinstance(data, list) else [data]):
            p = _coerce_op_proposal(d, ctx.vocab, valid_kinds=tuple(kinds),
                                    str_kinds=tuple(str_kinds))
            if p:
                out.append(p)
        return out


_ADV_WORKLOAD_SYS = (
    "You are the ADVERSARY in a discovery engine for streaming heavy-hitters. "
    "Your job is to attack the EVALUATION METRIC: choose stream configurations on "
    "which the given candidate's recall will COLLAPSE while a good method would "
    "still succeed (e.g. heavy hitters arriving late, hash-collision-heavy keys, "
    "more heavy items than a bounded summary can hold). You do NOT decide success "
    "-- a harness measures recall. Respond with ONLY a JSON array of objects, each "
    '{"dist": <one of the allowed names>, "n": <int>, "k": <int>, '
    '"hypothesis": "why this collapses the candidate"}.')

_ADV_DISTS = ["uniform", "near_uniform", "zipf", "few_heavy", "bursty",
              "front_loaded", "freq_late", "collision", "heavy_skew_overflow"]


class LLMAdversary:
    MAX_CASES = 8

    def attack(self, approach: Approach, ctx: RoundContext) -> AdversaryReport:
        assert ctx.llm is not None
        if getattr(ctx.domain, "uses_frontier", False):
            return self._attack_metric(approach, ctx)
        report = AdversaryReport(approach_id=approach.id)
        model = ctx.cfg.models["adversary"]
        user = (f"CONTRACT:\n{ctx.domain.contract}\n\n"
                f"CANDIDATE IMPLEMENTATION:\n```python\n{approach.code}\n```\n\n"
                f"Propose up to {self.MAX_CASES} adversarial (items, k) inputs "
                f"most likely to expose a disagreement with the contract/oracle.")
        try:
            reply = ctx.llm.chat(model, _ADVERSARY_SYS, user, temperature=0.6)
            proposals = extract_json(reply)
        except (LLMError, ValueError) as e:
            report.findings.append(AdversaryFinding(
                hypothesis=f"adversary unavailable: {e}", broke=False,
                grounded=False))
            return report

        cases = self._sanitize(proposals)
        # Surface how many proposals were rejected as ill-typed, so a model that
        # keeps hallucinating non-numeric cases is at least visible rather than
        # silently producing zero pressure.
        n_proposed = len(proposals) if isinstance(proposals, list) else 0
        dropped = n_proposed - len(cases)
        if dropped > 0:
            report.findings.append(AdversaryFinding(
                hypothesis=f"{dropped}/{n_proposed} proposed case(s) rejected as "
                           f"ill-typed (non-numeric items / bad k)",
                broke=False, grounded=False))
        if not cases:
            return report

        # The harness -- not the model -- decides what actually broke.
        res = check_cases(approach.code, [(n, it, k) for n, it, k, _ in cases],
                          ctx.domain.grounding_spec, ctx.cfg)
        case_results = res.get("cases", {})
        for name, items, k, hyp in cases:
            rec = case_results.get(name)
            if rec is None:
                continue
            broke = not rec.get("ok", False)
            report.findings.append(AdversaryFinding(
                hypothesis=hyp, broke=broke,
                grounded=res.get("import_ok", False),
                counterexample=({"items": list(items), "k": k,
                                 "expected": rec.get("expected"),
                                 "got": rec.get("got"),
                                 "error": rec.get("error")} if broke else None),
                notes=("mutated input" if rec.get("mutated") else "")))
        return report

    def _attack_metric(self, approach: Approach, ctx: RoundContext
                       ) -> AdversaryReport:
        """v2: attack the metric. The model proposes hostile STREAM configs; the
        harness measures the candidate's recall on them. Always includes the
        domain's deterministic arsenal so there is baseline pressure even if the
        model fails. The engine decides what counts as a collapse."""
        report = AdversaryReport(approach_id=approach.id)
        model = ctx.cfg.models["adversary"]
        user = (f"CONTRACT:\n{ctx.domain.contract}\n\n"
                f"CANDIDATE:\n```python\n{approach.code}\n```\n\n"
                f"Allowed dist names: {_ADV_DISTS}\n"
                f"Propose up to 4 stream configs to collapse this candidate.")
        specs = []
        try:
            specs = self._sanitize_workloads(extract_json(
                ctx.llm.chat(model, _ADV_WORKLOAD_SYS, user, temperature=0.6)))
        except (LLMError, ValueError):
            specs = []
        # Deterministic arsenal + v3 param attacks + v4 generator-level attacks
        # + LLM-proposed.
        wl0 = ctx.domain.grounding_spec.workloads
        k0 = wl0[0]["k"] if wl0 else 10
        param = (ctx.domain.adversary_param_attacks(k0)
                 if getattr(ctx.domain, "adversary_param_attacks", None) else [])
        gens = (ctx.domain.adversary_generators(k0)
                if getattr(ctx.domain, "adversary_generators", None) else [])
        wls, seen = [], set()
        for w in (list(ctx.domain.adversary_workloads) + param + gens
                  + list(ctx.proposed_generators) + specs):  # v5: proposed gens too
            if w["name"] not in seen:
                seen.add(w["name"])
                wls.append(w)
        recalls = measure_recall_on(approach.code, wls,
                                    ctx.live_spec(), ctx.cfg)
        for w in wls:
            r = recalls.get(w["name"])
            if r is None:
                continue
            report.findings.append(AdversaryFinding(
                hypothesis=w["name"], broke=r < ctx.cfg.metric_collapse_floor,
                grounded=True, counterexample={"workload": w, "recall": r}))
        return report

    # -- v5: the LLM Adversary authors a NEW generator as source ------------- #
    def propose_generators(self, ctx: RoundContext) -> list:
        if ctx.vocab is None or ctx.llm is None:
            return []
        model = ctx.cfg.models["adversary"]
        weak = ", ".join(ctx.guidance.get("weak", [])) or "general recall"
        # v6 self-play: name the recently-proposed ops so the Adversary can author a
        # generator specifically to DEFEAT them.
        recent_ops = ctx.vocab.proposed_op_names() if ctx.vocab else set()
        ops_line = (f"DEFEAT these recently-proposed operations: "
                    f"{sorted(recent_ops)}.\n" if recent_ops else "")
        user = (f"{ops_line}TARGET WEAKNESSES: {weak}.\n"
                f"Existing generators: {list(ctx.vocab.generators.keys())}.\n"
                f"Invent ONE new generator (not already listed) that exposes a "
                f"weakness those operations do not fix.")
        try:
            data = extract_json(ctx.llm.chat(model, _GEN_PROPOSER_SYS, user,
                                             temperature=0.8))
        except (LLMError, ValueError):
            return []
        out = []
        for d in (data if isinstance(data, list) else [data]):
            p = _coerce_gen_proposal(d, ctx.vocab)
            if p:
                out.append(p)
        return out

    def _sanitize_workloads(self, proposals) -> list[dict]:
        """Coerce model JSON into trustworthy stream specs (known dist, sane n/k)."""
        out: list[dict] = []
        if not isinstance(proposals, list):
            return out
        for i, p in enumerate(proposals[:4]):
            if not isinstance(p, dict):
                continue
            dist = str(p.get("dist", "")).strip()
            n, k = p.get("n"), p.get("k")
            if dist not in _ADV_DISTS:
                continue
            if not isinstance(n, int) or not isinstance(k, int):
                continue
            n = max(1000, min(n, 60000))
            k = max(1, min(k, 100))
            out.append({"name": f"llmadv{i}_{dist}", "n": n, "k": k, "dist": dist})
        return out

    def _sanitize(self, proposals) -> list[tuple]:
        """Coerce the model's JSON into trustworthy (name, items, k, hyp) tuples.
        Rejects anything that is not a list of comparable numbers + an int k."""
        out: list[tuple] = []
        if not isinstance(proposals, list):
            return out
        for i, p in enumerate(proposals[: self.MAX_CASES]):
            if not isinstance(p, dict):
                continue
            items, k = p.get("items"), p.get("k")
            if not isinstance(items, list) or not isinstance(k, int):
                continue
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                       for x in items):
                continue
            if k < 0 or len(items) > 5000:    # keep cases small + safe
                continue
            out.append((f"llmadv{i}", items, k, str(p.get("hypothesis", ""))[:200]))
        return out


class LLMSynthesizer:
    """The LLM decides, but inside hard guardrails: it is fed the objective
    signals, and the engine rejects decisions that contradict them (you cannot
    ACCEPT with no correct approach; you cannot CONTINUE through confirmed local
    patching). The deterministic policy is the fallback when the model fails."""

    def __init__(self):
        self._fallback = DetSynthesizer()

    def decide(self, ctx: RoundContext) -> Directive:
        assert ctx.llm is not None
        s = ctx.signals
        model = ctx.cfg.models["synthesizer"]
        round_summary = "\n".join(f"  - {a.short()}" for a in ctx.current_round) \
            or "  (no proposals this round)"
        wsum = "; ".join(f"{w}->{a.id}<{a.algo_class.value}>"
                         for w, a in ctx.winners.items()) or "(none)"
        user = (
            f"PROBLEM: {ctx.domain.problem_desc}\n\n"
            f"OBJECTIVE SIGNALS (you cannot override these):\n  {s.headline()}\n"
            f"  untried classes: {[c.value for c in untried_classes(s, ctx.domain)]}\n"
            f"  per-workload winners: {wsum}\n"
            f"  synthesis_available={s.synthesis_available} "
            f"recombination_dominant={s.recombination_dominant}\n\n"
            f"THIS ROUND'S EVALUATED PROPOSALS:\n{round_summary}\n\n"
            f"BOUNDED MEMORY:\n{ctx.memory.digest()}\n\n"
            f"Decide the next action. If recombination is dominant and a synthesis "
            f"opportunity exists, you should SYNTHESIZE. Any flagged stall / "
            f"recombination / recurring failure -- your reason MUST address it.")
        try:
            reply = ctx.llm.chat(model, _SYNTH_SYS, user, temperature=0.3)
            data = extract_json(reply)
            directive = self._parse(data, s)
            # Fill concrete synthesis material if the model asked to synthesize
            # but didn't name parents -- the Builder needs something to combine.
            if directive.action == Action.SYNTHESIZE and not directive.parent_ids:
                directive.parent_ids = synthesis_parents(ctx.winners)
                directive.synthesis_brief = (directive.synthesis_brief
                    or synthesis_brief(ctx.winners, ctx.domain))
        except (LLMError, ValueError, KeyError) as e:
            directive = self._fallback.decide(ctx)
            directive.reason = f"[LLM synthesizer fell back: {e}] " + directive.reason
        # BOTH the LLM and the fallback decision pass through the guardrails --
        # the fallback must not be a hole in the protective layer.
        return self._enforce_guardrails(directive, ctx)

    def _parse(self, data: dict, s: Signals) -> Directive:
        # Default an unrecognised/misspelled action to CONTINUE rather than
        # raising (which would discard the model's reasoning via the fallback).
        action_str = str(data.get("action", "continue")).lower().strip()
        try:
            action = Action(action_str)
        except ValueError:
            action = Action.CONTINUE
        tc = data.get("target_class")
        target_class = None
        if tc:
            try:
                target_class = AlgoClass(str(tc).lower().strip())
            except ValueError:
                target_class = None
        # Always merge the engine-computed drift flags in, so the model can never
        # silently drop a flagged condition.
        flags = list(data.get("drift_flags", []) or [])
        flags += [f for f in DetSynthesizer._drift_flags(s) if f not in flags]
        return Directive(
            action=action,
            reason=str(data.get("reason", "")).strip()[:1000],
            instruction=str(data.get("instruction", "")).strip()[:1000],
            target_class=target_class,
            synthesis_brief=str(data.get("synthesis_brief", "")).strip()[:1000],
            explore_target=str(data.get("explore_target", "")).strip()[:100],
            novelty_assessment=str(data.get("novelty_assessment", "")).strip()[:1000],
            drift_flags=flags)

    def _enforce_guardrails(self, d: Directive, ctx: RoundContext) -> Directive:
        s = ctx.signals
        mem = ctx.memory
        untried = untried_classes(s, ctx.domain)

        # G1: cannot ACCEPT without a grounded, correct winner -- and "winner"
        # means actually correct now, not merely non-None (an approach can be
        # downgraded by the Adversary after it became best).
        if d.action == Action.ACCEPT and not (mem.best and mem.best.is_correct):
            d.action = Action.CONTINUE
            d.reason = "[guardrail: nothing correct to accept] " + d.reason

        # G2: confirmed local patching cannot be waved through with CONTINUE.
        if s.patching and d.action in (Action.CONTINUE, Action.SYNTHESIZE):
            if untried:
                d.action = Action.RESET
                d.target_class = d.target_class or untried[0]
                d.reason = ("[guardrail: patching detected -> forced reset] "
                            + d.reason)
            else:
                d.action = Action.ACCEPT if mem.best else Action.TERMINATE
                d.reason = ("[guardrail: patching + nothing new -> "
                            f"{d.action.value}] " + d.reason)

        # G3: a forced RESET needs a concrete target. Supply one if missing; but
        # if there is NOTHING left to reset to, RESET is meaningless -> resolve to
        # a real terminal action instead of silently degrading to priority order.
        if d.action == Action.RESET and d.target_class is None:
            if untried:
                d.target_class = untried[0]
            else:
                d.action = Action.ACCEPT if (mem.best and mem.best.is_correct) \
                    else Action.TERMINATE
                d.reason = ("[guardrail: RESET with no target and no untried "
                            f"classes -> {d.action.value}] " + d.reason)

        # G4 (v1): recombination is the DOMINANT measured pattern -> the model may
        # NOT coast with CONTINUE/ACCEPT. It must do real work: synthesise if the
        # opportunity exists, else explore a new niche, else (truly exhausted)
        # accept the grounded best or terminate. This is the core v1 anti-collapse
        # upgrade, and it keys off a signal the LLM cannot fake.
        have_hybrid = bool(mem.best and mem.best.algo_class in _SYNTH_CLASSES
                           and mem.best.is_correct)
        synthesis_failed = (any(a.algo_class in _SYNTH_CLASSES
                                for a in mem.approaches) and not have_hybrid)
        if s.recombination_dominant and d.action in (Action.CONTINUE, Action.ACCEPT):
            if s.synthesis_available and not have_hybrid and not synthesis_failed:
                d.action = Action.SYNTHESIZE
                d.parent_ids = d.parent_ids or synthesis_parents(ctx.winners)
                d.synthesis_brief = (d.synthesis_brief
                                     or synthesis_brief(ctx.winners, ctx.domain))
                d.reason = "[guardrail: recombination dominant -> synthesize] " + d.reason
            elif untried:
                d.action = Action.EXPLORE
                d.target_class = d.target_class or untried[0]
                d.reason = "[guardrail: recombination dominant -> explore] " + d.reason
            elif d.action == Action.CONTINUE:
                d.action = (Action.ACCEPT if (mem.best and mem.best.is_correct)
                            else Action.TERMINATE)
                d.reason = (f"[guardrail: recombination dominant, nothing new -> "
                            f"{d.action.value}] " + d.reason)

        # G5: a SYNTHESIZE with no parents to combine is empty -> downgrade.
        if d.action == Action.SYNTHESIZE and not d.parent_ids:
            d.action = Action.EXPLORE if untried else Action.CONTINUE
            d.reason = "[guardrail: no synthesis material -> " \
                       f"{d.action.value}] " + d.reason

        # G6 (v1): convergence. A free LLM Synthesizer can keep CHOOSING to
        # synthesise to "break the plateau" forever. Once a correct hybrid is the
        # best AND synthesis has already been tried (>=2 hybrids exist), further
        # synthesis is not paying off -> redirect to genuine exploration of any
        # untried class, or accept the winning hybrid. This is the LLM-side analog
        # of the deterministic `synthesis_failed` guard, and guarantees the loop
        # converges instead of re-synthesising worse variants until exhaustion.
        n_hybrids = sum(1 for a in mem.approaches
                        if a.algo_class in _SYNTH_CLASSES)
        if d.action == Action.SYNTHESIZE and (
                (have_hybrid and n_hybrids >= 2)
                or s.synthesis_attempts >= ctx.cfg.max_synthesis_attempts):
            d.action = Action.EXPLORE if untried else (
                Action.ACCEPT if (mem.best and mem.best.is_correct)
                else Action.TERMINATE)
            d.reason = ("[guardrail: synthesis budget spent / winning hybrid "
                        f"exists -> {d.action.value}] " + d.reason)

        # G7 (v1): CONVERGENCE. Once progress has genuinely stalled and we hold a
        # correct best, stop exploring/synthesising indefinitely and ACCEPT it --
        # UNLESS a real, not-yet-attempted synthesis opportunity remains (give
        # that one shot first). The human checkpoint can still push further. This
        # mirrors the deterministic stall->accept and guarantees the LLM loop
        # converges instead of running to max_iterations.
        productive_synth = (s.synthesis_available and not have_hybrid
                            and s.synthesis_attempts < ctx.cfg.max_synthesis_attempts)
        if (s.stalled and mem.best and mem.best.is_correct
                and not productive_synth
                and d.action in (Action.CONTINUE, Action.EXPLORE,
                                 Action.SYNTHESIZE)):
            d.action = Action.ACCEPT
            d.reason = "[guardrail: stalled with a correct best -> accept] " + d.reason
        return d


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

@dataclass
class Roles:
    builder: Builder
    adversary: Adversary
    synthesizer: Synthesizer


def make_roles(cfg: Config) -> Roles:
    if cfg.backend == "llm":
        return Roles(LLMBuilder(), LLMAdversary(), LLMSynthesizer())
    return Roles(DetBuilder(), DetAdversary(), DetSynthesizer())
