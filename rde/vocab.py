"""v5: the OPEN VOCABULARY -- a provenance-tracked registry of the primitive
operations the Builder composes policies from, and the stream generators the
Adversary attacks with.

v4 fixed the basis: three hand-written dicts of victim / inherit / pre ops, and a
fixed `gen:` generator DSL. v5 keeps those as the SEED vocabulary (provenance
`human`) but lets the Builder and Adversary *extend* it at runtime by proposing
new operations / generators **as source code**. Every proposal is AST-screened and
empirically grounded before it is admitted here -- nothing influences the frontier
or the Adversary on the strength of a proposal alone.

This module is intentionally pure data + bookkeeping: the *screening* and
*grounding* gates live in the engine (which owns the harness), and the *proposers*
live in the roles / domain (which own the ideas). The registry only records what
survived, and who authored it -- so the run report can say, honestly, "N of the
ops on the accepted policy were proposed by the system, not designed by a human."
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class Provenance(str, enum.Enum):
    """Who authored a vocabulary entry. The whole point of v5 is to be able to
    tell these apart in the final assessment."""
    HUMAN = "human"                 # shipped in the seed vocabulary
    PROPOSED_DET = "proposed-det"   # invented by the deterministic meta-proposer
    PROPOSED_LLM = "proposed-llm"   # invented by the LLM Builder / Adversary

    @property
    def is_proposed(self) -> bool:
        return self is not Provenance.HUMAN


# Op kinds (the three composable slots of an eviction policy).
VICTIM, INHERIT, PRE = "victim", "inherit", "pre"
OP_KINDS = (VICTIM, INHERIT, PRE)


@dataclass
class OpEntry:
    """One primitive operation fragment in the vocabulary.

    `fragment` is the spliced-in source: a list[str] of statement lines for
    `victim` / `pre` ops, or a single str expression for an `inherit` op (the
    count an admitted item inherits). The shape mirrors v4's _VICTIM_OPS /
    _INHERIT_OPS / _PRE_OPS so assembly is unchanged."""
    kind: str                       # VICTIM | INHERIT | PRE
    name: str
    fragment: object                # list[str] (victim/pre) | str (inherit)
    origin: Provenance = Provenance.HUMAN
    iteration: int = 0              # when it was admitted (0 == seed)
    note: str = ""                  # the guidance that motivated a proposal
    used_on_frontier: bool = False  # set True if a frontier member uses it
    # v6: explicit utility vs the HUMAN Pareto frontier on the hardened suite.
    utility: str = "unknown"        # beats_human | matches_human | below_human
    utility_margin: float = 0.0     # best recall delta vs the human frontier @ <= mem
    # v10 P3: SECOND tier vs the best human INCLUDING the adaptive references
    # (ARC/LeCaR) at <= equal declared cost. `utility` keeps the v6 STATIC
    # semantics for cross-run history continuity; THIS is the headline tier.
    utility_adaptive: str = "unknown"
    utility_adaptive_margin: float = 0.0
    # v11 A3: boundary-straddling multi-seed verdicts are ANNOTATED, everywhere.
    utility_fragile: bool = False
    utility_adaptive_fragile: bool = False


@dataclass
class GenEntry:
    """One stream generator in the vocabulary. Human generators are the built-in
    `gen:`/`param:` DSL (no source block -- they live inside gen_workload). A
    PROPOSED generator carries `src_block`: the AST-screened `_genprop_<name>`
    definition appended to the live spec's workload_gen_src, dispatched by the
    `prop:<name>` dist."""
    name: str
    dist: str                       # e.g. "prop:comeback" or "gen:phase|..."
    origin: Provenance = Provenance.HUMAN
    src_block: str = ""             # the appended source (proposed only)
    iteration: int = 0
    note: str = ""
    exposed_a_candidate: bool = False  # set True if it ever collapsed someone


@dataclass
class Vocabulary:
    """The live, per-run registry. Seeded from the domain's human ops/generators,
    grown by admitted proposals.

    v7: op kinds are GENERIC -- ``slots`` maps a kind name to its ops, created on
    demand, so a domain may use any op-slot names (heavy-hitters: victim/inherit/pre;
    cache-eviction: score/update/admit). The victim/inherit/pre PROPERTIES are kept
    for backward compatibility with the heavy-hitters domain + tests."""
    slots: dict = field(default_factory=dict)        # kind -> {name: OpEntry}
    generators: dict[str, GenEntry] = field(default_factory=dict)
    # admission order of PROPOSED ops, as (kind, name) -- so the search composes
    # them in the order the Builder proposed (guidance-prioritised) them, not in
    # arbitrary slot order.
    _proposed_order: list = field(default_factory=list)

    # -- op access ---------------------------------------------------------- #
    def _slot(self, kind: str) -> dict[str, OpEntry]:
        """READ-ONLY slot view -- never creates a slot, so merely reading a kind (or
        the victim/inherit/pre back-compat properties) cannot inject phantom empty
        slots into a domain that does not use that kind."""
        return self.slots.get(kind, {})

    def _slot_mut(self, kind: str) -> dict[str, OpEntry]:
        """Writable slot (created on demand) -- used only when ADDING an op."""
        return self.slots.setdefault(kind, {})

    # Backward-compat accessors for the heavy-hitters op kinds.
    @property
    def victim(self) -> dict:
        return self._slot(VICTIM)

    @property
    def inherit(self) -> dict:
        return self._slot(INHERIT)

    @property
    def pre(self) -> dict:
        return self._slot(PRE)

    def has_op(self, kind: str, name: str) -> bool:
        return name in self._slot(kind)

    def name_in_any_slot(self, name: str) -> bool:
        """v5: a proposed op name must be unique across ALL slots, else name-keyed
        provenance (op_entry / used_on_frontier) would be ambiguous."""
        return any(name in slot for slot in self.slots.values())

    def add_op(self, entry: OpEntry) -> None:
        self._slot_mut(entry.kind)[entry.name] = entry      # creates the slot if new
        if entry.origin.is_proposed:
            self._proposed_order.append((entry.kind, entry.name))

    def fragment(self, kind: str, name: str):
        return self._slot(kind)[name].fragment

    def op_names(self, kind: str) -> list[str]:
        return list(self._slot(kind).keys())

    def proposed_ops(self) -> list[OpEntry]:
        """Admitted proposed ops, in the order they were admitted (which is the
        order the Builder proposed them -- highest-priority first)."""
        return [self._slot(kind)[name] for (kind, name) in self._proposed_order
                if name in self._slot(kind)]

    def proposed_op_names(self) -> set:
        return {e.name for e in self.proposed_ops()}

    def op_entry(self, name: str):
        """Resolve an op by name across all slots (names are unique in practice)."""
        for slot in self.slots.values():
            if name in slot:
                return slot[name]
        return None

    # -- generator access --------------------------------------------------- #
    def has_gen(self, name: str) -> bool:
        return name in self.generators

    def add_gen(self, entry: GenEntry) -> None:
        self.generators[entry.name] = entry

    def proposed_generators(self) -> list[GenEntry]:
        return [g for g in self.generators.values() if g.origin.is_proposed]

    # -- provenance reporting ----------------------------------------------- #
    def provenance_summary(self) -> dict:
        """Counts for the run report: human vs system-proposed, and how many of
        the proposed entries actually mattered (used on the frontier / exposed a
        candidate)."""
        ops = [e for slot in self.slots.values() for e in slot.values()]
        prop_ops = [e for e in ops if e.origin.is_proposed]
        prop_gens = self.proposed_generators()
        return {
            "human_ops": sum(1 for e in ops if not e.origin.is_proposed),
            "proposed_ops": len(prop_ops),
            "proposed_ops_used": sum(1 for e in prop_ops if e.used_on_frontier),
            # v6: explicit utility tally over the proposed ops (STATIC tier).
            "beats_human": sum(1 for e in prop_ops if e.utility == "beats_human"),
            "matches_human": sum(1 for e in prop_ops if e.utility == "matches_human"),
            "below_human": sum(1 for e in prop_ops if e.utility == "below_human"),
            # v10 P3: the ADAPTIVE tier (vs best human incl. ARC/LeCaR) -- headline.
            "beats_adaptive_human": sum(
                1 for e in prop_ops if e.utility_adaptive == "beats_human"),
            "matches_adaptive_human": sum(
                1 for e in prop_ops if e.utility_adaptive == "matches_human"),
            "below_adaptive_human": sum(
                1 for e in prop_ops if e.utility_adaptive == "below_human"),
            "proposed_op_utility_adaptive": {e.name: e.utility_adaptive
                                             for e in prop_ops},
            # v10 P5: persist the FRAGMENTS for provenance auditing.
            "proposed_op_fragments": {e.name: str(e.fragment)[:300]
                                      for e in prop_ops},
            "human_generators": sum(1 for g in self.generators.values()
                                    if not g.origin.is_proposed),
            "proposed_generators": len(prop_gens),
            "proposed_generators_exposing": sum(1 for g in prop_gens
                                                if g.exposed_a_candidate),
            "proposed_op_names": [e.name for e in prop_ops],
            "proposed_op_utility": {e.name: e.utility for e in prop_ops},
            "proposed_generator_names": [g.name for g in prop_gens],
        }


# --------------------------------------------------------------------------- #
# Proposals -- what a proposer emits BEFORE screening/grounding admits it.
# --------------------------------------------------------------------------- #

@dataclass
class OpProposal:
    """A Builder's candidate new operation, not yet screened or grounded."""
    kind: str                       # VICTIM | INHERIT | PRE
    name: str
    fragment: object                # list[str] | str (same shape as OpEntry)
    origin: Provenance = Provenance.PROPOSED_DET
    note: str = ""                  # which weakness / gap it targets


@dataclass
class GenProposal:
    """An Adversary's candidate new generator, not yet screened or grounded.
    `body` is the function BODY operating on (n, rng) and returning a list[int]."""
    name: str
    body: str
    origin: Provenance = Provenance.PROPOSED_DET
    note: str = ""
