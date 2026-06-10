"""The Domain interface -- everything the engine needs to explore one problem,
bundled so the orchestration loop stays problem-agnostic.

The engine in `engine.py` never imports `topk`; it only ever sees a `Domain`.
Adding a second problem (say, "near-sorted sorting") is therefore a matter of
writing one module that constructs a `Domain`, with zero changes to the loop,
the roles, the memory, or the protective mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .grounding import GroundingSpec
from .types import AlgoClass


@dataclass
class Strategy:
    """A concrete, runnable candidate the *deterministic* Builder can draw from.
    (The LLM Builder writes its own code instead.) `sound` is only the author's
    claim -- the grounding harness is what actually decides correctness."""
    name: str
    algo_class: AlgoClass
    code: str
    rationale: str
    sound: bool = True


@dataclass
class Domain:
    name: str
    problem_desc: str
    contract: str                          # exact function contract, fed to roles
    grounding_spec: GroundingSpec          # oracle + workloads + trusted suite
    classify: Callable[[str], AlgoClass]   # code -> algorithmic family

    # Deterministic-backend repertoire.
    seed_strategy: str = ""
    strategies: dict[str, Strategy] = field(default_factory=dict)
    builder_priority: list[str] = field(default_factory=list)
    # Adversary's deterministic arsenal: [(name, items, k), ...]
    adversary_cases: list = field(default_factory=list)

    # --- v1 hooks (optional; enable stronger synthesis + conceptual memory) - #
    # Deterministic synthesis: given parent Approaches, compose a hybrid solver
    # source, or None if no meaningful hybrid is possible.
    compose: Callable | None = None
    # Map a workload name to a human-readable regime (e.g. "small-k").
    regime: Callable[[str], str] | None = None
    # Explain a failure structurally: Approach -> (pattern, regime, detail) | None.
    explain_failure: Callable | None = None

    # --- v2 hooks (frontier-based discovery) -------------------------------- #
    # When True the engine rewards advancing the (quality, memory) Pareto frontier
    # rather than a scalar fitness, and the Adversary attacks the eval metric.
    uses_frontier: bool = False
    # Memory budget for the "recommended" point = this fraction of the seed's
    # peak memory (resolved at seed time).
    memory_budget_frac: float = 0.5
    # Algorithmic families the Builder should explore (defaults to the top-K set
    # in roles.py when None).
    meaningful_classes: list | None = None
    # Adversary's deterministic arsenal of WORKLOAD specs that attack the metric:
    # [{"name","n","k","dist"}, ...] (skewed/late/collision streams).
    adversary_workloads: list = field(default_factory=list)

    # --- v3 hooks (open-ended synthesis + generative adversary) ------------- #
    # Compositional grammar: assemble NEW programs from primitives. Returns a list
    # of (label, source) pairs the Builder grounds and the frontier filters.
    synthesize_programs: Callable | None = None
    # Generative adversary: compose NOVEL parameterised workload specs at runtime
    # given k. Returns [{"name","n","k","dist"}, ...].
    adversary_param_attacks: Callable | None = None
    # v4 generator-level adversary: compose NEW attack TYPES (generator-DSL specs)
    # at runtime given k. Adding any of these triggers frontier re-scoring.
    adversary_generators: Callable | None = None

    # --- v5 hooks (OPEN VOCABULARY: propose new ops/generators as source) ---- #
    # Build the per-run vocabulary (human seed ops + generators).
    seed_vocabulary: Callable | None = None
    # Compose policies over the live vocabulary (incl. admitted proposed ops).
    # Returns [(label, code, ops_used)].
    assemble_from_vocab: Callable | None = None
    # Builder op-proposer (deterministic): given the vocab + guidance, return new
    # OpProposals; `op_probe` builds a screenable/groundable probe policy from one;
    # `admit_op` adds a grounded+safe op to the vocabulary.
    propose_operations: Callable | None = None
    op_probe: Callable | None = None
    admit_op: Callable | None = None
    # Adversary generator-proposer (deterministic): new stream GENERATORS as source.
    # `generator_block` is the appended source; `generator_probe_spec` validates it
    # produces a stream; `admit_generator` registers it. `probe_solver` is a trusted
    # solver the engine measures to confirm a generator yields a runnable stream.
    propose_generators: Callable | None = None
    generator_block: Callable | None = None
    generator_probe_spec: Callable | None = None
    admit_generator: Callable | None = None
    probe_solver: str = ""
    # v6: the human-only policies (for the HUMAN Pareto frontier the utility flag
    # beats_human / matches_human / below_human is measured against), and the human
    # DEFAULT policy at a given capacity (the self-play baseline).
    human_baseline_policies: Callable | None = None
    human_default_policy: Callable | None = None

    # v7: domain-specific guidance for the LLM op-proposer, so the engine stays
    # domain-agnostic (heavy-hitters proposes victim/inherit/pre ops; cache-eviction
    # proposes score/update ops). op_str_kinds = the kinds whose fragment is a single
    # STRING expression (vs a list of statement lines). Empty -> heavy-hitters default.
    op_proposer_sys: str = ""
    op_kinds: tuple = ()
    op_str_kinds: tuple = ()

    # v10 P3: trusted ADAPTIVE human baselines (e.g. ARC / LeCaR references) --
    # a callable returning [{"label","driver_src","stub","cost_units"}, ...].
    # Measured through the normal harness with driver_src swapped in; the
    # beats_adaptive_human tier is graded against these + the static humans.
    adaptive_baselines: Callable | None = None
    # v11 C2: proposer-facing texts + the template library, for the per-run
    # audit bundle (evidence for an external reviewer; never scored in-repo).
    audit_texts: Callable | None = None
    # v11 A4b: the holdout generator family (parameters drawn at finalize).
    holdout_suite: Callable | None = None
    # v10 P5: the domain's maximum declared capacity in cost units (e.g. max CMUL)
    # -- used to set the frontier's memory_ref so every legal capacity has
    # positive resource headroom (0 = derive from the seed as before).
    max_policy_cost_units: int = 0

    # v9: a STANDARDISED, WEIGHTED benchmark suite for the UTILITY verdict
    # (beats/matches/below human): [{"name","n","k","dist","weight"}, ...]. When set,
    # the utility flag + human frontier are measured as the WEIGHTED mean over these
    # declared, multi-regime workloads -- reproducible AND demanding (the adversarial
    # tail is weighted in by declaration, not by whatever attacks a run admitted).
    # None -> the v8 fallback (natural + static adversary workloads, equal weight).
    benchmark_suite: list | None = None

    def strategies_of_class(self, cls: AlgoClass) -> list[str]:
        return [n for n, s in self.strategies.items() if s.algo_class == cls]
