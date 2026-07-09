"""Configuration: model assignment, thresholds, and .env loading.

Design note on thresholds: every protective mechanism in the spec is governed by
a *named, tunable* constant here rather than a magic number buried in code. That
keeps the engine's behaviour auditable -- you can see exactly what "drift" or
"recombination" means numerically and change it in one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# .env loading (no third-party dependency)
# --------------------------------------------------------------------------- #

def load_dotenv(path: str | os.PathLike | None = None) -> None:
    """Minimal .env reader: KEY=VALUE lines, '#' comments, no export keyword.
    Values already present in the real environment win (never overwritten)."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / ".env"
    path = Path(path)
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


# --------------------------------------------------------------------------- #
# Model profiles -- different families per role to fight shared blind spots
# --------------------------------------------------------------------------- #

# Three DISTINCT vendors per profile -- the cheapest structural defence against
# the spec's "shared blind spots": if all three roles ran the same base model
# they would share systematic errors and the Adversary could not catch what the
# Builder cannot see. Roles are matched to vendor strengths: a coder model
# builds, a strong critical reasoner attacks, the best high-level judge decides.
# (All IDs validated against the live OpenRouter catalogue.)

# Capable profile -- top current model from each of three vendors.
MODELS_DEFAULT = {
    "builder": "openai/gpt-5.1-codex",            # OpenAI -- purpose-built coder
    "adversary": "google/gemini-3.1-pro-preview",  # Google -- critical reasoning
    "synthesizer": "anthropic/claude-opus-4.8",    # Anthropic -- meta judgement
}

# Cheap profile for live demos -- still three distinct families.
MODELS_CHEAP = {
    "builder": "openai/gpt-5-mini",
    "adversary": "google/gemini-2.5-flash",
    "synthesizer": "anthropic/claude-haiku-4.5",
}


@dataclass
class Config:
    # --- backends ---------------------------------------------------------- #
    backend: str = "deterministic"        # "deterministic" | "llm" | "codex-cli"
    profile: str = "default"              # "default" | "cheap" (remote llm only)

    # --- LLM client -------------------------------------------------------- #
    api_base: str = field(default_factory=lambda: os.environ.get(
        "RDE_API_BASE", "https://openrouter.ai/api/v1"))
    api_key_env: str = "OPENROUTER_API_KEY"
    # Env seam like RDE_API_BASE: a local CLI-backed endpoint (atlas shim) can
    # legitimately take minutes per call -- 120s is right for OpenRouter but
    # starves behind a serial local backend, and client-side retries then stack
    # orphaned calls on the server (the 2026-07-01 campaign cascade).
    request_timeout_s: float = field(default_factory=lambda: float(
        os.environ.get("RDE_REQUEST_TIMEOUT_S", "120")))
    max_retries: int = 3
    # Codex Desktop/CLI transport. It is deliberately a separate backend from
    # OpenRouter: no API key, no provider-model comparison, and no inherited
    # user MCP/plugin configuration unless explicitly requested.
    codex_bin: str = field(default_factory=lambda: os.environ.get("RDE_CODEX_BIN", ""))
    codex_model: str = field(default_factory=lambda: os.environ.get("RDE_CODEX_MODEL", ""))
    codex_use_user_config: bool = field(default_factory=lambda: os.environ.get(
        "RDE_CODEX_USE_USER_CONFIG", "").lower() in {"1", "true", "yes", "on"})

    # --- grounding harness ------------------------------------------------- #
    subprocess_timeout_s: float = 20.0    # hard kill for a runaway candidate
    # v5: kill a candidate/proposal subprocess the moment its RSS exceeds this many
    # MB (a psutil watchdog in the parent), so an unbounded allocation cannot thrash
    # the host before the wall-clock timeout fires. 0 disables. `resource.setrlimit`
    # is unavailable on Windows, so this is the cross-platform guard.
    subprocess_mem_cap_mb: int = 1536
    bench_repeats: int = 5               # median of N timing samples
    bench_seed: int = 1234               # fixed -> reproducible benchmarks

    # --- fitness weights (correctness is a hard gate, not a weight) -------- #
    # v1 leans further into performance: synthesis is largely about combining
    # performance strengths across regimes, and an adaptive hybrid is necessarily
    # a little less "simple" than a one-liner -- we don't want the simplicity term
    # to veto a genuinely faster hybrid.
    w_perf: float = 0.70
    w_mem: float = 0.15
    w_simplicity: float = 0.15

    # --- novelty / recombination ------------------------------------------ #
    # If structural similarity to ANY prior approach exceeds this AND the algo
    # class matches, the proposal is flagged as recombination.
    recombination_similarity: float = 0.85
    # v1: behavioural recombination -- a proposal whose per-workload performance
    # PROFILE is this close to a prior is "the same thing in disguise", even if
    # its source looks different. Harder for an LLM to game than structure.
    behavioral_similarity: float = 0.93
    behavioral_novelty_floor: float = 0.10   # below this == behaviourally stale

    # --- drift / stall / collapse detection -------------------------------- #
    stall_window: int = 3                 # look back this many accepted-best points
    stall_epsilon: float = 0.02          # min fitness gain to count as "progress"
    recurring_failure_limit: int = 2      # same failure class N times -> drift
    patching_limit: int = 2               # N consecutive recombinations -> reset
    # v1: a fitness "gain" is only real if it also clears the benchmark noise
    # band by this many sigma -- stops noise from masquerading as progress.
    noise_sigmas: float = 1.0
    # v1: recombination is DOMINANT if this many of the last `recomb_window`
    # rounds were structurally-recombinant OR behaviourally-stale.
    recomb_window: int = 3
    recomb_dominant_limit: int = 2
    # v1: a SYNTHESIS only counts as a real hybrid if it matches/beats each
    # parent on that parent's strongest workload to within this factor. Set a
    # little below 1.0 to tolerate benchmark noise (a 0.94 gain at ~6% noise is a
    # genuine match, not a regression).
    synthesis_min_gain: float = 0.90
    # v1: hard cap on synthesis attempts per run -- a free LLM Synthesizer can
    # keep choosing to "break the plateau" forever; this guarantees convergence.
    # v3 raises it so the compositional grammar gets explored across rounds.
    max_synthesis_attempts: int = 5
    # v3: how many grammar compositions the deterministic Builder proposes per
    # synthesis round (a small batch -> the frontier filters the winners).
    synthesis_batch: int = 3

    # --- v2 frontier + adversarial-metric -------------------------------- #
    # A candidate's marginal Pareto hypervolume must exceed this to count as
    # "advancing the frontier" (filters out noise-level contributions).
    frontier_gain_eps: float = 0.005
    # A synthesis opportunity exists when this much more recall is reachable above
    # the memory budget than within it.
    frontier_gap_threshold: float = 0.15
    # Metric-gaming exposure: a candidate is exposed when its recall on an
    # adversarial workload is BOTH this far below its base recall AND under the
    # floor -- i.e. it looked good on easy streams but collapses on a hard one.
    metric_collapse_margin: float = 0.30
    metric_collapse_floor: float = 0.40

    # --- v5 open vocabulary (propose new ops/generators as source) --------- #
    # Master switch. When True (and the domain provides the v5 hooks) the Builder
    # may propose NEW primitive ops and the Adversary NEW generators, each
    # AST-screened + grounded before it can join the vocabulary / harden the suite.
    open_vocabulary: bool = True
    # v6 raises proposal VOLUME and leans on post-grounding filtering (frontier gain
    # + utility flag) rather than upstream pre-filtering -- accepting more short-term
    # noise in exchange for faster iteration on proposal quality.
    max_proposed_ops_per_round: int = 8       # how many op proposals to screen/round
    max_proposed_generators_per_round: int = 3
    proposed_ops_cap: int = 16                # total admitted proposed ops per run
    proposed_generators_cap: int = 6          # total admitted proposed generators
    # v6: utility flag tolerance. A proposed op's best policy "beats" the human
    # frontier when its recall exceeds the best human at <= its memory by this much.
    utility_eps: float = 0.01
    # v6: append per-op utility to this cross-run history file (relative to cwd) so
    # the trend (are proposed primitives getting better?) is visible across runs.
    utility_history_path: str = "runs/utility_history.jsonl"
    # v6 self-play: enable Builder-vs-Adversary scoring (win rate + margin).
    self_play: bool = True
    # v7: cheap surrogate pre-filter -- estimate a proposed op's value on ONE small
    # workload (via the grounded-code-execution skill) before full grounding, and
    # skip ops clearly below the human default by more than `surrogate_margin`.
    surrogate_prefilter: bool = True
    surrogate_margin: float = 0.12
    surrogate_n: int = 1500              # tiny workload size for the surrogate probe

    # --- memory ------------------------------------------------------------ #
    memory_max_approaches: int = 12       # bounded: Pareto frontier + recent
    memory_max_insights: int = 40

    # --- loop -------------------------------------------------------------- #
    max_iterations: int = 8
    builder_fanout: int = 1               # approaches proposed per exploration round

    # populated from the active profile at construction
    models: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.models:
            if self.backend == "codex-cli":
                model = self.codex_model or "codex-cli"
                self.models = {
                    "builder": os.environ.get("RDE_BUILDER_MODEL", model),
                    "adversary": os.environ.get("RDE_ADVERSARY_MODEL", model),
                    "synthesizer": os.environ.get("RDE_SYNTHESIZER_MODEL", model),
                }
                return
            base = MODELS_CHEAP if self.profile == "cheap" else MODELS_DEFAULT
            self.models = {
                "builder": os.environ.get("RDE_BUILDER_MODEL", base["builder"]),
                "adversary": os.environ.get("RDE_ADVERSARY_MODEL", base["adversary"]),
                "synthesizer": os.environ.get(
                    "RDE_SYNTHESIZER_MODEL", base["synthesizer"]),
            }

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)
