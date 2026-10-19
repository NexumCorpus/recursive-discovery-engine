<div align="center">
  <img src="./nexum-hero.svg" alt="Recursive Discovery Engine — discovery that exposes overfit" width="100%" />

  <p><strong>A discovery loop is only as real as the holdout that can erase its winners.</strong></p>

  <p><a href="#five-minutes">Five minutes</a> · <a href="#evidence">Evidence</a> · <a href="./docs/V11_NOTES.md">v11 notes</a> · <a href="https://github.com/NexumCorpus">Nexum Corpus</a></p>
</div>

---

## What it is

A discovery loop with three roles. A **Builder** proposes algorithms, an
**Adversary** tries to break them with executable counterexamples, and a
**Synthesizer** decides when to continue, combine, reset or stop. Every
candidate is executed in an isolated subprocess, checked against a brute-force
oracle, benchmarked, and graded by code. No grade comes from a model's say-so.
It runs fully offline with a deterministic reference policy, or live with three
model vendors through OpenRouter.

## Why it matters

Automated discovery loops are good at producing winners that only win on the
benchmark they were tuned against. This one ships the instrument that catches
that: a frozen bar of validated human baselines, a charge sweep, and a holdout
redrawn from fresh seeds. The headline below is that the holdout erased every
winner, and anyone can watch it do so again.

## Five minutes

Python 3.11+, standard library only. No install step, no API key.

```bash
git clone https://github.com/NexumCorpus/recursive-discovery-engine
cd recursive-discovery-engine

python -m rde --backend deterministic        # toy domain (top-k), ~2 s: watch the loop disqualify, synthesize, accept
python scripts/reproduce.py                  # the v11 headline, ~3 min: re-runs cache eviction, diffs vs the cited run
```

`reproduce.py` prints `REPRODUCED: 24/24 headline fields identical` and the
fresh holdout margins. Your holdout seed differs from ours, so your margins
differ; the verdict (every winner OVERFIT) should not.

Optional:

```bash
pip install pytest && python -m pytest       # 170 tests, ~15 min (the engine tests run real loops)
python scripts/check_claims.py evidence/runs/20260610_115502_cache_eviction_deterministic   # README vs claims.json
python -m rde --backend llm --profile cheap --iterations 6   # live, three vendors; needs OPENROUTER_API_KEY (.env.example)
```

## Evidence

**Nothing achieves `robust_beats` against the frozen v11 bar — 0/10 prior
winners, 0/14 in the citable run — and the holdout annotates the two remaining
declared-charge winners as OVERFIT.** This is the expected outcome, reported
plainly, and it is the most informative result the project has produced: every
prior `beats` headline was real *under its own version's bar*, and each bar
improvement (v8 unfakeable metric → v10 adaptive tier → v11 validated, broadened,
multi-seed, charge-swept, holdout-checked tier) removed another way to be
accidentally impressive. Run: `20260610_115502_cache_eviction_deterministic`
(`claims.json` + `audit/` bundle; `scripts/check_claims.py` passes). In-run
adaptive tier: 2 beats, 6 matches, 6 below (n_graded=14, selection factor 1.95);
the recommended artifact grades beats_human at the declared charge (+0.0126
adaptive-tier margin) — and both in-run beats carry the holdout OVERFIT
annotation below, which is the operative caveat on that line.

**What the dual-implementation test found (A1):** nothing — and that is the
finding. An independently written ARC (list-based, from the Megiddo & Modha
Fig. 4 case structure) agrees **hit-for-hit with the canonical implementation
across all 63 trace×capacity combinations**; adaptation/scan/loop properties
hold; LeCaR's deterministic deviation is documented and pinned. The bar's
references are now validated, not just present.

**The frozen-bar re-grade of every v9/v10 winner** (n_graded=10, selection
factor 1.83, `scripts/regrade_frozen_bar.py`; full table in
[docs/V11_NOTES.md](docs/V11_NOTES.md)): 7/10 still beat at the *declared* ghost
charge (all aux-adaptive — every static blend fell to fragile-matches), but
**0/10 survive the charge sweep**: at `GHOST_FRAC = 0`, W-TinyLFU (0.930 wCR)
and 2Q at resident-only cost dominate everything. v10's "survives ghost-free
accounting" was true against ARC/LeCaR only; the broadened tier closes that gap.

**Benchmark-vs-holdout gaps (A4b):** decisive. In the citable run the two
in-run winners grade `beats` on the declared benchmark (+0.035 / +0.059 vs the
reachable tier) yet **lose to W-TinyLFU on the freshly-drawn holdout family by
−0.171 / −0.137** — gaps of ~0.2 against a 0.05 overfit threshold. The
threshold-switch idiom's wins are calibrated to the declared suite; they do not
transfer to drawn cousins of the same regimes. Both ops carry `overfit: true`
in `claims.json`.

**What changed structurally (the design constitution):** every metric is now
classified unfakeable / alignable / insulatable before it is wired in. The repo
**stopped answering "was this invented?"** — its only in-loop novelty metric is
behavioral profile distance (alignable: gaming it = being behaviorally
different), and each run emits an `audit/` bundle (post-scrub prompts, all
fragments including rejected, template-library hash stamped against the
benchmark version) for an *external* reviewer to judge. Leak rule enforced and
pinned: prompts carry problems, never rubrics — the worked threshold-switch
idiom, EWMA constants, quality rubric, and what-scores-well language were all
scrubbed from every proposer-facing text (full removal diff in
[docs/V11_NOTES.md](docs/V11_NOTES.md); `test_rubric_scrub_is_in_effect` pins it).

**New bottlenecks:** (1) the holdout result says the real frontier work is now
*generalization*, not benchmark margin — candidate mechanisms need to win on
families, not points; (2) the adaptive tier still lacks LIRS and uses one
deterministic LeCaR variant; (3) the scrubbed-prompt capable-model run — the
first clean reading of capability-without-idiom — is pending operator
confirmation; "the models produce nothing admissible without the worked
example" is a live and informative possibility.

**v11 sequencing (visible in git):** `dc91fe1` v10 baseline → `7d4f4c9`
**A-bar FROZEN** (dual-validated ARC, 2Q + W-TinyLFU, 5-seed benchmark, declared
beats/fragile semantics with selection-penalised threshold, charge sweep,
holdout) → `6b175f8` Half B (ghost observable, `adyn` dynamics, rubric scrub) →
this commit (Half C: behavioral provenance, audit bundles, claim completions).

## What it does not claim

- No discovered policy beats the validated human tier robustly. 0/10 prior winners
  and 0/14 in the citable run survive the frozen bar; the two in-run `beats` are
  holdout OVERFIT.
- It does not judge novelty ("was this invented?"). The only in-loop novelty metric
  is behavioral distance; each run writes an `audit/` bundle for an external reviewer.
- The domains are small, classical problems (top-k, heavy hitters, cache eviction).
  This is an instrument for honest discovery loops, not a source of new algorithms.
- The grounding harness executes generated code behind an AST screen and a
  subprocess. It is not a hardened sandbox (see [Security note](#security-note)).

## Where it sits

One of the research instruments of [Nexum Corpus](https://github.com/NexumCorpus):
instruments for verifiable machine work. Its generator-≠-grader rule is shared with
the [Emergent Geometry Engine](https://github.com/NexumCorpus/emergent-geometry-engine)
and [Director 2](https://github.com/NexumCorpus/director2). Version-by-version
history (v1 → v11, including every corrected headline) is in
[docs/HISTORY.md](docs/HISTORY.md). Built by Claude, directed by Nexum Corpus. MIT licensed.

---

## The three roles

| Role | Responsibility | Deterministic flavour | LLM flavour (distinct vendor) |
|------|----------------|------------------------|-------------------------------|
| **Builder** | Propose new approaches / syntheses | Curated strategy library | a *coder* model |
| **Adversary** | Try to **disprove** proposals with executable evidence | Fixed attack arsenal | a strong *critical reasoner* |
| **Synthesizer** | Maintain the high-level view; **the only role that may declare convergence or terminate**; detect drift/recombination | Explicit, auditable policy | the best *meta-judge* |

Each role has **two implementations**. The deterministic trio is the *reference
control logic*: it runs with zero API cost, is what the test-suite exercises, and
is the behaviour the LLM trio is held to (and falls back to on failure). The loop
in `engine.py` is identical for both — it just receives a different trio.

> **Why three different model vendors?** Assigning a different model family to
> each role is the cheapest structural defence against the spec's *shared blind
> spots*. If Builder, Adversary, and Synthesizer all ran the same base model they
> would share systematic errors and the Adversary could not catch what the
> Builder cannot see. In the live run the `claude-haiku` Synthesizer correctly
> labelled the `gpt-5-mini` Builder's full-sort proposal "recombination of known
> techniques" — outside-perspective friction that a single model cannot give
> itself.

---

## How the spec's protective mechanisms are implemented

| Spec mechanism | Where | How |
|----------------|-------|-----|
| **Empirical Grounding Rule** — no approach advances without runnable, falsifiable evidence | `grounding.py` | Every candidate is executed in an **isolated subprocess** with a hard timeout, checked against a brute-force **oracle**, benchmarked, and memory-profiled. Correctness is a **hard gate**: an incorrect approach is *disqualified* (fitness `None`), never down-weighted. No field of `Evidence` is ever set by an LLM assertion. |
| **Novelty / Recombination Check** | `fingerprint.py` | Each approach is reduced to a structural fingerprint (algorithmic call-set, imports, AST-shape histogram). `is_recombination` ⇔ a prior approach in the **same class** exceeds the similarity threshold. Generic builtins (`list`,`len`,…) are filtered so similarity tracks *algorithm*, not boilerplate. |
| **Adversarial Pressure** | `roles.py` + `engine.py` | The Adversary only *proposes* inputs; the **harness executes them against the oracle** and decides what broke. A confirmed counterexample on an approach that passed the suite **downgrades it AND is folded into the trusted suite** — the harness's eyes grow over the run. |
| **Drift Awareness** | `signals.py` | The engine *computes* objective signals (stall over a fitness window, consecutive-recombination "patching", recurring failure classes) and hands them to the Synthesizer **as facts it cannot override**. The Synthesizer must address any flagged condition. |
| **Bounded Memory** | `memory.py` | The approach archive is capped to the **Pareto frontier** (+ best, + one failure per class, + recency); insights are a capped ring. `digest()` — the *only* view handed to LLM roles — is size-bounded, so a 50-iteration run prompts with as much context as a 5-iteration run. |
| **Human Override Points** | `cli.py` (`--interactive`) | At each synthesis checkpoint a hook receives the state + proposed directive and may accept, force a different action, or terminate. |

### The anti-rationalisation design (the key idea)

The single most important decision: **the Synthesizer does not get to decide
whether the process is drifting.** `signals.compute()` derives stall / patching /
recurring-failure deterministically from history; the LLM Synthesizer is *given*
those facts and its decision is then **checked against hard guardrails** — it
cannot `ACCEPT` with no correct approach, and it cannot `CONTINUE` through
confirmed local patching. This is what stops an LLM from cheerfully answering
"yes, great progress!" while emitting the third cosmetic variant of one idea.

---

## The core loop (`engine.py`)

```
seed → [ Builder.propose
         → GROUND each (isolated subprocess, oracle, benchmark, memory)
         → score + fingerprint + novelty/recombination
         → Adversary.attack the survivors (confirmed breaks harden the suite)
         → remember (bounded) + record insights
         → compute OBJECTIVE signals
         → Synthesizer.decide  (continue / synthesize / reset / accept / terminate)
         → human checkpoint (optional override)
         → act ] × N
       → accept / terminate / exhausted
```

Ordering is load-bearing: evaluate → adversary → commit, so a confirmed
counterexample can downgrade a candidate *before* it is admitted as "best."

---

## Project layout

```
rde/
  types.py        typed records exchanged between roles (lineage + evidence first-class)
  config.py       model assignment + every protective threshold as a named tunable
  grounding.py    the Empirical Grounding Rule: isolated execution, oracle, scoring
  fingerprint.py  structural fingerprint + similarity (the novelty check)
  signals.py      stall / patching / recurring-failure detection
  memory.py       bounded Pareto-frontier memory + digest
  llm.py          stdlib-only OpenRouter client + robust JSON extraction
  roles.py        Builder / Adversary / Synthesizer  (deterministic + LLM)
  engine.py       the orchestration loop + run persistence + report
  domain.py       the Domain interface (keeps the loop problem-agnostic)
  vocab.py        v5+ open-vocabulary registry (generic op-slot kinds since v7)
  selfplay.py     v6 self-play ledger, causal guidance, utility flags + history
  surrogate.py    v7 cheap pre-filter that composes the grounded-code-execution skill
  domains/topk.py            the v0 problem: oracle, trusted suite, strategy library
  domains/heavy_hitters.py   v2–v6: streaming heavy-hitters (recall@K x memory)
  domains/cache_eviction.py  v7: cache eviction (hit rate vs Belady OPT x memory)
  cli.py          `python -m rde ... --domain {topk,heavy_hitters,cache_eviction}`
skills/           v6.5 extracted skills (ast-safety-screen, grounded-code-execution,
                  pareto-frontier, multi-role-discovery) — loadable, stdlib-only
tests/            170 tests: grounding, fingerprint, signals, llm, engine, v3–v11, skills
scripts/          reproduce.py (headline re-derivation), check_claims.py (README vs claims.json)
evidence/runs/    the cited v11 run: claims.json + audit bundle, committed so it can be checked
runs/             per-run artifacts: result.json, iterations.jsonl, report.txt, best_solution.py
```

Adding a second problem domain means writing one module that constructs a
`Domain` — **zero** changes to the loop, roles, memory, or mechanisms (v7's cache
eviction is exactly this: a new domain module + a generic op-slot registry).

---

## Configuration & models

Models are assigned per-role in `config.py` (validated against the live
OpenRouter catalogue) and overridable via env vars. Two profiles:

- `--profile default`: `openai/gpt-5.1-codex` · `google/gemini-3.1-pro-preview` · `anthropic/claude-opus-4.8`
- `--profile cheap`: `openai/gpt-5-mini` · `google/gemini-2.5-flash` · `anthropic/claude-haiku-4.5`

Every protective threshold (stall window/epsilon, recombination similarity,
patching limit, memory caps, fitness weights) lives in `Config` so the engine's
behaviour is auditable and tunable in one place.

### Secrets

`OPENROUTER_API_KEY` is read from `.env` (gitignored — see `.env.example`). It is
never hardcoded and never committed.

---

## Security note

The grounding harness **executes model-generated code**. Defences, in layers:

- a separate Python process per candidate (`-E -s`, stripped `PYTHONPATH`) so
  candidates can't import the parent's packages, with no state leakage and an
  OS-level timeout that kills runaways;
- an **AST-based static screen** (the primary static defence): imports must be on
  an allow-list of safe stdlib modules, code-execution/introspection builtins
  (`eval`/`exec`/`compile`/`getattr`/`open`/…) are forbidden by name, and dunder
  attribute/name access (`__globals__`, `__subclasses__`, `__builtins__`, …) —
  the backbone of classic sandbox escapes — is rejected outright;
- a regex denylist as a secondary net; best-effort temp cleanup.

This is **still not a hardened sandbox** — a determined adversary with a fully
unrestricted stdlib could find gaps. It is adequate for a trusted local research
setting. Do not point the LLM backend at untrusted/hostile models on a machine
you care about without adding OS-level isolation (container / seccomp / restricted
user).

---
