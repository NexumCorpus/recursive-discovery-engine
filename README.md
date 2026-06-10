# Recursive Discovery Engine (RDE) — v11

A small, runnable system that **explores approaches to a problem, evaluates them
with real executable evidence, synthesizes genuinely better solutions, and guards
against collapse** (recombination, drift, shared blind spots, low-value churn).

## Honest assessment (v11)

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

---

# (v10, historical) Recursive Discovery Engine — v10

**v10 closes the seams.** A full technical review of v9 found the per-component
invariants strong but the *boundaries between subsystems* unowned: a verdict that
flickered across identical runs, a cost axis that was declared but not enforced, a
human baseline missing the humans' own adaptive policies, a discovery loop
optimising a different metric than the verdict judged, and a README that could
drift from what the code produces. v10 fixes the places where declared and actual
behavior had come apart — and re-grades everything under the stricter rules.

## What changed in v10 (and why)

| Fix | Mechanism |
|---|---|
| **Phase 0 — diagnose before fixing** | The observed 6↔7 `beats_human` flicker was NOT the hypothesized suite-truncation/renormalization (measurements complete in ~0.3 s vs a 20 s timeout). The actual mechanism: **representative-variant instability** — the finalize catch-all recorded only the first capacity variant per op, a load-induced `ground()` timeout silently flipped *which capacity represented an op*, and two ops sat 1–5×10⁻⁴ from the eps boundary. Full evidence in [docs/V10_NOTES.md](docs/V10_NOTES.md). |
| **P1 — completeness gate** | A weighted benchmark score is valid only with a measurement for EVERY declared workload: retry once, then score missing workloads **0.0** with an explicit `incomplete: [names]` flag in the report and `utility_history.jsonl`. Applied **symmetrically to human baselines**. Silent renormalization is structurally unreachable; all capacity variants are recorded and the representative is chosen deterministically on the judgment metric. |
| **P2 — statelessness enforced** | Driver-domain candidates are now shape-locked: module top level admits only imports/defs/docstrings/the literal `CMUL`/a literal `AUX0`; **mutable default params** and **attribute stores** are rejected; `global`/`nonlocal` stays banned. Each stash vector has a pinned exploit test. A tracemalloc cross-check (measured ≤ 8× declared, honest policies measure 1.1–1.4×) backstops the AST rule. Legacy domains untouched. |
| **P3 — adaptive human baselines** | **ARC** and a deterministic **LeCaR-style** ghost policy as trusted driver-level references, measured through the same harness/verdict path. One ghost-charging rule (`GHOST_ENTRY_COST_FRAC = 0.25`, applied in exactly one place) governs the cost of ghost metadata. Two tiers reported side by side: `beats_static_human` (v6-compatible) and **`beats_adaptive_human` (the headline)**; history records are versioned (`baseline: static|adaptive`). |
| **P4 — loop re-coupled to verdict** | The frontier's quality axis IS the weighted benchmark CR (adversary attacks remain pressure/diagnosis, not the denominator); admitted ops are composed into frontier candidates **in the round they are admitted**; an **INCOHERENT-RESULT assertion** fails the run loudly if a beating op out-scores the entire frontier — it FIRED on its first test run, catching the finalize-graded-but-frontier-invisible seam, which is now closed by routing catch-all policies through the same admission machinery; the **recommended artifact is graded next to the headline**; LLM contract rejections are fed back into the next Builder prompt verbatim. |
| **P5 — cleanups** | `memory_ref` from the declared capacity range (max-capacity points no longer score `fit=0.000` for lack of headroom); static recency+frequency blends are now `BLENDED` (`ADAPTIVE` = actually reads aux); proposed-op *fragments* persisted in run artifacts; every run emits **claims.json** and `scripts/check_claims.py` diffs the README against a named run. |

### Verified result (v10 — run `20260610_072254_cache_eviction_deterministic`)

```
HEADLINE -- vs best ADAPTIVE human (incl. ARC/LeCaR): 4 beats, 5 matches, 5 below
  beats: ad_thrash, ad_guard15 (aux-adaptive), rec_logf, rec_plus_f (static)
RECOMMENDED ARTIFACT: i8.0_mut4 CR 0.84 @ 6KB -- graded next to the headline:
  beats_human on BOTH tiers (+0.0224 vs best human at <= its memory)
completeness flags: none | INCOHERENT-RESULT: none | claims.json: pinned
```

**Re-grade of every v9 winner under the stricter v10 rules** (per-cap representative
chosen on the judgment metric; both tiers; `+margin` vs the best human at ≤ cost):

| op | origin | wCR (best cap) | static tier | adaptive tier | ghost-FREE accounting |
|---|---|---|---|---|---|
| `ad_guard15` | det template | 0.892 (c4) | beats +0.070 | beats +0.070 | **beats +0.055** |
| `adaptive_mixguard` | **LLM-invented** | 0.884 (c4) | beats +0.062 | beats +0.062 | **beats +0.047** |
| `adaptive_hotset` | **LLM-invented** | 0.873 (c4) | beats +0.051 | beats +0.051 | beats +0.036 |
| `adaptive_freq_guard` | **LLM-invented** | 0.873 (c4) | beats +0.051 | beats +0.051 | beats +0.036 |
| `ad_thrash` | det template | 0.868 (c4) | beats +0.046 | beats +0.046 | beats +0.031 |
| `rec_logf` | det static | 0.842 (c4) | beats +0.020 | beats +0.020 | **match +0.005** |

**The expected outcome was that ARC would end the streak — it did not, and the
evidence is robust to the accounting.** Under the declared ghost charge (0.25), the
ARC/LeCaR points are dominated by static classics at every reachable cost, so the
two tiers coincide. Under the *most ARC-favorable* accounting (ghosts entirely
free), every **aux-adaptive** op still beats the best adaptive human
(ad_guard15 +0.055), while the static champion `rec_logf` drops to *match* —
the accounting-robust wins are precisely the adaptive inventions. The honest
caveat stands: the benchmark's 10 %-weight pure-loop regime is ARC's known blind
spot (ARC is scan-resistant, not loop-resistant), and that regime is where the
adaptive ops' margin comes from; removing the loop entirely still leaves
ad_guard15 ahead, but by less.

## Honest assessment (v10)

**Which review gaps were closed, and to what degree.** All five priority areas
shipped with pinned regression tests: the verdict path can no longer silently
renormalize (P1 — structurally unreachable, not just unobserved), candidate
statelessness is enforced rather than declared (P2 — every named stash vector is
attempted by a test and rejected), the human baseline now contains the humans'
adaptive policies measured through the identical harness (P3), the loop optimizes
the metric the verdict judges (P4 — and the coherence assertion that guards this
seam *fired during development*, caught a real gap, and now passes structurally),
and the README's numbers are diffable against a named run (P5, this section cites
`20260610_072254_cache_eviction_deterministic`; `scripts/check_claims.py` verifies it).

**What Phase 0 actually found.** Not the hypothesized timeout-renormalization
(which was real but latent — measurements run 60× under the timeout), but
representative-variant instability: *which capacity variant represented an op*
was load-dependent, and two ops sat 10⁻⁴ from the eps boundary. The fix was
determinism of the variant set, not the suite gate alone. Lesson recorded in
[docs/V10_NOTES.md](docs/V10_NOTES.md): the verdict was only as stable as its
least-owned seam.

**Was `beats_adaptive_human` achieved?** Yes — contrary to the stated expectation
that ARC would end the streak. Four ops beat the best adaptive human at ≤ equal
declared cost, and the three strongest survive even ghost-free accounting. Zero
would have been an acceptable answer; it is not what the measurements say. The
margin is concentrated in the loop regime (ARC's structural blind spot), which is
declared in the benchmark at 10 % weight — a legitimate but narrow edge.

**New bottlenecks.** (1) The adaptive-tier bar is only as strong as the reference
implementations: classic ARC and a deterministic LeCaR variant, at three
capacities, one seed per regime — a stronger tier would add LIRS/2Q/W-TinyLFU and
multi-seed workloads. (2) The benchmark is still single-seed per regime; the
knife-edge ops (`lfu_aged` at 10⁻⁴ from the boundary) are now *deterministic* but
their flags remain epistemically fragile — confidence intervals need seed
replicates. (3) Self-play win-rate remains ~0.04; attack-specialist wins are still
unsolved. (4) The capable-model LLM path is unverified under v10's stricter gates
(run pending operator confirmation — see below).

**How much more trustworthy is the signal now — evidence, not adjectives.** The
P1 acceptance battery (five consecutive identical-config runs, results recorded in
docs/V10_NOTES.md) replaces the observed 6↔7 flicker; the cheat/stash/cost
exploits are all *attempted* by the test suite and rejected; the recommended
artifact carries its own grade so the shipped policy and the headline cannot
diverge silently; and claims.json makes every number in this section mechanically
checkable against the named run.

---

# (v9, historical) Recursive Discovery Engine — v9

**v9 moves from regime-specific wins to ROBUST, regime-aware improvement** — and
delivers the project's first **open-invented** `beats_human` primitives. v8 made the
metric unfakeable but its honest finding was that *no static eviction score dominates
across loop, drift, and skew*. v9 attacks that ceiling on three fronts:

- **Adaptive primitives — bounded aux state, driven by the trusted driver.** A
  candidate may now declare `AUX0` (≤ 8 finite numbers), `_on_access(aux, hit, t)`
  (called every access with the per-access hit *flag* — what any online algorithm
  observes; never a count), and `_evict_score_aux(la, fr, ins, t, aux)`. The driver
  owns the state (reset per trace, passed as an immutable tuple, malformed updates
  reset to `AUX0` — degrade-only). This is enough for **regime detection**: an EWMA
  of the hit flag collapses when the cache thrashes on a loop/scan, and the score
  *switches eviction behaviour mid-trace* — what no static score can do.
- **Competitive-ratio quality + a standardised weighted benchmark.** Quality is now
  `hit_rate / Belady-OPT` per trace (the anti-cheat stays), so easy regimes can't
  dominate a mean; `beats_human` is the **weighted CR mean over a declared 7-regime
  benchmark** (natural 0.7 / adversarial tail 0.3) — reproducible *and* demanding.
- **A generic trusted-driver gate.** The required candidate symbols are now
  spec-driven (`GroundingSpec.driver_funcs` / `driver_literal_ints`), so any future
  domain gets verifiable-by-construction grounding with its own symbol names.

**The headline results (both verified end-to-end; numbers are post-review, on the
hardened benchmark with the period-60 loop and CMUL ∈ [1,4]):**
- *Deterministic*: the adaptive thrash-guard scores **0.776 weighted CR vs the best
  equal-cost human's 0.687 (+0.089) at c2, and 0.868 vs 0.822 at c4** — beating the
  v8 static champion `rec_logf` (0.732/0.842) while never losing to it on any
  single regime and rescuing the loop at *both* capacities (CR 0.45/0.26 where
  every LRU-family static records 0.00). Run report: **4 `beats_human` ops, two of
  them adaptive; the recommended frontier point is itself an adaptive-class
  policy**.
- *Capable-model LLM* (gpt-5.1-codex / gemini-3.1-pro / claude-opus-4.8): the
  Builder authored **3 adaptive `ascore` ops — `proposed-llm`, all 3 `beats_human`**
  on the standardised benchmark, measured by the unfakeable driver and composed by
  the close-the-loop machinery. v5–v8's LLM ops never beat; this is the first time
  **open invention clears the bar** (with a caveat below: the proposer prompt
  teaches the adaptive idiom by example, so this is guided invention, not de novo).

Every protective invariant holds: the v8 trusted driver remains the only metric
authority (the aux extension is driver-owned and the cheat tests still pass under
CR), candidate ops are untrusted and AST-screened + fragment-validated + grounded,
the `CMUL` literal rule is generalised, the cost axis stays the exact declared
capacity, and self-play remains descriptive, never gating.

```
python -m rde --backend deterministic --domain cache_eviction   # zero cost
python -m rde --backend llm --profile default --domain cache_eviction   # capable models
python -m rde --backend deterministic --domain heavy_hitters     # the v2–v6 domain
```

### Verified result (v9 — deterministic + capable-model LLM, cache_eviction)

```
ADAPTIVE CONTRACT: a candidate may declare AUX0 / _on_access(aux, hit, t) /
_evict_score_aux(la, fr, ins, t, aux). The TRUSTED driver carries the (<= 8-float,
degrade-only) aux through the trace; the candidate sees the per-access hit FLAG only.

QUALITY: competitive ratio hit_rate/Belady-OPT, weighted over the DECLARED 7-regime
benchmark (zipf .20, recency .15, drift .20, mixed .15 | period-60 loop .10,
scan .10, antag .10). CMUL is clamped to [1, 4] so no legal capacity can swallow a
benchmark universe and make CR vacuous.

DETERMINISTIC RUN: 14 ops proposed -> 4 beats_human:
  ad_thrash, ad_guard15 (ADAPTIVE: "(-la) if aux[0] < 0.08 else (la + 30*fr.bit_length())")
  rec_logf, rec_plus_f  (static)
  adaptive vs best equal-cost human:  c2: 0.776 vs 0.687 (+0.089)   c4: 0.868 vs 0.822
  adaptive vs v8 static champion:     c2: 0.776 vs 0.732 (+0.044); never worse on ANY
  regime, and rescues the loop at BOTH capacities: CR 0.45/0.26 vs statics' 0.00
  frontier: an adaptive policy ON the frontier; RECOMMENDED point is adaptive-class

CAPABLE-MODEL LLM RUNS (gpt-5.1-codex / gemini-3.1-pro / claude-opus-4.8, 4 iters,
2 independent runs -- the second under the post-review hardened rules):
  run 1: adaptive_hotshield / adaptive_agefreq / adaptive_mixedshield -- 3/3 beats_human
  run 2 (hardened benchmark, CMUL [1,4]):
    adaptive_hotset    "(-la) if aux[0] < 0.1  else (la + (fr.bit_length() << 5))"
    adaptive_mixguard  "(-la) if aux[0] < 0.18 else (la + fr.bit_length()*24 + (t-ins)//8)"
    adaptive_freq_guard "(-la) if aux[0] < 0.12 else (la + (fr.bit_length() << 5))"
    -- ALL proposed-llm, ALL beats_human again (3/3). mixguard's residency-age term
    (t-ins)//8 appears in NO curated template: a novel component inside the taught
    idiom. First open-invented beats_human in the project (v5-v8 LLM ops: 0).
  ~15 calls / ~50k tokens per run.

SOUNDNESS (all test-pinned): a candidate's own solve() is still discarded (records
the honest CR, not a sub-OPT fabrication); adaptive candidates without a valid AUX0
are rejected; oversized aux (9 floats) rejected; garbage _on_access returns reset to
AUX0 (degrade-only); aux is passed immutably; CR cannot exceed 1.0 (anti-cheat).
```

---

### Verified result (v8, historical — superseded by the v9 evaluation standard)

```
CONTRACT: a candidate provides _evict_score / _on_hit / CMUL -- NOT a solve(). A TRUSTED
driver runs the eviction sim and counts hits, so the metric is unfakeable by construction.

CHEAT TEST (H1 closure): a candidate that ALSO writes `def solve(...): return (1.0, cap)`
records the driver's honest hit rate (~0.0 on a loop, where LRU collapses), NOT 1.0.

PROPOSAL QUALITY (utility vs the human Pareto frontier, on the STABLE suite, exact cost):
  rec_logf : beats_human   (la + 30*fr.bit_length(); 0.822 vs LRU@equal-cap 0.802, +0.020)
  lfu_aged / lfu_aged2 / aged_logf / idle_penalty / decay_hit : matches_human
  -> 1 REPRODUCIBLE beats_human   (v7 reported 3 -- inflated mainly by a run-variable
     suite, compounded by cost-axis noise; v8's stable suite + exact cost -> honest 1)
  close-the-loop: every admitted op (incl. live LLM-invented ops) earns a flag, not 'unknown'

heavy_hitters (unchanged path): declow still beats_human -> backward-compat intact.
```

The honest headline: v8's contribution is **trust**, not a bigger number. The metric
is now *unfakeable* (trusted driver, cheat-tested) and *reproducible* (exact cost +
stable suite); applying that rigor to v7's own claims revealed the “3” was
measurement-assisted, and the genuine, reproducible win is **1** (`rec_logf`, a
recency-with-log-frequency score that beats LRU at exactly equal capacity).

**Stated with equal honesty:**
- **`beats_human` is still suite-sensitive.** `rec_logf` beats on the declared
  comprehensive suite (4 natural + 3 adversary), but on a *loop-heavy* suite LFU's
  robustness makes it the best human and the aging ops only tie. No *static* score
  dominates every regime — robust, comprehensive beats_human needs *adaptive* ops
  (humans lack a static answer too). A **standardised benchmark suite** is the #1 v9
  item.
- **The cost-axis calibration (`160 B/entry`) is a constant**, chosen to match v7's
  tracemalloc scale so budgets carry over; it is exact and capacity-proportional but
  not a physical measurement.
- **The deterministic op library is still human-curated.** `rec_logf` is a
  hand-authored template the engine *selected* via causal guidance; genuine open
  invention happens on the LLM path, which now *composes* its inventions (utility
  flags, not `unknown`) but — under the cheap synthesizer — produces *matches*, not
  beats.

---

### Verified result (deterministic, cache_eviction) — v7, since CORRECTED by v8

> **Correction:** the “3 `beats_human` / rate 0.6” below was measured on a
> **run-variable** suite (whichever attacks the Adversary admitted that run),
> compounded by a noisy tracemalloc cost axis. v8's stable suite + exact-capacity
> cost reduce the honest, reproducible count to **1** (`rec_logf`). The v7 narrative
> is kept for the historical record; treat its numbers as superseded.

```
ADVERSARY authors new trace generators as SOURCE -> grounded: gen_slow_drift, gen_sawtooth (both expose a candidate)
CAUSAL: best policy collapses on a loop/drift workload -> the regime SELECTS aging/forgetting score ops
BUILDER authors new SCORE ops as SOURCE -> AST/fragment-screened + grounded:
   lfu_aged   = "frequency AGED by idle time"        (fr - (t - la))      -- not in the human {LRU,LFU,FIFO,MRU}
   aged_logf  = "log-freq boost minus idle penalty"  (la + 20*fr.bit_length() - (t-la))
   lfu_aged2, idle_penalty, decay_hit (update)

PROPOSAL QUALITY (utility vs the human Pareto frontier, PRIMARY metric = hit rate):
  lfu_aged : beats_human      lfu_aged2 : beats_human      aged_logf : beats_human
  idle_penalty : matches_human    decay_hit : below_human
  -> 3 beats_human / rate 0.6   (v6: 1 beats_human, and only on a SECONDARY axis)
  self-play (vs the BEST HUMAN per generator, capacity-matched): win_rate 0.2;
             net-positive frontier movement = YES

PARETO FRONTIER (hit-rate x memory):
  lfu(0.41,3KB)  aged_logf/c2(0.42,4KB)  lfu_aged2/c4(0.74,7KB)  [mut of a proposed op](0.75,7KB)
  RECOMMENDED within budget: ~0.75 hit rate @ 7KB  (seed LRU: 0.38)
```

The honest headline: on a domain with real headroom, the diagnosed loop/drift failure
drove the Builder to author **aging score ops** (e.g. `lfu_aged` = frequency decayed
by idle time) that are **outside the human vocabulary** `{LRU, LFU, FIFO, MRU}` and
**beat the best human policy on the primary metric (hit rate)** at equal memory —
**three** such `beats_human` ops (rate 0.6), versus v6's single memory-axis win on a
saturated domain. This is the clearer signal v7 set out to produce.

**Stated with equal honesty:**
- **The win is real and on the primary axis, but the recommended policy is an AST
  *mutation* of a proposed-op policy** (0.75 vs the proposed `lfu_aged2/c4`'s 0.74) —
  classified `other`, so the frontier headline traces to the proposed ops even though
  the single top point is a one-step mutation of one.
- **Self-play measures generalists, not specialists.** Against the *best human per
  attack* the proposed ops mostly tie/lose on the individual adversary generators
  (win_rate 0.2) — they are better *generalists* (beats_human on the diverse main
  suite) than *specialists*. That is the honest, nuanced reading of the unified
  baseline, not a forced win.
- **The scalar-hit-rate contract is verifiable only up to OPT.** All *demonstrated*
  policies are **op-composed** into a trusted simulation template (so they cannot
  misreport), but a hypothetical *free-form* `solve` — which receives the whole trace
  — could in principle compute Belady itself and report a rate it did not achieve. v7
  closes the easy lever (clamps an absurd reported capacity; validates every op
  fragment is a single safe expression); a harness-driven simulation contract is the
  v8 fix (below).
- **The LLM path proposes the right ops but the cheap synthesizer rarely composes
  them.** With the v7 domain-driven prompt the LLM Builder now authors genuine cache
  score ops (`la_plus_fr_bits`, `decay_bump`); they read `unknown` utility only
  because the cheap synthesizer over-explores (a documented v3–v6 trait) and never
  grounds them into a frontier policy. The `beats_human` evidence is from the
  **deterministic** op templates.

---

## What changed in v9 (and why)

v8's honest finding set the agenda: the metric was finally trustworthy, but **no
static eviction score robustly dominates across regimes** — beats_human stayed
suite-sensitive and regime-local. v9 implements v8's own top recommendations.

| Change | What it does | Why |
|--------|--------------|-----|
| **Adaptive primitives** (`cache_eviction._driver_solve` aux support; `_assemble_adaptive`; `ASCORE` op kind) | Optional candidate symbols `AUX0` / `_on_access(aux, hit, t)` / `_evict_score_aux(..., aux)`. The **driver** owns the state: ≤ 8 finite floats, reset per trace, passed as a tuple, malformed updates reset to `AUX0`, malformed `AUX0` hard-fails. The `ascore` kind makes adaptive expressions first-class proposable ops (`fragment_is_safe` admits `aux`). | Regime detection is the only road past the static ceiling: an EWMA of the hit flag detects thrash (loop/scan) mid-trace and switches eviction behaviour — validated grounding-first before building (the adaptive strictly Pareto-dominates the v8 champion per-regime). Soundness is untouched: the hit *flag* is what any online algorithm observes; aux only steers eviction. |
| **Competitive-ratio quality** (`cache_eviction.verdict`) | Quality = `hit_rate / Belady-OPT(cap)` per trace, clamped to [0, 1]; `OPT = 0 → 1.0` (zero hits *is* optimal there); the `hr > OPT` anti-cheat and capacity clamp stay. | Raw hit-rate means are dominated by easy regimes; CR measures "how close to optimal *here*", making cross-regime means meaningful and the bar uniform. |
| **Standardised weighted benchmark** (`cache_eviction.BENCHMARK_SUITE`; `Domain.benchmark_suite`; `engine._utility_workloads`/`_weighted_quality`) | `beats_human` = higher **weighted CR mean** than every human at ≤ equal declared capacity, over 7 declared regimes (weights sum to 1; natural 0.7, adversarial tail 0.3). Candidates and humans are measured on the identical suite + weights. | Makes suite-relativity *declared* instead of implicit, keeps the v8 reproducibility, and stresses exactly the regimes (loop/drift/skew) where static policies disagree. |
| **Generic trusted-driver gate** (`grounding.driver_symbols_ok`; `GroundingSpec.driver_funcs`/`driver_literal_ints`) | The required candidate symbols are spec-driven; the literal-int (source == runtime) soundness rule applies to any declared cost knob. `defines_driver_symbols` stays as the cache alias. | Any future synthesis domain gets verifiable-by-construction grounding for free — v8's pattern, generalised (unit-tested with toy symbol names). |
| **Op proposals on EXPLORE rounds** (`engine.run`) | The vocabulary extends on EXPLORE as well as SYNTHESIZE directives (caps unchanged). | Proposals were hostage to synthesis scheduling — under CR-shifted dynamics a run could explore forever and propose zero ops. Exploration is exactly when new primitives are wanted. |
| **Surrogate repair** (`rde/surrogate.py`) | For driver domains the pre-filter now uses the engine harness on one tiny workload (the skill harness requires a `solve()` and rejects driver text — the surrogate had been a **silent no-op for cache since v8**). | An advisory filter that silently never runs is worse than none: it reads as coverage that doesn't exist. Still advisory-only; the full AST + grounding gate is unchanged. |

### v9 adversarial review — what it caught (all fixed)

A multi-agent review (3 dimensions, per-finding verification; the verify phase was
partly cut short by an account spend limit, so the remaining findings were verified
manually) raised 10 findings. The real ones, now fixed and regression-tested:

- **HIGH — literal-int gate bypass by rebinding** (reproduced by the verifier as a
  frontier-dominating exploit): the gate only counted top-level `Assign`, so
  `CMUL += 3`, a nested `if: CMUL = 4`, `for CMUL in [4]`, `CMUL = X = 4`, an
  `AnnAssign` rebind, or a `global CMUL` mutation let a free-form candidate charge
  `cmul=1` *cost* while running `cmul=4` *quality*. The gate is now a **whole-tree
  binding census** (exactly one single-target literal binding, no deletes, no
  global/nonlocal — which the AST screen now also forbids outright); all six
  vectors are pinned as regressions.
- **HIGH — capacity inflation made CR vacuous**: at `CMUL=8` (cap 80) four of the
  seven benchmark universes fit entirely in cache, giving CR 1.0 to *any* policy on
  0.60 of the weight. `CMUL` is now clamped to **[1, 4]** (contract, driver, and
  verdict), and the benchmark loop is **period-60**, so it pressures *both* legal
  capacities (the old period-40 loop degenerated at cap 40 — the "adversarial tail
  evaporates at c4" finding).
- **MEDIUM/LOW** — proposed-generator name collisions vs not-yet-hardened arsenal/
  benchmark names (collision set now covers everything the run can measure under);
  per-workload persistence in the quality loop (a hang on workload 7 no longer
  discards 6 finished measurements); op utility now selected per-capacity **on the
  judgment metric** (each cap variant measured on the weighted benchmark);
  `classify` no longer labels a no-op-aux policy ADAPTIVE (the active score must
  actually read `aux`); rejected proposals are memoised (no re-screening every
  round); the idiomatic `CMUL: int = 2` is now legal (and cost-parsable).

Re-verified after the fixes: adaptive c2 **0.776 vs best human-at-equal-cost
0.687**, c4 **0.868 vs 0.822**; the loop discriminates at both capacities
(0.45/0.26 vs statics' 0.00); per-regime the adaptive still never loses to the
static champion.

## Honest assessment (v9): did adaptive + competitive-ratio deliver robust progress?

**Yes — this is the first version where the wins are robust rather than localized,
and the first where open invention clears the bar.** The adaptive thrash-guard beats
every static human *and* the v8 champion on the weighted benchmark **without losing
any single regime**, and the capable-model run produced 3 LLM-authored adaptive ops
that all grade `beats_human` under the strictest evaluation the project has had
(unfakeable driver, exact cost, declared weighted CR suite). The v8→v9 arc is the
intended one: first make the metric trustworthy, then earn a bigger number on it.

**What still limits consistent beats_human performance:**
1. **The human baseline is still the four static classics.** Humans also invented
   adaptive policies — ARC, LeCaR — and they are *not* in the baseline. v9's
   `beats_human` means "beats the best static classic at equal capacity", not
   "beats ARC". This is now the weakest link in the claim.
2. **Guided, not de novo, invention.** The op-proposer prompt teaches the adaptive
   idiom with an example; the LLM's 3 winners are genuine, grounded variations
   within that taught pattern (model-authored thresholds/blends/rationales), not
   independently conceived mechanisms. Also, run artifacts persist op *names and
   rationales* but not fragments — provenance auditing needs the expressions.
3. **Benchmark weights are designer-chosen** (0.7/0.3) and each regime runs one
   seed. Reproducible given the declaration, but the declaration itself is a choice;
   multi-seed workloads would close the residual single-seed luck.
4. **Aux persistence is per-trace** (reset between workloads) — deliberate, so
   benchmark traces stay independent and order-insensitive; it does mean no policy
   can learn *across* workloads (that would also break comparability).
5. **Self-play remains specialist-dominated** (win-rate 0.04 vs the best human per
   attack): the adaptives win the declared benchmark, not every targeted attack.

## Highest-leverage direction for v10

1. **Adaptive human baselines (the real bar).** Add ARC and a LeCaR-style
   ghost-list policy to `human_baseline_policies` — likely as driver-level
   reference implementations, since they need more than 8 floats of state — and
   re-grade everything. `beats_human` should mean *beating the best human adaptive
   policy*, the strongest claim this domain can support.
2. **Richer (still bounded, still driver-owned) adaptive state**: a small ghost
   list / limited history as a trusted driver facility candidates can rent, so
   ARC-class mechanisms are *expressible* by proposed ops — paired with proposing
   the `_on_access` dynamics themselves, not just the score.
3. **A second trusted-driver domain** (online admission control or bin packing
   with a poly-time reference) using the now-generic gate, to show driver + CR +
   declared-benchmark is an architecture, not a cache trick.
4. **An invention-provenance study**: persist proposed fragments in run artifacts,
   run N capable-model cycles, and measure invention rate / variance / similarity
   to the taught examples — turning "open invention" from an anecdote into a
   measured property.

---

## What changed in v8 (and why)

v7's honest limitation was a **soundness boundary**: a free-form candidate could
report a hit rate it never achieved (it sees the whole trace, so it can compute
Belady itself). v8 closes that — and the same rigor, applied to v8's own cost axis
and evaluation suite, corrected v7's inflated headline. Design was chosen by a
**3-design judge panel** (trusted-driver vs capability-factory vs verified-replay);
the winner (trusted driver) scored 5/5 soundness from every judge with **zero
residual cheats found**, and grafted the runners-up's hardening.

| Change | What it does | Why |
|--------|--------------|-----|
| **Trusted simulation driver** (`grounding.GroundingSpec.driver_src`, `_RUNNER_TEMPLATE`; `cache_eviction._driver_solve`/`DRIVER_SRC`) | A cache candidate exposes only `_evict_score`/`_on_hit`/`CMUL`; when `driver_src` is set the runner binds `solve` to the **domain-owned** driver and discards any candidate `solve`. The driver owns the hit counter; clamps `CMUL∈[1,8]`; coerces a NaN/non-numeric score to `+inf`; normalises a bad `_on_hit` to the neutral default. | The metric is **computed by trusted code**, never reported by the candidate — closing v7's H1 self-report hole. Hardening means a pathological op can only *degrade*, never inflate, the count. |
| **Exact-capacity cost axis** (`engine._apply_cost_axis`/`_policy_cost`; `_policy_cap` also reads `CMUL = N`) | The frontier cost for a driver domain is `CMUL·k·160 B` — a deterministic integer — instead of tracemalloc peak. `_policy_cap` reads the `CMUL` assignment so it survives `ast.unparse` (mutation strips comments). | Removes measurement noise; two policies at the same `CMUL` are exactly cost-comparable. This is what **exposed** that v7's `beats_human` relied on a proposed op looking marginally cheaper than an equal-capacity human. |
| **Stable utility suite** (`engine._utility_workloads`) | `beats_human`/`matches_human`/`below_human` and the human frontier are measured on the domain's **declared natural + static-adversary** workloads, not the run-variable hardened suite. | Makes the verdict **reproducible** — independent of which attacks the Adversary admitted this run. On it, `rec_logf` genuinely beats LRU at equal capacity (0.822 vs 0.802). `declow` still beats for heavy-hitters. |
| **Closed propose→compose loop** (`engine._compose_uncovered_ops`) | At finalize, the engine composes a policy (via the domain grammar) for every admitted op the Builder never composed, then measures it. | The deterministic Builder composes ops via the grammar; the **LLM Builder writes free-form code and does not** — so an LLM invention would read `unknown`. Now every invention earns an honest flag. Descriptive only; never gates admission. |
| **Generator-targeted self-play** (`cache_eviction._causal_failure_tags` folds in `recent_generators`; `scan_resist` op) | The op-proposer's regime tags include the Adversary's recent generators, so it authors ops aimed at the live attacks. | Aims self-play at the actual attacks, not only the worst main workload. |

**Protective architecture intact.** The driver is *trusted* domain text the engine
controls; candidate ops are *untrusted* and still pass the AST safety screen +
fragment validation + isolated `-E -s` subprocess + RSS watchdog + the grounding
gate. Marginal hypervolume is still the only fitness; comparability,
frontier-protected memory, and strict technical rejection are unchanged; the
trusted-driver mechanism is **opt-in per domain** (`driver_src=""` → byte-identical
legacy path), so heavy-hitters and topk are untouched.

This version was **adversarially reviewed** (a multi-agent workflow across soundness /
backward-compat / honesty / close-the-loop / cost-axis / tests) which confirmed the
driver is sound (no residual cheat), the legacy path is byte-identical, and the
stable-suite change is a reproducibility fix rather than goalpost-moving — and caught
the issues now fixed (see commit history).

## Honest assessment (v8): is the metric now trustworthy?

**Yes — and that mattered more than the number.** The cache metric is now
*unfakeable* (a candidate has no channel to a hit rate it didn't earn — proven by the
cheat test) and *reproducible* (exact-capacity cost + a stable suite). Turning that
rigor on v7's own results showed its “3 `beats_human`” was **measurement-inflated**:
the honest, reproducible count is **1** (`rec_logf`, +0.020 over LRU at equal
capacity). A research engine that exposes its *own* inflated claims is doing the right
thing; v8's deliverable is **trust in the discovery signal**, plus the closed loop
that lets LLM-invented ops actually be measured.

**What still limits the result:**
1. **`beats_human` is suite-sensitive, and no static cache score dominates.** Aging
   ops win on natural/light-loop suites; LFU/MRU win loop-heavy ones. Robust
   comprehensive wins need *adaptive* ops (regime detection), which a pure
   `(la,fr,ins,t)` score cannot express — a real expressivity ceiling.
2. **Invention is still mostly curated.** Causal guidance *selects* `rec_logf` from a
   human-authored template library; the LLM path *invents* but, under the cheap
   model, only *matches*.
3. **The cost calibration is a constant**, not a measured resource.

## Highest-leverage direction for v9

1. **A standardised, weighted benchmark suite** (the #1 item). Fix `beats_human`'s
   suite-sensitivity: a declared, representative mixture (natural-weighted, with a
   bounded adversarial tail) so the verdict is both reproducible *and* meaningful —
   and add a **competitive-ratio** quality (`hit_rate / Belady`) so cross-workload
   means aren't dominated by easy regimes.
2. **Adaptive primitives.** Extend the contract so a policy may keep a *small bounded
   aux state* (still driven by the trusted driver), enabling regime-detecting
   (ARC/LeCaR-style) scores — the only way to *robustly* beat the best human across
   loop + drift + skew at once.
3. **Generalise the trusted driver.** Make `driver_src` the *norm* for synthesis
   domains (not just cache): any domain where the metric is a function of
   candidate-provided ops gets verifiable-by-construction grounding for free.
4. **A capable-model LLM run** to test whether open invention (not curation) clears
   the bar the deterministic templates set.

---

## What changed in v7 (and why)

v6's honest limitation was the **domain**: heavy-hitters is saturated, so even a
genuine discovery (`declow`) could only win on a secondary axis. v7 attacks that
directly — a less-saturated domain where the primary metric has headroom — and
sharpens the three mechanisms that decide *proposal quality*.

| Change | What it does | Why |
|--------|--------------|-----|
| **New domain: cache eviction** (`rde/domains/cache_eviction.py`) | A policy is composed from a **score op** (rank a key for eviction; evict argmin), an **update op** (on a hit), and a **capacity multiplier** (the memory/cost axis). Oracle = **Belady OPT**; quality = **hit rate**. The verdict recomputes OPT at the policy's reported capacity and rejects any rate above it. | Human policies are good but **far from OPT** on loop/drift traces, and **none dominates** — so `beats_human` on the *primary* metric is achievable and measurable, and "best human per generator" is a real moving target. |
| **Causal guidance selects the op SET** (`cache_eviction._causal_failure_tags`, `_OP_TEMPLATES`) | The diagnosed failure's regime (drift/loop/scan/skew) **picks which new ops to author** from a library *larger than the per-round cap*; no padding with unrelated templates. | v6's deterministic proposer merely *re-ordered* a fixed list. Now a drift failure genuinely proposes a different set (aging ops) than a skew failure (frequency-boost ops). |
| **Unified, capacity-matched baselines** (`engine._finalize_v6`, `human_baseline_policies`) | Self-play **and** the utility flag compare against the **best human per generator at the same capacity** (fallback: nearest human cap *not exceeding* the op's). The run reports **net-positive frontier movement** (won self-play ∧ on the frontier). | v6's self-play used a fixed incumbent default (a strawman that overstated wins). The unified baseline is honest and tells generalist (utility) apart from specialist (self-play) gains. |
| **Surrogate pre-filter that composes the skills** (`rde/surrogate.py`, `engine._surrogate_workload`) | Before full grounding, a proposed op is scored on the **highest-headroom** workload (where the human default is weakest) via the extracted `grounded-code-execution` **skill**; clearly-bad ops are skipped. | Trims chaff as proposal volume grows, and is a real "engine uses the skills" link. Probing the *hard* workload (not a default-friendly one) is what stops it suppressing the aging ops v7 is built to find. |
| **Domain-driven LLM op-proposer** (`Domain.op_proposer_sys`/`op_kinds`; `roles._coerce_op_proposal`) | The op-proposer prompt + valid op kinds are **provided by the domain**, so the LLM proposes cache score/update ops (not heavy-hitters victim/inherit/pre). | Keeps the engine domain-agnostic. Before this, the LLM proposed wrong-domain ops that could never compose. |
| **Hardened op fragments + capacity parsing** (`cache_eviction.fragment_is_safe`; `# cmul=` marker) | Every score/update fragment must be a **single safe expression** over `{la,fr,ins,t}` (+ `max/min/abs`, `.bit_length()`); the capacity is read from a structured marker, not a brittle regex. | Op-composed policies provably cannot smuggle statements past the module AST screen, and the cost axis is parsed reliably. |

**Protective architecture intact.** Marginal-hypervolume is still the only fitness;
comparability, frontier-protected memory, isolated grounding, and **strict technical
rejection** are unchanged; the surrogate is **purely subtractive** (it runs *after*
the AST screen and can only skip, never admit); self-play is **descriptive, never
gating**. Set `--domain heavy_hitters` for the v2–v6 path.

This version was **adversarially reviewed** (3 parallel reviewers). They confirmed
`_belady` is the true optimum (brute-forced on 3000 traces, 0 mismatches), the
frontier fairness is real, and all four invariants hold — and caught real issues now
fixed: an unfair self-play capacity fallback, a surrogate that probed a
default-friendly workload (could suppress the very ops v7 targets), a vacuous-OPT lever
(unclamped reported capacity), a fragment-injection surface, and stale report/docstring
labels. They also flagged the **scalar-contract soundness boundary** (above), the v8
priority.

## Honest assessment (v7): did the domain change deliver clearer signal?

**Yes, on the headline question.** Moving off the saturated heavy-hitters domain
turned the primary-metric `beats_human` from *artificially hard* into *routine*: a
deterministic run produces **3 `beats_human` ops on hit rate (rate 0.6)** with
**net-positive frontier movement**, versus v6's **single** win that lived only on the
memory axis. The new domain is a strictly better *measuring instrument* for "can the
architecture beat strong humans on the metric that matters" — there is real,
quantifiable headroom (best-human-vs-OPT gaps of 0.08–0.24 across workloads) and no
dominant human policy, so a win is unambiguous rather than a memory-axis technicality.

**What still limits proposal quality:**
1. **The deterministic op library is human-curated.** The three `beats_human` ops are
   hand-authored templates (`lfu_aged`, `aged_logf`, …); causal guidance *selects* among
   them well, but it does not *invent* the expression. Genuine open invention only
   happens on the LLM path — which proposes good ops but, under the cheap synthesizer,
   rarely composes them onto the frontier (so they read `unknown`).
2. **Self-play wins are scarce against the strongest human per attack.** The proposed
   ops are better generalists than specialists; pushing self-play to *net* wins needs
   the proposer to target the adversary's generators directly, not just the worst main
   workload.
3. **The scalar contract is only verifiable up to OPT.** Demonstrated results are sound
   (op-composed, trusted simulation), but the contract cannot falsify a free-form
   policy that reports an OPT-bounded rate it did not achieve.
4. **Cost.** The surrogate adds up to ~`max_proposed_ops_per_round` cheap subprocesses
   per synthesis round on top of grounding; bounded, but it grew with v6/v7 volume.

## Highest-leverage direction for v8

1. **A harness-driven simulation contract (closes the soundness boundary).** Stop
   trusting a scalar return: have the policy provide *only* the score/update ops, and
   let a **trusted harness** run the eviction simulation and compute the hit rate.
   Then *no* policy — free-form or op-composed, deterministic or LLM — can misreport,
   and the cost axis becomes the exact declared capacity rather than noisy tracemalloc.
   This is the natural endpoint of the op-composition architecture.
2. **Close the LLM-propose → compose loop.** Make the LLM-proposed ops actually reach
   the frontier (compose admitted LLM ops via `assemble_from_vocab` even when the cheap
   synthesizer over-explores, or trigger a synthesis pass whenever a new op is
   admitted). That is the path to *invented* (not curated) `beats_human` primitives.
3. **Generator-targeted self-play.** Aim the proposer at the Adversary's specific
   generators (not only the worst main workload) so self-play can produce *net* wins —
   turning robustness pressure into measurable frontier movement.
4. **A second non-saturated domain** to show the gains are architectural, not
   cache-specific (e.g. online bin-packing or admission control, both with poly-time
   or near-optimal reference oracles).

---

## What changed in v6 (and why)

v5's honest limitation was **proposal quality**: an open vocabulary that only ever
*matched* human ops. v6 adds tighter feedback so a proposal can be *meaningfully
better* — accepting more short-term noise (higher volume, looser upstream filtering)
in exchange for faster iteration on quality, while keeping every protective invariant.

| Change | What it does | Why |
|--------|--------------|-----|
| **Causal guidance** (`grounding.diagnose`; `engine._build_causal_guidance`) | Diagnoses the best policy's worst workload — the oracle top-k vs the policy top-k and the items it *missed* — and feeds that concrete failure to the proposer as the **primary** signal (keyword tags + frontier gap demoted to secondary). | A proposal aimed at a *measured* failure (`decoy` inflates transients) is better than one aimed at a keyword. The threshold-prune `declow` came directly from this. |
| **Self-play** (`rde/selfplay.py` `SelfPlayLedger`; `engine._finalize_v6`) | Builder authors ops to beat the Adversary's recent generators; Adversary authors generators to defeat recent ops. **Win-rate + improvement-margin** vs the v4 incumbent default are first-class metrics. | Iterative pressure toward robustness. It is **descriptive only** — `self.selfplay` is written once after the loop and read only by the report; it never touches fitness, the frontier, or admission, so the engine **cannot overfit to its own attacks**. |
| **Utility flags + history** (`selfplay.UtilityFlag`; `engine._classify_utility`, `_finalize_v6`) | Every proposed op is labelled `beats_human` / `matches_human` / `below_human` vs the **full human op-combo Pareto frontier**; a per-op margin and a **cross-run `utility_history.jsonl`** make the trend visible across runs. | Makes "did this proposal actually help?" explicit and auditable — and surfaced the one genuine `beats_human` (`declow`). |
| **Higher volume + post-grounding filtering** (`config`: 8 ops/round, cap 16; broader `_proposed_op_genomes`) | More proposals, each composed over more backbones; the **frontier + utility flag** filter *after* grounding. Technical rejection (AST screen, bounded/non-crashing/stream gates, RSS watchdog) stays strict. | Trades upstream pre-filtering for grounded selection — the brief's "accept short-term noise for faster quality iteration." |

**Protective architecture intact.** Marginal-hypervolume is still the only fitness;
comparability, frontier-protected memory (v5), and isolated grounding are unchanged;
self-play is non-gating. Set `open_vocabulary=False` for the v4 path.

This version was **adversarially reviewed** (a 21-agent workflow). It confirmed the
`declow` discovery is real (brute-forcing all 480 human op-combos — none dominate it)
and that self-play is structurally non-gating, and it caught **reporting overclaims**
— a `-1.0` sentinel that inflated the utility margin, a strawman self-play baseline,
an over-asserted causal hypothesis, and trend-computation bugs — **all now fixed**.

---

## What changed in v5 (and why)

v4's honest limitation was that the op vocabulary was **hand-designed** — the engine
composed a *combination* of ops a human wrote. v5 lets the Builder and Adversary
**extend the vocabulary at runtime, as source code**, under the same Empirical
Grounding Rule that governs everything else: nothing influences the frontier or the
Adversary on the strength of a *proposal* — only after it is screened and run.

| Change | What it does | Why |
|--------|--------------|-----|
| **Provenance-tracked vocabulary** (`rde/vocab.py`: `Vocabulary` / `OpEntry` / `GenEntry` / `Provenance`) | The live registry of ops + generators, each tagged `human` / `proposed-det` / `proposed-llm`. `_assemble_policy` reads ops from it; the run report prints a PROVENANCE block. | Lets the system say, honestly, "this op on the frontier was *proposed by the system*, not designed by a human" — and distinguish *matched* from *beaten*. |
| **Open op proposal** (`heavy_hitters.propose_operations` / `op_probe`; `roles.LLMBuilder.propose_operations`; `engine._extend_op_vocabulary`) | The Builder authors a new victim/inherit/pre op **as source**, guided by frontier-gap + weakness tags. The engine AST-screens it, splices it into a probe policy, and grounds it on a correctness-focused spec; admits only safe+correct ops. | The basis is no longer fixed: a *proposed* op (`min_or_one`, `prune_ones`) composes into policies and is filtered by the frontier exactly like a human op. |
| **Open generator proposal** (`heavy_hitters.propose_generators` / `generator_block`; `roles.LLMAdversary.propose_generators`; `engine._extend_generator_vocabulary`) | The Adversary authors a new `gen(n, rng)` body **as source**; it dispatches through gen_workload's `prop:<name>` branch. The engine AST-screens it and **grounds it at production scale**, rejecting any generator that raises, returns a non-stream, or is unbounded, before it can harden the suite. | New attack *structures*, not DSL parameterisations — `decoy`, `comeback`, and (live) LLM-authored `gen_oscillation` etc. |
| **Hardened subprocess** (`grounding._run_with_limits`) | A psutil RSS watchdog kills a proposal subprocess the instant it breaches a memory cap (`subprocess_mem_cap_mb`), so an unbounded allocation cannot thrash the host before the wall-clock timeout. (`resource.setrlimit` is POSIX-only; this is the cross-platform guard.) | Open proposal means *untrusted* code authored by an LLM; the sandbox must be robust to a buggy/adversarial proposer, not just a benign one. |

**The protective architecture is preserved and extended.** Marginal Pareto
*hypervolume* is still the **primary fitness**; comparability (`_ensure_full_coverage`
/ `_rescore_frontier`, now measuring against the *live* spec that carries proposed
generators), frontier-protected bounded memory (protected members are now **exempt
from cap truncation**, so v5's larger frontier can't evict a Pareto point), behavioural
novelty, recombination flagging, and isolated grounding all stay active. Set
`open_vocabulary=False` to fall back exactly to v4.

This version was **adversarially reviewed** (a 22-agent workflow); the review confirmed
the result is honest and the architecture intact, and surfaced soundness gaps in the
proposal gates (a raising/non-stream/unbounded generator could be admitted; op
admission was coupled to benchmark wall-clock; an unbounded allocation could destabilise
the host) — **all now fixed** and covered by tests.

---

## What changed in v4 (and why)

v3's honest limitation was that the synthesis grammar enumerated **whole named
policies** — the engine composed a *combination*, but every policy in the grammar
was one a human had named. v4 drops the basis to **low-level operations**, makes the
Adversary **invent new attack types**, and makes the **frontier re-score itself** so
hardening can't make it dishonest. It extends the v1/v2/v3 architecture, not
replaces it.

| Change | What it does | Why |
|--------|--------------|-----|
| **Eviction-policy GENOME** (`domains/heavy_hitters.py`: `_VICTIM_OPS` / `_INHERIT_OPS` / `_PRE_OPS` / `_assemble_policy` / `_GENOME`) | The Builder is given only low-level ops (victim choice × inherited-count × global pre-step) and **synthesises the policy by composing them**. Most of the 4×5×3 op space is policies no one named; `synthesize_programs()` emits the genome, the frontier keeps the winners, all tagged `AlgoClass.INVENTED`. The winner `min`/`victim÷2`/`none` is **outside** `V3_POLICY_GENOMES`. | The basis is now *operations*, not *policies* — the engine assembles a policy whose core rule was never written down, and a `V3_POLICY_GENOMES` set lets the tests *prove* it lies outside the prior grammar. |
| **Generator-level Adversary** (`heavy_hitters.adversary_generators` + `gen_workload "gen:"`) | A bounded generator DSL lets the Adversary compose **new attack TYPES** — `phase` (heavy set shifts mid-stream), `spike` (quiet background + bursts), `mixture` (two interleaved heavy sets) — not just re-parameterise one template. | The metric attack is now *structural*: a non-stationary stream a single `param:` schema (one fixed heavy set) cannot express. Adding one triggers re-scoring. |
| **Frontier re-scoring on hardening** (`engine._rescore_frontier` + `_ensure_full_coverage`) | When the Adversary hardens the suite, **every existing frontier member is re-measured** on the new workloads and the frontier is rebuilt; fresh candidates are filled to full suite coverage before admission. Bounded memory is **frontier-aware** (`Memory.protected_ids`) so a Pareto member is never evicted. | Closes v3's comparability gap: all frontier points are always scored on the *same growing metric*, so the marginal-hypervolume reward stays honest as the suite hardens. |

**The protective architecture is preserved and strengthened.** Marginal Pareto
*hypervolume* is still the **primary fitness** (dominated/recombinant points score
0); behavioural novelty, recombination flagging, bounded memory, objective signals,
and isolated-subprocess grounding all stay active. v4 makes the frontier *stricter*
(re-scoring) and memory *safer* (frontier-aware pruning).

---

## What changed in v3 (and why)

v2's honest limitation was that synthesis recombined a **fixed library**. v3 opens
the generation space with two grounded mechanisms, plus a generative Adversary —
extending the v1/v2 architecture, not replacing it.

| Change | What it does | Why |
|--------|--------------|-----|
| **Compositional grammar** (`domains/heavy_hitters.py`) | The Builder *assembles* `solve` from primitive fragments: a `(capacity × eviction-policy)` grammar (`_SKELETON` + `_EVICTIONS`). Eviction = *replace-min-and-inherit* composes **Space-Saving**; *drop-min* and *drop-arbitrary* are also novel. `synthesize_programs()` walks this space; the frontier keeps the winners. Composed programs are tagged `AlgoClass.INVENTED`. | The combination space contains algorithms **not in the library** — genuine compositional invention, not selection. |
| **AST-level mutation** (`mutate.py`) | `EXPLORE` parses a frontier member, perturbs a numeric literal or flips a comparison operator, and re-emits source. | A second open-ended source: find better parameterizations / variants by search. |
| **Generative Adversary** (`heavy_hitters.adversary_param_attacks` + `gen_workload "param:"`) | The Adversary *composes* parameterized hostile streams at runtime from knobs (`late`, `heavy`, `coll`) rather than picking from a fixed named set. | Attacks the metric with novel failure modes nobody pre-listed — bounded by the param schema, verified by measured recall. |

**Grounding is preserved end-to-end:** every composed or mutated candidate goes
through the same isolated-subprocess execution + recall@K oracle, the same
Adversary metric-attack and suite-hardening, the same frontier (marginal
hypervolume) admission. The convergence guards generalise — a winning *synthesised
or invented* program (`HYBRID` **or** `INVENTED`) stops the synthesis loop, and the
grammar is walked in bounded batches under the synthesis-attempt cap.

---

## What changed in v2 (and why)

v1 reached its ceiling not because the engine was weak but because top-K is a
*closed* problem (known optimum, pass/fail oracle): recombination was sufficient
and the synthesis win was a noise-sized ~0.02. v2 opens the problem so genuine
synthesis has clear, measurable value — keeping the v1 engine architecture intact.

| Change | What it does | Why |
|--------|--------------|-----|
| **Open domain** (`domains/heavy_hitters.py`) | Streaming frequent-items under a memory budget. Continuous **recall@K** quality vs an exact `Counter` oracle; peak memory the binding resource. Textbook strategies (exact, Misra-Gries, Count-Min, sampling) + a deliberately metric-gaming `first_k_distinct`. | No single method is best → an adaptive hybrid can reach a region none occupy. Discovery has headroom. |
| **Frontier reward** (`frontier.py`) | Tracks the (recall × memory) Pareto frontier; a candidate's **fitness = its marginal hypervolume contribution**. Dominated/recombinant points score 0; points reaching a new region score positive *even if worse on one axis*. | Rewards *reaching new value*, not *being different*. Plugs into the v1 fitness-driven loop: "fitness gain" becomes "frontier advanced", "stall" becomes "frontier stalled". |
| **Continuous verdict** (`grounding.py`) | `GroundingSpec.verdict_src` lets a domain return `(ok, quality)` instead of exact equality. The gate becomes *well-formedness*; recall is the quality axis. | Approximate methods are admissible and ranked by the frontier, not pass/fail. |
| **Adversary attacks the metric** (`roles.py` + `engine.py`) | Measures candidate recall on hostile streams (skew-overflow, collision, heavy-hitters-arrive-late). On a confirmed *collapse* (looked good on easy streams, fails a hard one) it **downgrades the candidate AND permanently hardens the eval suite** with that workload — the v1 suite-hardening mechanism, lifted from correctness to the metric. | Stops the engine accepting solutions that only look good because the benchmark is weak/gameable. |

Everything else is reused: isolated-subprocess grounding, behavioural-novelty,
recombination/drift signals, bounded conceptual memory, the human-override
checkpoint, and the convergence guardrails. The deterministic roles remain the
zero-cost reference; the LLM roles write streaming code / propose hostile streams.

---

## What changed in v1 (and why)

v0 was a stable *structured empirical search*: it explored a class library,
graded by execution, and stopped without collapsing. But its "synthesis" was
really selection, its memory was approaches-and-scores, and its novelty check was
purely structural. v1 targets the gap between *search* and *discovery*:

| # | v1 upgrade | What it actually does | Why it matters |
|---|------------|-----------------------|----------------|
| 1 | **Stronger synthesis** | The Builder gains a real `SYNTHESIZE` mode: it composes an **adaptive hybrid** that dispatches on the input regime (e.g. `k`) to route to the strategy that wins it. The Synthesizer *pushes* this when distinct approaches own distinct regimes. New `synthesis_gain` metric checks the hybrid beats **each parent on that parent's home workload**. | Produces something strictly better than the best single parent, not a re-pick. |
| 2 | **Conceptual memory** (`concepts.py`) | A `ConceptLedger` distils transferable structural lessons from evidence + fingerprints: *"value-indexed counting fails on negatives/floats"*, *"heap wins small-k, sort wins large-k"*, *"quickselect: better asymptotics, worse wall-clock"*. Bounded, merge-and-strengthen. | Future exploration builds on *why*, not just *what*; dead ends aren't re-explored. |
| 3 | **Deliberate exploration** | New `EXPLORE` action + behavioural-niche map (per-workload winners). When work goes stale, the Synthesizer directs the Builder at an under-owned niche (LLM: diversity-constrained, high-temperature prompt) instead of waiting for it to "think of something new". | Diversity becomes a driven mechanism, not luck. |
| 4 | **Tighter, harder-to-game drift** (`behavior.py`) | **Behavioural novelty** compares *measured timing profiles*, catching "sophisticated local patching" (rewritten source, identical behaviour). **Noise-aware progress** uses benchmark stddev so a within-noise "speedup" isn't counted. **`recombination_dominant`** keys off measured structure-or-behaviour staleness over a window. | The Synthesizer is handed facts it cannot rationalise away. |
| 5 | **Reduced scaffolding** | The LLM Builder writes free-form code with `synthesize`/`explore` modes; novelty/exploration are driven by *measured behaviour* and runtime niche coverage rather than the hand-coded class list; the LLM Synthesizer chooses freely inside guardrails. The deterministic roles remain as the zero-cost reference/test harness. | More reasoning from the models, same protective floor. |

**Protective mechanisms are untouched:** empirical grounding via isolated
subprocess + oracle, adversarial pressure (with suite-hardening), bounded memory,
and objective signals the Synthesizer cannot override — all retained, and the
anti-collapse guardrails were *strengthened* (recombination-dominant now forces
synthesis or exploration; a failed synthesis can't loop).

### v1 architecture delta

```
NEW   rde/behavior.py   behavioural descriptor + similarity + synthesis_gain
NEW   rde/concepts.py   ConceptLedger: distil/store/merge structural lessons
EDIT  rde/types.py      Action.EXPLORE, AlgoClass.HYBRID, behavioural+synthesis fields,
                        Evidence.timing_stddev/noise, Lesson record
EDIT  rde/grounding.py  per-workload stddev; PER-WORKLOAD perf scoring (each regime equal)
EDIT  rde/signals.py    synthesis_available, recombination_dominant, behavioural novelty,
                        synthesis_gain, noise floor
EDIT  rde/memory.py     ConceptLedger wired in; richer digest (concepts first)
EDIT  rde/roles.py      Builder SYNTHESIZE/EXPLORE modes (det + LLM); Synthesizer pushes
                        synthesis/exploration; guardrails G4 (recomb-dominant), G5
EDIT  rde/domains/topk.py  small-k + large-k regimes; hybrid composer; regime/failure hooks
```

### v1 hardening (from a 51-agent adversarial review)

v1 was put through the same adversarial multi-agent review as v0 (5 dimensions,
every finding independently verified). It confirmed 31 findings; the substantive
ones were fixed and locked with tests:

- **Completeness gate (critical).** A candidate whose benchmark crashed or timed
  out *part-way* was being recorded as `correct` with a **partial** set of
  timings — which would let it "win" some regimes while hiding a blow-up in
  another, and could establish a degenerate baseline from a half-benchmarked
  seed. Now: a correct candidate must have **complete** benchmark coverage of
  every workload, or it is not admitted as correct.
- **Convergence guarantees.** A free LLM Synthesizer could keep choosing
  `SYNTHESIZE`/`EXPLORE` to "break the plateau" forever. Added a hard
  synthesis-attempt cap, a "synthesis already failed → don't retry the identical
  attempt" guard, and **G7: stalled + a correct best ⇒ ACCEPT** — so the loop
  converges on the winner instead of running to `max_iterations`.
- **Lineage protection.** Memory pruning now never evicts an approach that is a
  *parent* of a retained one, so `synthesis_gain` and parent resolution can't
  silently break.
- Plus smaller fixes: full-window guard on `recombination_dominant`, safe parse
  of an unrecognised LLM action, redundant-regex cleanup.

---

## TL;DR — run it

```bash
# Zero cost, no API key: the deterministic reference policy.
python -m rde --backend deterministic

# Live, three different model vendors via OpenRouter (needs a key in .env).
python -m rde --backend llm --profile cheap --iterations 6

# Pause for human review/override at every synthesis checkpoint.
python -m rde --backend llm --interactive
```

Only dependency is `pytest` (for tests). The engine core, the grounding harness,
and the OpenRouter client use **only the Python standard library** — by design,
so the empirical harness runs anywhere.

### Verified results (v1, deterministic)

The engine explores 4 algorithmic classes, **disqualifies the broken bucket
strategy by execution**, scores quickselect *below* the heap (the empirical trap,
found by measurement), distils the conceptual lessons, then **synthesizes an
adaptive hybrid that beats every single approach** and accepts it:

```
SEED:  naive_select       fit 0.500   12.1 ms
  it1  heapq_nlargest     -> wins small-k
  it2  sorted_topk        -> wins large-k
  it3  quickselect_3way   -> correct but slow  (trap distilled)
  it4  bucket_nonneg      -> DISQUALIFIED on negatives  (cause distilled) + SYNTHESIZE pushed
  it5  HYBRID (k-dispatch)-> fit 0.704, 2.2 ms, synthesis_gain 0.94  -> ACCEPT
BEST:  adaptive hybrid    fitness ×1.41 over seed, wall-clock ×5.5, beats best single approach
```

The hybrid is a genuine combination — `if k >= n//2: sorted(...)[:k] else
heapq.nlargest(k, items)` — that wins **both** regimes, which no pure strategy
does. Conceptual lessons retained include *heap→small-k*, *sort→large-k*,
*quickselect is the wall-clock trap*, and *value-indexed counting breaks on
negatives/floats*.

**Live-LLM backend** (gpt-5-mini / gemini-2.5-flash / claude-haiku-4.5) runs the
identical loop: the Builder writes a winning hybrid free-form (×4.6 over seed,
status `ACCEPTED`), the Adversary attacks each candidate, and the Synthesizer's
guardrails drive convergence — notably the AST screen rejected an LLM-proposed
candidate that used a `__slots__` dunder, and the stall-→-accept guardrail
stopped the loop on the winner instead of churning.

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
tests/            138 tests: grounding, fingerprint, signals, llm, engine, v3–v9, skills
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

## Honest assessment (v6): did aggressive pursuit of proposal quality pay off?

Being blunt, because the brief asks for it. **Mostly yes — with one genuine win and
a lot of measured noise around it.**

**The win is real.** For the first time in the project a *system-proposed* primitive
**beats the human baseline**, not just matches it: `declow` (prune count ≤ 2 on
overflow) is structurally outside the human pre-op vocabulary, and `min/one/declow/32`
lands at **0.84 recall @ 13 KB**, a `(recall, memory)` point that Pareto-dominates the
*entire* 480-policy human op-combo space (the adversarial review brute-forced it; none
dominates or ties). And it arrived through the intended mechanism — a **concrete
diagnosed failure** (the Adversary's `decoy` attack) steered the proposer to a
threshold-prune op. Causal guidance + self-play + the Adversary's open generators
genuinely composed into a better proposal.

**But the win is narrow, and several v6 signals are weaker than they first look** (the
review's confirmed corrections, now applied honestly):

- **It's a memory-axis win, not a recall win.** declow reaches a recall *no cheaper
  than* the best human op — it just reaches it at *less memory*. On the primary axis
  (recall), the suite's ceiling is still human-reachable (`min/one` matches declow).
  The recommended shippable point is a `min/one` policy (≈0.90), essentially human.
- **The self-play "+0.30 win" is over the v4 *incumbent* (`vhalf`), not the best
  human.** Against `min/one`, declow *ties*. The win means "better than what v4
  shipped," which is true and useful, but it is not "better than the best human."
- **Causal guidance is shallow for the deterministic Builder** — it mostly *re-orders*
  a fixed 4-template set; it is genuinely free-form only for the LLM Builder (which
  the cheap synthesizer rarely reaches). The "primary signal" claim is real for the
  LLM path, weak for the deterministic one.
- **The proposal mechanism is noisy:** of every proposed-op batch, ~1 beats, ~1
  matches, ~2 are below human; the cross-run `beats_human_rate` is a steady **0.25**.
  Higher volume bought one good op surrounded by chaff that the frontier discards.

So: proposal quality improved **meaningfully but locally** — the system can now author
a primitive that genuinely beats the best human on a *secondary* axis, driven by a
real failure signal. It cannot yet beat the best human on the *primary* axis, and the
"quality machinery" is theatre on the deterministic path (a real result arrived through
a largely cosmetic guidance loop). This is a step past v5's "admissible-but-matching,"
not a leap to "the system out-designs humans."

## New bottlenecks once you push proposal quality

1. **The domain ceiling is human-saturated on the primary axis.** Every recall
   improvement is reachable by some human op-combo; the only open axis was memory.
   You cannot demonstrate a primary-axis `beats_human` on a problem humans already
   solve optimally — the *problem* now bounds the *result*.
2. **Causal guidance is only as good as the proposer it feeds.** A fixed template set
   turns "guidance" into "re-ordering." Real causal benefit needs a proposer that can
   *generate* an op shaped by the failure, not select one.
3. **Self-play vs a saturated best-human collapses to ties.** A meaningful win-rate
   needs either a moving incumbent (beat your own previous best) or a domain where the
   best human is genuinely beatable — otherwise the metric reads ~0.
4. **Cost.** v6's finalize adds ~40 grounding subprocesses/run (human frontier +
   self-play tournament + re-measurement); runs are ~3× v5. Aggressive volume without
   a cheap surrogate pre-filter does not scale.

## Highest-leverage direction for v7

**Make causal guidance change the proposed *set*, not its order — and move to a
regime where the best human is genuinely beatable.** Concretely: (a) give the
deterministic Builder *many more* op templates than `proposed_ops_cap` and drop the
"+2 general" padding, so a concrete diagnosis genuinely *selects* survivors instead of
re-sorting a tiny fixed list; make `diagnose` **tie-aware** so it never steers on a
count-floor boundary artifact. (b) **Unify the self-play and utility baselines** into
one per-generator *best-human* frontier, so every reported "win" means "beats the
strongest human a person would ship" (no incumbent strawman). (c) Add a **cheap
surrogate pre-filter** (score a proposed op on one small hostile stream before paying
for full grounding) so volume scales. (d) Most importantly, **change the problem**:
run on a regime/domain where bounded summaries do *not* saturate the recall ceiling
(e.g. tighter budgets, adversarial drift, or a different problem) so a proposed
primitive can beat the best human on the *primary* axis — the real remaining gap.

## Success criteria for v6 — met

1. ✅ **Causal guidance as the primary signal**: `diagnose` feeds the actual failing
   stream + missed items to the proposer; keyword tags + frontier gap are secondary.
2. ✅ **Self-play with first-class metrics**: win-rate + improvement-margin tracked
   (`SelfPlayLedger`); the Adversary authors generators to defeat recent ops; it is
   **non-gating** (verified: cannot overfit to its own attacks).
3. ✅ **Explicit utility flags**: every proposed op labelled `beats_human` /
   `matches_human` / `below_human` vs the full human frontier; flag + margin recorded.
4. ✅ **Higher volume + post-grounding filtering**: 8 ops/round, broader composition,
   filtered by frontier gain + utility; strict technical rejection preserved.
5. ✅ **Measurable improvement over v5**: **1 `beats_human` per run (declow)** vs v5's
   **0**; confirmed genuine by a 480-combo brute-force, not a baseline artifact.
6. ✅ **Provenance + utility history across runs**: `utility_history.jsonl` +
   `summarize_utility_history` (beats-rate, early→late trend).
7. ✅ **Expanded coverage**: 89 → **97 tests** (causal diagnose, self-play ledger,
   utility flags/history, the declow-dominates-human invariant, higher volume).

---

> **Verification footnote.** v6 was checked by a deterministic run (above), 3 stacked
> runs to populate the cross-run trend (beats_human_rate 0.25, flat — deterministic),
> a short live LLM run (the Adversary authored 4 generators incl. `gen_cascading_decoy`
> in visible response to the self-play prompt; the cheap synthesizer over-explores so
> LLM *op* proposal isn't reached — deterministic is the clean op demonstration), the
> 97-test suite, and a **21-agent adversarial review**. The review confirmed the
> `declow` `beats_human` is real (480-combo brute force) and self-play is non-gating,
> and surfaced reporting overclaims now **fixed and tested**: the `-1.0` utility-margin
> sentinel (→ compare vs the cheapest real human point; +1.7 became +0.18), the
> strawman self-play baseline (→ relabelled honestly as "vs the v4 incumbent vhalf,
> not the best human"), the over-asserted causal hypothesis (→ factual + tie-aware
> wording), a single-run trend bug and a lexicographic run-id sort (→ "n/a" until ≥2
> runs; numeric sort), and a run-id race (→ timestamp+pid id).

---

## Honest assessment: how close is v5 to conceptual discovery?

Being blunt, because the brief asks for it.

**What v5 genuinely achieves.** v5 delivers a *working, grounded, provenance-tracked
open-vocabulary pipeline*. The Builder and Adversary author **new ops and generators as
source code**; each runs the gauntlet — AST safety screen → splice/wrap → isolated
subprocess execution → oracle/recall grounding → frontier or exposure filtering — and
only survivors join the vocabulary. Concretely and verified:

- The deterministic Builder **proposed a new eviction op** (`min_or_one`: evict a
  count-1 *transient* first, protecting built-up heavy counters), the engine
  AST-screened and grounded it, and a policy built from it **landed on the Pareto
  frontier** at recall 0.76 @ 18KB. Provenance records it as system-proposed.
- The deterministic Adversary **proposed a new generator** (`decoy`), grounded it, and
  it **collapsed a candidate below the floor** — the invented attack did real work.
- Live, the **LLM Adversary authored four structurally-new generators**
  (`gen_oscillation`, `gen_drifting_heavy_set`, `gen_intermittent_heavy_decoy`,
  `gen_interleaved_heavy_decoy`), all screened + grounded + admitted, **two of which
  exposed candidates** — open generator synthesis working with a real model.

**The load-bearing honesty caveat:** the proposed op only **MATCHES** the best human op
at equal memory; it never strictly advances the frontier *beyond human reach*. Its
frontier slot is an ordering+dedup artifact — proposed policies are synthesised first
and win first-seen ties; the byte-equivalent human policy is batch-cut. On this
near-saturated suite the human inheritance rules are already close to optimal, so v5
demonstrates **admissibility, grounding, and honest credit for an open vocabulary, not
superhuman discovery**. The **recommended shippable point stays the human `min_half32`**
(0.88 @ 36KB). The report and tests say this explicitly — the credit signal is labelled
"matching, not beating," and a test asserts the tie.

**And the deeper caveat — the *generator* of proposals is still mostly human.** The
deterministic op/generator templates are hand-written; the LLM proposer is genuinely
open but unguaranteed. So v5 is **"the system can author and ground a new primitive that
was not in its vocabulary,"** which clears the brief's bar ("an accepted result that uses
a newly proposed operation that advances the frontier" — a non-dominated point was
added). It is **not** "the system invented a fundamentally new algorithmic idea that
beats the human design." The ceiling moved from *fixed op set* to *open op set under a
human-or-LLM proposer*, but proposal *quality* (does an authored op beat the baseline?)
is the new wall.

## New bottlenecks once the vocabulary is opened

1. **Proposal quality is the new ceiling.** Admissibility (safe + correct + grounded) is
   solved; *utility* (does a proposed op beat the human baseline?) is not. Every v5
   proposal ties or is dominated at equal memory. The bottleneck is now *how good are the
   proposals*, which is bounded by the proposer (template heuristics, or LLM judgement)
   and by guidance quality.
2. **Search cost explodes.** Each proposal costs a screen + a full grounding subprocess;
   composing proposed ops multiplies candidates per round. v5 runs are ~3× slower than
   v4. Without a cheap *surrogate* pre-filter, an open search won't scale.
3. **Verification of untrusted authored code is hard.** The AST screen is defence in
   depth, not a boundary; v5 added an RSS watchdog + bounded-stream + correctness-only
   op grounding because an LLM/adversarial proposer *will* emit raising, unbounded, and
   slow code. Each new degree of freedom is a new way to be wrong-but-plausible, and the
   grounding harness must stay ahead of it.
4. **Guidance is shallow.** v5 derives weakness *tags* (late/overflow/collision) from
   text; it does not reason about *why* a policy fails or *what op shape* would fix it.
   The proposer is only as good as the guidance, and the guidance is keyword-level.

## Highest-leverage direction for v6

**Close the loop from grounded failure → targeted proposal → measured improvement, and
make the proposer earn its keep.** v5 proved proposals can be authored and grounded
honestly; v6 should make them *good*: (a) **causal guidance** — instead of weakness
tags, feed the proposer the *specific* streams a frontier policy fails on and *why* (a
diff of where its top-k diverges from the oracle), so an authored op targets the actual
failure; (b) a **surrogate/cheap pre-filter** (measure a proposed op on one small hostile
stream before paying for full grounding) so an open op space becomes searchable; (c) a
**utility gate with honest credit** — admit an op to the *frontier* only when it reaches
a (recall, memory) point no human-only policy ties, and report `beats_human` vs
`matched_human` as first-class provenance (v5 reports the distinction in prose; make it a
measured flag); (d) **self-play between the two open vocabularies** — the Adversary's new
generators should *drive* the Builder's next op (decoy neutralised prune_ones in v5; v6
should let the Builder author an op specifically to beat the decoy). That is the move
from "can author a primitive" to "authors a primitive that *wins*," which is the real
remaining gap to conceptual discovery — and v5's grounded, provenance-tracked pipeline is
exactly the substrate for it.

## Success criteria for v5 — met

1. ✅ **Open op proposal (Builder)**: new ops are authored *as source*, AST-screened +
   grounded before admission, guided by frontier gaps + weakness concepts
   (`propose_operations` / `op_probe` / `_extend_op_vocabulary`).
2. ✅ **Open generator proposal (Adversary)**: new generators are authored *as source*,
   AST-screened + grounded (well-formed, bounded, runnable) before they harden the suite
   (`propose_generators` / `generator_block` / `_extend_generator_vocabulary`).
3. ✅ **Protective architecture preserved + extended**: hypervolume-primary fitness,
   comparability (against the *live* spec), frontier-protected memory (now exempt from
   cap truncation), novelty/recombination, isolated grounding — all intact; plus an RSS
   watchdog for untrusted code. `open_vocabulary=False` restores v4 exactly.
4. ✅ **Provenance tracking**: every op/generator is `human` / `proposed-det` /
   `proposed-llm`; the report prints human-vs-proposed counts and which proposed ops
   reached the frontier.
5. ✅ **An accepted result uses a newly-proposed op that advances the frontier**:
   `policy_min_or_one_vhalf16` (proposed op `min_or_one`) is a non-dominated frontier
   point at 18KB — *honestly reported as matching, not beating, the human op*.
6. ✅ **The Adversary uses a newly-proposed generator structure**: `decoy` (det) and two
   LLM-authored generators collapsed candidates below the floor.
7. ✅ **Expanded coverage**: 73 → **89 tests** (op/generator proposal safety + grounding,
   raising/non-stream/unbounded-generator rejection, the matches-not-beats honesty
   invariant, frontier protection under v5 memory pressure, and provenance).

---

> **Verification footnote.** v5 was checked by a deterministic run (above), a short live
> LLM run (the LLM Adversary authored 4 new generators, 2 exposing candidates; the cheap
> synthesizer over-explores so LLM *op* proposal isn't reached in a short budget —
> deterministic is the clean op demonstration), the 89-test suite, and a **22-agent
> adversarial review workflow** (5 lenses → per-finding adversarial verification →
> synthesis). The review confirmed the result is **honest** (not overclaimed) and the
> architecture intact, and surfaced soundness gaps now **fixed and tested**: a raising /
> non-stream / unbounded generator could be admitted and poison the suite (→ split the
> runner's generator-vs-solve failure, probe at production scale, reject on error +
> length); op admission was coupled to benchmark wall-clock (→ ground on a
> correctness-only spec); an unbounded allocation could destabilise the host (→ psutil
> RSS watchdog); a frontier member could be evicted when the cap < frontier size (→
> exempt protected members from truncation); and cross-slot op-name / generator-name
> collisions could miscredit provenance (→ rejected at admission).

---

## Honest assessment (v4 history): how close is v4 to conceptual discovery?

Being blunt, because the brief asks for it.

**What v4 genuinely achieves.** The accepted, recommended solution is a policy the
engine **assembled from low-level ops, not selected from the library and not a v3
named policy**. Its inheritance rule — admit at *half* the evicted victim's count
(`max(1, vv // 2)`) — is provably **outside `V3_POLICY_GENOMES`**, and the test
suite asserts both that membership and that the source is byte-distinct from every
library strategy. That clears the brief's actual bar: *an accepted candidate whose
core logic was not present in any human-designed grammar or initial library*. The
win is **real and order-independent** — scored on the full hostile suite it ranks
**#1 of all ten genome policies** and is the apex of the v4-only Pareto frontier
regardless of synthesis order (the genome's ordering is a demo convenience, not the
cause). It is grounded (isolated execution + recall oracle), it **strictly
dominates** v3's best policy Space-Saving at equal capacity (recall 0.89 @ 36 KB vs
0.84–0.86 @ 45 KB, a +0.05 recall gap reproducible to **zero variance** across
deterministic trials, concentrated on hostile late-arrival/overflow streams), and
the generator-level Adversary **exposed the winner itself** on `param_l97_h400`.
The Adversary now invents structurally new attacks (`gen:phase/spike/mixture`), and
the frontier re-scores on hardening so the marginal-hypervolume reward stays honest.

**But this is composition within a hand-designed op-basis, not invention of a new
primitive concept.** Honest caveats:

- **I designed the ops.** `vhalf = max(1, vv // 2)` was authored by a human as a
  candidate inherit-op; `_VICTIM_OPS` / `_INHERIT_OPS` / `_PRE_OPS` are all
  hand-written. What the engine did autonomously was **discover, by grounded
  measurement, that the `(min, vhalf, none, 32)` composition Pareto-dominates the
  best human-named policy** — a combination no human wrote down as a named
  algorithm. The ceiling moved one notch (whole policies → compositions of finer
  ops) but it is the same *kind* of ceiling: this is **"a novel composition from a
  designed op-basis," not "a primitive concept not designed in advance."**
- **The genome is enumerated, not searched.** v4 walks a fixed 10-point list in
  batches; it does not *propose* new ops, do typed program/AST search, or grow
  structure. The op vocabulary is the wall.
- **The generator DSL is still a bounded schema.** `phase/spike/mixture` are
  structurally richer than `param:`, but the Adversary picks among three authored
  generator *types* — it does not write a genuinely new generator from scratch.
- **The metric saturates.** recall hits 1.0 for strong methods on clean skew,
  costing axis resolution; the hardest signal lives in a few adversarial workloads.

Net: v4 crosses the brief's line cleanly — an accepted result's core logic was
outside *every* human-designed grammar, it dominates the best prior policy, and the
Adversary pressures it — but it is **disciplined compositional discovery within a
designed op-space**, a real step beyond v3's named-policy grammar, not yet
autonomous invention of new primitives or paradigms.

## (v4 history) New bottlenecks once the op-basis is opened

1. **The op vocabulary is the new ceiling.** Discovery is bounded by which victim/
   inherit/pre ops were supplied; the hard problem is now "invent an op", not
   "compose ops". (v3's ceiling was "invent a policy"; v4 lowered it one level.)
2. **Enumeration doesn't scale.** A 4×5×3 genome is fine; an open op space is not
   walkable. Composition must become *guided* (typed search / LLM-proposed ops
   directed by frontier gaps + the ConceptLedger), not enumerated.
3. **Verification must scale with openness.** The oracle, the generator-DSL safety
   clamps, and the AST screen (which rejects *all* dunders — possible
   over-rejection) must catch subtler wrong-but-plausible programs as the space
   widens; the Adversary is the only thing standing between "plausible" and "real".
4. **Comparability is fixed but costly.** Re-scoring every frontier member on every
   new attack keeps the metric honest but multiplies grounding cost as the suite
   grows — the honest tax for an honest frontier.

## (v4 history) Highest-leverage direction for v5 — now realized in v5 above

**Let the Builder propose new *ops*, and the Adversary write new *generators* —
both as grounded, typed fragments, not fixed lists.** v4 proved the machinery
(grounding + marginal-hypervolume frontier + re-scoring + adversarial exposure) can
keep a *composition* honest; v5 should point that same machinery at an **open op
space**: an LLM (or typed program search) proposes a candidate inherit/victim/pre
op *as code*, the AST screen + oracle admit only sound ones, and frontier gaps +
ConceptLedger lessons *direct* which ops to try — so the engine can exceed the
supplied vocabulary instead of enumerating it. Symmetrically, let the Adversary
**synthesise a new grounded stream *generator*** (a `gen_*` function body), not pick
among three authored types, with re-scoring keeping the metric honest. That is the
move from "compose a designed op-basis" to "search an open op/generator space" — the
real path toward primitive invention — and v4's grounding + honest-frontier
machinery is exactly what would bound its cost and keep it from fooling itself.

## Success criteria for v4 — met

1. ✅ **Opened the synthesis basis below named policies**: an eviction-policy genome
   (`victim × inherit × pre` ops) synthesises policies absent from the v3 grammar,
   all grounded; `V3_POLICY_GENOMES` lets the tests *prove* the winner is outside it.
2. ✅ **Generative Adversary at the generator level**: composes new attack *types*
   via a bounded `gen:` DSL (phase/spike/mixture), and **frontier members are
   re-scored** whenever a new attack hardens the suite.
3. ✅ **Protective architecture intact + strengthened**: frontier marginal-
   hypervolume stays the primary fitness (0 for dominated/recombinant); behavioural
   novelty, recombination flagging, objective signals all active; bounded memory is
   now *frontier-aware*; the frontier is now *re-scored* for comparability.
4. ✅ **An accepted candidate whose core logic was in no human grammar/library**:
   `policy_min_half32` (recall 0.89 @ 36 KB), provably outside `V3_POLICY_GENOMES`,
   strictly dominating the best v3 policy at equal capacity.
5. ✅ **Adversary surfaces weaknesses in the new structures**: it exposed the winner
   itself on `param_l97_h400` and collapsed the v3 Space-Saving baseline on the
   v4-invented `gen_spike` attack.
6. ✅ **Improved test coverage**: 64 → **73 tests** (genome-outside-grammar, the
   invented op beating Space-Saving at equal budget, generator-DSL safety + phase
   shift, frontier re-scoring comparability, and a memory-pressure regression that
   proves a frontier member is never evicted).

---

> **Verification footnote.** v4 was checked by deterministic run (above), a short
> live LLM run (backend + generative Adversary + re-scoring work end-to-end; the
> *cheap* synthesizer over-explores and does not reach the genome path within a
> short budget — deterministic is the clean synthesis demonstration), the 73-test
> suite, and a 3-lens adversarial code review. The review confirmed the result is
> **honest and order-independent** and the protective architecture intact, and
> surfaced two real defects that are now fixed: bounded memory could evict a
> frontier member (→ frontier-aware `protected_ids`), and the LLM Adversary's
> *per-candidate* attacks could score candidates on different sub-suites (→
> `_ensure_full_coverage` forces full-suite coverage before admission).
