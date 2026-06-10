# v11 notes

## Design constitution (operative for every metric added in this repo)

Unfakeable (trusted-code-computed) → loop + claims.json. Alignable (gaming it IS
the goal, e.g. behavioral profile distance) → loop-visible. Insulatable
(judgments over the generator: invention-vs-interpolation, narrative honesty) →
NEVER computed in-repo; the repo emits evidence bundles (`audit/`), the judgment
stays external, and reviewer findings come back as problems, never as scoring
heuristics. Leak rule: evaluation→generation carries PROBLEMS (failing traces,
missed keys, verbatim contract errors), never RUBRICS (eps, weights, worked
idioms, thresholds).

## Half A — the bar, made real (FROZEN at commit "v11 A-bar frozen")

**A1 — dual-implementation ARC validation.** An independent ARC was written from
the Megiddo & Modha Fig. 4 case structure (list-based queues, different
traversal; identical declared conventions: dict/list-LRU order, integer-division
deltas floored at 1, no-ghost drop in Case IV(a) with B1 empty, charging-bound
ghost trim). **Hit-for-hit agreement across 63 (trace, cap) combinations — all 7
regimes × 3 seeds × 3 capacities: EXACT.** No divergence finding. Property tests
pin: p moves toward recency under recency traffic vs frequency traffic; scan
resistance ≥ LRU; the loop blind spot behaves as documented (≤0.05 on a
period-40 loop at cap 20). LeCaR's deviation from the published randomized
algorithm (deterministic argmax, single c-capped history, exponential regret) is
documented at the implementation and pinned by test.

**A2 — broadened tier.** 2Q (Johnson & Shasha, full version) and W-TinyLFU
(Einziger et al., deterministic simplifications declared at the implementation —
exact dict sketch is *stronger* than a real CM sketch, so the reference errs
demanding) joined ARC and LeCaR at capacities {1,2,4}. All metadata flows
through the single `_baseline_cost_units` site: ghosts at
`GHOST_ENTRY_COST_FRAC = 0.25`, sketch at `SKETCH_SLOT_COST_FRAC = 1/16` × 8
declared slots/entry (doorkeeper folded in, no side deals). Sanity pinned before
joining: 2Q scan-resistant, W-TinyLFU ≥ LFU on zipf.

**A3 — multi-seed + declared flag semantics.** S=5 declared seeds per benchmark
workload, looped inside ONE runner subprocess per policy (`quality_seeds`
recorded). Declared rule (verbatim in `_classify_vs_ms`): comparator = highest
mean at ≤ memory; margins are pairwise per-seed; `beats` ⇔ mean margin >
eps × selection_factor(n_graded); `matches` ⇔ mean ≥ −eps; `fragile` ⇔ any
per-seed margin falls in a different flag region than the mean. Selection
penalty: `1 + log2(max(n,2))/4` (declared, recorded per run in claims.json).
History records versioned `benchmark: "v11-multiseed"`.

**A4 — robustness + holdout.** Charge sweep at FRAC ∈ {0, 0.25, 0.5, 1.0};
`robust_beats` = beats at every point; `sensitivity` block in claims.json.
Holdout family (loop period ~U[45,90], zipf exponent ~U[2.0,3.0], drawn seeds)
drawn at finalize seeded by the run tag; `holdout_margin` reported beside the
benchmark margin; gap > 0.05 = overfit annotation.

## THE FROZEN-BAR RE-GRADE (scripts/regrade_frozen_bar.py, 2026-06-10)

n_graded=10, selection_factor=1.830, eps_beats=0.0183. Adaptive front (declared
charge): (0.637@2000) arc1 · (0.771@2400) wtlfu1 · (0.832@4800) wtlfu2 ·
(0.897@8000) · (0.930@9600) wtlfu4.

| op | origin | wCR | adaptive tier | sweep (0/.25/.5/1) | fragile |
|---|---|---|---|---|---|
| ad_guard15 | det | 0.8911 | beats +0.059 | matches/beats/beats/beats | no |
| adaptive_mixguard | v9-LLM | 0.8840 | beats +0.052 | below/beats/beats/beats | no |
| adaptive_hotset | v9-LLM | 0.8726 | beats +0.041 | below/beats/beats/beats | no |
| adaptive_freq_guard | v9-LLM | 0.8727 | beats +0.041 | below/beats/beats/beats | no |
| hybrid_hot_guard | v10-LLM | 0.8723 | beats +0.040 | below/beats/beats/beats | no |
| ad_thrash | det | 0.8669 | beats +0.035 | below/beats/beats/beats | no |
| adaptive_freq_stick | v10-LLM | 0.8572 | beats +0.025 | below/beats/beats/beats | **yes** |
| rec_logf | det | 0.8413 | matches +0.009 | below/matches/matches/matches | **yes** |
| rec_plus_f | det | 0.8400 | matches +0.008 | below/… | **yes** |
| warm_freq_bias | v10-LLM | 0.8409 | matches +0.009 | below/… | **yes** |

**Headline: 7/10 beats_adaptive_human at the declared charge; 0/10
robust_beats.** The robust streak ENDS — the expected outcome — and the
mechanism is precise: at FRAC=0 the broadened references at resident-only cost
(W-TinyLFU 0.93, 2Q/ARC ≈0.90 at 6400B) dominate every current op. v10's
"survives ghost-free accounting" was true against ARC/LeCaR only; 2Q and
W-TinyLFU close that gap. Every op that still beats at the declared charge is
aux-adaptive; every static blend fell to matches and is FRAGILE
(boundary-straddling across seeds — the exact knife-edge ops the multi-seed
semantics were built to expose). This table is the bar Half B must clear.


## Half B -- making the bar reachable (landed AFTER the A-bar freeze commit)

**B1 -- driver-owned ghost observable.** The trusted driver keeps a bounded
ghost list (<= cap, FIFO of evicted keys, reset per trace) and passes a
per-access `ghost_hit` 0/1 flag into 4-argument `_on_access` dynamics; 3-argument
v9 candidates keep working (arity-dispatched, pinned). Opt-in via the literal
`USE_GHOST` knob: single literal 0/1 enforced by the shape screen + a rebind
census; cost charged through the SAME rule as the references (engine
GHOST_FRAC_DEFAULT pinned == domain GHOST_ENTRY_COST_FRAC). Soundness stated in
the driver: the flag is information any ghost-list algorithm legitimately has;
the candidate sees the flag, never the list, and steers only eviction.

**B2 -- proposable aux dynamics (`adyn`).** A list-expression over
(aux, hit, ghost_hit, t), following the UPDATE-op fragment precedent;
`fragment_is_safe` extended with those names for this kind only; malformed
returns degrade to AUX0 (driver-enforced); rides the domain-curated
DEFAULT_AUX_SCORE (template text, audit-hashed, never prompt-visible).

**B3 -- rubric scrub (gates the LLM run). Removed from proposer-facing texts:**

From CONTRACT:
  - the worked EWMA dynamics body `[aux[0] * 0.98 + 0.02 * hit]` ("e.g. an EWMA
    of the hit flag") -- the canonical taught idiom;
  - the worked threshold-switch ascore example
    `(-la) if aux[0] < 0.08 else (la + 30 * fr.bit_length())` and its constants;
  - the regime-detection coaching sentence ("DETECT a regime (e.g. a loop the
    cache cannot hold: the EWMA hit-rate collapses) and switch...");
  - the quality rubric ("QUALITY is the COMPETITIVE RATIO ... weighted across a
    declared multi-regime benchmark (skew, reuse, drift, loop, scan) -- so
    robustness across regimes is what scores, not specialisation") and the
    benchmark-degeneracy rationale for the CMUL bound.

From OP_PROPOSER_SYS:
  - the seed-op idiom map ("LRU is `la`, LFU is `fr`, FIFO is `ins`, MRU is
    `-la`") and the worked example "`la + 30 * fr.bit_length()` keeps a hot key
    warm";
  - the ENTIRE worked ascore example with thresholds, the EWMA decay/start
    constants (0.98 / 0.5) and the thrash-collapse coaching;
  - "QUALITY is the competitive ratio ... ROBUSTNESS across regimes is what
    scores";
  - the rationale-shape hint `"<why it helps across regimes>"` (replaced by
    `"<your reasoning>"`);
  - the update-op default hint `(default [t, fr + 1, ins])`.

From the self-play steering line (roles.py): "author an op so a policy using it
BEATS the best human on the Adversary's recent generators" -> "The Adversary's
recent attack generators are {...}; current policies degrade on them" (problem
framing, no scoring language).

KEPT (and strengthened): the mechanical contract rules, the full causal
diagnosis channel (failing trace, dist, missed keys, verbatim contract errors),
and the NAMES of the observables (`aux`, `hit`, `ghost_hit`, `t`) without usage
examples.


## The citable v11 run (20260610_115502_cache_eviction_deterministic)

In-run: 14 ops graded (selection factor 1.95); adaptive tier 2 beats / 6
matches / 6 below; robust_beats = []; the only beats are the two aux-adaptive
templates, with engine-derived mechanism lines printed beside every flag.

**HOLDOUT FINDING (the run's headline).** Both in-run winners grade beats on
the declared benchmark yet LOSE to W-TinyLFU on the freshly-drawn holdout
family:

    ad_thrash   bench +0.0349  holdout -0.1706  (comparator wtlfu2)  OVERFIT
    ad_guard15  bench +0.0591  holdout -0.1367  (comparator wtlfu2)  OVERFIT

Gap ~0.2 against the 0.05 threshold. The threshold-switch idiom's wins are
calibrated to the declared suite's exact parameters and weights; on drawn
cousins of the same regimes (loop period ~U[45,90], jittered zipf exponent,
fresh drift seeds, equal weights) W-TinyLFU dominates. Combined with the sweep
(0/14 robust), v11's verdict layer now says, with evidence at every level:
the current op family does not robustly beat the best adaptive humans -- the
real frontier work is GENERALIZATION, not benchmark margin.

Behavioral provenance (alignable, in-loop): the two adaptive winners are also
the two most behaviorally distant ops from any vocabulary member (ad_thrash
0.256 from lru4; ad_guard15 0.225 from twoq4) -- profile distance and the
mechanism lines agree with what the flags say, without the repo ever scoring
"inventedness".


## Multi-seed stability battery -- first attempt: a FINDING, then the clean pass

The first 5-run battery (12:05-12:21) was launched while the full 169-test suite
was still finishing on the same machine. Result, read directly from the five
claims.json artifacts (the wrapper process zombied without flushing output --
scripts/stability_check.py now flushes):

  run 1 (120852, UNDER SUITE LOAD): n_graded=8 -- six ops never proposed
  runs 2-5 (121201/121516/121828/122141, idle): n_graded=14, and their
  op_utility maps (both tiers) + fragile annotations are BIT-IDENTICAL to each
  other AND to the citable run 115502.

Diagnosis (recorded, not hidden): the VERDICT layer is deterministic -- five
distinct runs with the same admitted vocabulary produce identical maps. What is
load-sensitive is the PROPOSAL TRAJECTORY: directives consume timing-noise
signals (the v1 noise-aware stall detector, by design wall-clock-aware), so
heavy machine load changes which rounds propose, hence WHICH ops exist to be
graded. The claim layer makes this detectable rather than silent: `n_graded`
is recorded per run, and the selection-penalised beats threshold scales with
it. Acceptance therefore requires an UNLOADED machine -- rerun below.


## Multi-seed stability acceptance -- CLEAN PASS (idle machine, 2026-06-10)

    run 1/5: 20260610_164517_cache_eviction_deterministic  beats(adaptive)=2
    run 2/5: 20260610_164808_cache_eviction_deterministic  beats(adaptive)=2
    run 3/5: 20260610_165100_cache_eviction_deterministic  beats(adaptive)=2
    run 4/5: 20260610_165352_cache_eviction_deterministic  beats(adaptive)=2
    run 5/5: 20260610_165644_cache_eviction_deterministic  beats(adaptive)=2
    STABLE: identical op_utility maps across 5 consecutive runs

The maps also equal the citable run 20260610_115502 -- ten runs total now share
the identical 14-op verdict (both tiers, fragile annotations, robust_beats=[]).
Acceptance criterion: met. The documented caveat stands: vocabulary FORMATION
is load-sensitive (n_graded in claims detects it); the VERDICT given a
vocabulary is bit-stable.
