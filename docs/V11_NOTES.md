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
