# v10 notes

## Phase 0 — verdict-instability diagnosis (2026-06-10)

**Symptom (reproduced from artifacts).** Seven back-to-back deterministic
`cache_eviction` runs (`runs/20260610_06*`) alternate between two `op_utility`
patterns: {6 beats: `lfu_aged`=match, `rec_sq_freq`=match} and {7 beats:
`lfu_aged`=beats, `rec_sq_freq`=below}. The two ops always flip TOGETHER —
pointing at a shared cause, not per-op noise.

**Hypothesis tested and rejected.** The review hypothesized subprocess-timeout
truncation of the benchmark suite interacting with `_weighted_quality`'s silent
renormalization. Instrumented `_run_subprocess` on the exact finalize-path
measurements: every relevant policy (both `lfu_aged` caps, both `rec_sq_freq`
caps, `lru4`) measures **7/7 workloads in ~0.3 s** against a 20 s timeout —
60× headroom; humans measured 7/7 in every probe. Suite truncation cannot
plausibly explain flips on an idle-to-moderately-loaded machine. (The silent-
renormalization gap is REAL but LATENT — nothing observed triggers it. P1
closes it anyway: missing measurements are now retried once, then scored 0.0
with an explicit `incomplete` flag, never renormalized away. Symmetrically for
human baselines.)

**Actual mechanism (evidence-backed): representative-variant instability.**
1. Offline replication of the finalize classification is bit-stable across
   trials and shows the knife edges: `lfu_aged` c2 weighted-CR = 0.6967 vs
   beats-threshold 0.6966 (**margin +0.0001**, via the 4-dp rounding in
   `_weighted_quality`); its c4 variant grades *match*. `rec_sq_freq` c4 sits
   0.0005 from its boundary. So the flag an op gets depends on WHICH capacity
   variant represents it.
2. `_compose_uncovered_ops` (finalize catch-all) had an early-`discard`: once
   an op's c2 policy was recorded, the c4 policy was skipped — finalize-only
   ops were judged c2-only. In the observed runs NO synthesize round fired
   (directives: continue x2, explore x4 in both patterns), so ALL 14 ops took
   this path.
3. Each variant is admitted to the record only if `ground(code,
   _op_probe_spec())` returns correct — and `ground()` (correctly) hard-fails
   when its 20 s wall-clock truncates the 5-repeat benchmark phase
   (`benchmark incomplete`). Under the review's parallel load, a c2 variant's
   grounding could time out → variant silently skipped → op judged at c4 →
   flag flips across the 1e-4 boundary. This explains correlated flips
   (machine load affects neighbouring subprocesses) on an otherwise fully
   seeded, deterministic pipeline.

**Fixes (fix what was found, not just what was hypothesized):**
- `_compose_uncovered_ops`: record **all** capacity variants (early-discard
  removed); a failed variant grounding is **retried once**, and a variant
  dropped after retry is recorded in the op's `incomplete` flag rather than
  silently changing which capacity represents the op.
- P1 completeness gate (suite level): every utility/human measurement must
  cover the full declared suite; one retry; then missing workloads score 0.0
  with an explicit `incomplete: [names]` flag propagated to the report and
  `utility_history.jsonl`. `_weighted_quality` therefore always runs with the
  full denominator — silent renormalization is structurally unreachable.
- Acceptance: 5 consecutive deterministic runs must produce identical
  `op_utility` maps (recorded below when run).

## P4 — the INCOHERENT-RESULT assertion fired on its first test run

Worth recording: the coherence assertion (any `beats_*_human` op must be matched
by the frontier) FIRED the first time the v10 e2e test ran — beating-op CR 0.8893
vs frontier max 0.8406. It caught a real seam: the finalize CATCH-ALL composes and
grades policies the frontier never saw (the in-round composer caps at 6/round; a
6-iteration run cannot cover 28 variants). Per the ground rules the assertion was
not weakened; the seam was closed by routing finalize-graded, frontier-EXTENDING
policies through the SAME `_admit_frontier` machinery (full coverage + marginal-
hypervolume reward; a dominated point earns zero gain and changes nothing). The
assertion now holds structurally and remains armed.

## P3 — adaptive-tier outcome (against expectation)

Expected: ARC in the baseline ends the beats streak. Measured: it does not.
Under the declared ghost charge (GHOST_ENTRY_COST_FRAC = 0.25) every ARC/LeCaR
point is Pareto-dominated by a static classic at the reachable costs, so the
adaptive and static tiers coincide. Under ghost-FREE accounting (maximally
ARC-favorable), the aux-adaptive ops still beat the best adaptive human
(ad_guard15 +0.055, adaptive_mixguard +0.047, ad_thrash +0.031) while the static
champion rec_logf drops to match (+0.005). The accounting-robust wins are the
adaptive inventions only; their margin is concentrated in the 10%-weight pure-loop
regime, ARC's structural blind spot (scan-resistant, not loop-resistant) -- stated
as a declared caveat, not hidden.

## P1 acceptance — five-run stability (PASSED)

BEFORE (v9 code, runs/20260610_06*): seven back-to-back identical-config runs
alternated 6 <-> 7 beats_human (3x pattern A, 4x pattern B; `lfu_aged` and
`rec_sq_freq` flipping together).

AFTER (v10, `scripts/stability_check.py 5`, 2026-06-10):

    run 1/5: 20260610_072538_cache_eviction_deterministic  beats(adaptive)=4
    run 2/5: 20260610_072740_cache_eviction_deterministic  beats(adaptive)=4
    run 3/5: 20260610_072943_cache_eviction_deterministic  beats(adaptive)=4
    run 4/5: 20260610_073144_cache_eviction_deterministic  beats(adaptive)=4
    run 5/5: 20260610_073346_cache_eviction_deterministic  beats(adaptive)=4
    STABLE: identical op_utility maps across 5 consecutive runs

Harsher than required: the battery ran CONCURRENTLY with the full test suite
(dozens of competing grounding subprocesses) -- the very machine-load condition
that produced the original flips. Identical maps on both tiers, zero
`incomplete` flags, across all five runs.

## Capable-model LLM run under the v10 gates (run 20260610_080513, operator-approved)

**Open invention clears the ADAPTIVE bar: 3/3 proposed-llm ops grade beats_human
on BOTH tiers** (static classics AND ARC/LeCaR), zero `incomplete` flags,
`incoherent: null`. 21 calls / ~87k tokens (gpt-5.1-codex / gemini-3.1-pro /
claude-opus-4.8, 4 iters). Fragments (persisted per P5):

    warm_freq_bias      la + (fr << 4)                                      [BLENDED, static]
    hybrid_hot_guard    (-la) if aux[0] < 0.08 else (la + fr.bit_length() * 16)
    adaptive_freq_stick (-la) if aux[0] < 0.07 else (la + (fr << 6) - ((t - ins) >> 2))

**P4 re-coupling visibly worked vs the v9 LLM run**: the recommended artifact is
now itself an LLM invention ON the frontier — `i2.v1_policy_hybrid_hot_guard_c4`,
weighted CR 0.87 @ 6KB, recommended_grade beats_human +0.0506 on both tiers
(v9's LLM run recommended seed LRU while its wins were graded off-stage). A real
SYNTHESIZE fired (it3); the Adversary's proposed `gen_hot_flush` exposed
`adaptive_freq_stick` and hardened the suite (declared, descriptive).

**Declared caveats**: (1) still GUIDED invention — both aux ops are the taught
threshold-switch idiom (novel blend terms within it); (2) under the declared
ghost charge the tiers coincide, so "beats adaptive" is materially the static
comparison — see the ghost-free sensitivity below; (3) free-form 'other'-class
candidates still failed the driver contract twice (the verbatim-error feedback
made the Synthesizer steer away rather than loop).

### Ghost-free sensitivity for the LLM inventions (closes caveat 2)

Re-graded against the maximally ARC-favorable accounting (ARC/LeCaR at
resident-only cost; front = (0.636, 1600) / (0.722, 3200) / (0.837, 6400)):

    hybrid_hot_guard      c4  wCR 0.873  -> beats (+0.0357)   ROBUST
    adaptive_freq_stick   c4  wCR 0.859  -> beats (+0.0220)   ROBUST
    warm_freq_bias        c4  wCR 0.841  -> match (+0.0039)   NOT robust (static)

Same pattern as the deterministic ops: the accounting-robust wins are exclusively
the AUX-ADAPTIVE mechanism (2/3 robust; the static blend's beat exists only under
the declared ghost charge, exactly like rec_logf). The shipped artifact
(`hybrid_hot_guard_c4`) is itself robustly-beating under BOTH accountings — the
recommended policy and the strongest claim now coincide. Final form of the v10
LLM result: **3/3 beats the adaptive tier under the declared accounting; the two
aux-adaptive inventions (including the recommended artifact) survive ghost-free
accounting; the static invention does not.**
