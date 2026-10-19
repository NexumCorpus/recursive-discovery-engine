# Proposer-facing texts (post-scrub), verbatim

## CONTRACT

```
Implement a CACHE EVICTION policy by defining EXACTLY these three module-top-level
symbols. Do NOT define solve(); a TRUSTED harness runs the cache simulation and
computes the hit rate. You provide ONLY the eviction LOGIC.

    # 1) The eviction SCORE. On a miss with a FULL cache, the key with the MINIMUM
    #    _evict_score is evicted. Each cached key has integer state: la = last
    #    access time, fr = frequency (access count), ins = insert time; t = the
    #    current clock (access index). LRU = `la`, LFU = `fr`, FIFO = `ins`,
    #    MRU = `-la`. Return a number; lower means evicted first.
    def _evict_score(la, fr, ins, t):
        return la

    # 2) The per-HIT state UPDATE. Return the key's NEW [la, fr, ins] after a hit.
    #    Default (track recency + frequency): [t, fr + 1, ins].
    def _on_hit(la, fr, ins, t):
        return [t, fr + 1, ins]

    # 3) The capacity multiplier. The cache holds CMUL * k keys (a SMALL int, 1..4).
    CMUL = 2

OPTIONAL ADAPTIVE EXTENSION -- per-trace state your own dynamics maintain. You may
ALSO define, at module top level (signatures only; the bodies are yours):

    AUX0 = [...]                          # initial aux state: <= 8 finite numbers
    def _on_access(aux, hit, ghost_hit, t):   # called on EVERY access
        ...                               # return the new aux (list, <= 8 numbers)
    def _evict_score_aux(la, fr, ins, t, aux):  # scoring that may READ aux
        ...

Observables available to your dynamics (names only; how you use them is yours):
  * aux        -- your own state vector, carried by the trusted harness through the
                  trace (reset per trace), passed read-only;
  * hit        -- the per-access hit FLAG, 0 or 1 (what any online algorithm
                  observes; you never see a count);
  * ghost_hit  -- 0/1: this missed key was RECENTLY EVICTED (a bounded,
                  harness-owned ghost list; you see the flag, never the list).
                  Using it requires declaring `USE_GHOST = 1` (a literal int),
                  which is charged on the declared cost axis;
  * t          -- the access clock.
A 3-argument _on_access(aux, hit, t) is also accepted.

RULES. _evict_score and _on_hit MUST be PURE functions of (la, fr, ins, t) only:
no globals, no I/O, no randomness, no reading any hit count. Return a number from
_evict_score and a 3-element list of numbers from _on_hit. _on_access must return a
list of <= 8 finite numbers (a malformed return is reset to AUX0). CMUL must be a
literal int in [1, 4], assigned exactly once. USE_GHOST, if present, must be a
single literal 0 or 1. Module top level may contain ONLY: imports, function defs,
docstrings, CMUL, USE_GHOST, and AUX0. You CANNOT report a hit rate and you CANNOT
beat Belady's offline optimum, because you never see the future -- the harness
feeds you one key at a time and counts hits itself. Define the three required
symbols at module top level. Do not print.

```


## OP_PROPOSER_SYS

```
You are the BUILDER extending the primitive vocabulary of a CACHE EVICTION engine. A policy keeps a bounded cache; on a miss with a FULL cache it evicts the cached key with the MINIMUM score. Each cached key has integer state: la (last access time), fr (frequency / access count), ins (insert time); t is the current clock. Propose ONE new primitive OPERATION that addresses the concrete failing trace described to you. Allowed kinds: 'score' = a Python EXPRESSION over la, fr, ins, t giving the eviction priority (lower = evicted first); 'update' = a Python LIST expression [<new_la>, <new_fr>, <new_ins>] for a key's state on a HIT; 'ascore' = a score EXPRESSION over la, fr, ins, t AND aux (aux = a small state vector maintained for you across the trace); 'adyn' = a LIST expression over (aux, hit, ghost_hit, t) defining how aux itself updates on every access (hit and ghost_hit are 0/1 flags; ghost_hit means this missed key was recently evicted). Use ONLY the listed names, numbers, arithmetic, comparisons, conditional expressions, and int methods like .bit_length() -- no imports, no I/O. Respond with ONLY a JSON object: {"kind":"score"|"update"|"ascore"|"adyn","name":"<short_identifier>","fragment":"<expr>","rationale":"<your reasoning>"}.
```


## template_library

```
[('score', 'rec_logf', 'la + 30 * fr.bit_length()', {'skew', 'drift', 'mixed'}, 'recency with a LOGARITHMIC frequency boost -- keeps a hot key warm without letting raw frequency dominate (adapts as popularity shifts)'), ('score', 'score_half', 'la + (fr * (t - ins)) // 4', {'skew', 'mixed'}, 'recency offset by frequency x age -- favours items that have earned reuse'), ('score', 'lfu_aged', 'fr - (t - la)', {'drift', 'loop', 'shift'}, 'frequency AGED by idle time -- forgets stale-popular keys when the working set drifts (an ARC/LRFU-flavoured score)'), ('score', 'lfu_aged2', '2 * fr - (t - la)', {'drift', 'shift'}, 'frequency (weighted) aged by idle time'), ('score', 'rec_plus_f', 'la + 8 * fr', {'skew'}, 'recency with a linear frequency bonus'), ('score', 'freq_then_rec', 'fr * 100000 + la', {'skew', 'frequency'}, 'frequency primary, recency tie-break (LFU with an LRU tie-break)'), ('score', 'idle_penalty', 'fr - 2 * (t - la)', {'drift', 'loop', 'scan'}, 'strong idle penalty -- evict keys not touched recently regardless of count'), ('score', 'aged_logf', 'la + 20 * fr.bit_length() - (t - la)', {'drift', 'shift'}, 'log-frequency boost minus idle penalty -- warm-but-recent keys survive'), ('score', 'freq_over_age', 'fr * 1000 - (t - la) * 10', {'skew', 'recency'}, 'frequency credit decayed linearly by idle time'), ('score', 'rec_sq_freq', 'la + fr * fr', {'skew'}, 'recency with a SUPER-linear frequency bonus (very hot keys pinned)'), ('score', 'scan_resist', 'fr * 100000 - (t - ins)', {'shift', 'loop', 'scan'}, "scan/loop-resistant (2Q/LIRS-flavoured): protect frequently-reused keys; among equal-frequency one-timers evict the longest-resident -- a generator-targeted op for the Adversary's scan/loop attacks"), ('ascore', 'ad_thrash', '(-la) if aux[0] < 0.08 else (la + 30 * fr.bit_length())', {'scan', 'drift', 'loop', 'mixed', 'shift'}, 'ADAPTIVE thrash guard: when the EWMA hit-rate collapses (a loop/scan the cache cannot hold -- LRU-family thrashes to 0), switch to MRU-like eviction (keeps a stable resident subset); otherwise behave as rec_logf. Detects and switches regime DURING the trace -- what no static score can do.'), ('ascore', 'ad_guard15', '(-la) if aux[0] < 0.15 else (la + 30 * fr.bit_length())', {'loop', 'shift'}, 'adaptive thrash guard with an EAGER threshold (trips earlier, recovers more of a long loop, slightly jumpier on borderline regimes)'), ('update', 'decay_hit', '[t, max(1, fr // 2 + 1), ins]', {'drift', 'shift'}, 'on a hit, DECAY the accumulated frequency (forget old bursts) then +1')]
```
