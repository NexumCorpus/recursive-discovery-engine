"""Domain (v7): CACHE EVICTION policy synthesis under a memory budget.

Why this domain: heavy-hitters (v2-v6) became *saturated* -- human ops (min/one)
were already near-optimal on recall, so beats_human on the PRIMARY metric was
artificially hard. Cache eviction is the opposite: human policies (LRU, LFU, FIFO,
MRU) are good but provably FAR from Belady's offline optimum (OPT) on many traces
(LRU is 0% on a loop while OPT is high), so there is real headroom on the primary
metric -- HIT RATE -- and no single human policy dominates (LRU wins recency, LFU
wins skew, MRU wins loops). That makes beats_human achievable and measurable on the
main axis, which is exactly what v7 needs.

Contract: a policy processes an access trace with a bounded cache and decides, on a
miss with a full cache, which key to EVICT. It is composed from low-level OPS:
  * a SCORE op -- rank a cached key for eviction (evict argmin score), an expression
    over the key's state (la = last access time, fr = frequency, ins = insert time)
    and the clock t. LRU = `la`, LFU = `fr`, FIFO = `ins`, MRU = `-la`.
  * an UPDATE op -- how a key's state changes on a HIT.
  * a capacity multiplier cmul -- the policy's memory budget (cap = cmul * k), the
    cost axis of the (hit-rate, memory) frontier (exactly analogous to the summary
    size in heavy-hitters).

`solve(trace, k)` returns `(hit_rate, capacity)`. The verdict recomputes Belady OPT
at the policy's REPORTED capacity and rejects any hit rate above it (you cannot beat
the offline optimum -- the anti-cheat), with quality = the raw hit rate.
"""

from __future__ import annotations

import ast as _ast
import inspect

from ..domain import Domain, Strategy
from ..grounding import GroundingSpec
from ..types import AlgoClass
from ..vocab import (GenEntry, GenProposal, OpEntry, OpProposal, Provenance,
                     Vocabulary)

# Op-slot kind names for this domain (the generic Vocabulary supports any kinds).
# v9 adds ASCORE: an ADAPTIVE score expression over (la, fr, ins, t, aux) where
# aux[0] is a trusted EWMA of the per-access hit flag -- enabling regime detection
# (loop/scan thrash vs skew/reuse) DURING a trace.
SCORE, UPDATE, ASCORE = "score", "update", "ascore"


CONTRACT = """\
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

OPTIONAL ADAPTIVE EXTENSION (v9) -- regime detection during the trace. You may ALSO
define, at module top level:

    AUX0 = [0.5]                          # initial aux state: <= 8 finite numbers
    def _on_access(aux, hit, t):          # called on EVERY access; hit is 0 or 1
        return [aux[0] * 0.98 + 0.02 * hit]   # e.g. an EWMA of the hit flag
    def _evict_score_aux(la, fr, ins, t, aux):    # scoring that may READ aux
        return (-la) if aux[0] < 0.08 else (la + 30 * fr.bit_length())

The trusted harness carries `aux` through the trace (reset per trace) and uses
_evict_score_aux instead of _evict_score when present. This lets a policy DETECT a
regime (e.g. a loop the cache cannot hold: the EWMA hit-rate collapses) and switch
eviction behaviour mid-trace -- something no static score can do. `hit` is the
per-access hit FLAG (what any online algorithm observes); you still never see a hit
COUNT and cannot report one.

RULES. _evict_score and _on_hit MUST be PURE functions of (la, fr, ins, t) only:
no globals, no I/O, no randomness, no reading any hit count. Return a number from
_evict_score and a 3-element list of numbers from _on_hit. _on_access must return a
list of <= 8 finite numbers (a malformed return is reset to AUX0). CMUL must be a
literal int in [1, 4], assigned exactly once (caps above 4*k would swallow whole
benchmark universes and make the competitive ratio vacuously 1.0). You CANNOT report a hit rate and you
CANNOT beat Belady's offline optimum, because you never see the future -- the
harness feeds you one key at a time and counts hits itself. QUALITY is the
COMPETITIVE RATIO hit_rate / Belady-OPT on each trace, weighted across a declared
multi-regime benchmark (skew, reuse, drift, loop, scan) -- so robustness across
regimes is what scores, not specialisation. Define the three required symbols at
module top level. Do not print.
"""

PROBLEM_DESC = (
    "Cache eviction: process an access trace with a bounded cache and choose which "
    "key to evict on a miss, to MAXIMISE the hit rate. Human policies (LRU/LFU/FIFO/"
    "MRU) are good but far from Belady's optimum on many traces -- the goal is to "
    "synthesise an eviction policy that advances the (hit-rate x memory) frontier.")


# --------------------------------------------------------------------------- #
# Oracle: Belady's offline optimum (evict the key used furthest in the future).
# --------------------------------------------------------------------------- #
def _belady(trace, cap):
    """Exact offline-optimal hit rate at capacity `cap` -- the achievable ceiling.
    O(n * cap) with monotone next-use pointers (cap is small)."""
    n = len(trace)
    if n == 0:
        return 1.0
    if cap < 1:
        return 0.0
    pos = {}
    for i, key in enumerate(trace):
        pos.setdefault(key, []).append(i)
    ptr = {key: 0 for key in pos}

    def next_use(key, after):
        lst = pos[key]
        p = ptr[key]
        while p < len(lst) and lst[p] <= after:
            p += 1
        ptr[key] = p
        return lst[p] if p < len(lst) else n      # n == never used again

    cache = set()
    hits = 0
    for i, key in enumerate(trace):
        if key in cache:
            hits += 1
        else:
            if len(cache) >= cap:
                victim = max(cache, key=lambda c: next_use(c, i))
                cache.discard(victim)
            cache.add(key)
    return hits / n


def oracle(trace, k):
    """Reference ceiling at the BASE capacity (2*k). The custom verdict recomputes
    OPT at the policy's own reported capacity; this is used for reporting."""
    return _belady(trace, max(2 * k, 4))


# --------------------------------------------------------------------------- #
# Verdict: quality = raw hit rate; reject any rate above Belady OPT at the
# policy's REPORTED capacity (cannot beat the offline optimum -- the anti-cheat).
# --------------------------------------------------------------------------- #
def verdict(got, expected, items, k):
    if not (isinstance(got, (tuple, list)) and len(got) == 2):
        return (False, 0.0)
    hr, cap = got
    if not isinstance(hr, (int, float)) or isinstance(hr, bool):
        return (False, 0.0)
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        return (False, 0.0)
    # The contract says C is a SMALL multiple of k. Reject an absurdly large reported
    # cap, which would make OPT(cap) -> 1.0 and the anti-cheat ceiling vacuous (a
    # policy reporting a capacity far above what it used to dodge the OPT check).
    if cap > 4 * max(int(k), 1):
        return (False, 0.0)
    if hr < -1e-9 or hr > 1.0 + 1e-9:
        return (False, 0.0)
    opt = _belady(items, cap)
    if hr > opt + 1e-6:
        return (False, 0.0)             # beat the offline optimum -> cheating / bug
    # v9: quality is the COMPETITIVE RATIO hr / OPT(cap) in [0, 1] -- "how close to
    # the offline optimum did this policy get on THIS trace", so cross-workload
    # means are not dominated by easy regimes (a 0.4 hit rate where OPT is 0.48 is
    # EXCELLENT; a 0.7 where OPT is 0.95 is mediocre). The anti-cheat above still
    # rejects hr > OPT, so the ratio is well-defined. On a trace where OPT is 0
    # (no repeats), every legal policy scores 0 hits -> ratio defined as 1.0 (you
    # did as well as anything could).
    if opt <= 1e-9:
        return (True, 1.0)
    return (True, round(min(float(max(0.0, hr)) / opt, 1.0), 6))


# --------------------------------------------------------------------------- #
# Trace generators. `dist` selects an access pattern; the adversarial ones
# (loop/scan/antagonistic) defeat specific human policies.
# --------------------------------------------------------------------------- #
_GENPROP_REGISTRY: dict = {}


def gen_workload(n, dist, seed):
    """Deterministic access-trace generation. Keys are ints; the universe is sized
    so the cache (a few * k) cannot hold everything -> real eviction pressure."""
    import random
    rng = random.Random(seed)
    if dist.startswith("prop:"):
        nm = dist[len("prop:"):].split("|")[0]
        fn = _GENPROP_REGISTRY.get(nm)
        if fn is not None:
            return fn(n, rng)
        return [rng.randint(0, 60) for _ in range(n)]
    if dist == "zipf":                          # skewed popularity (LFU/LRU decent)
        U = 60
        return [int(U * (rng.random() ** 2.5)) for _ in range(n)]
    if dist == "recency":                       # bursty re-reference (LRU shines)
        out = []
        while len(out) < n:
            k = rng.randint(0, 60)
            out.extend([k] * rng.randint(1, 6))
        return out[:n]
    if dist == "drift":                         # popularity set SHIFTS over time
        out = []
        seg = max(n // 4, 1)
        for i in range(n):
            base = (i // seg) * 50
            out.append(base + int(20 * (rng.random() ** 2)) if rng.random() < 0.8
                       else rng.randint(0, 200))
        return out
    if dist == "mixed":                         # interleave recency + skew regimes
        out = []
        for i in range(n):
            if (i // 200) % 2 == 0:
                out.append(int(60 * (rng.random() ** 2.5)))
            else:
                k = rng.randint(0, 40)
                out.append(k)
        return out
    # --- adversarial patterns (attack specific policies) ------------------- #
    if dist == "loop":                          # cyclic scan > cap -> LRU collapses
        U = 40
        return [i % U for i in range(n)]
    if dist == "scan":                          # one-shot sweep + a small hot set
        hot = list(range(5))
        return [rng.choice(hot) if rng.random() < 0.5 else 1000 + i for i in range(n)]
    if dist == "antagonistic":                  # hot set just over capacity
        U = 35
        return [rng.choice(range(U)) if rng.random() < 0.9 else 2000 + i
                for i in range(n)]
    if dist.startswith("param:"):
        # Generative adversary: a parameterised loop+noise trace. Knobs:
        #   period (loop length -> defeats LRU when > cap), noise (fraction of
        #   one-shot misses), skew (popularity concentration in [0,1]).
        import math
        allowed = {"period", "noise", "skew"}
        params = {}
        for kv in dist[len("param:"):].split(";"):
            if "=" not in kv:
                continue
            key, _, val = kv.partition("=")
            key = key.strip()
            if key not in allowed:
                continue
            try:
                f = float(val)
            except ValueError:
                continue
            if math.isfinite(f):
                params[key] = f
        period = max(2, min(int(params.get("period", 30)), 5000))
        noise = min(max(params.get("noise", 0.1), 0.0), 0.9)
        skew = min(max(params.get("skew", 0.0), 0.0), 1.0)
        out = []
        for i in range(n):
            if rng.random() < noise:
                out.append(10000 + i)                       # one-shot rare
            elif rng.random() < skew:
                out.append(int(period * (rng.random() ** 3)))   # skewed within loop
            else:
                out.append(i % period)                      # the loop
        return out
    if dist.startswith("gen:"):
        # v4-style generator DSL: NEW structural trace TYPES (phase/burst).
        import math
        parts = dist[len("gen:"):].split("|")
        gtype = parts[0].strip()
        gp = {}
        for kv in parts[1:]:
            if "=" not in kv:
                continue
            key, _, val = kv.partition("=")
            try:
                f = float(val)
            except ValueError:
                continue
            if math.isfinite(f):
                gp[key.strip()] = f

        def _g(name, lo, hi, d):
            return min(max(gp.get(name, d), lo), hi)

        if gtype == "phase":                    # popularity set A then set B
            ha = int(_g("ha", 1, 500, 20)); hb = int(_g("hb", 1, 500, 20))
            split = _g("split", 0.1, 0.9, 0.5)
            A = list(range(ha)); B = list(range(500, 500 + hb))
            cut = int(n * split)
            return [rng.choice(A if i < cut else B) if rng.random() < 0.8
                    else rng.randint(9000, 9000 + n) for i in range(n)]
        if gtype == "burst":                    # quiet rares + sudden re-reads
            h = list(range(int(_g("h", 1, 200, 8))))
            burst = _g("burst", 0.0, 0.9, 0.3)
            return [rng.choice(h) if rng.random() < burst else rng.randint(9000, 9000 + n)
                    for _ in range(n)]
        return [rng.randint(0, max(n // 4, 1)) for _ in range(n)]   # fallback
    raise ValueError("unknown dist: " + dist)


ORACLE_SRC = inspect.getsource(_belady) + "\n\n" + inspect.getsource(oracle)
WORKLOAD_GEN_SRC = "_GENPROP_REGISTRY = {}\n\n" + inspect.getsource(gen_workload)
VERDICT_SRC = inspect.getsource(verdict)


# --------------------------------------------------------------------------- #
# Trusted criteria suite -- correctness sanity (capacity-robust known bounds).
# --------------------------------------------------------------------------- #
CRITERIA_CASES = [
    ("all_distinct", list(range(60)), 5),       # no repeats -> hit_rate must be 0
    ("all_same",     [7] * 40, 5),              # one key -> hit_rate ~ (n-1)/n
    ("tiny_loop",    [0, 1, 2, 3] * 10, 2),     # loop period 4, small cap
    ("repeat_hot",   [1, 1, 2, 2, 3, 3, 1, 2], 3),
]


# --------------------------------------------------------------------------- #
# Evaluation workloads (hit-rate + memory) and the memory probe.
# --------------------------------------------------------------------------- #
WORKLOADS = [
    {"name": "zipf",    "n": 6000, "k": 10, "dist": "zipf"},
    {"name": "recency", "n": 6000, "k": 10, "dist": "recency"},
    {"name": "drift",   "n": 6000, "k": 10, "dist": "drift"},
    {"name": "mixed",   "n": 6000, "k": 10, "dist": "mixed"},
]
MEM_WORKLOAD = {"name": "mem", "n": 6000, "k": 10, "dist": "zipf"}
ADVERSARY_WORKLOADS = [
    {"name": "adv_loop", "n": 6000, "k": 10, "dist": "loop"},
    {"name": "adv_scan", "n": 6000, "k": 10, "dist": "scan"},
    {"name": "adv_antag", "n": 6000, "k": 10, "dist": "antagonistic"},
]
MEANINGFUL_CLASSES = [AlgoClass.RECENCY, AlgoClass.FREQUENCY,
                      AlgoClass.INSERTION, AlgoClass.ADAPTIVE]

# v9: the STANDARDISED, WEIGHTED benchmark the utility verdict is measured on.
# Multi-regime by declaration: natural access patterns carry 0.7 of the weight, the
# adversarial tail (loop/scan/antagonistic -- the regimes that expose static
# policies) carries 0.3. Quality per trace is the COMPETITIVE RATIO vs Belady, so
# no regime dominates the mean just by being easy. This is what `beats_human`
# means in v9: a higher WEIGHTED COMPETITIVE RATIO than every human policy at <=
# equal declared capacity, on a suite fixed in advance.
BENCHMARK_SUITE = [
    {"name": "zipf",      "n": 6000, "k": 10, "dist": "zipf",          "weight": 0.20},
    {"name": "recency",   "n": 6000, "k": 10, "dist": "recency",       "weight": 0.15},
    {"name": "drift",     "n": 6000, "k": 10, "dist": "drift",         "weight": 0.20},
    {"name": "mixed",     "n": 6000, "k": 10, "dist": "mixed",         "weight": 0.15},
    {"name": "bench_loop", "n": 6000, "k": 10,
     "dist": "param:period=60;noise=0.0;skew=0.0",                  "weight": 0.10},
    {"name": "bench_scan", "n": 6000, "k": 10, "dist": "scan",         "weight": 0.10},
    {"name": "bench_antag", "n": 6000, "k": 10, "dist": "antagonistic", "weight": 0.10},
]


# --------------------------------------------------------------------------- #
# Policy assembly: splice a SCORE expr + an UPDATE expr + a capacity multiplier
# into a complete cache-simulation solve().
# --------------------------------------------------------------------------- #
def _assemble_core(score_expr: str, update_expr: str, cmul: int) -> str:
    """v8: a candidate emits only the eviction LOGIC -- three top-level symbols
    (_evict_score, _on_hit, CMUL). The TRUSTED driver (DRIVER_SRC) runs the
    simulation and computes the hit rate, so no candidate can misreport the metric.
    The score_expr / update_expr splice points are IDENTICAL to v7, so _SCORE_OPS,
    _UPDATE_OPS, fragment_is_safe, op_probe and assemble_from_vocab are unchanged."""
    L = [
        "# v8-cache-policy",
        f"# cmul={cmul}",                   # marker (stripped by ast.unparse on
        "def _evict_score(la, fr, ins, t):",  # mutation -- CMUL=N below survives)
        f"    return {score_expr}",
        "def _on_hit(la, fr, ins, t):",
        f"    return {update_expr}",
        f"CMUL = {cmul}",
    ]
    return "\n".join(L)


# v8 TRUSTED DRIVER. Defined as a real function (unit-testable), exported as source
# via inspect.getsource and spliced into the grounding runner. It imports the
# candidate's three pure symbols and computes the hit rate ITSELF -- the candidate
# never sees `hits`, `cache`, `cap`, or `trace`, so the metric is honest by
# construction. Hardening (from the design panel): CMUL clamped to [1,8]; a
# non-numeric/NaN score coerced to +inf so it can neither crash min() nor be
# evicted-by-cheat; a malformed _on_hit return replaced by the neutral default so a
# garbage update can only DEGRADE, never inflate, the hit count.
def _driver_solve(trace, k):
    import candidate as _c
    score = _c._evict_score
    on_hit = _c._on_hit
    cmul = _c.CMUL
    if not isinstance(cmul, int) or isinstance(cmul, bool) or not (1 <= cmul <= 4):
        raise ValueError("CMUL must be an int in [1, 4]")
    cap = max(cmul * k, 4)

    # v9 OPTIONAL adaptive symbols. The DRIVER carries a small bounded aux state
    # through the trace (reset per trace, so traces stay independent/comparable):
    #   AUX0                              -- initial aux: a list of <= 8 finite numbers
    #   _on_access(aux, hit, t) -> aux'   -- called on EVERY access with the hit FLAG
    #   _evict_score_aux(la, fr, ins, t, aux) -- scoring that may READ aux
    # Soundness is unchanged: the candidate sees only PAST information (the per-access
    # hit flag is what any online algorithm observes; never the count), it still has
    # no reference to hits/cache/cap/trace, and aux can only influence WHICH key is
    # evicted -- a legitimate simulation input. A malformed aux return is RESET to
    # AUX0 (degrade-only); a malformed AUX0 on an adaptive candidate is a hard error.
    on_access = getattr(_c, "_on_access", None)
    score_aux = getattr(_c, "_evict_score_aux", None)

    def _aux_ok(v):
        return (isinstance(v, (list, tuple)) and len(v) <= 8
                and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                        and x == x and -1e18 < x < 1e18 for x in v))

    aux = None
    aux0 = None
    if on_access is not None or score_aux is not None:
        raw0 = getattr(_c, "AUX0", None)
        if not _aux_ok(raw0):
            raise ValueError("adaptive candidate must declare AUX0: "
                             "a list of <= 8 finite numbers")
        aux0 = [float(x) for x in raw0]
        aux = list(aux0)

    def _num(v):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
            return float("inf")
        return v

    cache = {}
    hits = 0
    n = len(trace)
    for t in range(n):
        key = trace[t]
        st = cache.get(key)
        hit = st is not None
        if on_access is not None:
            new_aux = on_access(tuple(aux), 1 if hit else 0, t)
            aux = ([float(x) for x in new_aux] if _aux_ok(new_aux)
                   else list(aux0))
        if hit:
            hits += 1
            la, fr, ins = st
            new = on_hit(la, fr, ins, t)
            if (isinstance(new, (list, tuple)) and len(new) == 3
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                            and x == x for x in new)):
                cache[key] = [new[0], new[1], new[2]]
            else:
                cache[key] = [t, fr + 1, ins]
        else:
            if len(cache) >= cap:
                if score_aux is not None:
                    _ax = tuple(aux) if aux is not None else ()
                    vk = min(cache, key=lambda _k: _num(score_aux(
                        cache[_k][0], cache[_k][1], cache[_k][2], t, _ax)))
                else:
                    vk = min(cache, key=lambda _k: _num(score(
                        cache[_k][0], cache[_k][1], cache[_k][2], t)))
                del cache[vk]
            cache[key] = [t, 1, t]
    return (hits / n if n else 1.0, cap)


DRIVER_SRC = inspect.getsource(_driver_solve)


# --------------------------------------------------------------------------- #
# v10 P3: ADAPTIVE HUMAN BASELINES -- trusted, domain-owned reference policies.
# Humans also invented adaptive eviction (ARC, LeCaR); grading proposals only
# against the four static classics overstated them. These references are DRIVER-
# LEVEL implementations (they need ghost lists -- more than the candidate
# contract's 8 floats), are NEVER importable by candidates (they live in the
# runner via driver_src, not in the candidate module), and are measured through
# the SAME harness/verdict path as every candidate.
# --------------------------------------------------------------------------- #

# THE ghost-charging rule -- single source of truth, applied in exactly one place
# (`_baseline_cost_units`). A ghost entry stores a KEY ONLY (no value, no per-key
# state), so it is charged GHOST_ENTRY_COST_FRAC of a resident entry on the
# declared-capacity cost axis. Both reference implementations bound their ghost
# population to <= 1x resident capacity (ARC by the DBL(2c) invariant + an
# explicit trim; LeCaR by a single history dict capped at c), so:
#   cost_units(cmul) = cmul * (1 + GHOST_ENTRY_COST_FRAC)
# Rationale for 0.25: a resident entry carries key + [la, fr, ins] + dict slot
# (~4 machine words of payload); a ghost carries the key + dict slot (~1 word of
# payload) -- a quarter is the honest order of magnitude, declared rather than
# measured so the axis stays exact. ANY future candidate-rentable ghost facility
# MUST be charged through this same constant -- two charging rules for one
# resource is exactly the declared-vs-actual drift v10 exists to eliminate.
GHOST_ENTRY_COST_FRAC = 0.25


def _baseline_cost_units(cmul: int, ghost_ratio: float = 1.0) -> float:
    """Declared cost units for a policy with ghosts (the ONE place the ghost rule
    is applied). ghost_ratio = ghost capacity as a fraction of resident capacity."""
    return cmul * (1.0 + ghost_ratio * GHOST_ENTRY_COST_FRAC)


def _arc_reference(trace, k):
    """ARC (Megiddo & Modha) -- trusted reference. T1/T2 resident, B1/B2 ghosts,
    adaptive target p. Deterministic: dicts in insertion order are the LRU lists.
    Capacity from the stub candidate's CMUL (the stub's score/update are unused).
    Ghost population obeys |B1|+|B2| <= c (DBL invariant once resident is full);
    an explicit trim enforces the declared bound in transient states."""
    import candidate as _c
    cmul = _c.CMUL
    if not isinstance(cmul, int) or isinstance(cmul, bool) or not (1 <= cmul <= 4):
        raise ValueError("CMUL must be an int in [1, 4]")
    c = max(cmul * k, 4)
    t1, t2, b1, b2 = {}, {}, {}, {}
    p = 0
    hits = 0
    n = len(trace)

    def _replace(in_b2):
        if t1 and (len(t1) > p or (in_b2 and len(t1) == p)):
            lru = next(iter(t1))
            del t1[lru]
            b1[lru] = True
        elif t2:
            lru = next(iter(t2))
            del t2[lru]
            b2[lru] = True

    def _trim_ghosts():
        while len(b1) + len(b2) > c:                 # declared ghost bound: <= c
            g = b1 if len(b1) >= len(b2) else b2
            del g[next(iter(g))]

    for x in trace:
        if x in t1:
            del t1[x]
            t2[x] = True
            hits += 1
        elif x in t2:
            del t2[x]
            t2[x] = True                              # move to MRU
            hits += 1
        elif x in b1:
            p = min(c, p + max(1, len(b2) // max(1, len(b1))))
            _replace(False)
            del b1[x]
            t2[x] = True
        elif x in b2:
            p = max(0, p - max(1, len(b1) // max(1, len(b2))))
            _replace(True)
            del b2[x]
            t2[x] = True
        else:
            if len(t1) + len(b1) == c:
                if len(t1) < c:
                    del b1[next(iter(b1))]
                    _replace(False)
                elif t1:
                    del t1[next(iter(t1))]            # L1 all-resident: drop LRU
            elif (len(t1) + len(b1) < c
                    and len(t1) + len(t2) + len(b1) + len(b2) >= c):
                if len(t1) + len(t2) + len(b1) + len(b2) >= 2 * c and b2:
                    del b2[next(iter(b2))]
                _replace(False)
            t1[x] = True
        _trim_ghosts()
    return (hits / n if n else 1.0, c)


def _lecar_reference(trace, k):
    """LeCaR-style reference -- DETERMINISTIC variant (argmax policy choice, no
    sampling; exponential regret on ghost hits; ONE shared ghost history capped at
    c). Documented deviation from the sampled original so runs are reproducible."""
    import math
    import candidate as _c
    cmul = _c.CMUL
    if not isinstance(cmul, int) or isinstance(cmul, bool) or not (1 <= cmul <= 4):
        raise ValueError("CMUL must be an int in [1, 4]")
    c = max(cmul * k, 4)
    cache = {}                       # key -> [last_access, freq]
    hist = {}                        # evicted key -> policy that evicted it
    w_lru = w_lfu = 0.5
    lr = 0.45
    hits = 0
    n = len(trace)
    for t in range(n):
        x = trace[t]
        st = cache.get(x)
        if st is not None:
            hits += 1
            st[0] = t
            st[1] += 1
            continue
        if x in hist:                # regret: the evicting policy was wrong
            by = hist.pop(x)
            if by == "L":
                w_lru *= math.exp(-lr)
            else:
                w_lfu *= math.exp(-lr)
            s = w_lru + w_lfu
            w_lru, w_lfu = w_lru / s, w_lfu / s
        if len(cache) >= c:
            if w_lru >= w_lfu:       # deterministic argmax; tie -> LRU
                victim = min(cache, key=lambda kk: cache[kk][0])
                by = "L"
            else:
                victim = min(cache, key=lambda kk: (cache[kk][1], cache[kk][0]))
                by = "F"
            del cache[victim]
            hist[victim] = by
            if len(hist) > c:        # declared ghost bound: <= c
                del hist[next(iter(hist))]
        cache[x] = [t, 1]
    return (hits / n if n else 1.0, c)


ARC_DRIVER_SRC = inspect.getsource(_arc_reference).replace(
    "def _arc_reference(", "def _driver_solve(")
LECAR_DRIVER_SRC = inspect.getsource(_lecar_reference).replace(
    "def _lecar_reference(", "def _driver_solve(")


def adaptive_baselines() -> list:
    """The adaptive HUMAN tier: ARC + LeCaR-style at each legal capacity. Each
    entry is measured through the normal harness with its driver_src swapped in
    and an inert stub candidate carrying only the capacity (CMUL)."""
    out = []
    for cmul in (1, 2, 4):
        stub = _assemble_policy("lru", "bump_both", cmul)
        units = _baseline_cost_units(cmul)           # THE ghost rule, applied once
        out.append({"label": f"arc{cmul}", "driver_src": ARC_DRIVER_SRC,
                    "stub": stub, "cost_units": units})
        out.append({"label": f"lecar{cmul}", "driver_src": LECAR_DRIVER_SRC,
                    "stub": stub, "cost_units": units})
    return out


# Seed (human) ops. SCORE: evict argmin. UPDATE: new [la, fr, ins] on a hit.
_SCORE_OPS = {
    "lru":  "la",          # evict least-recently-used (oldest access)
    "lfu":  "fr",          # evict least-frequently-used
    "fifo": "ins",         # evict oldest-inserted
    "mru":  "-la",         # evict most-recently-used
}
_UPDATE_OPS = {
    "bump_both": "[t, fr + 1, ins]",   # track recency AND frequency (default)
    "bump_rec":  "[t, fr, ins]",       # track recency only
    "bump_freq": "[la, fr + 1, ins]",  # track frequency only
}

# A score/update FRAGMENT is spliced as raw text into the policy template, so it must
# be a SINGLE expression over {la, fr, ins, t} (+ int literals, arithmetic, and the
# int method .bit_length()) -- not a statement, not a call to an arbitrary name, not
# code smuggled past the module-level AST screen via `;`/newline (which `mode="eval"`
# already rejects). This is defence-in-depth on top of the engine's AST screen.
_FRAG_NAMES = {"la", "fr", "ins", "t"}
_FRAG_FUNCS = {"max", "min", "abs"}           # safe pure builtins fragments may call


def fragment_is_safe(expr, extra_names=()) -> bool:
    """`extra_names` admits kind-specific identifiers (the ASCORE kind exposes
    `aux`, the trusted EWMA state tuple)."""
    if not isinstance(expr, str) or not expr.strip():
        return False
    try:
        tree = _ast.parse(expr, mode="eval")        # statements/`;`/newline -> error
    except SyntaxError:
        return False
    allowed = _FRAG_NAMES | _FRAG_FUNCS | set(extra_names)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Name) and node.id not in allowed:
            return False
        if isinstance(node, _ast.Attribute) and node.attr != "bit_length":
            return False
        if isinstance(node, _ast.Call):
            f = node.func
            if not ((isinstance(f, _ast.Attribute) and f.attr == "bit_length")
                    or (isinstance(f, _ast.Name) and f.id in _FRAG_FUNCS)):
                return False
        if isinstance(node, (_ast.Lambda, _ast.comprehension, _ast.Await,
                             _ast.NamedExpr)):
            return False
    return True


def _assemble_policy(score: str, update: str, cmul: int) -> str:
    return _assemble_core(_SCORE_OPS[score], _UPDATE_OPS[update], cmul)


def _frag_extra(kind: str) -> tuple:
    """Extra identifiers a fragment of this kind may reference."""
    return ("aux",) if kind == ASCORE else ()


def _assemble_adaptive(ascore_expr: str, cmul: int) -> str:
    """v9 ADAPTIVE policy: a fixed, TRUSTED EWMA hit-flag tracker (aux[0], decay
    0.98) plus a spliced adaptive score expression over (la, fr, ins, t, aux). Only
    the score expression is candidate-varying; the aux dynamics are template-owned,
    so the search composes over `ascore` fragments exactly like score/update ones."""
    L = [
        "# v9-cache-policy",
        f"# cmul={cmul}",
        "def _evict_score(la, fr, ins, t):",
        "    return la",                    # static fallback (driver prefers aux path)
        "def _on_hit(la, fr, ins, t):",
        "    return [t, fr + 1, ins]",
        "AUX0 = [0.5]",
        "def _on_access(aux, hit, t):",
        "    return [aux[0] * 0.98 + 0.02 * hit]",
        "def _evict_score_aux(la, fr, ins, t, aux):",
        f"    return {ascore_expr}",
        f"CMUL = {cmul}",
    ]
    return "\n".join(L)


def assemble_policy_from(vocab: Vocabulary, score: str, update: str,
                         cmul: int) -> str:
    return _assemble_core(vocab.fragment(SCORE, score),
                          vocab.fragment(UPDATE, update), cmul)


def classify(code: str) -> AlgoClass:
    """Classify a cache policy by which state its _evict_score actually reads. Keys
    off the AST FunctionDef (not the `# v8-cache-policy` comment), so a MUTATED
    candidate -- whose comments ast.unparse stripped -- and an arbitrary free-form
    (multi-statement) score body both classify correctly."""
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return AlgoClass.OTHER
    fn_defs = {n.name: n for n in _ast.walk(tree)
               if isinstance(n, _ast.FunctionDef)}
    # v9: a policy is ADAPTIVE only if its ACTIVE score actually READS the aux
    # state (a no-op _on_access with an aux-blind score is just its static policy
    # in disguise -- the v9 review caught the presence-only check inflating the
    # class). The driver scores with _evict_score_aux when present, so that is
    # the function whose reads matter.
    fn = fn_defs.get("_evict_score_aux") or fn_defs.get("_evict_score")
    if fn is None:
        return AlgoClass.OTHER
    names = {n.id for n in _ast.walk(fn) if isinstance(n, _ast.Name)}
    if "aux" in names and fn.name == "_evict_score_aux":
        return AlgoClass.ADAPTIVE
    has_la, has_fr, has_ins = "la" in names, "fr" in names, "ins" in names
    if has_la and has_fr:
        # v10 P5: a STATIC recency+frequency blend is BLENDED, not ADAPTIVE --
        # ADAPTIVE is reserved for scores that actually read the aux state.
        return AlgoClass.BLENDED
    if has_fr:
        return AlgoClass.FREQUENCY
    if has_ins and not has_la:
        return AlgoClass.INSERTION
    if has_la:
        return AlgoClass.RECENCY
    return AlgoClass.INVENTED


def regime(workload_name: str) -> str:
    return {"zipf": "skewed-popularity", "recency": "bursty-reuse",
            "drift": "shifting-popularity", "mixed": "mixed-regime",
            "adv_loop": "cyclic-scan", "adv_scan": "one-shot-sweep"}.get(
        workload_name, workload_name)


def explain_failure(approach):
    """-> (pattern, regime, detail) | None (the engine builds a Lesson from it)."""
    ev = approach.evidence
    if not ev:
        return None
    return ("fixed recency/frequency rule", "shifting / cyclic traces",
            "a fixed eviction rule mis-ranks keys when the access pattern changes "
            "regime (loops, drift) -- it cannot forget a once-popular key")


# --------------------------------------------------------------------------- #
# Seed strategy library (the deterministic Builder's repertoire).
# --------------------------------------------------------------------------- #
STRATEGIES = {
    "lru":  Strategy("lru",  AlgoClass.RECENCY,
                     _assemble_policy("lru", "bump_both", 2),
                     "LRU: evict the least-recently-used key."),
    "lfu":  Strategy("lfu",  AlgoClass.FREQUENCY,
                     _assemble_policy("lfu", "bump_both", 2),
                     "LFU: evict the least-frequently-used key."),
    "fifo": Strategy("fifo", AlgoClass.INSERTION,
                     _assemble_policy("fifo", "bump_both", 2),
                     "FIFO: evict the oldest-inserted key."),
    "mru":  Strategy("mru",  AlgoClass.RECENCY,
                     _assemble_policy("mru", "bump_both", 2),
                     "MRU: evict the most-recently-used key (good on loops)."),
}
SEED_STRATEGY = "lru"
BUILDER_PRIORITY = ["lru", "lfu", "fifo", "mru"]
PROBE_SOLVER = STRATEGIES["lru"].code            # trusted solver for generator probes


# --------------------------------------------------------------------------- #
# v4-style genome (human policies) the synthesiser walks.
# --------------------------------------------------------------------------- #
_GENOME = [
    ("lru2",  "lru",  "bump_both", 2),
    ("lfu2",  "lfu",  "bump_both", 2),
    ("fifo2", "fifo", "bump_both", 2),
    ("mru2",  "mru",  "bump_both", 2),
    ("lru4",  "lru",  "bump_both", 4),
    ("lfu4",  "lfu",  "bump_both", 4),
    ("mru4",  "mru",  "bump_both", 4),
]


def synthesize_programs(parents=None) -> list[tuple]:
    return [(f"policy_{label}", _assemble_policy(s, u, c))
            for (label, s, u, c) in _GENOME]


# --------------------------------------------------------------------------- #
# v5/v6 open vocabulary.
# --------------------------------------------------------------------------- #
def seed_vocabulary() -> Vocabulary:
    v = Vocabulary()
    for name, frag in _SCORE_OPS.items():
        v.add_op(OpEntry(SCORE, name, frag, Provenance.HUMAN))
    for name, frag in _UPDATE_OPS.items():
        v.add_op(OpEntry(UPDATE, name, frag, Provenance.HUMAN))
    for g in adversary_generators(10):
        v.add_gen(GenEntry(name=g["name"], dist=g["dist"], origin=Provenance.HUMAN))
    return v


def _proposed_op_genomes(entry: OpEntry):
    """Compose a proposed op onto cmul {2, 4}. A proposed SCORE op rides the default
    `bump_both` update; a proposed UPDATE op rides the strong `lru` score."""
    nm = entry.name
    if entry.kind == SCORE:
        return [(f"{nm}_c2", nm, "bump_both", 2),
                (f"{nm}_c4", nm, "bump_both", 4)]
    if entry.kind == UPDATE:
        return [(f"lru_{nm}_c2", "lru", nm, 2),
                (f"lfu_{nm}_c2", "lfu", nm, 2)]
    return []


def assemble_from_vocab(vocab: Vocabulary, rng=None) -> list[tuple]:
    out: list[tuple] = []
    for e in vocab.proposed_ops():
        if e.kind == ASCORE:
            # v9: an adaptive score composes into the adaptive template (trusted
            # EWMA aux) at both capacities.
            for c in (2, 4):
                out.append((f"policy_{e.name}_c{c}",
                            _assemble_adaptive(vocab.fragment(ASCORE, e.name), c),
                            [e.name]))
            continue
        for (lbl, sc, up, c) in _proposed_op_genomes(e):
            out.append((f"policy_{lbl}", assemble_policy_from(vocab, sc, up, c),
                        [sc, up]))
    for (label, sc, up, c) in _GENOME:
        out.append((f"policy_{label}", assemble_policy_from(vocab, sc, up, c),
                    [sc, up]))
    return out


# --- the Builder's deterministic OP meta-proposer ------------------------- #
# A LIBRARY of new SCORE ops (none in the seed vocabulary), each tagged with the
# trace regime it targets. v7: many MORE templates than the per-round cap, so causal
# guidance SELECTS which to propose (it does not merely reorder a tiny fixed list).
_OP_TEMPLATES = [
    # (kind, name, fragment, target_tags, note)
    (SCORE, "rec_logf", "la + 30 * fr.bit_length()", {"skew", "drift", "mixed"},
     "recency with a LOGARITHMIC frequency boost -- keeps a hot key warm without "
     "letting raw frequency dominate (adapts as popularity shifts)"),
    (SCORE, "score_half", "la + (fr * (t - ins)) // 4", {"skew", "mixed"},
     "recency offset by frequency x age -- favours items that have earned reuse"),
    (SCORE, "lfu_aged", "fr - (t - la)", {"drift", "loop", "shift"},
     "frequency AGED by idle time -- forgets stale-popular keys when the working "
     "set drifts (an ARC/LRFU-flavoured score)"),
    (SCORE, "lfu_aged2", "2 * fr - (t - la)", {"drift", "shift"},
     "frequency (weighted) aged by idle time"),
    (SCORE, "rec_plus_f", "la + 8 * fr", {"skew"},
     "recency with a linear frequency bonus"),
    (SCORE, "freq_then_rec", "fr * 100000 + la", {"skew", "frequency"},
     "frequency primary, recency tie-break (LFU with an LRU tie-break)"),
    (SCORE, "idle_penalty", "fr - 2 * (t - la)", {"drift", "loop", "scan"},
     "strong idle penalty -- evict keys not touched recently regardless of count"),
    (SCORE, "aged_logf", "la + 20 * fr.bit_length() - (t - la)", {"drift", "shift"},
     "log-frequency boost minus idle penalty -- warm-but-recent keys survive"),
    (SCORE, "freq_over_age", "fr * 1000 - (t - la) * 10", {"recency", "skew"},
     "frequency credit decayed linearly by idle time"),
    (SCORE, "rec_sq_freq", "la + fr * fr", {"skew"},
     "recency with a SUPER-linear frequency bonus (very hot keys pinned)"),
    (SCORE, "scan_resist", "fr * 100000 - (t - ins)", {"scan", "loop", "shift"},
     "scan/loop-resistant (2Q/LIRS-flavoured): protect frequently-reused keys; among "
     "equal-frequency one-timers evict the longest-resident -- a generator-targeted "
     "op for the Adversary's scan/loop attacks"),
    # v9 ADAPTIVE score ops (aux[0] = trusted EWMA of the hit flag, decay 0.98).
    (ASCORE, "ad_thrash", "(-la) if aux[0] < 0.08 else (la + 30 * fr.bit_length())",
     {"loop", "scan", "shift", "drift", "mixed"},
     "ADAPTIVE thrash guard: when the EWMA hit-rate collapses (a loop/scan the cache "
     "cannot hold -- LRU-family thrashes to 0), switch to MRU-like eviction (keeps a "
     "stable resident subset); otherwise behave as rec_logf. Detects and switches "
     "regime DURING the trace -- what no static score can do."),
    (ASCORE, "ad_guard15", "(-la) if aux[0] < 0.15 else (la + 30 * fr.bit_length())",
     {"loop", "shift"},
     "adaptive thrash guard with an EAGER threshold (trips earlier, recovers more "
     "of a long loop, slightly jumpier on borderline regimes)"),
    (UPDATE, "decay_hit", "[t, max(1, fr // 2 + 1), ins]", {"drift", "shift"},
     "on a hit, DECAY the accumulated frequency (forget old bursts) then +1"),
]


_REGIME_NEEDLES = (("drift", "drift"), ("loop", "loop"), ("scan", "scan"),
                   ("sawtooth", "scan"), ("zipf", "skew"), ("mixed", "mixed"),
                   ("phase", "shift"), ("burst", "shift"), ("antag", "loop"),
                   ("param", "loop"), ("recency", "recency"))


def _causal_failure_tags(guidance: dict) -> set:
    """v7: turn the CAUSAL guidance (the failing trace's dist) into op-target tags --
    the PRIMARY driver of which ops to propose. v8: ALSO fold in the Adversary's
    recent GENERATORS, so self-play is generator-targeted -- the Builder authors ops
    aimed at the very attacks the Adversary just built (a loop/scan generator steers
    the proposer toward loop/scan-resistant scores, not just the worst main workload)."""
    tags: set = set()
    causal = guidance.get("causal")
    signals = []
    if causal is not None:
        signals.append((getattr(causal, "dist", "") or "").lower())
        tags |= set(getattr(causal, "tags", set()) or set())
    signals += [str(g).lower() for g in guidance.get("recent_generators", [])]
    for sig in signals:
        for needle, tag in _REGIME_NEEDLES:
            if needle in sig:
                tags.add(tag)
    if tags & {"drift", "loop", "scan", "shift"}:
        tags.add("shift")           # any non-stationary failure -> aging/forgetting ops
    return tags


def propose_operations(vocab: Vocabulary, guidance: dict, rng=None
                       ) -> list[OpProposal]:
    """v7 deterministic op-proposer. CAUSAL guidance SELECTS which new ops to author
    (matched by the diagnosed failure's regime tags); the matched set is what gets
    proposed -- not a reordering of a fixed list. Falls back to a spread only when
    no failure was diagnosed."""
    tags = _causal_failure_tags(guidance) | set(guidance.get("weak", []))
    matched, general = [], []
    for (kind, name, frag, ttags, note) in _OP_TEMPLATES:
        if vocab.has_op(kind, name):
            continue
        prop = OpProposal(kind, name, frag, Provenance.PROPOSED_DET, note)
        (matched if (ttags & tags) else general).append(prop)
    # v7: when a concrete failure is diagnosed, propose the MATCHED ops (the causal
    # selection); no padding with unrelated templates. Spread only with no signal.
    return matched if matched else general


def op_probe(vocab: Vocabulary, proposal: OpProposal) -> str:
    """A probe policy that USES the proposed op (safe human defaults elsewhere).
    Rejects any fragment that is not a single safe expression BEFORE splicing it into
    the template (closes a statement-injection surface; the engine wraps this in
    try/except and logs the rejection)."""
    frag = proposal.fragment
    if isinstance(frag, list):
        frag = "\n".join(frag)
    if not fragment_is_safe(frag, extra_names=_frag_extra(proposal.kind)):
        raise ValueError(f"unsafe/non-expression fragment: {frag!r}")
    if proposal.kind == SCORE:
        return _assemble_core(frag, _UPDATE_OPS["bump_both"], 2)
    if proposal.kind == UPDATE:
        return _assemble_core(_SCORE_OPS["lru"], frag, 2)
    if proposal.kind == ASCORE:
        return _assemble_adaptive(frag, 2)
    return _assemble_core(_SCORE_OPS["lru"], _UPDATE_OPS["bump_both"], 2)


def admit_op(vocab: Vocabulary, proposal: OpProposal, iteration: int) -> None:
    frag = proposal.fragment
    if isinstance(frag, list):
        frag = "\n".join(frag)
    if not fragment_is_safe(frag, extra_names=_frag_extra(proposal.kind)):
        raise ValueError(f"unsafe fragment at admit: {frag!r}")   # defence-in-depth
    vocab.add_op(OpEntry(proposal.kind, proposal.name, frag,
                         proposal.origin, iteration, proposal.note))


# --- generative adversary (parameterised + structural trace attacks) ------ #
def adversary_param_attacks(k: int) -> list[dict]:
    out = []
    for period in (25, 45):
        for noise in (0.05, 0.3):
            out.append({"name": f"param_p{period}_n{int(noise * 100)}",
                        "n": 6000, "k": k,
                        "dist": f"param:period={period};noise={noise};skew=0.2"})
    return out


def adversary_generators(k: int) -> list[dict]:
    return [
        {"name": "gen_phase", "n": 6000, "k": k, "dist": "gen:phase|ha=20|hb=20|split=0.5"},
        {"name": "gen_burst", "n": 6000, "k": k, "dist": "gen:burst|h=8|burst=0.3"},
    ]


# --- open generator proposal (Adversary authors new trace generators) ----- #
_GEN_DRIFT = (
    "out = []\n"
    "base = 0\n"
    "for i in range(n):\n"
    "    if i % 800 == 0:\n"
    "        base += 60\n"
    "    if rng.random() < 0.85:\n"
    "        out.append(base + int(15 * (rng.random() ** 2)))\n"
    "    else:\n"
    "        out.append(rng.randint(9000, 9000 + n))\n"
    "return out"
)
_GEN_SAWTOOTH = (
    "out = []\n"
    "U = 40\n"
    "i = 0\n"
    "while len(out) < n:\n"
    "    for x in range(U):\n"
    "        out.append(x)\n"
    "        if rng.random() < 0.15:\n"
    "            out.append(rng.choice(range(3)))\n"
    "return out[:n]"
)
_GEN_TEMPLATES = [
    ("gen_slow_drift", _GEN_DRIFT,
     "a SLOWLY drifting working set -- punishes policies that cannot forget a "
     "once-popular key"),
    ("gen_sawtooth", _GEN_SAWTOOTH,
     "a sawtooth scan with a tiny hot set -- LRU collapses, MRU/aging survive"),
]


def propose_generators(vocab: Vocabulary, guidance: dict, rng=None
                       ) -> list[GenProposal]:
    return [GenProposal(name, body, Provenance.PROPOSED_DET, note)
            for (name, body, note) in _GEN_TEMPLATES if not vocab.has_gen(name)]


def make_generator_block(name: str, body: str) -> str:
    lines = [f"def _genprop_{name}(n, rng):"]
    for ln in body.splitlines():
        lines.append(("    " + ln) if ln.strip() else "")
    lines.append(f'_GENPROP_REGISTRY["{name}"] = _genprop_{name}')
    return "\n".join(lines) + "\n"


def generator_block(proposal: GenProposal) -> str:
    return make_generator_block(proposal.name, proposal.body)


def generator_probe_spec(name: str, k: int = 10) -> dict:
    return {"name": name, "n": 6000, "k": k, "dist": f"prop:{name}"}


def admit_generator(vocab: Vocabulary, proposal: GenProposal, iteration: int,
                    k: int = 10) -> GenEntry:
    block = make_generator_block(proposal.name, proposal.body)
    entry = GenEntry(name=proposal.name, dist=f"prop:{proposal.name}",
                     origin=proposal.origin, src_block=block,
                     iteration=iteration, note=proposal.note)
    vocab.add_gen(entry)
    return entry


# --- human baselines (for the utility frontier + self-play) --------------- #
def human_baseline_policies(vocab: Vocabulary) -> list[tuple]:
    """The policies a HUMAN-only vocabulary can build (every seed score x update at
    each capacity), so the utility flag is measured against the BEST human, not a
    curated subset."""
    out = []
    for sc in _SCORE_OPS:
        for up in ("bump_both",):
            for c in (2, 4):
                out.append((f"{sc}_{up}_{c}", _assemble_policy(sc, up, c)))
    return out


def human_default_policy(vocab: Vocabulary, cap: int) -> str:
    """The incumbent human default at a capacity multiplier -- LRU (what most
    systems ship). The engine's v7 self-play instead compares against the BEST human
    per generator; this remains as a stable fallback."""
    cmul = max(cap, 2)
    return _assemble_policy("lru", "bump_both", cmul)


# LLM op-proposer guidance (domain-specific, so the LLM proposes CACHE ops).
OP_PROPOSER_SYS = (
    "You are the BUILDER extending the primitive vocabulary of a CACHE EVICTION "
    "engine. A policy keeps a bounded cache; on a miss with a FULL cache it evicts "
    "the cached key with the MINIMUM score. Each cached key has integer state: la "
    "(last access time), fr (frequency / access count), ins (insert time); t is the "
    "current clock. Propose ONE new primitive OPERATION expected to improve the HIT "
    "RATE on the listed failing trace. QUALITY is the competitive ratio vs Belady's "
    "offline optimum, weighted across skew/reuse/drift/loop/scan regimes -- "
    "ROBUSTNESS across regimes is what scores. Allowed kinds: 'score' = a Python "
    "EXPRESSION over la, fr, ins, t giving the eviction priority (lower = evicted "
    "first; LRU is `la`, LFU is `fr`, FIFO is `ins`, MRU is `-la`, and e.g. `la + 30 "
    "* fr.bit_length()` keeps a hot key warm); 'update' = a Python LIST expression "
    "[<new_la>, <new_fr>, <new_ins>] for a key's state on a HIT (default "
    "[t, fr + 1, ins]); 'ascore' = an ADAPTIVE score EXPRESSION over la, fr, ins, t "
    "AND aux, where aux[0] is a trusted EWMA of the per-access hit flag (decay 0.98, "
    "starts 0.5) -- it collapses toward 0 when the cache is thrashing (a loop/scan "
    "it cannot hold), so an ascore can SWITCH regimes mid-trace, e.g. `(-la) if "
    "aux[0] < 0.08 else (la + 30 * fr.bit_length())` (MRU under thrash, else "
    "recency+log-frequency). Use ONLY the listed names, numbers, arithmetic, "
    "comparisons, conditional expressions, and int methods like .bit_length() -- no "
    "imports, no I/O. Respond with ONLY a JSON object: "
    '{"kind":"score"|"update"|"ascore","name":"<short_identifier>",'
    '"fragment":"<expr>","rationale":"<why it helps across regimes>"}.')


# --------------------------------------------------------------------------- #
# Grounding spec + Domain object.
# --------------------------------------------------------------------------- #
GROUNDING_SPEC = GroundingSpec(
    oracle_src=ORACLE_SRC,
    workload_gen_src=WORKLOAD_GEN_SRC,
    criteria_cases=CRITERIA_CASES,
    workloads=WORKLOADS,
    mem_workload=MEM_WORKLOAD,
    verdict_src=VERDICT_SRC,
    driver_src=DRIVER_SRC,                  # v8: trusted-by-construction simulation
)

DOMAIN = Domain(
    name="cache_eviction",
    problem_desc=PROBLEM_DESC,
    contract=CONTRACT,
    grounding_spec=GROUNDING_SPEC,
    classify=classify,
    seed_strategy=SEED_STRATEGY,
    strategies=STRATEGIES,
    builder_priority=BUILDER_PRIORITY,
    adversary_cases=[],
    compose=None,
    regime=regime,
    explain_failure=explain_failure,
    uses_frontier=True,
    # v8: with the EXACT-capacity cost axis the seed (LRU at cmul=2) is the CHEAPEST
    # policy, so the old 0.6*seed budget excluded every point (the "recommended within
    # budget" label was then a lie). 2.5*seed admits the genome's cmul range (2..4),
    # so the recommended pick genuinely sits within the stated budget.
    memory_budget_frac=2.5,
    meaningful_classes=MEANINGFUL_CLASSES,
    adversary_workloads=ADVERSARY_WORKLOADS,
    synthesize_programs=synthesize_programs,
    adversary_param_attacks=adversary_param_attacks,
    adversary_generators=adversary_generators,
    seed_vocabulary=seed_vocabulary,
    assemble_from_vocab=assemble_from_vocab,
    propose_operations=propose_operations,
    op_probe=op_probe,
    admit_op=admit_op,
    propose_generators=propose_generators,
    generator_block=generator_block,
    generator_probe_spec=generator_probe_spec,
    admit_generator=admit_generator,
    probe_solver=PROBE_SOLVER,
    human_baseline_policies=human_baseline_policies,
    human_default_policy=human_default_policy,
    op_proposer_sys=OP_PROPOSER_SYS,                 # v7: domain-specific LLM op prompt
    op_kinds=(SCORE, UPDATE, ASCORE),                # v9: + adaptive score kind
    op_str_kinds=(SCORE, UPDATE, ASCORE),            # all fragments are expressions
    benchmark_suite=BENCHMARK_SUITE,                 # v9: weighted utility benchmark
    adaptive_baselines=adaptive_baselines,           # v10 P3: ARC/LeCaR human tier
    max_policy_cost_units=4,                         # v10 P5: max legal CMUL
)
