"""Domain: streaming heavy-hitters / frequent-items under a memory budget (v2).

Process a stream and return the k most frequent items. Unlike top-K selection,
this problem has NO single best solution and NO closed pass/fail oracle:

  * QUALITY is continuous -- recall@K against an exact Counter oracle. Approximate
    methods are allowed; they trade recall for memory.
  * RESOURCE is peak memory (the binding axis: "under a memory budget"), with
    throughput recorded secondarily.

So the objective is a genuine (recall x memory) Pareto FRONTIER:
  - exact counting       -> recall 1.0 but O(distinct) memory (unbounded);
  - Misra-Gries          -> tiny bounded memory, recall depends on skew;
  - Count-Min + heap     -> bounded memory, probabilistic, hurt by collisions;
  - sampling             -> bounded memory, recall depends on skew;
  - a budget-switch HYBRID (exact until distinct exceeds a budget, then MG) can
    reach HIGH recall at BOUNDED memory -- a point no single textbook method
    occupies. That is the discovery target, and the engine can only find it
    because the frontier *rewards* reaching a previously-unreachable combination.

There is also a deliberately METRIC-GAMING strategy (return the first k distinct
items) that looks fine on easy streams but collapses when the heavy hitters
arrive late -- the Adversary's job is to expose it by attacking the metric.
"""

from __future__ import annotations

import inspect
import re

from ..domain import Domain, Strategy
from ..grounding import GroundingSpec
from ..types import AlgoClass
from ..vocab import (GenEntry, GenProposal, INHERIT, OpEntry, OpProposal, PRE,
                     Provenance, VICTIM, Vocabulary)


CONTRACT = """\
Implement a single top-level function with EXACTLY this signature:

    def solve(stream, k):
        ...

Semantics (preconditions: k is an int with k >= 0; stream is a list of hashable,
comparable items -- here, integers):
  * Return a list of up to k items: your estimate of the k MOST FREQUENT items
    in the stream, most-frequent first.
  * The returned items must be DISTINCT and must all actually occur in `stream`.
  * If k == 0 or the stream is empty -> return [].
  * You are scored on recall@k against the exact top-k AND on peak memory; you may
    be approximate to save memory. MUST NOT mutate `stream`.

You may import from the standard library (collections, heapq, random, math, ...).
Define `solve` at module top level. Do not print anything.
"""

PROBLEM_DESC = (
    "Streaming heavy-hitters: estimate the k most frequent items of a stream "
    "under a memory budget. Approaches trade recall@K against peak memory "
    "(and throughput); no single method is best -- the goal is to advance the "
    "recall x memory Pareto frontier.")


# --------------------------------------------------------------------------- #
# Oracle, workload generation, and the continuous verdict (recall@K).
# All three are spliced into the isolated grounding subprocess, so each is
# SELF-CONTAINED (imports live inside the function bodies).
# --------------------------------------------------------------------------- #

def oracle(stream, k):
    """Exact top-k by (count desc, item desc). The single source of truth."""
    from collections import Counter
    c = Counter(stream)
    items = sorted(c.items(), key=lambda kv: (-kv[1], -kv[0]))
    return [it for it, _ in items[:k]]


def verdict(got, expected, items, k):
    """(ok, quality). ok == well-formed prediction (the hard gate); quality ==
    recall@k vs the exact top-k (the continuous frontier axis). Recall is NOT
    gated -- approximate methods are admissible; they simply score lower."""
    if not isinstance(got, (list, tuple)):
        return (False, 0.0)
    g = list(got)
    if len(g) > k:
        return (False, 0.0)
    if len(set(g)) != len(g):           # no duplicates
        return (False, 0.0)
    present = set(items)
    for x in g:
        if x not in present:            # must be items that actually occurred
            return (False, 0.0)
    truth = set(expected)
    if not truth:
        return (True, 1.0)
    recall = len(set(g) & truth) / len(truth)
    return (True, recall)


# v5: registry of PROPOSED stream generators (Adversary-authored source). Empty in
# the base module; the engine appends `_genprop_<name>` definitions (which register
# themselves here) to the LIVE spec's workload_gen_src, so the `prop:<name>` dist
# below dispatches to them inside the grounding subprocess. Kept module-level so a
# native call to gen_workload also resolves it.
_GENPROP_REGISTRY: dict = {}


def gen_workload(n, dist, seed):
    """Deterministic stream generation. `dist` selects the distribution; the
    adversarial dists (freq_late/collision/overflow) attack specific methods."""
    import random
    rng = random.Random(seed)
    if dist.startswith("prop:"):
        # v5: a PROPOSED generator (Adversary-authored, AST-screened + grounded
        # before it ever reaches here). Look it up in the registry; fall back to a
        # uniform stream if it is somehow absent (never crash the benchmark).
        nm = dist[len("prop:"):].split("|")[0]
        fn = _GENPROP_REGISTRY.get(nm)
        if fn is not None:
            return fn(n, rng)
        return [rng.randint(0, max(n // 2, 1)) for _ in range(n)]
    if dist == "uniform":                       # many distinct, ~flat counts
        D = max(n // 2, 1)
        return [rng.randint(0, D) for _ in range(n)]
    if dist == "near_uniform":                  # many items, near-equal counts
        D = max(n // 20, 1)
        return [rng.randint(0, D) for _ in range(n)]
    if dist == "zipf":                          # heavy skew toward small ids
        D = max(n // 4, 50)
        return [int(D * (rng.random() ** 3)) for _ in range(n)]
    if dist == "few_heavy":                     # a few heavy + a long singleton tail
        heavy = list(range(20))
        out = []
        for _ in range(n):
            out.append(rng.choice(heavy) if rng.random() < 0.6
                       else rng.randint(100, 100 + n))
        return out
    if dist == "bursty":                        # long runs of one item
        out = []
        while len(out) < n:
            item = rng.randint(0, n // 8 + 1)
            out.extend([item] * rng.randint(1, 50))
        return out[:n]
    if dist == "competing_heavy":
        # ~150 moderately-frequent items competing for the top-k slots: ranking the
        # true top-k requires tracking many of them, so recall trades off against
        # the summary's capacity (small m misses some; larger m recovers them).
        heavy = list(range(150))
        out = []
        for _ in range(n):
            out.append(rng.choice(heavy) if rng.random() < 0.85
                       else rng.randint(2000, 2000 + n))
        return out
    if dist == "front_loaded":
        # Heavy hitters (ids 0..11) appear DENSELY in the first 30%; the rest are
        # singletons. The prefix heuristic looks great here -- and the Adversary's
        # freq_late stream (same hitters, but late) is what collapses it.
        heavy = list(range(12))
        head = [rng.choice(heavy) for _ in range(int(n * 0.3))]
        tail = [rng.randint(1000, 1000 + n) for _ in range(n - len(head))]
        return head + tail
    # --- adversarial distributions (attack the metric) -------------------- #
    if dist == "freq_late":
        # The heavy hitters (ids 0..11) appear ONLY in the second half; the first
        # half is a flood of distinct rares. Wrecks "first k distinct".
        heavy = list(range(12))
        first = [rng.randint(1000, 1000 + n) for _ in range(n // 2)]
        second = []
        for _ in range(n - n // 2):
            second.append(rng.choice(heavy) if rng.random() < 0.6
                          else rng.randint(1000, 1000 + n))
        return first + second
    if dist == "collision":
        # Dense medium band to maximise sketch hash collisions around the true
        # heavy hitters.
        heavy = list(range(8))
        out = []
        for _ in range(n):
            r = rng.random()
            if r < 0.4:
                out.append(rng.choice(heavy))
            elif r < 0.9:
                out.append(rng.randint(8, 250))      # dense medium band
            else:
                out.append(rng.randint(1000, 1000 + n))
        return out
    if dist == "heavy_skew_overflow":
        # MANY moderately-frequent items -> overflows small bounded summaries.
        heavy = list(range(300))
        out = []
        for _ in range(n):
            out.append(rng.choice(heavy) if rng.random() < 0.85
                       else rng.randint(2000, 2000 + n))
        return out
    if dist.startswith("param:"):
        # v3 GENERATIVE adversary: a parameterised stream the Adversary composes
        # at runtime from knobs -- not a fixed named distribution. Knobs:
        #   late  (fraction of the stream before heavy hitters appear),
        #   heavy (number of heavy items -- stresses bounded summaries),
        #   coll  (fraction drawn from a dense medium band -> hash collisions).
        # HARDENED parsing: whitelist keys, reject non-finite values, and clamp
        # every knob to a sane range. A hostile param string (heavy=1e100, NaN,
        # unknown keys) must NOT be able to exhaust memory or crash the runner.
        import math
        allowed = {"late", "heavy", "coll"}
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
        n_heavy = max(1, min(int(params.get("heavy", 12)), 100000))
        late = min(max(params.get("late", 0.0), 0.0), 0.99)
        coll = min(max(params.get("coll", 0.0), 0.0), 0.9)
        heavy = list(range(n_heavy))
        cut = int(n * late)
        out = []
        for i in range(n):
            if i < cut:
                out.append(rng.randint(10000, 10000 + n))     # rares first
            else:
                r = rng.random()
                if r < 0.6:
                    out.append(rng.choice(heavy))
                elif r < 0.6 + coll:
                    out.append(rng.randint(n_heavy, n_heavy + 200))
                else:
                    out.append(rng.randint(10000, 10000 + n))
        return out
    if dist.startswith("gen:"):
        # v4 GENERATOR-DSL: the Adversary composes a stream from STRUCTURAL
        # segments -- a new attack TYPE, not just a parameterisation of one
        # template. Format: "gen:<type>|key=val|...". Bounded + validated.
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

        if gtype == "phase":
            # heavy set A early, set B late: a "stale winners" trap.
            hA = int(_g("ha", 1, 5000, 8)); hB = int(_g("hb", 1, 5000, 8))
            split = _g("split", 0.1, 0.9, 0.4)
            A = list(range(hA)); B = list(range(1000, 1000 + hB))
            cut = int(n * split); out = []
            for i in range(n):
                pool = A if i < cut else B
                out.append(rng.choice(pool) if rng.random() < 0.7
                           else rng.randint(5000, 5000 + n))
            return out
        if gtype == "spike":
            # quiet background of rares with sudden bursts of a small set.
            hh = list(range(int(_g("h", 1, 5000, 5))))
            burst = _g("burst", 0.0, 0.5, 0.05)
            return [rng.choice(hh) if rng.random() < burst
                    else rng.randint(5000, 5000 + n) for _ in range(n)]
        if gtype == "mixture":
            # two heavy sets interleaved at different densities -> ranking stress.
            hA = int(_g("ha", 1, 5000, 6)); hB = int(_g("hb", 1, 5000, 60))
            A = list(range(hA)); B = list(range(1000, 1000 + hB))
            out = []
            for _ in range(n):
                r = rng.random()
                if r < 0.45:
                    out.append(rng.choice(A))
                elif r < 0.85:
                    out.append(rng.choice(B))
                else:
                    out.append(rng.randint(5000, 5000 + n))
            return out
        return [rng.randint(0, max(n // 2, 1)) for _ in range(n)]   # fallback
    raise ValueError("unknown dist: " + dist)


ORACLE_SRC = inspect.getsource(oracle)
# v5: the subprocess needs `_GENPROP_REGISTRY` defined BEFORE gen_workload (which
# references it) and before any appended `_genprop_<name>` blocks register into it.
WORKLOAD_GEN_SRC = "_GENPROP_REGISTRY = {}\n\n" + inspect.getsource(gen_workload)
VERDICT_SRC = inspect.getsource(verdict)


# --------------------------------------------------------------------------- #
# Trusted criteria suite -- well-formedness + recall sanity on small streams.
# --------------------------------------------------------------------------- #

CRITERIA_CASES = [
    ("tiny",         [1, 1, 1, 2, 2, 3],                         2),
    ("single",       [5, 5, 5],                                  1),
    ("all_distinct", [1, 2, 3, 4],                               10),
    ("empty",        [],                                         3),
    ("k_zero",       [1, 1, 2],                                  0),
    ("ties",         [1, 1, 2, 2, 3, 3, 4],                      3),
    ("skew_small",   [7] * 20 + [8] * 10 + [9] * 5 + list(range(100, 130)), 3),
]


# --------------------------------------------------------------------------- #
# Evaluation workloads (quality + throughput + memory) and the memory probe.
# --------------------------------------------------------------------------- #

# Recall is only meaningful where heavy hitters EXIST, so the eval workloads are
# skewed; flat/uniform streams (no heavy hitters) are used only for the memory
# probe. `competing_heavy` keeps recall-vs-memory a real trade-off.
WORKLOADS = [
    {"name": "zipf",            "n": 40000, "k": 10, "dist": "zipf"},
    {"name": "few_heavy",       "n": 40000, "k": 10, "dist": "few_heavy"},
    {"name": "front_loaded",    "n": 40000, "k": 10, "dist": "front_loaded"},
    {"name": "competing_heavy", "n": 40000, "k": 10, "dist": "competing_heavy"},
]

# Memory probed on a HIGH-DISTINCT stream so exact's O(distinct) cost is visible.
MEM_WORKLOAD = {"name": "mem", "n": 40000, "k": 10, "dist": "uniform"}

# The Adversary's deterministic arsenal of metric attacks.
ADVERSARY_WORKLOADS = [
    {"name": "adv_freq_late", "n": 40000, "k": 10, "dist": "freq_late"},
    {"name": "adv_collision", "n": 40000, "k": 10, "dist": "collision"},
    {"name": "adv_overflow",  "n": 40000, "k": 10, "dist": "heavy_skew_overflow"},
]


# --------------------------------------------------------------------------- #
# Strategy library (the deterministic Builder's repertoire). Recall/memory are
# decided by EXECUTION, never by these labels.
# --------------------------------------------------------------------------- #

_EXACT = '''
from collections import Counter
def solve(stream, k):
    c = Counter(stream)
    items = sorted(c.items(), key=lambda kv: (-kv[1], -kv[0]))
    return [it for it, _ in items[:k]]
'''

_MISRA_GRIES = '''
def solve(stream, k):
    m = max(4 * k, 16)                 # bounded number of counters
    counters = {}
    for x in stream:
        if x in counters:
            counters[x] += 1
        elif len(counters) < m:
            counters[x] = 1
        else:
            dead = []
            for key in counters:
                counters[key] -= 1
                if counters[key] == 0:
                    dead.append(key)
            for key in dead:
                del counters[key]
    items = sorted(counters.items(), key=lambda kv: (-kv[1], -kv[0]))
    return [it for it, _ in items[:k]]
'''

_COUNT_MIN = '''
def solve(stream, k):
    W = 1024
    D = 4
    salts = (1, 1000003, 2000011, 3000017)
    table = [[0] * W for _ in range(D)]
    cap = max(8 * k, 64)
    cand = {}
    for x in stream:
        est = None
        for i in range(D):
            j = hash((salts[i], x)) % W
            table[i][j] += 1
            v = table[i][j]
            if est is None or v < est:
                est = v
        cand[x] = est
        if len(cand) > cap:
            drop = min(cand, key=lambda y: cand[y])
            del cand[drop]
    items = sorted(cand.items(), key=lambda kv: (-kv[1], -kv[0]))
    return [it for it, _ in items[:k]]
'''

# Deliberately METRIC-GAMING: ignores frequency entirely. Looks fine when heavy
# hitters happen to appear early; the Adversary's freq_late stream collapses it.
_FIRST_K_DISTINCT = '''
def solve(stream, k):
    if k <= 0:
        return []
    out = []
    seen = set()
    for x in stream:
        if x not in seen:
            seen.add(x)
            out.append(x)
            if len(out) >= k:
                break
    return out
'''

_RESERVOIR = '''
def solve(stream, k):
    import random
    from collections import Counter
    rng = random.Random(12345)
    cap = 3000
    sample = []
    i = 0
    for x in stream:
        if len(sample) < cap:
            sample.append(x)
        else:
            j = rng.randint(0, i)
            if j < cap:
                sample[j] = x
        i += 1
    c = Counter(sample)
    items = sorted(c.items(), key=lambda kv: (-kv[1], -kv[0]))
    return [it for it, _ in items[:k]]
'''

# Budget-switch HYBRID: exact while distinct <= budget, Misra-Gries beyond it.
# High recall at BOUNDED memory -- the synthesis target.
_HYBRID = '''
def solve(stream, k):
    budget = 4096
    counts = {}
    for x in stream:
        if x in counts:
            counts[x] += 1
        elif len(counts) < budget:
            counts[x] = 1
        else:
            dead = []
            for key in counts:
                counts[key] -= 1
                if counts[key] == 0:
                    dead.append(key)
            for key in dead:
                del counts[key]
            if len(counts) < budget:
                counts[x] = 1
    items = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))
    return [it for it, _ in items[:k]]
'''

STRATEGIES: dict[str, Strategy] = {
    "exact_counter": Strategy("exact_counter", AlgoClass.EXACT, _EXACT,
        "Exact Counter over all distinct items. Recall 1.0, O(distinct) memory.",
        sound=True),
    "misra_gries": Strategy("misra_gries", AlgoClass.MISRA_GRIES, _MISRA_GRIES,
        "Misra-Gries with 4k bounded counters. Tiny memory; recall tracks skew.",
        sound=True),
    "count_min": Strategy("count_min", AlgoClass.COUNT_MIN, _COUNT_MIN,
        "Count-Min sketch + bounded candidate set. Bounded memory, probabilistic.",
        sound=True),
    "first_k_distinct": Strategy("first_k_distinct", AlgoClass.SAMPLE,
        _FIRST_K_DISTINCT,
        "Return the first k DISTINCT items. Ignores frequency (metric-gaming).",
        sound=False),
    "reservoir": Strategy("reservoir", AlgoClass.SAMPLE, _RESERVOIR,
        "Reservoir-sample the stream, then count the sample. Bounded memory.",
        sound=True),
}

SEED_STRATEGY = "exact_counter"

# Explore one distinct family each; first_k_distinct is the SAMPLE representative
# so the Adversary gets a metric-gamer to expose.
BUILDER_PRIORITY = [
    "misra_gries",
    "count_min",
    "first_k_distinct",
    "reservoir",
]

MEANINGFUL_CLASSES = [AlgoClass.EXACT, AlgoClass.MISRA_GRIES,
                      AlgoClass.COUNT_MIN, AlgoClass.SAMPLE]


# --------------------------------------------------------------------------- #
# Classifier, regime labels, failure explanation, hybrid composer.
# --------------------------------------------------------------------------- #

def classify(code: str) -> AlgoClass:
    low = code.lower()
    # v4: a synthesised eviction policy (genome marker) is an invented structure.
    if "v4-synth-policy" in code:
        return AlgoClass.INVENTED
    # v3: a min-eviction-with-inheritance summary is Space-Saving -- a structure
    # absent from the library; tag it INVENTED so it is visibly "new".
    if re.search(r"min\s*\(\s*counts", code) and ("pop(" in code or "del " in code):
        return AlgoClass.INVENTED
    if "budget" in low:
        return AlgoClass.HYBRID
    if "hash(" in code and ("table" in low or "sketch" in low or "* w" in low):
        return AlgoClass.COUNT_MIN
    if "counter(" in low or "most_common" in low:
        if "reservoir" in low or "sample" in low:
            return AlgoClass.SAMPLE
        return AlgoClass.EXACT
    if "-= 1" in code and ("counters" in low or "counts" in low):
        return AlgoClass.MISRA_GRIES
    if ("seen" in low and "break" in low) or "sample" in low or "reservoir" in low:
        return AlgoClass.SAMPLE
    return AlgoClass.OTHER


def regime(workload_name: str) -> str:
    if "uniform" in workload_name:
        return "high-distinct/flat"
    if workload_name.startswith("adv_"):
        return "adversarial"
    return "skewed"


def explain_failure(approach) -> tuple | None:
    ev = approach.evidence
    if ev is None:
        return None
    if ev.mutated_input:
        return ("in-place mutation of the stream", "any input",
                "modifying `stream` violates the no-mutation contract")
    if "seen" in approach.code and "break" in approach.code:
        return ("first-k-distinct / prefix heuristic",
                "streams where heavy hitters arrive late",
                "returning the first k distinct items ignores frequency and "
                "collapses when the heavy hitters do not appear early")
    return None


def compose_bounded_hybrid(parents) -> str | None:
    """Deterministic synthesis: if the parents include the EXACT approach AND a
    bounded one (Misra-Gries / Count-Min / sample), compose the budget-switch
    hybrid -- exact while it fits, bounded-summary beyond -- which reaches high
    recall at bounded memory (a frontier point no single parent occupies)."""
    classes = {p.algo_class for p in parents if p.is_correct}
    bounded = {AlgoClass.MISRA_GRIES, AlgoClass.COUNT_MIN, AlgoClass.SAMPLE}
    if AlgoClass.EXACT not in classes or not (bounded & classes):
        return None
    return _HYBRID


# --------------------------------------------------------------------------- #
# v3: COMPOSITIONAL PRIMITIVE GRAMMAR
# --------------------------------------------------------------------------- #
# A bounded streaming summary = (capacity m) x (eviction policy). The engine
# ASSEMBLES `solve` from these primitive fragments, so it can produce algorithms
# that are NOT any single library strategy -- most importantly Space-Saving
# (replace the min-count item, inheriting its count), which the library does not
# contain. This is genuine compositional synthesis, not selection from a library;
# the combination space is searched and every candidate is grounded.

_SKELETON = '''
def solve(stream, k):
    m = max({mmul} * k, {mbase})
    counts = {{}}
    for x in stream:
        if x in counts:
            counts[x] += 1
        elif len(counts) < m:
            counts[x] = 1
        else:
{evict}
    items = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))
    return [it for it, _ in items[:k]]
'''

_EVICTIONS = {
    # Space-Saving (Metwally et al.): replace min-count item, inherit its count.
    "replace_min_inherit": (
        "            mi = min(counts, key=counts.get)\n"
        "            mc = counts.pop(mi)\n"
        "            counts[x] = mc + 1"),
    # Misra-Gries discipline: global decrement.
    "global_decrement": (
        "            dead = []\n"
        "            for key in counts:\n"
        "                counts[key] -= 1\n"
        "                if counts[key] == 0:\n"
        "                    dead.append(key)\n"
        "            for key in dead:\n"
        "                del counts[key]\n"
        "            if len(counts) < m:\n"
        "                counts[x] = 1"),
    # Plain drop-min, no inheritance.
    "drop_min": (
        "            mi = min(counts, key=counts.get)\n"
        "            del counts[mi]\n"
        "            counts[x] = 1"),
    # Deliberately weak: drop an arbitrary item (the frontier should reject it).
    "drop_first": (
        "            victim = next(iter(counts))\n"
        "            del counts[victim]\n"
        "            counts[x] = 1"),
}


def _assemble(evict_name: str, mmul: int, mbase: int = 64) -> str:
    return _SKELETON.format(mmul=mmul, mbase=mbase, evict=_EVICTIONS[evict_name])


# Curated search order: the interesting compositions first (Space-Saving), then
# capacity variants and weaker policies. Returns (label, source) pairs.
_COMPOSITIONS = [
    ("space_saving_x8",   "replace_min_inherit", 8),
    ("space_saving_x32",  "replace_min_inherit", 32),
    ("drop_min_x16",      "drop_min", 16),
    ("mg_skeleton_x16",   "global_decrement", 16),
    ("space_saving_x128", "replace_min_inherit", 128),
    ("drop_first_x16",    "drop_first", 16),
]


# --------------------------------------------------------------------------- #
# v4: EVICTION-POLICY SYNTHESIS (primitive synthesis from low-level operations)
# --------------------------------------------------------------------------- #
# v3 enumerated whole eviction POLICIES. v4 provides only low-level OPERATIONS --
# how to choose a victim, what count the admitted item inherits, an optional
# global pre-step -- and SYNTHESISES the policy by composing them. Most resulting
# policies were never written by the implementer (e.g. "evict the 2nd-smallest,
# inherit the victim's count, halve-all on overflow"). The OP vocabulary is still
# designed (that is v5's wall), but the POLICIES now emerge from search; the
# frontier + Adversary decide which survive.

_VICTIM_OPS = {
    "min":        ["vk = min(counts, key=counts.get)"],
    "max":        ["vk = max(counts, key=counts.get)"],
    "second_min": ["vk = min(counts, key=counts.get)",
                   "_rest = [kk for kk in counts if kk != vk]",
                   "if _rest:",
                   "    vk = min(_rest, key=counts.get)"],
    "first":      ["vk = next(iter(counts))"],
}
_INHERIT_OPS = {
    "one": "1", "vplus": "vv + 1", "v": "vv",
    "vdec": "max(1, vv - 1)", "vhalf": "max(1, vv // 2)",
}
_PRE_OPS = {
    "none": [],
    "dec_all": ["for _kk in list(counts):",
                "    counts[_kk] -= 1",
                "    if counts[_kk] <= 0:",
                "        del counts[_kk]"],
    "halve_all": ["for _kk in list(counts):",
                  "    counts[_kk] = counts[_kk] // 2",
                  "    if counts[_kk] <= 0:",
                  "        del counts[_kk]"],
}

# v3's named eviction policies, expressed as genome points -- used to PROVE an
# accepted v4 policy is genuinely outside the v3 grammar.
V3_POLICY_GENOMES = {("min", "vplus", "none"),   # Space-Saving
                     ("min", "one", "none"),     # drop_min
                     ("first", "one", "none")}   # drop_first


def _assemble_core(victim_lines: list, inherit_expr: str, pre_lines: list,
                   mmul: int) -> str:
    """Splice three op FRAGMENTS into a complete `solve()`. v5: this takes the raw
    fragments (not dict keys), so it serves both the human seed ops AND any op the
    Builder proposed and the engine admitted to the vocabulary."""
    L = ["# v4-synth-policy",
         "def solve(stream, k):",
         f"    m = max({mmul} * k, 64)",
         "    counts = {}",
         "    for x in stream:",
         "        if x in counts:",
         "            counts[x] += 1",
         "        elif len(counts) < m:",
         "            counts[x] = 1",
         "        else:"]
    if pre_lines:
        # A global pre-step (decay) can FREE slots, so re-check whether the new item
        # now fits before evicting; eviction is the fallback. Emitted ONLY when the
        # pre-step exists -- for pre="none" the re-check is dead code, so we drop it
        # and evict directly (keeps synthesised output free of unreachable branches).
        L += ["            " + ln for ln in pre_lines]
        L += ["            if len(counts) < m:",
              "                counts[x] = 1",
              "            else:"]
        ind = "                "          # inside the re-check else (16 spaces)
    else:
        ind = "            "              # directly under the outer else (12 spaces)
    L += [ind + ln for ln in victim_lines]
    L += [ind + "vv = counts.pop(vk)",
          ind + f"counts[x] = {inherit_expr}",
          "    items = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))",
          "    return [it for it, _ in items[:k]]"]
    return "\n".join(L)


def _assemble_policy(victim: str, inherit: str, pre: str, mmul: int) -> str:
    """Back-compat: assemble from the human seed dicts by KEY (v4 callers/tests)."""
    return _assemble_core(_VICTIM_OPS[victim], _INHERIT_OPS[inherit],
                          _PRE_OPS[pre], mmul)


def assemble_policy_from(vocab, victim: str, inherit: str, pre: str,
                         mmul: int) -> str:
    """v5: assemble from the live VOCABULARY by op name -- so a policy may use a
    PROPOSED op (a victim/inherit/pre fragment the Builder authored and the engine
    grounded) exactly as it uses a human seed op."""
    return _assemble_core(vocab.fragment("victim", victim),
                          vocab.fragment("inherit", inherit),
                          vocab.fragment("pre", pre), mmul)


# Genome the search walks (stable order). Most points are NOT a v3 named policy.
# Order matters: the Builder synthesises in batches (cfg.synthesis_batch), so the
# FIRST batch should carry structurally-DISTINCT, robust candidates -- the search
# must surface a winner before the Synthesizer converges. Empirically (measured on
# the full hardened suite incl. every adversary attack), `min_half32` -- min-evict
# with HALVED inheritance, a rule absent from the v3 grammar -- reaches mean recall
# 0.88 @ 36KB, strictly DOMINATING the best v3 policy ss_x32 (0.83 @ 45KB): higher
# recall at less memory. That is the v4 headline (an accepted structure outside any
# human-designed grammar). Batch 1 = [winner, a distinct cheaper v4-only point, the
# v3 Space-Saving baseline that the Adversary collapses] so a single synthesis round
# demonstrates A (novel accepted) AND B (Adversary exposes weak structures) at once.
_GENOME = [
    ("min_half32",   "min",        "vhalf", "none",      32),    # v4-only WINNER: dominates v3 Space-Saving
    ("min_dec_keep", "min",        "v",     "dec_all",   16),    # v4-only, distinct op (decrement-all + keep count)
    ("ss",           "min",        "vplus", "none",      8),     # v3 Space-Saving baseline (Adversary collapses it)
    ("min_half",     "min",        "vhalf", "none",      16),    # v4-only cheaper memory point
    ("ss_x32",       "min",        "vplus", "none",      32),    # v3 Space-Saving, larger cap
    ("secmin_inh",   "second_min", "vplus", "none",      8),     # v4-only: evict 2nd-smallest
    ("ss_decay",     "min",        "vplus", "halve_all", 8),     # v4-only: SS + periodic forgetting
    ("secmin_decay", "second_min", "vplus", "halve_all", 16),    # v4-only
    ("max_evict",    "max",        "one",   "none",      16),    # v4-only: evict TOP (deliberately bad)
    ("dropfirst",    "first",      "one",   "none",      16),    # v3 drop-first
]


def synthesize_programs(parents=None) -> list[tuple[str, str]]:
    """v4: synthesise eviction POLICIES by composing low-level operations. Returns
    (label, source); the Builder walks them and the frontier keeps the winners."""
    return [(f"policy_{label}", _assemble_policy(v, i, p, m))
            for (label, v, i, p, m) in _GENOME]


def adversary_param_attacks(k: int) -> list[dict]:
    """v3 generative Adversary: compose NOVEL parameterised hostile streams at
    runtime (late-arrival x #heavy x collision), not from a fixed named set."""
    out = []
    for late in (0.85, 0.97):
        for heavy in (120, 400):
            out.append({"name": f"param_l{int(late * 100)}_h{heavy}",
                        "n": 40000, "k": k,
                        "dist": f"param:late={late};heavy={heavy};coll=0.25"})
    return out


def adversary_generators(k: int) -> list[dict]:
    """v4 generator-level Adversary: compose NEW attack TYPES via the generator
    DSL (phase shift / spike / mixture) -- structurally new streams, not just a
    parameterisation of one template. Triggers frontier RE-SCORING when added."""
    return [
        {"name": "gen_phase", "n": 40000, "k": k,
         "dist": "gen:phase|ha=8|hb=8|split=0.4"},
        {"name": "gen_spike", "n": 40000, "k": k,
         "dist": "gen:spike|h=6|burst=0.04"},
        {"name": "gen_mixture", "n": 40000, "k": k,
         "dist": "gen:mixture|ha=6|hb=80"},
    ]


# =========================================================================== #
# v5: OPEN VOCABULARY -- the Builder proposes new OPS, the Adversary proposes new
# GENERATORS, both as source. The seed vocabulary is the v4 human op/gen set; the
# engine screens (AST) + grounds every proposal before admitting it here.
# =========================================================================== #

def seed_vocabulary() -> Vocabulary:
    """Build the per-run vocabulary from the v4 human ops + generators (provenance
    `human`). The engine grows it with admitted proposals."""
    v = Vocabulary()
    for name, frag in _VICTIM_OPS.items():
        v.add_op(OpEntry(VICTIM, name, frag, Provenance.HUMAN))
    for name, frag in _INHERIT_OPS.items():
        v.add_op(OpEntry(INHERIT, name, frag, Provenance.HUMAN))
    for name, frag in _PRE_OPS.items():
        v.add_op(OpEntry(PRE, name, frag, Provenance.HUMAN))
    # The v3/v4 generator DSL streams, tracked for provenance (built into
    # gen_workload -- no source block needed).
    for g in adversary_generators(10):
        v.add_gen(GenEntry(name=g["name"], dist=g["dist"], origin=Provenance.HUMAN))
    return v


def _proposed_op_genomes(entry: OpEntry):
    """Genome points that USE a proposed op. The capacities are chosen (from the
    measured cap-sweep) so a genuinely-helpful op surfaces a NON-dominated point:
    a `pre` op that prunes transients pays off at LOW capacity (frequent overflow),
    so it is composed at cap 16 where it can beat the human inheritance rule at the
    same memory; inheritance/victim ops are composed where they are most distinct."""
    nm = entry.name
    if entry.kind == PRE:
        # v6: compose a pre-op with the `one` inheritance (no count-inflation), where
        # a threshold-prune like `declow` reaches a point that dominates human ops,
        # plus the v5 vplus/vhalf backbones -- higher volume, frontier filters.
        return [(f"min_one_{nm}32", "min", "one", nm, 32),
                (f"min_one_{nm}16", "min", "one", nm, 16),
                (f"min_vplus_{nm}16", "min", "vplus", nm, 16),
                (f"min_vhalf_{nm}32", "min", "vhalf", nm, 32)]
    if entry.kind == INHERIT:
        return [(f"min_{nm}32", "min", nm, "none", 32),
                (f"min_{nm}16", "min", nm, "none", 16),
                (f"min_{nm}12", "min", nm, "none", 12)]
    if entry.kind == VICTIM:
        return [(f"{nm}_one32", nm, "one", "none", 32),
                (f"{nm}_one16", nm, "one", "none", 16),
                (f"{nm}_vhalf16", nm, "vhalf", "none", 16)]
    return []


def assemble_from_vocab(vocab: Vocabulary, rng=None) -> list[tuple]:
    """v5: compose policies over the LIVE vocabulary. Proposed-op policies lead
    (so a single synthesis round surfaces a proposed-op winner), then the v4 human
    genome. Returns (label, code, ops_used) -- ops_used names the three ops the
    policy is built from, so the engine can flag which proposed ops reached the
    frontier."""
    out: list[tuple] = []
    for e in vocab.proposed_ops():
        for (lbl, vi, ih, pr, m) in _proposed_op_genomes(e):
            out.append((f"policy_{lbl}", assemble_policy_from(vocab, vi, ih, pr, m),
                        [vi, ih, pr]))
    for (label, vi, ih, pr, m) in _GENOME:
        out.append((f"policy_{label}", assemble_policy_from(vocab, vi, ih, pr, m),
                    [vi, ih, pr]))
    return out


# --- the Builder's deterministic OP meta-proposer ------------------------- #
# A small library of NEW operation fragments (none in the v4 seed vocabulary),
# each TAGGED with the weakness it is meant to address. The proposer emits the
# ones whose tag matches the current guidance (frontier gap / exposed workloads),
# so proposals are GUIDED, not random. Every one is still AST-screened + grounded
# by the engine before it can join the vocabulary.
_VIC_MIN_OR_ONE = [
    "vk = min(counts, key=counts.get)",
    "if counts[vk] > 1:",
    "    for _kk in counts:",
    "        if counts[_kk] == 1:",
    "            vk = _kk",
    "            break",
]
_PRE_PRUNE_ONES = [
    "for _kk in [kk for kk in counts if counts[kk] == 1]:",
    "    del counts[_kk]",
]
# v6: a THRESHOLD-prune pre-op (delete every counter <= 2) -- NOT in the human pre
# vocabulary (which only has none / decrement-all / halve-all). Composed with the
# `one` inheritance (no count-inflation) it both fixes the Adversary's decoy/spike
# attacks AND keeps the dict tiny, reaching a (recall, memory) point that strictly
# DOMINATES the best human-only policy. This is the v6 beats_human primitive.
_PRE_DECLOW = [
    "for _kk in [kk for kk in counts if counts[kk] <= 2]:",
    "    del counts[_kk]",
]
_OP_TEMPLATES = [
    # (kind, name, fragment, target_tags, note). Order = proposal priority. v6 tags
    # are matched against CAUSAL guidance (the dist + missed-items of an actual
    # failure), not just keyword lessons.
    (PRE, "declow", _PRE_DECLOW,
     {"transient", "decoy", "spike", "collision", "overflow", "inflation"},
     "on overflow, prune every counter with count<=2 -- clears transient bursts "
     "(which a halving-inherit would otherwise inflate into the top-k) and keeps "
     "the summary small"),
    (PRE, "prune_ones", _PRE_PRUNE_ONES,
     {"transient", "decoy", "overflow", "collision"},
     "on overflow, drop ALL count==1 transients to free room for real hitters"),
    (VICTIM, "min_or_one", _VIC_MIN_OR_ONE,
     {"transient", "overflow", "collision", "late"},
     "evict a count==1 TRANSIENT first when one exists, protecting built-up "
     "heavy counters from being knocked out by churn"),
    (INHERIT, "keep34", "max(1, vv - vv // 4)", {"late", "gap"},
     "gentler decay: an admitted item keeps 3/4 of the victim's count, so a "
     "LATE-arriving heavy hitter is not under-counted into eviction"),
]


def _causal_failure_tags(guidance: dict) -> set:
    """v6: turn the CAUSAL guidance (the dist + missed items of a concrete failure)
    into op-target tags -- the PRIMARY driver of what to propose. A bursty/decoy/
    spike failure means transients are being inflated into the top-k -> tags that
    select threshold-pruning + no-inflation ops."""
    tags: set = set()
    causal = guidance.get("causal")
    dist = ""
    if causal is not None:
        dist = (getattr(causal, "dist", "") or "").lower()
        tags |= set(getattr(causal, "tags", set()) or set())
    for needle, tag in (("decoy", "decoy"), ("spike", "spike"),
                        ("comeback", "transient"), ("mixture", "collision"),
                        ("collision", "collision"), ("overflow", "overflow"),
                        ("prop:", "transient"), ("param", "late"),
                        ("late", "late"), ("front", "late")):
        if needle in dist:
            tags.add(tag)
    # A burst/decoy/spike failure is fundamentally a count-INFLATION failure.
    if tags & {"decoy", "spike", "transient", "collision", "overflow"}:
        tags.add("inflation")
    return tags


def propose_operations(vocab: Vocabulary, guidance: dict, rng=None
                       ) -> list[OpProposal]:
    """Deterministic Builder op-proposer. v6: the PRIMARY signal is CAUSAL guidance
    (the actual failing workload + the heavy hitters the best policy missed); the
    keyword `weak` tags + frontier gap are only a fallback. Emit the new ops whose
    target tags match the diagnosed failure; if nothing matches, emit the spread.
    (Aggressive: propose broadly and let post-grounding utility + the frontier
    filter, rather than pre-filtering hard.)"""
    causal_tags = _causal_failure_tags(guidance)
    weak = set(guidance.get("weak", [])) | causal_tags
    if guidance.get("frontier_gap", 0.0) > 0:
        weak.add("gap")
    matched, general = [], []
    for (kind, name, frag, tags, note) in _OP_TEMPLATES:
        if vocab.has_op(kind, name):
            continue
        prop = OpProposal(kind, name, frag, Provenance.PROPOSED_DET, note)
        (matched if (tags & weak) else general).append(prop)
    # v6 higher volume: when a concrete failure was diagnosed, propose the matched
    # ops AND keep a couple of general ones so the frontier has more to filter.
    return (matched + general[:2]) if matched else general


def human_baseline_policies(vocab: Vocabulary) -> list[tuple]:
    """v6: the policies a HUMAN-only vocabulary can build, for the HUMAN Pareto
    frontier the utility flag is measured against. To be HONEST, this is the BROAD
    human op-combo space (not just the curated v4 genome) -- so a proposed op only
    earns `beats_human` if it beats the best policy a human could write from the
    seed ops, including strong combos like min/one that the genome omits. Returns
    (label, code)."""
    out = []
    # min-victim across the human inheritance rules at the capacities that span the
    # frontier (incl. cap 8 for the low-memory end where a prune-op would compete,
    # and the strong min/one combo the curated genome omits).
    for inh in ("one", "vplus", "vhalf"):
        for cap in (8, 16, 32):
            out.append((f"min_{inh}_none_{cap}",
                        _assemble_policy("min", inh, "none", cap)))
    # the human decay pre-steps (incl. cap 8 -- the CHEAPEST human policies, which
    # reach the low-memory region a prune-op targets, so the utility comparison is
    # honest there) + second-min victim (a distinct human eviction discipline).
    for cap in (8, 16, 32):
        out.append((f"min_v_decall_{cap}",
                    _assemble_policy("min", "v", "dec_all", cap)))
    out.append(("min_one_decall_8", _assemble_policy("min", "one", "dec_all", 8)))
    out.append(("secmin_vplus_none_32",
                _assemble_policy("second_min", "vplus", "none", 32)))
    return out


def human_default_policy(vocab: Vocabulary, cap: int) -> str:
    """v6 self-play baseline: the v4 INCUMBENT default policy at a given capacity --
    min eviction, halved inheritance, no pre-step (the inheritance rule v4 shipped).
    A self-play 'win' means the proposed op beats WHAT V4 SHIPPED on an Adversary
    generator -- improvement-over-incumbent. NB: this is NOT the best human op:
    against min/one (an equally-human inheritance that does not inflate transients)
    the strong proposed ops merely TIE on recall -- their real edge is memory (the
    utility flag, measured vs the full human Pareto frontier)."""
    return assemble_policy_from(vocab, "min", "vhalf", "none", cap)


def op_probe(vocab: Vocabulary, proposal: OpProposal) -> str:
    """A probe policy that USES the proposed op (safe human ops in the other two
    slots), for the engine to AST-screen + ground before admitting the op."""
    vlines = proposal.fragment if proposal.kind == VICTIM else _VICTIM_OPS["min"]
    iexpr = proposal.fragment if proposal.kind == INHERIT else _INHERIT_OPS["vplus"]
    plines = proposal.fragment if proposal.kind == PRE else _PRE_OPS["none"]
    return _assemble_core(vlines, iexpr, plines, 16)


def admit_op(vocab: Vocabulary, proposal: OpProposal, iteration: int) -> None:
    vocab.add_op(OpEntry(proposal.kind, proposal.name, proposal.fragment,
                         proposal.origin, iteration, proposal.note))


# --- the Adversary's deterministic GENERATOR meta-proposer ---------------- #
# New stream STRUCTURES (not gen:-DSL parameterisations), each a function body over
# (n, rng) returning a list[int]. AST-screened + dry-run-grounded before use.
_GEN_COMEBACK = (
    "heavy = list(range(8))\n"
    "out = []\n"
    "lo = n // 3\n"
    "hi = 2 * n // 3\n"
    "for i in range(n):\n"
    "    if lo <= i < hi:\n"
    "        out.append(rng.randint(10000, 10000 + n))\n"
    "    else:\n"
    "        out.append(rng.choice(heavy) if rng.random() < 0.7 "
    "else rng.randint(10000, 10000 + n))\n"
    "return out"
)
_GEN_DECOY = (
    "true_heavy = list(range(5))\n"
    "out = []\n"
    "decoy = 1000\n"
    "while len(out) < n:\n"
    "    r = rng.random()\n"
    "    if r < 0.30:\n"
    "        out.append(rng.choice(true_heavy))\n"
    "    elif r < 0.55:\n"
    "        out.extend([decoy] * rng.randint(20, 60))\n"
    "        decoy += 1\n"
    "    else:\n"
    "        out.append(rng.randint(5000, 5000 + n))\n"
    "return out[:n]"
)
_GEN_TEMPLATES = [
    ("comeback", _GEN_COMEBACK,
     "heavy hitters dominate the ends but VANISH for the middle third -- punishes "
     "summaries that forget (decay/halving) and cannot recover a lapsed count"),
    ("decoy", _GEN_DECOY,
     "many DECOY items each spike briefly then go silent, churning out the "
     "steadily-accumulating true hitters under min-eviction"),
]


def propose_generators(vocab: Vocabulary, guidance: dict, rng=None
                       ) -> list[GenProposal]:
    """Deterministic Adversary generator-proposer: new stream structures the
    `gen:` DSL cannot express. Skips any already admitted."""
    return [GenProposal(name, body, Provenance.PROPOSED_DET, note)
            for (name, body, note) in _GEN_TEMPLATES if not vocab.has_gen(name)]


def make_generator_block(name: str, body: str) -> str:
    """Wrap a proposed generator BODY into a `_genprop_<name>(n, rng)` definition
    that self-registers, to be appended to the live spec's workload_gen_src."""
    lines = [f"def _genprop_{name}(n, rng):"]
    for ln in body.splitlines():
        lines.append(("    " + ln) if ln.strip() else "")
    lines.append(f'_GENPROP_REGISTRY["{name}"] = _genprop_{name}')
    return "\n".join(lines) + "\n"


def generator_block(proposal: GenProposal) -> str:
    return make_generator_block(proposal.name, proposal.body)


def generator_probe_spec(name: str, k: int = 10) -> dict:
    """A workload spec to validate a proposed generator produces a usable stream.
    Probed at the SAME n it will be ADMITTED/run at (40000), so a generator that is
    fine at small n but raises or allocates unbounded at scale is caught at the gate
    rather than after it has poisoned the live suite."""
    return {"name": name, "n": 40000, "k": k, "dist": f"prop:{name}"}


def admit_generator(vocab: Vocabulary, proposal: GenProposal, iteration: int,
                    k: int = 10) -> GenEntry:
    block = make_generator_block(proposal.name, proposal.body)
    entry = GenEntry(name=proposal.name, dist=f"prop:{proposal.name}",
                     origin=proposal.origin, src_block=block,
                     iteration=iteration, note=proposal.note)
    vocab.add_gen(entry)
    return entry


# A trusted probe solver (exact counter) the engine measures to confirm a proposed
# generator yields a runnable stream (no timing/memory -- just "does it produce a
# stream the harness can score").
PROBE_SOLVER = STRATEGIES["exact_counter"].code


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
)

DOMAIN = Domain(
    name="heavy_hitters",
    problem_desc=PROBLEM_DESC,
    contract=CONTRACT,
    grounding_spec=GROUNDING_SPEC,
    classify=classify,
    seed_strategy=SEED_STRATEGY,
    strategies=STRATEGIES,
    builder_priority=BUILDER_PRIORITY,
    adversary_cases=[],                     # correctness arsenal not used here
    compose=compose_bounded_hybrid,
    regime=regime,
    explain_failure=explain_failure,
    uses_frontier=True,
    memory_budget_frac=0.5,
    meaningful_classes=MEANINGFUL_CLASSES,
    adversary_workloads=ADVERSARY_WORKLOADS,
    synthesize_programs=synthesize_programs,    # v4: eviction-policy genome
    adversary_param_attacks=adversary_param_attacks,  # v3: generative adversary
    adversary_generators=adversary_generators,  # v4: generator-level adversary
    # v5: open vocabulary -- propose new ops (Builder) + generators (Adversary).
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
    human_baseline_policies=human_baseline_policies,   # v6: utility baseline
    human_default_policy=human_default_policy,         # v6: self-play baseline
)
