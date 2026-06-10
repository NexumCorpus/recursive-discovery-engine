"""Domain: Top-K selection.

Problem: given a list of mutually comparable items and an integer k, return the k
largest items in non-increasing order.

Why this domain (it exercises EVERY protective mechanism in the spec):
  * Genuinely different algorithmic *classes* (full-sort O(n log n), bounded-heap
    O(n log k), quickselect O(n) avg, bucket O(n) bounded) -> "novelty vs
    recombination" is a real distinction, not cosmetic.
  * An airtight brute-force oracle -> every correctness claim is falsifiable.
  * A rich adversarial surface (all-equal, presorted, k>=n, input mutation,
    negative/float inputs that break range assumptions).
  * A famous empirical trap: the asymptotically-superior *pure-Python*
    quickselect tends to LOSE on wall-clock to the naive-but-C-backed builtins.
    The engine can only learn that by *measuring* -- which is the whole point.
"""

from __future__ import annotations

import inspect
import random
import re

from ..types import AlgoClass
from ..grounding import GroundingSpec
from ..domain import Domain, Strategy


# --------------------------------------------------------------------------- #
# The contract (fed verbatim to the LLM Builder; enforced by the trusted suite)
# --------------------------------------------------------------------------- #

CONTRACT = """\
Implement a single top-level function with EXACTLY this signature:

    def solve(items, k):
        ...

Semantics (preconditions: k is an int with k >= 0; items is a list of mutually
comparable values):
  * Return the k largest items, in NON-INCREASING (descending) order.
  * If k == 0            -> return [].
  * If k >= len(items)   -> return ALL items, sorted descending.
  * Duplicates are kept (top-k by value, with multiplicity).
  * MUST NOT mutate the input list `items`.

You may import from the standard library (e.g. heapq, random). Define `solve`
at module top level. Do not print anything.
"""

PROBLEM_DESC = (
    "Top-K selection: return the k largest items of a list in descending order. "
    "Competing approaches trade off correctness, wall-clock performance across "
    "input distributions, peak memory, and code simplicity."
)


# --------------------------------------------------------------------------- #
# Oracle -- the single source of truth for correctness
# --------------------------------------------------------------------------- #

def oracle(items, k):
    """Reference implementation: intentionally the simplest possible correct
    code. The grounding subprocess embeds an identical copy (see ORACLE_SRC)."""
    return sorted(items, reverse=True)[:k]


# --------------------------------------------------------------------------- #
# Deterministic workload generation (shared by harness + subprocess)
# --------------------------------------------------------------------------- #

def gen_workload(n, dist, seed):
    """Build a reproducible benchmark input from a (size, distribution, seed)."""
    rng = random.Random(seed)
    if dist == "random":
        return [rng.randint(0, 1_000_000) for _ in range(n)]
    if dist == "ascending":
        return list(range(n))
    if dist == "descending":
        return list(range(n, 0, -1))
    if dist == "all_equal":
        return [42] * n
    if dist == "few_unique":
        return [rng.randint(0, 9) for _ in range(n)]
    if dist == "floats":
        return [rng.random() * 1_000_000 for _ in range(n)]
    raise ValueError("unknown dist: " + dist)


# Source text of the two helpers above, embedded verbatim into the isolated
# subprocess runner. Derived via inspect so there is exactly ONE definition of
# each and no risk of the in-process and subprocess copies drifting apart.
ORACLE_SRC = inspect.getsource(oracle)
WORKLOAD_GEN_SRC = inspect.getsource(gen_workload)


# Benchmark workloads. v1 deliberately spans BOTH a small-k regime (where a
# bounded heap wins) and a large-k regime (where a full sort wins) -- measured on
# the target machine. Because no single strategy wins both, an adaptive
# dispatcher (the synthesis target) can genuinely beat every pure approach, and
# the split shows up objectively as different elites owning different workloads.
# NB the large-k regimes use a smaller n: the naive O(n*k) seed must survive
# benchmarking on every workload, and n*k blows the subprocess timeout if both are
# large. Smaller n keeps the seed runnable while still making the large-k regime
# contribute enough wall-clock that the heap-vs-sort split is real and the hybrid
# win is meaningful (verified by micro-benchmark).
# Balanced 3 small-k : 3 large-k so neither regime dominates the per-workload
# perf average -- the hybrid that wins BOTH should clearly out-score a pure
# strategy that wins only one.
WORKLOADS = [
    {"name": "random_k10",   "n": 40000, "k": 10,   "dist": "random"},      # heap
    {"name": "random_k50",   "n": 40000, "k": 50,   "dist": "random"},      # heap
    {"name": "floats_k10",   "n": 40000, "k": 10,   "dist": "floats"},      # heap
    {"name": "random_klarge","n": 4000,  "k": 2000, "dist": "random"},      # sort
    {"name": "random_kmid",  "n": 4000,  "k": 3000, "dist": "random"},      # sort
    {"name": "random_kfull", "n": 4000,  "k": 4000, "dist": "random"},      # sort
]


# --------------------------------------------------------------------------- #
# Trusted criteria suite -- the harness's OWN eyes (independent of both roles)
# --------------------------------------------------------------------------- #
# This is the structural defense against *shared blind spots*: an approach only
# advances if it passes this fixed battery, which neither the Builder nor the
# Adversary authored. Each case is (name, items, k).

def _rng_list(seed, n, lo=-1000, hi=1000):
    r = random.Random(seed)
    return [r.randint(lo, hi) for _ in range(n)]

CRITERIA_CASES = [
    ("empty_k0",          [],                         0),
    ("empty_k3",          [],                         3),    # k > len -> []
    ("single_k1",         [9],                        1),
    ("single_k3",         [9],                        3),    # k > len
    ("k0_nonempty",       [3, 1, 2],                  0),    # -> []
    ("all_dupes",         [5] * 10,                   3),
    ("ascending",         list(range(20)),            5),
    ("descending",        list(range(19, -1, -1)),    5),
    ("negatives",         [-5, -1, -3, -2, -4],       2),
    ("floats_mixed",      [1.5, -2.0, 3.25, 0.0],     2),
    ("k_eq_len",          [4, 2, 7, 1],               4),
    ("k_eq_len_minus1",   [4, 2, 7, 1],               3),
    ("tie_at_boundary",   [5, 4, 3, 3, 2],            3),    # one of the 3s
    ("medium_random",     _rng_list(7, 200),          17),
    ("near_dupes",        [1, 1, 2, 2, 3, 3],         4),
]


# --------------------------------------------------------------------------- #
# Adversary arsenal -- deterministic attack generators
# --------------------------------------------------------------------------- #
# The deterministic Adversary throws these at a candidate; the grounding harness
# RUNS them against the oracle and only credits *confirmed* discrepancies.

ADVERSARY_CASES = [
    ("attack_negatives",      [-5, -1, -3, -2, -4, -100, -7],  3),
    ("attack_floats",         [1.5, 2.5, 0.5, -3.3, 9.9],      2),
    ("attack_all_equal_big",  [7] * 800,                       4),
    ("attack_k_eq_len",       [3, 1, 2, 5, 4],                 5),
    ("attack_k_gt_len",       [3, 1, 2],                       9),
    ("attack_k_zero",         [3, 1, 2],                       0),
    ("attack_single",         [9],                             1),
    ("attack_empty",          [],                              3),
    ("attack_presorted_desc", list(range(300, 0, -1)),         5),
    ("attack_dup_boundary",   [5, 4, 3, 3, 3, 2, 1],           4),
    ("attack_mixed_sign",     [-2, 0, 2, -1, 1, 3, -3],        3),
]


# --------------------------------------------------------------------------- #
# Strategy library -- the deterministic Builder's repertoire
# --------------------------------------------------------------------------- #
# Each entry is a concrete, runnable implementation. Some are deliberately flawed
# so the *grounding gate* and the *Adversary* have something real to catch. The
# engine never trusts these labels: correctness/perf are decided by execution.
# (Strategy is defined once in rde.domain and shared across domains.)

STRATEGIES: dict[str, Strategy] = {
    # The deliberately-naive SEED. Correct, but O(n*k): re-scans for each pick.
    "repeated_max": Strategy(
        "repeated_max", AlgoClass.OTHER,
        '''
def solve(items, k):
    arr = list(items)              # copy: do not mutate the caller's list
    out = []
    for _ in range(min(k, len(arr))):
        m = max(arr)
        arr.remove(m)
        out.append(m)
    return out
''',
        "Naive first attempt: repeatedly take the running max. O(n*k).",
        sound=True),

    # The clumsy-but-reasonable SEED baseline (v1): a full C-sort followed by a
    # deliberately Python-level extraction loop. O(n log n) -- non-pathological on
    # every workload (so perf scores don't saturate the way the O(n*k) seed did),
    # yet beatable everywhere: a heap wins small-k, a clean full-sort wins large-k,
    # and an adaptive hybrid wins both.
    "naive_select": Strategy(
        "naive_select", AlgoClass.OTHER,
        '''
def solve(items, k):
    arr = list(items)
    arr.sort()                       # ascending, C-level
    k = min(k, len(arr))
    out = []
    for i in range(k):               # Python-level extraction (intentionally naive)
        out.append(arr[len(arr) - 1 - i])
    return out
''',
        "Naive baseline: full sort, then a Python loop to pull the top k.",
        sound=True),

    # Full-sort. Correct, dead simple, C-backed Timsort.
    "sorted_topk": Strategy(
        "sorted_topk", AlgoClass.FULL_SORT,
        '''
def solve(items, k):
    return sorted(items, reverse=True)[:k]
''',
        "Sort everything descending, slice the first k. O(n log n), C-level.",
        sound=True),

    # Bounded heap via the C-accelerated builtin. Correct, O(n log k).
    "heapq_nlargest": Strategy(
        "heapq_nlargest", AlgoClass.HEAP,
        '''
import heapq
def solve(items, k):
    return heapq.nlargest(k, items)
''',
        "Bounded heap of size k via heapq.nlargest. O(n log k).",
        sound=True),

    # Trivial variation of heapq_nlargest -> should be flagged RECOMBINATION.
    "nlargest_wrapped": Strategy(
        "nlargest_wrapped", AlgoClass.HEAP,
        '''
import heapq
def solve(items, k):
    result = heapq.nlargest(k, items)
    return list(result)
''',
        "Same as nlargest but wraps the result in list(). (Cosmetic variation.)",
        sound=True),

    # Genuine new class: 3-way quickselect (Dutch-flag, duplicate-robust).
    "quickselect_3way": Strategy(
        "quickselect_3way", AlgoClass.SELECTION,
        '''
import random
def solve(items, k):
    n = len(items)
    if k <= 0:
        return []
    if k >= n:
        return sorted(items, reverse=True)
    arr = list(items)
    target = n - k                 # arr[target:] will hold the k largest
    lo, hi = 0, n - 1
    while lo < hi:
        pivot = arr[random.randint(lo, hi)]
        lt, gt, i = lo, hi, lo     # 3-way partition (handles duplicates in O(n))
        while i <= gt:
            if arr[i] < pivot:
                arr[lt], arr[i] = arr[i], arr[lt]; lt += 1; i += 1
            elif arr[i] > pivot:
                arr[gt], arr[i] = arr[i], arr[gt]; gt -= 1
            else:
                i += 1
        if target < lt:
            hi = lt - 1
        elif target > gt:
            lo = gt + 1
        else:
            break                  # target lands inside the equal block
    top = arr[target:]
    top.sort(reverse=True)
    return top
''',
        "Quickselect with 3-way partition: O(n) average, robust to duplicates.",
        sound=True),

    # Tempting O(n) bucket count -- but secretly assumes non-negative ints.
    # The grounding gate will DISQUALIFY it on negatives/floats.
    "bucket_nonneg": Strategy(
        "bucket_nonneg", AlgoClass.BUCKET,
        '''
def solve(items, k):
    if k <= 0 or not items:
        return []
    m = max(items)
    counts = [0] * (m + 1)         # BUG: breaks on negatives / floats
    for x in items:
        counts[x] += 1
    out = []
    v = m
    while v >= 0 and len(out) < k:
        c = counts[v]
        while c > 0 and len(out) < k:
            out.append(v)
            c -= 1
        v -= 1
    return out
''',
        "Counting/bucket sort: O(n) for bounded non-negative ints.",
        sound=False),

    # Correct values but MUTATES the caller's list -> fails the mutation check.
    "inplace_sort": Strategy(
        "inplace_sort", AlgoClass.FULL_SORT,
        '''
def solve(items, k):
    items.sort(reverse=True)       # BUG: mutates the caller's list in place
    return items[:k]
''',
        "Sort in place, slice k. Saves a copy -- but corrupts the input.",
        sound=False),
}

# Global preference order the deterministic Builder explores when unsteered.
# v1: one distinct algorithmic CLASS each, so the early rounds quickly populate
# the per-regime winners that synthesis then combines. (nlargest_wrapped stays in
# the library for the recombination tests but is off the default exploration path.)
BUILDER_PRIORITY = [
    "heapq_nlargest",
    "sorted_topk",
    "quickselect_3way",
    "bucket_nonneg",
]

SEED_STRATEGY = "naive_select"


def strategies_of_class(cls: AlgoClass) -> list[str]:
    return [n for n, s in STRATEGIES.items() if s.algo_class == cls]


# --------------------------------------------------------------------------- #
# Algorithmic classifier -- maps free-form (LLM-written) code to a family.
# --------------------------------------------------------------------------- #
# The deterministic strategies carry explicit AlgoClass labels; this is for the
# LLM Builder's code, which arrives unlabelled. The novelty/recombination check
# leans on the result: "same class + near-identical fingerprint == variation."

def classify_topk(code: str) -> AlgoClass:
    low = code.lower()
    has_heap = bool(re.search(r"\bnlargest\b|\bnsmallest\b|\bheappush\b|"
                              r"\bheapreplace\b|\bheappushpop\b|\bheapify\b", code))
    has_sort = bool(re.search(r"\bsorted\s*\(|\.sort\s*\(", code))
    branches_on_k = bool(re.search(r"\bk\s*[<>]=?", code))
    # Hybrid/adaptive: combines >= 2 families and dispatches (typically on k).
    if has_heap and has_sort and branches_on_k:
        return AlgoClass.HYBRID
    # Counting/bucket: an index-by-value table like `counts = [0] * (m + 1)`.
    if re.search(r"\[\s*0\s*\]\s*\*", code) and ("count" in low or "bucket" in low):
        return AlgoClass.BUCKET
    if has_heap:
        return AlgoClass.HEAP
    if "pivot" in low or "partition" in low or "quickselect" in low:
        return AlgoClass.SELECTION
    if has_sort:
        return AlgoClass.FULL_SORT
    return AlgoClass.OTHER


# --------------------------------------------------------------------------- #
# v1 hooks: hybrid composition, regime labels, structural failure explanation.
# --------------------------------------------------------------------------- #

def regime(workload_name: str) -> str:
    if "klarge" in workload_name or "kfull" in workload_name:
        return "large-k"
    return "small-k"


def explain_failure(approach) -> tuple | None:
    """Map a disqualified approach to a transferable structural lesson."""
    ev = approach.evidence
    if ev is None:
        return None
    if ev.mutated_input:
        return ("in-place mutation of input", "any input",
                "modifying `items` in place violates the no-mutation contract")
    if (approach.algo_class == AlgoClass.BUCKET
            or (re.search(r"\[\s*0\s*\]\s*\*", approach.code)
                and "count" in approach.code.lower())):
        return ("value-indexed count table", "negative / non-integer inputs",
                "indexing a counts array by value breaks on negatives and floats")
    return None


def compose_adaptive(parents) -> str | None:
    """Deterministic synthesis: if the parents include a HEAP winner and a
    FULL_SORT winner, emit ONE adaptive dispatcher that uses the heap for small k
    and the sort for large k -- a genuine hybrid that wins both regimes. Returns
    None if no meaningful hybrid is possible from the given parents.

    (This is the scaffolded *reference* of synthesis; the LLM Builder composes
    hybrids open-endedly. See README for the scaffolding trade-off.)"""
    classes = {p.algo_class for p in parents if p.is_correct}
    if not ({AlgoClass.HEAP, AlgoClass.FULL_SORT} <= classes):
        return None
    return (
        "import heapq\n"
        "def solve(items, k):\n"
        "    n = len(items)\n"
        "    if k <= 0:\n"
        "        return []\n"
        "    # learned: for large k a full C-sort beats a big heap;\n"
        "    #          for small k a bounded heap beats a full sort.\n"
        "    if k >= n // 2:\n"
        "        return sorted(items, reverse=True)[:k]\n"
        "    return heapq.nlargest(k, items)\n")


# --------------------------------------------------------------------------- #
# Grounding spec + Domain object -- the single handle the engine consumes.
# --------------------------------------------------------------------------- #

GROUNDING_SPEC = GroundingSpec(
    oracle_src=ORACLE_SRC,
    workload_gen_src=WORKLOAD_GEN_SRC,
    criteria_cases=CRITERIA_CASES,
    workloads=WORKLOADS,
    mem_workload={"name": "mem_probe", "n": 40000, "k": 50, "dist": "random"},
)

DOMAIN = Domain(
    name="topk",
    problem_desc=PROBLEM_DESC,
    contract=CONTRACT,
    grounding_spec=GROUNDING_SPEC,
    classify=classify_topk,
    seed_strategy=SEED_STRATEGY,
    strategies=STRATEGIES,
    builder_priority=BUILDER_PRIORITY,
    adversary_cases=ADVERSARY_CASES,
    compose=compose_adaptive,
    regime=regime,
    explain_failure=explain_failure,
)
