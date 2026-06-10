"""Self-contained example + smoke test for grounded-code-execution.

Problem: return the k most-frequent items of a stream (streaming heavy-hitters).
Quality is recall@k vs an exact Counter oracle. Shows the full path: a custom
oracle + generator + verdict, grounding an exact solver and an approximate one,
measuring quality on a hostile workload, and diagnosing a failure.
"""
import inspect
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from grounding import (GroundingSpec, ground, measure_quality_on, diagnose)


# --- the problem definition (these get spliced into the subprocess as SOURCE) --- #
def oracle(items, k):
    from collections import Counter
    c = Counter(items)
    return [it for it, _ in sorted(c.items(), key=lambda kv: (-kv[1], -kv[0]))[:k]]


def gen_workload(n, dist, seed):
    import random
    rng = random.Random(seed)
    if dist == "zipf":
        D = max(n // 4, 50)
        return [int(D * (rng.random() ** 3)) for _ in range(n)]
    if dist == "uniform":
        return [rng.randint(0, max(n // 2, 1)) for _ in range(n)]
    if dist == "few_heavy":
        heavy = list(range(10))
        return [rng.choice(heavy) if rng.random() < 0.6 else rng.randint(100, 100 + n)
                for _ in range(n)]
    if dist == "many_heavy":
        # 200 moderately-frequent items -> overflows a SMALL bounded summary while a
        # large one (or exact) recovers them. A valid adversary workload: a GOOD
        # method succeeds, so low recall signals under-capacity, not inherent hardness.
        heavy = list(range(200))
        return [rng.choice(heavy) if rng.random() < 0.85 else rng.randint(2000, 2000 + n)
                for _ in range(n)]
    return [rng.randint(0, 9) for _ in range(n)]


def verdict(got, expected, items, k):
    if len(got) != len(set(got)) or any(x not in items for x in got):
        return (False, 0.0)          # must return distinct items that occur
    if len(got) > k:
        return (False, 0.0)
    hit = len(set(got) & set(expected))
    return (True, hit / len(expected) if expected else 1.0)


SPEC = GroundingSpec(
    oracle_src=inspect.getsource(oracle),
    workload_gen_src=inspect.getsource(gen_workload),
    criteria_cases=[("tiny", [1, 1, 1, 2, 2, 3], 2), ("empty", [], 3),
                    ("ties", [1, 1, 2, 2, 3], 2)],
    workloads=[{"name": "zipf", "n": 8000, "k": 10, "dist": "zipf"},
               {"name": "few_heavy", "n": 8000, "k": 10, "dist": "few_heavy"}],
    mem_workload={"name": "mem", "n": 8000, "k": 10, "dist": "uniform"},
    verdict_src=inspect.getsource(verdict),
)

# --- candidate solvers --- #
EXACT = (
    "from collections import Counter\n"
    "def solve(stream, k):\n"
    "    c = Counter(stream)\n"
    "    return [it for it,_ in sorted(c.items(), key=lambda kv:(-kv[1],-kv[0]))[:k]]\n")

# An approximate, bounded-memory Space-Saving-style solver.
APPROX = (
    "def solve(stream, k):\n"
    "    m = max(8 * k, 64); counts = {}\n"
    "    for x in stream:\n"
    "        if x in counts: counts[x] += 1\n"
    "        elif len(counts) < m: counts[x] = 1\n"
    "        else:\n"
    "            vk = min(counts, key=counts.get); vv = counts.pop(vk)\n"
    "            counts[x] = vv + 1\n"
    "    return [it for it,_ in sorted(counts.items(), key=lambda kv:(-kv[1],-kv[0]))[:k]]\n")

UNSAFE = "import os\ndef solve(stream, k):\n    return os.listdir('.')[:k]\n"

if __name__ == "__main__":
    ev_exact = ground(EXACT, SPEC, bench_repeats=1)
    print("EXACT :", ev_exact.summary())
    assert ev_exact.correct and ev_exact.quality > 0.99

    ev_approx = ground(APPROX, SPEC, bench_repeats=1)
    print("APPROX:", ev_approx.summary(),
          "(bounded memory, approximate recall)")
    assert ev_approx.correct
    assert ev_approx.peak_memory_bytes < ev_exact.peak_memory_bytes  # cheaper

    bad = ground(UNSAFE, SPEC, bench_repeats=1)
    print("UNSAFE:", bad.error)
    assert not bad.correct and "static-safety" in bad.error

    # Measure recall on a hostile high-cardinality stream (no benchmark).
    q = measure_quality_on(APPROX, [{"name": "hostile", "n": 8000, "k": 10,
                                     "dist": "uniform"}], SPEC, bench_repeats=1)
    print("APPROX recall on hostile uniform stream:", q.get("hostile"))

    # Diagnose what the approx solver missed on the hostile stream.
    d = diagnose(APPROX, {"name": "hostile", "n": 8000, "k": 10, "dist": "uniform"},
                 SPEC)
    print("missed (first few):", d.get("missed", [])[:5])
    print("SMOKE_OK")
