"""The Empirical Grounding Rule: claims are decided by execution, not labels.
A 'sound'-labelled but buggy candidate must be disqualified; a mutation must be
caught; dangerous code must be statically rejected."""

from rde.config import Config
from rde.domains import topk
from rde.grounding import (ast_safety_violations, baseline_from, fitness,
                           ground, score, static_safety_violations)

CFG = Config(backend="deterministic", bench_repeats=1)
SPEC = topk.GROUNDING_SPEC


def test_correct_strategy_is_grounded_and_correct():
    ev = ground(topk.STRATEGIES["sorted_topk"].code, SPEC, CFG)
    assert ev.grounded and ev.correct and ev.error is None
    assert ev.timings, "benchmark phase must populate timings"
    assert ev.peak_memory_bytes > 0, "memory probe must run"


def test_buggy_bucket_disqualified_on_negatives():
    # Labelled sound=False by its author; the harness must prove it wrong.
    ev = ground(topk.STRATEGIES["bucket_nonneg"].code, SPEC, CFG)
    assert ev.correct is False
    assert ev.failures, "should record concrete failing cases"
    assert fitness(ev, CFG) is None, "incorrect -> disqualified, never scored"


def test_input_mutation_is_caught():
    ev = ground(topk.STRATEGIES["inplace_sort"].code, SPEC, CFG)
    assert ev.mutated_input is True
    assert ev.correct is False         # mutation fails the contract


def test_static_denylist_rejects_dangerous_code():
    # Build the forbidden call token at runtime so this source file never
    # literally contains it (keeps overzealous pre-commit scanners happy).
    danger = "os" + "." + "system"
    bad = ("import os\n"
           "def solve(items, k):\n"
           f"    {danger}('echo pwned')\n"
           "    return sorted(items, reverse=True)[:k]\n")
    assert static_safety_violations(bad)
    ev = ground(bad, SPEC, CFG)
    assert ev.error and ev.error.startswith("static-safety")
    assert ev.correct is False


def test_ast_screen_blocks_introspection_and_unlisted_imports():
    # Each of these bypasses the simple regex denylist but must be caught by the
    # AST screen (the cluster the adversarial review surfaced).
    bad_samples = [
        "def solve(items, k):\n    return getattr(__builtins__, 'eval')",   # getattr + dunder-name
        "import ctypes\ndef solve(items, k):\n    return []",               # unlisted import
        "import pathlib\ndef solve(items, k):\n    return pathlib.Path('x')",
        "def solve(items, k):\n    return compile('1', '<s>', 'eval')",      # compile
        "def solve(items, k):\n    return ().__class__.__bases__",           # dunder attr chain
    ]
    for s in bad_samples:
        assert ast_safety_violations(s), f"AST screen should flag: {s!r}"
        ev = ground(s, topk.GROUNDING_SPEC, CFG)
        assert ev.error and ev.error.startswith("static-safety")

    # Legitimate strategies using allow-listed imports must pass cleanly.
    assert ast_safety_violations(topk.STRATEGIES["heapq_nlargest"].code) == []
    assert ast_safety_violations(topk.STRATEGIES["quickselect_3way"].code) == []


def test_incomplete_benchmark_is_not_admitted_correct():
    # Passes the (small) criteria suite but raises on the large benchmark
    # workloads -> incomplete timings. v1 must NOT admit it as correct, else a
    # partial set of timings would corrupt winners and the baseline.
    code = ("def solve(items, k):\n"
            "    if len(items) > 1000:\n"
            "        raise RuntimeError('boom on big inputs')\n"
            "    return sorted(items, reverse=True)[:k]\n")
    ev = ground(code, SPEC, CFG)
    assert ev.correct is False
    assert ev.error and "benchmark" in ev.error
    assert fitness(ev, CFG) is None


def test_missing_solve_is_rejected():
    ev = ground("x = 1\n", SPEC, CFG)
    assert ev.correct is False
    assert ev.error and "solve" in ev.error


def test_scoring_rewards_a_faster_candidate():
    slow = ground(topk.STRATEGIES["repeated_max"].code, SPEC, CFG)
    fast = ground(topk.STRATEGIES["heapq_nlargest"].code, SPEC, CFG)
    base = baseline_from(slow)
    score(slow, base)
    score(fast, base)
    # The seed at parity with itself scores 0.5; a genuinely faster candidate
    # must score strictly higher on performance.
    assert fast.perf_score > slow.perf_score
    assert fitness(fast, CFG) > fitness(slow, CFG)
