"""v3 open-ended synthesis: the compositional grammar invents programs absent
from the library, AST mutation perturbs structure, and the Adversary composes
novel parameterised workloads."""

import random

from rde.config import Config
from rde.domains import heavy_hitters as hh
from rde.grounding import ground
from rde.mutate import mutate
from rde.types import AlgoClass


# --------------------------------------------------------------------------- #
# Compositional grammar
# --------------------------------------------------------------------------- #

def test_grammar_produces_programs_absent_from_library():
    lib = {s.code.strip() for s in hh.STRATEGIES.values()}
    progs = hh.synthesize_programs()
    assert len(progs) >= 4
    labels = [label for label, _ in progs]
    # v4: the basis is an eviction-policy GENOME (composed from low-level ops).
    assert all(label.startswith("policy_") for label in labels)
    assert "policy_min_half32" in labels        # the v4-only winner is in the genome
    # Every composed program is NEW -- not byte-identical to a library strategy.
    for _label, code in progs:
        assert code.strip() not in lib


def test_synthesised_policy_classified_as_invented():
    code = dict(hh.synthesize_programs())["policy_min_half32"]
    assert hh.classify(code) == AlgoClass.INVENTED
    # Its core move (evict the min-count item, inherit a TRANSFORMED count) is in NO
    # library strategy -- the engine assembled it from low-level ops.
    assert "min(counts" in code and "pop(" in code
    for s in hh.STRATEGIES.values():
        assert "min(counts" not in s.code


def test_composed_program_grounds_correct_with_recall():
    # Run the invented program through the REAL grounding harness (subprocess +
    # oracle), not inline -- it must be well-formed and recover heavy hitters.
    code = dict(hh.synthesize_programs())["policy_min_half32"]
    ev = ground(code, hh.GROUNDING_SPEC, Config(bench_repeats=1))
    assert ev.correct                       # passes the well-formedness gate
    assert ev.quality > 0.5                 # decent recall on skewed streams
    assert ev.peak_memory_bytes < 300 * 1024    # bounded memory (<< exact)


# --------------------------------------------------------------------------- #
# AST mutation
# --------------------------------------------------------------------------- #

def test_mutate_changes_a_constant_and_stays_valid():
    code = "def solve(stream, k):\n    m = 8 * k\n    return stream[:m]\n"
    res = mutate(code, random.Random(0))
    assert res is not None
    mutated, desc = res
    assert mutated != code
    compile(mutated, "<mut>", "exec")       # still parses
    assert "->" in desc                     # records what it changed


def test_mutate_returns_none_when_nothing_mutable():
    res = mutate("def solve(stream, k):\n    return []\n", random.Random(0))
    assert res is None


# --------------------------------------------------------------------------- #
# Generative Adversary
# --------------------------------------------------------------------------- #

def test_adversary_composes_parameterised_workloads():
    specs = hh.adversary_param_attacks(10)
    assert len(specs) >= 3
    fixed = {w["name"] for w in hh.ADVERSARY_WORKLOADS}
    for s in specs:
        assert s["dist"].startswith("param:")   # runtime-composed, not named
        assert s["name"] not in fixed


def test_param_parser_hardened_against_hostile_input():
    # Huge / non-finite values and unknown keys must be clamped/ignored, never
    # exhaust memory or crash the (subprocess) generator.
    s = hh.gen_workload(500, "param:heavy=1e100;late=nan;coll=inf;evil=9", seed=1)
    assert len(s) == 500                       # heavy clamped, NaN/inf rejected
    s2 = hh.gen_workload(500, "param:heavy=abc;late=2.0", seed=1)
    assert len(s2) == 500                       # malformed ignored, late clamped


def test_param_stream_generates_late_arriving_heavy_hitters():
    # late=0.9 -> the heavy hitters appear only in the last ~10% of the stream.
    stream = hh.gen_workload(2000, "param:late=0.9;heavy=5;coll=0.0", seed=1)
    assert len(stream) == 2000
    heavy = set(range(5))
    assert not (heavy & set(stream[:1000]))     # heavy absent early
    assert heavy & set(stream[1000:])           # present late
