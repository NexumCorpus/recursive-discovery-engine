"""v9: ADAPTIVE PRIMITIVES (bounded aux state in the trusted driver) +
COMPETITIVE-RATIO quality + a STANDARDISED WEIGHTED benchmark + a GENERIC
trusted-driver gate.

The v9 thesis these tests pin: no STATIC eviction score robustly dominates across
loop/drift/skew regimes; a policy with a tiny trusted aux state (EWMA of the hit
flag) that detects regimes DURING the trace beats every static human -- and the v8
static champion -- on the declared weighted competitive-ratio benchmark, while every
v8 soundness property (unfakeable metric, degrade-only hardening) is preserved.
"""

import ast
import dataclasses

from rde.config import Config
from rde.domains import cache_eviction as ce
from rde.engine import Engine
from rde.grounding import (GroundingSpec, candidate_ok, defines_driver_symbols,
                           driver_symbols_ok, ground, measure_recall_on)
from rde.roles import make_roles
from rde.types import AlgoClass

CFG = Config(bench_repeats=1)
SPEC = ce.GROUNDING_SPEC
LOOP = [{"name": "adv_loop", "n": 4000, "k": 10, "dist": "loop"}]
_BASE = ("def _evict_score(la, fr, ins, t):\n    return la\n"
         "def _on_hit(la, fr, ins, t):\n    return [t, fr + 1, ins]\n")
AD_EXPR = "(-la) if aux[0] < 0.08 else (la + 30 * fr.bit_length())"


def _wmean(code):
    bs = ce.BENCHMARK_SUITE
    rec = measure_recall_on(code, bs, SPEC, CFG)
    num = sum(rec[w["name"]] * w["weight"] for w in bs if w["name"] in rec)
    den = sum(w["weight"] for w in bs if w["name"] in rec)
    return (num / den if den else 0.0), rec


# --------------------------------------------------------------------------- #
# A. Aux state: strictly bounded, auditable, degrade-only -- and still unfakeable
# --------------------------------------------------------------------------- #

def test_adaptive_without_aux0_is_rejected():
    code = _BASE + "def _on_access(aux, hit, t):\n    return [0.5]\nCMUL = 2\n"
    assert not ground(code, SPEC, CFG).correct


def test_oversized_aux0_is_rejected():
    code = (_BASE + "AUX0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]\n"
            "def _on_access(aux, hit, t):\n    return list(aux)\nCMUL = 2\n")
    assert not ground(code, SPEC, CFG).correct      # 9 > the 8-float bound


def test_garbage_on_access_degrades_never_inflates():
    code = (_BASE + "AUX0 = [0.5]\n"
            "def _on_access(aux, hit, t):\n    return 'junk'\n"
            "def _evict_score_aux(la, fr, ins, t, aux):\n    return la\nCMUL = 2\n")
    ev = ground(code, SPEC, CFG)
    # malformed aux returns are RESET to AUX0: the run completes and quality stays
    # a valid bounded competitive ratio (garbage can only degrade the policy).
    assert ev.correct and 0.0 <= ev.quality <= 1.0


def test_adaptive_candidate_cannot_fake_the_metric():
    # An adaptive candidate that ALSO ships its own solve returning a sub-OPT (so
    # verdict-acceptable) fabricated rate: the trusted driver still discards it and
    # records the HONEST adaptive result on the loop (~0.5 CR, not ~0.94).
    ad = ce._assemble_adaptive(AD_EXPR, 2)
    cheat = ad + "\ndef solve(trace, k):\n    return (0.45, max(2 * k, 4))\n"
    rate = measure_recall_on(cheat, LOOP, SPEC, CFG).get("adv_loop", 1.0)
    assert 0.3 < rate < 0.75, f"want the honest adaptive CR (~0.5), got {rate}"


def test_aux_is_passed_immutably():
    # _on_access receives aux as a TUPLE; in-place mutation raises -> the case
    # fails (strict), it cannot silently corrupt driver-held state.
    code = (_BASE + "AUX0 = [0.5]\n"
            "def _on_access(aux, hit, t):\n"
            "    aux[0] = 1.0\n"
            "    return [aux[0]]\n"
            "def _evict_score_aux(la, fr, ins, t, aux):\n    return la\nCMUL = 2\n")
    assert not ground(code, SPEC, CFG).correct


# --------------------------------------------------------------------------- #
# B. Competitive-ratio verdict
# --------------------------------------------------------------------------- #

def test_cr_verdict_properties():
    distinct = list(range(30))                       # OPT = 0
    ok, q = ce.verdict((0.0, 8), None, distinct, 4)
    assert ok and q == 1.0                           # zero hits IS optimal here
    assert ce.verdict((0.5, 8), None, distinct, 4)[0] is False   # still anti-cheat
    rep = [1, 2, 3, 4, 5] * 200                      # OPT ~ 0.995 at cap 20
    ok2, q2 = ce.verdict((0.5, 20), None, rep, 10)
    assert ok2 and 0.45 < q2 < 0.55                  # CR = hr / OPT, bounded


# --------------------------------------------------------------------------- #
# C. Weighted utility machinery
# --------------------------------------------------------------------------- #

def test_weighted_quality_math():
    suite = [{"name": "a", "weight": 0.75}, {"name": "b", "weight": 0.25},
             {"name": "missing", "weight": 5.0}]
    assert Engine._weighted_quality({"a": 1.0, "b": 0.0}, suite) == 0.75


def test_utility_suite_is_the_declared_benchmark_for_cache():
    eng = Engine(CFG, ce.DOMAIN, make_roles(CFG))
    suite = eng._utility_workloads()
    assert [w["name"] for w in suite] == [w["name"] for w in ce.BENCHMARK_SUITE]
    assert abs(sum(w["weight"] for w in suite) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# D. Generic trusted-driver gate (any domain's symbol names)
# --------------------------------------------------------------------------- #

def test_generic_driver_gate_with_toy_symbols():
    toy = "def _foo(x):\n    return x\nBARS = 3\n"
    assert driver_symbols_ok(toy, funcs=("_foo",), literal_ints=("BARS",))
    assert not driver_symbols_ok(toy, funcs=("_missing",), literal_ints=("BARS",))
    assert not driver_symbols_ok("def _foo(x):\n    return x\nBARS = 1 + 2\n",
                                 funcs=("_foo",), literal_ints=("BARS",))
    spec = GroundingSpec(oracle_src="", workload_gen_src="", criteria_cases=[],
                         workloads=[], mem_workload={}, driver_src="x",
                         driver_funcs=("_foo",), driver_literal_ints=("BARS",))
    assert candidate_ok(toy, spec)
    assert not candidate_ok("def _foo(x):\n    return x\n", spec)   # BARS missing


def test_literal_gate_rejects_every_rebinding_vector():
    # v9 review HIGH (reproduced by the verifier): the cost axis reads CMUL from
    # SOURCE while the driver reads it at RUNTIME, so ANY second binding the regex
    # cannot see would let a free-form candidate charge cmul=1 cost while running
    # cmul=4 quality. The gate is a WHOLE-TREE binding census: exactly one
    # single-target literal-int binding, no deletes, no global/nonlocal.
    assert ce.fragment_is_safe is not None     # anchor import
    legal = _BASE + "CMUL = 2\n"
    legal_ann = _BASE + "CMUL: int = 2\n"      # idiomatic annotated form is legal
    assert defines_driver_symbols(legal) and defines_driver_symbols(legal_ann)
    assert Engine._policy_cap(legal_ann) == 2  # and the cost axis can read it
    exploits = [
        _BASE + "CMUL = 1\nCMUL += 3\n",                       # augassign rebind
        _BASE + "CMUL = 1\nif True:\n    CMUL = 4\n",          # nested rebind
        _BASE + "CMUL = 1\nCMUL: int = 4\n",                   # annassign rebind
        _BASE + "CMUL = 1\nfor CMUL in [4]:\n    pass\n",      # for-target rebind
        _BASE + "CMUL = X = 4\n",                              # tuple-target
        ("def _evict_score(la, fr, ins, t):\n    return la\n"
         "def _on_hit(la, fr, ins, t):\n    global CMUL\n    CMUL = 4\n"
         "    return [t, fr + 1, ins]\nCMUL = 1\n"),           # runtime global rebind
    ]
    for code in exploits:
        assert not defines_driver_symbols(code), f"gate must reject:\n{code}"
        assert not candidate_ok(code, ce.GROUNDING_SPEC)


# --------------------------------------------------------------------------- #
# E. THE v9 HEADLINE: adaptive robustly beats every static on the benchmark
# --------------------------------------------------------------------------- #

def test_adaptive_beats_every_static_on_the_weighted_benchmark():
    ad_q, ad_rec = _wmean(ce._assemble_adaptive(AD_EXPR, 2))
    static_champion_q, sc_rec = _wmean(ce._assemble_core(
        "la + 30 * fr.bit_length()", ce._UPDATE_OPS["bump_both"], 2))
    for s in ("lru", "lfu", "fifo", "mru"):
        hq, _ = _wmean(ce._assemble_policy(s, "bump_both", 2))
        assert ad_q > hq + 0.02, f"adaptive must beat human {s}: {ad_q:.3f} vs {hq:.3f}"
    assert ad_q > static_champion_q + 0.02, (
        f"adaptive must beat the v8 static champion: "
        f"{ad_q:.3f} vs {static_champion_q:.3f}")
    # ROBUSTNESS, not just a better mean: per-regime the adaptive never loses to
    # the static champion (>= everywhere within noise) and strictly rescues the
    # loop regime where every LRU-family static collapses.
    for w in ce.BENCHMARK_SUITE:
        nm = w["name"]
        assert ad_rec[nm] >= sc_rec[nm] - 0.03, (
            f"adaptive regressed on {nm}: {ad_rec[nm]:.2f} vs {sc_rec[nm]:.2f}")
    assert ad_rec["bench_loop"] > sc_rec["bench_loop"] + 0.3


# --------------------------------------------------------------------------- #
# F. Engine integration: classification, capacity, mutation-survival, e2e flags
# --------------------------------------------------------------------------- #

def test_adaptive_classifies_and_caps_even_after_unparse():
    code = ce._assemble_adaptive(AD_EXPR, 4)
    assert ce.classify(code) == AlgoClass.ADAPTIVE
    assert Engine._policy_cap(code) == 4
    stripped = ast.unparse(ast.parse(code))          # mutation strips comments
    assert ce.classify(stripped) == AlgoClass.ADAPTIVE
    assert Engine._policy_cap(stripped) == 4         # CMUL = 4 survives


def _fast_cache_domain():
    small = [dict(w, n=2500) for w in ce.WORKLOADS]
    spec = dataclasses.replace(ce.GROUNDING_SPEC, workloads=small,
                               mem_workload=dict(ce.MEM_WORKLOAD, n=2500))
    return dataclasses.replace(
        ce.DOMAIN, grounding_spec=spec,
        adversary_workloads=[dict(w, n=2500) for w in ce.ADVERSARY_WORKLOADS],
        benchmark_suite=[dict(w, n=2500) for w in ce.BENCHMARK_SUITE])


def test_v9_engine_admits_adaptive_op_and_flags_all(tmp_path):
    cfg = Config(backend="deterministic", bench_repeats=1, max_iterations=6,
                 utility_history_path=str(tmp_path / "uh.jsonl"))
    eng = Engine(cfg, _fast_cache_domain(), make_roles(cfg))
    result = eng.run()
    v6 = result.v6_summary
    assert v6
    util = v6["op_utility"]
    assert util and all(f != "unknown" for f in util.values()), util
    # at least one ADAPTIVE (ascore) op was admitted into the vocabulary
    kinds = {e.kind for e in eng.vocab.proposed_ops()}
    assert ce.ASCORE in kinds, f"no adaptive op admitted; kinds={kinds}"
