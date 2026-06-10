"""v8: VERIFIABLE BY CONSTRUCTION. A cache candidate provides only the eviction
LOGIC (_evict_score / _on_hit / CMUL); a TRUSTED driver runs the simulation and
computes the hit rate, so NO candidate -- deterministic, op-composed, or free-form
LLM -- can misreport the metric (closing v7's H1 self-report hole). These tests are
the closure proof + the hardening the design panel grafted in.
"""

import dataclasses

from rde.config import Config
from rde.domains import cache_eviction as ce
from rde.domains import heavy_hitters as hh
from rde.engine import Engine
from rde.grounding import (candidate_ok, defines_driver_symbols, defines_solve,
                           ground, measure_recall_on)
from rde.roles import make_roles
from rde.types import AlgoClass

CFG = Config(bench_repeats=1)
SPEC = ce.GROUNDING_SPEC
LOOP = [{"name": "loop", "n": 4000, "k": 10, "dist": "loop"}]


# --------------------------------------------------------------------------- #
# A. THE CHEAT IS GONE -- the candidate cannot report a hit rate it did not earn
# --------------------------------------------------------------------------- #

def test_candidate_solve_is_discarded_for_the_trusted_driver():
    # ISOLATE the driver-discard from mere verdict-rejection. A cheat ABOVE Belady OPT
    # would also be rejected by the verdict, so it proves nothing. Here the cheat
    # reports a hit rate BELOW OPT (the verdict would ACCEPT it if the solve were
    # used): on 'loop' OPT ~0.48, cheat 0.3 -> would record CR ~0.63 if the solve
    # were used. Honest LRU collapses on a loop (CR ~0.0). Recording ~0.0 proves the
    # driver genuinely discarded the candidate's own solve (the H1 closure), under
    # the v9 competitive-ratio quality.
    lru = ce._assemble_policy("lru", "bump_both", 2)
    cheat = lru + "\ndef solve(trace, k):\n    return (0.3, max(2 * k, 4))\n"
    rate = measure_recall_on(cheat, LOOP, SPEC, CFG).get("loop", 1.0)
    assert rate < 0.1, f"driver must record honest LRU (~0), not the cheat (~0.63): {rate}"
    # and the blatant case: a fabricated 1.0 on a loop records the driver's ~0 collapse.
    cheat2 = lru + "\ndef solve(trace, k):\n    return (1.0, max(2 * k, 4))\n"
    assert measure_recall_on(cheat2, LOOP, SPEC, CFG).get("loop", 1.0) < 0.2


def test_fabricated_score_cannot_beat_belady():
    # No matter what _evict_score/_on_hit return, the recorded quality is what the
    # eviction behaviour actually EARNS. A constant score + a wildly inflated
    # frequency update still thrashes on a loop the cache cannot hold -> CR ~0
    # (v9 note: the all-distinct-trace variant is no longer usable here because
    # competitive ratio DEFINES quality as 1.0 when OPT is 0 -- everyone is equally
    # optimal at zero hits -- so the loop regime is the discriminating probe).
    evil = (
        "# v9-cache-policy\n# cmul=2\n"
        "def _evict_score(la, fr, ins, t):\n    return -999999\n"   # tries to bias
        "def _on_hit(la, fr, ins, t):\n    return [t, fr + 1000, ins]\n"
        "CMUL = 2\n")
    rec = measure_recall_on(evil, LOOP, SPEC, CFG)
    assert rec.get("loop", 1.0) < 0.2


# --------------------------------------------------------------------------- #
# B. Hardening (CMUL clamp, garbage update/score) -- can only DEGRADE, never cheat
# --------------------------------------------------------------------------- #

def test_non_literal_or_duplicate_cmul_is_rejected():
    # v8 review HIGH: the cost axis reads CMUL from SOURCE while the driver reads it at
    # RUNTIME -- a non-literal (`CMUL = 2 if cond else 8`) or duplicate CMUL would let a
    # free-form candidate charge a cheap cmul while running an expensive one (a
    # recall-per-cost spoof). Only a SINGLE LITERAL int CMUL is admitted.
    base = ("def _evict_score(la, fr, ins, t):\n    return la\n"
            "def _on_hit(la, fr, ins, t):\n    return [t, fr + 1, ins]\n")
    assert defines_driver_symbols(base + "CMUL = 2\n")                      # literal -> ok
    assert not defines_driver_symbols(base + "CMUL = 2 if True else 8\n")   # expression
    assert not defines_driver_symbols(base + "CMUL = 1 * 4\n")              # expression
    assert not defines_driver_symbols(base + "CMUL = 2\nCMUL = 8\n")        # duplicate
    spoof = base + "CMUL = 2 if (1 < 2) else 8\n"
    assert not candidate_ok(spoof, ce.GROUNDING_SPEC)        # rejected at the gate


def test_cmul_out_of_range_fails_the_candidate():
    for bad in ("CMUL = 99", "CMUL = 0", "CMUL = True"):
        code = ("# v8-cache-policy\ndef _evict_score(la, fr, ins, t):\n    return la\n"
                "def _on_hit(la, fr, ins, t):\n    return [t, fr + 1, ins]\n" + bad + "\n")
        ev = ground(code, SPEC, CFG)
        assert not ev.correct, f"{bad} must be rejected (driver raises), got correct"


def test_garbage_on_hit_and_nan_score_do_not_crash_or_inflate():
    garbage = (
        "# v8-cache-policy\ndef _evict_score(la, fr, ins, t):\n    return float('nan')\n"
        "def _on_hit(la, fr, ins, t):\n    return 'not a list'\n"
        "CMUL = 2\n")
    ev = ground(garbage, SPEC, CFG)
    # It runs (no crash) and stays a VALID, bounded hit rate -- garbage can only
    # degrade the policy (driver normalises the update + coerces the NaN score).
    assert ev.correct and 0.0 <= ev.quality <= 1.0


# --------------------------------------------------------------------------- #
# C. Backward-compatibility -- heavy_hitters/topk untouched (opt-in driver)
# --------------------------------------------------------------------------- #

def test_gate_accepts_symbols_only_under_driver_only():
    sym = ce._assemble_policy("lru", "bump_both", 2)   # defines symbols, no solve
    assert defines_driver_symbols(sym) and not defines_solve(sym)
    assert candidate_ok(sym, ce.GROUNDING_SPEC)               # cache: driver set -> ok
    assert not candidate_ok(sym, hh.GROUNDING_SPEC)           # hh: no driver -> needs solve
    # a heavy_hitters solve-style module is still accepted there
    solve_only = "def solve(items, k):\n    return sorted(set(items))[:k]\n"
    assert candidate_ok(solve_only, hh.GROUNDING_SPEC)


def test_heavy_hitters_has_no_driver():
    assert hh.GROUNDING_SPEC.driver_src == ""
    assert ce.GROUNDING_SPEC.driver_src != ""


# --------------------------------------------------------------------------- #
# D. Mutation-safety: classify + capacity survive ast.unparse comment-stripping
# --------------------------------------------------------------------------- #

def test_classify_handles_freeform_and_comment_stripped_bodies():
    # multi-statement free-form score body, NO marker comment (as ast.unparse leaves it)
    body = ("def _evict_score(la, fr, ins, t):\n    x = fr - (t - la)\n    return x\n"
            "def _on_hit(la, fr, ins, t):\n    return [t, fr + 1, ins]\nCMUL = 2\n")
    # v10 P5: a STATIC la+fr blend is BLENDED; ADAPTIVE = reads aux.
    assert ce.classify(body) == AlgoClass.BLENDED
    pure_fr = ("def _evict_score(la, fr, ins, t):\n    return fr\n"
               "def _on_hit(la, fr, ins, t):\n    return [t, fr + 1, ins]\nCMUL = 2\n")
    assert ce.classify(pure_fr) == AlgoClass.FREQUENCY
    assert ce.classify("def solve(x, k):\n    return x\n") == AlgoClass.OTHER


def test_policy_cap_reads_cmul_assignment():
    # CMUL = N survives ast.unparse even when the `# cmul=` comment is stripped.
    no_comment = ("def _evict_score(la, fr, ins, t):\n    return la\n"
                  "def _on_hit(la, fr, ins, t):\n    return [t, fr + 1, ins]\nCMUL = 4\n")
    assert Engine._policy_cap(no_comment) == 4


# --------------------------------------------------------------------------- #
# E. Exact-capacity cost axis -- deterministic, no tracemalloc noise
# --------------------------------------------------------------------------- #

def _fast_cache_domain():
    base = ce.GROUNDING_SPEC
    small = [{**w, "n": 2500} for w in ce.WORKLOADS]
    spec = dataclasses.replace(base, workloads=small,
                               mem_workload={**ce.MEM_WORKLOAD, "n": 2500})
    return dataclasses.replace(ce.DOMAIN, grounding_spec=spec,
                               adversary_workloads=[{**w, "n": 2500}
                                                    for w in ce.ADVERSARY_WORKLOADS])


def test_cost_axis_is_exact_and_capacity_proportional():
    eng = Engine(CFG, _fast_cache_domain(), make_roles(CFG))
    c_lru2 = eng._policy_cost(ce._assemble_policy("lru", "bump_both", 2))
    c_lfu2 = eng._policy_cost(ce._assemble_policy("lfu", "bump_both", 2))
    c_lru4 = eng._policy_cost(ce._assemble_policy("lru", "bump_both", 4))
    assert c_lru2 == c_lfu2                       # same CMUL -> identical cost (no noise)
    assert c_lru4 == 2 * c_lru2                   # CMUL 2->4 doubles the cost exactly


# --------------------------------------------------------------------------- #
# F. End-to-end: trusted-driver run is sound + every op gets an HONEST flag
# --------------------------------------------------------------------------- #

def test_v8_engine_run_is_sound_and_utility_is_complete(tmp_path):
    cfg = Config(backend="deterministic", bench_repeats=1, max_iterations=6,
                 utility_history_path=str(tmp_path / "uh.jsonl"))
    eng = Engine(cfg, _fast_cache_domain(), make_roles(cfg))
    result = eng.run()
    v6 = result.v6_summary
    assert v6
    # close-the-loop: NO admitted op is left 'unknown' (every one earns a flag).
    util = v6["op_utility"]
    assert util and all(f != "unknown" for f in util.values()), util
    # every frontier member's recorded hit rate is <= Belady OPT at its capacity
    # (the driver cannot produce a super-optimal rate).
    for p in eng.frontier.points:
        a = eng.memory.get(p.approach_id)
        assert a is not None and 0.0 <= a.evidence.quality <= 1.0 + 1e-9
