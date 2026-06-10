"""v11 Half A: make the bar real.

A1: an INDEPENDENT ARC implementation (written from the Megiddo & Modha Fig. 4
case structure using list-based LRU queues -- deliberately different data
structures and traversal from the dict-based canonical `_arc_seq`) must agree
HIT-FOR-HIT with the canonical one across 60+ traces spanning every regime and
capacity. Plus adaptation/scan/loop property tests and the LeCaR deviation doc.
A2: 2Q + W-TinyLFU sanity before joining the tier.
A3: multi-seed runner + the DECLARED beats/fragile flag rule + selection factor.
A4: holdout family draws within declared ranges; zipfx dist; sweep wiring.
"""

import dataclasses
import random

from rde.config import Config
from rde.domains import cache_eviction as ce
from rde.engine import Engine
from rde.grounding import measure_recall_full
from rde.selfplay import UtilityFlag

CFG = Config(bench_repeats=1)


# --------------------------------------------------------------------------- #
# A1 -- the independent ARC (lists, written from the published case structure;
# declared conventions identical to _arc_seq's docstring)
# --------------------------------------------------------------------------- #

def _arc_independent(trace, c):
    T1, T2, B1, B2 = [], [], [], []          # index 0 = LRU end; append = MRU
    p = 0
    hits = []

    def replace(in_b2):
        if T1 and (len(T1) > p or (in_b2 and len(T1) == p)):
            B1.append(T1.pop(0))
        elif T2:
            B2.append(T2.pop(0))

    for x in trace:
        if x in T1:                          # Case I: hit in T1 -> MRU of T2
            T1.remove(x)
            T2.append(x)
            hits.append(True)
        elif x in T2:                        # Case I: hit in T2 -> MRU of T2
            T2.remove(x)
            T2.append(x)
            hits.append(True)
        elif x in B1:                        # Case II: ghost hit favours recency
            p = min(c, p + max(1, len(B2) // max(1, len(B1))))
            replace(False)
            B1.remove(x)
            T2.append(x)
            hits.append(False)
        elif x in B2:                        # Case III: ghost hit favours frequency
            p = max(0, p - max(1, len(B1) // max(1, len(B2))))
            replace(True)
            B2.remove(x)
            T2.append(x)
            hits.append(False)
        else:                                # Case IV: complete miss
            L1 = len(T1) + len(B1)
            total = L1 + len(T2) + len(B2)
            if L1 == c:
                if len(T1) < c:
                    B1.pop(0)
                    replace(False)
                elif T1:
                    T1.pop(0)                # B1 empty: drop T1 LRU, no ghost
            elif L1 < c and total >= c:
                if total >= 2 * c and B2:
                    B2.pop(0)
                replace(False)
            T1.append(x)
            hits.append(False)
        while len(B1) + len(B2) > c:         # declared charging-bound trim
            (B1 if len(B1) >= len(B2) else B2).pop(0)
    return hits


_REGIMES = ["zipf", "recency", "drift", "mixed", "loop", "scan", "antagonistic"]


def test_arc_dual_implementation_hit_for_hit_agreement():
    """A1: >= 60 (trace, cap) combinations across all regimes/capacities; EXACT
    per-access agreement. Disagreement is a finding to resolve, never suppress."""
    combos = 0
    for dist in _REGIMES:
        for seed in (1, 2, 3):
            trace = ce.gen_workload(1500, dist, seed)
            for cap in (8, 20, 40):
                canonical, _p = ce._arc_seq(trace, cap)
                independent = _arc_independent(trace, cap)
                assert canonical == independent, (
                    f"ARC implementations DISAGREE on dist={dist} seed={seed} "
                    f"cap={cap} at access "
                    f"{next(i for i, (a, b) in enumerate(zip(canonical, independent)) if a != b)}")
                combos += 1
    assert combos >= 60, combos


def test_arc_adaptation_property():
    """p moves toward recency (T1 target UP) under recency traffic relative to
    frequency-skewed traffic, at a capacity that forces ghost activity."""
    rec_trace = ce.gen_workload(6000, "recency", 7)
    zipf_trace = ce.gen_workload(6000, "zipf", 7)
    _h1, p_rec = ce._arc_seq(rec_trace, 20)
    _h2, p_zipf = ce._arc_seq(zipf_trace, 20)
    assert p_rec >= p_zipf, (p_rec, p_zipf)


def _lru_rate(trace, cap):
    cache = {}
    hits = 0
    for t, x in enumerate(trace):
        if x in cache:
            hits += 1
            del cache[x]
        elif len(cache) >= cap:
            del cache[next(iter(cache))]
        cache[x] = True
    return hits / len(trace)


def test_arc_scan_resistance_and_loop_blind_spot():
    scan = ce.gen_workload(6000, "scan", 7)
    arc_rate = sum(ce._arc_seq(scan, 20)[0]) / 6000
    assert arc_rate >= _lru_rate(scan, 20) - 0.02      # scan resistance
    loop = ce.gen_workload(6000, "loop", 7)            # period 40 > cap
    arc_loop = sum(ce._arc_seq(loop, 20)[0]) / 6000
    assert arc_loop <= 0.05, "the documented loop blind spot must behave as such"


def test_lecar_deviation_is_documented():
    doc = ce._lecar_reference.__doc__ or ""
    assert "DETERMINISTIC" in doc.upper() and "argmax" in doc, (
        "LeCaR's deviation from the published randomized algorithm must be "
        "documented at the implementation")


# --------------------------------------------------------------------------- #
# A2 -- 2Q + W-TinyLFU sanity before they join the tier
# --------------------------------------------------------------------------- #

def _lfu_rate(trace, cap):
    cache = {}
    hits = 0
    for t, x in enumerate(trace):
        st = cache.get(x)
        if st is not None:
            hits += 1
            st[0] += 1
        else:
            if len(cache) >= cap:
                victim = min(cache, key=lambda kk: (cache[kk][0], cache[kk][1]))
                del cache[victim]
            cache[x] = [1, t]
    return hits / len(trace)


def test_twoq_scan_resistance():
    scan = ce.gen_workload(6000, "scan", 7)
    assert sum(ce._twoq_seq(scan, 20)) / 6000 >= _lru_rate(scan, 20) - 0.02


def test_wtlfu_frequency_strength():
    zipf = ce.gen_workload(6000, "zipf", 7)
    assert sum(ce._wtlfu_seq(zipf, 20)) / 6000 >= _lfu_rate(zipf, 20) - 0.02
    loop = ce.gen_workload(6000, "loop", 7)
    r = sum(ce._wtlfu_seq(loop, 20)) / 6000
    assert 0.0 <= r <= 1.0


# --------------------------------------------------------------------------- #
# A3 -- multi-seed measurement + the DECLARED flag rule
# --------------------------------------------------------------------------- #

def test_runner_multiseed_records_per_seed_values():
    wl = [dict(ce.BENCHMARK_SUITE[0], n=1500)]          # zipf, 5 declared seeds
    r = measure_recall_full(ce._assemble_policy("lru", "bump_both", 2),
                            wl, ce.GROUNDING_SPEC, CFG)
    seeds = r["quality_seeds"]["zipf"]
    assert len(seeds) == len(ce.BENCH_SEEDS) == 5
    assert abs(r["quality"]["zipf"] - sum(seeds) / len(seeds)) < 1e-9


def _mk_engine():
    return Engine(CFG, ce.DOMAIN, make_roles_cached(), )


def make_roles_cached():
    from rde.roles import make_roles
    return make_roles(CFG)


def test_declared_multiseed_flag_rule_and_fragility():
    eng = Engine(CFG, ce.DOMAIN, make_roles_cached())
    # comparator: mean 0.70, per-seed [0.70 x5], cost 1000
    front = [(0.70, 1000, (0.70, 0.70, 0.70, 0.70, 0.70))]
    eps = CFG.utility_eps                               # 0.01
    sel = eng._selection_factor(2)                      # n_graded=2 -> 1.25
    # clean beats: every per-seed margin above eps*sel
    f, m, frag = eng._classify_vs_ms(front, 0.75, [0.75] * 5, 2000, 2)
    assert f is UtilityFlag.BEATS_HUMAN and not frag
    # boundary-straddling: mean beats but one seed dips below the beats region
    f2, m2, frag2 = eng._classify_vs_ms(
        front, 0.72, [0.74, 0.74, 0.74, 0.74, 0.70], 2000, 2)
    assert frag2, "a seed in a different flag region must annotate fragile"
    # selection pressure: the SAME margin that beats at n=2 must NOT beat at n=64
    margin = eps * sel * 1.05                           # just over the n=2 bar
    q = 0.70 + margin
    f3, _m3, _ = eng._classify_vs_ms(front, q, [q] * 5, 2000, 2)
    f4, _m4, _ = eng._classify_vs_ms(front, q, [q] * 5, 2000, 64)
    assert f3 is UtilityFlag.BEATS_HUMAN and f4 is not UtilityFlag.BEATS_HUMAN


def test_selection_factor_monotone():
    eng = Engine(CFG, ce.DOMAIN, make_roles_cached())
    assert (eng._selection_factor(2) < eng._selection_factor(14)
            < eng._selection_factor(30))


# --------------------------------------------------------------------------- #
# A4 -- holdout family + zipfx dist
# --------------------------------------------------------------------------- #

def test_holdout_family_draws_within_declared_ranges():
    for seed in (1, 99, 12345):
        suite = ce.holdout_suite(random.Random(seed))
        names = [w["name"] for w in suite]
        assert names == ["hold_loop", "hold_zipfx", "hold_drift"]
        period = int(suite[0]["dist"].split("period=")[1].split(";")[0])
        assert 45 <= period <= 90
        expo = float(suite[1]["dist"].split("e=")[1])
        assert 2.0 <= expo <= 3.0
        assert all(len(w["seeds"]) == 3 for w in suite)
    # deterministic given the rng seed (reproducible from the recorded draw)
    a = ce.holdout_suite(random.Random(7))
    b = ce.holdout_suite(random.Random(7))
    assert a == b


def test_zipfx_dist_is_deterministic_and_clamped():
    t1 = ce.gen_workload(1000, "zipfx:e=2.4", 7)
    t2 = ce.gen_workload(1000, "zipfx:e=2.4", 7)
    assert t1 == t2 and all(isinstance(x, int) for x in t1)
    t3 = ce.gen_workload(500, "zipfx:e=999", 7)         # clamped, no blow-up
    assert max(t3) <= 60
