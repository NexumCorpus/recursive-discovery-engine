# HISTORICAL (pre-v11): kept for the record. The adv_loop trace it reads was renamed
# when the v11 benchmark froze, so it no longer runs; scripts/reproduce.py is the
# current reproduction.
# Reproduction for the "adversarial tail evaporates at CMUL=4" finding.
import sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rde.domains import cache_eviction as ce

SEED, K = 1234, 10

def make_candidate(code):
    mod = types.ModuleType("candidate")
    cobj = compile(code, "<candidate>", "exec")
    eval(cobj, mod.__dict__)
    return mod

def run_policy(code, trace, k):
    mod = make_candidate(code)
    saved = sys.modules.get("candidate")
    sys.modules["candidate"] = mod
    try:
        return ce._driver_solve(trace, k)
    finally:
        if saved is not None:
            sys.modules["candidate"] = saved
        else:
            del sys.modules["candidate"]

REC_LOGF = "la + 30 * fr.bit_length()"
AD_THRASH = "(-la) if aux[0] < 0.08 else (la + 30 * fr.bit_length())"

policies = {
    "rec_logf_c2": ce._assemble_core(REC_LOGF, ce._UPDATE_OPS["bump_both"], 2),
    "rec_logf_c4": ce._assemble_core(REC_LOGF, ce._UPDATE_OPS["bump_both"], 4),
    "ad_thrash_c2": ce._assemble_adaptive(AD_THRASH, 2),
    "ad_thrash_c4": ce._assemble_adaptive(AD_THRASH, 4),
}
for sc in ce._SCORE_OPS:
    for c in (2, 4):
        policies.setdefault("%s%d" % (sc, c), ce._assemble_policy(sc, "bump_both", c))

suite = ce.BENCHMARK_SUITE
traces = {w["name"]: ce.gen_workload(w["n"], w["dist"], SEED) for w in suite}

results = {}
hitrates = {}
for nm, code in policies.items():
    qs, hrs = {}, {}
    for w in suite:
        tr = traces[w["name"]]
        got = run_policy(code, tr, K)
        ok, q = ce.verdict(got, None, tr, K)
        assert ok, (nm, w["name"], got)
        qs[w["name"]] = q
        hrs[w["name"]] = got[0]
    results[nm] = qs
    hitrates[nm] = hrs

def weighted(qs):
    num = sum(qs[w["name"]] * w["weight"] for w in suite if w["name"] in qs)
    den = sum(w["weight"] for w in suite if w["name"] in qs)
    return round(num / den, 4)

MAIN = ["zipf", "recency", "drift", "mixed"]
def unweighted_main(qs):
    vals = [qs[m] for m in MAIN]
    return round(sum(vals) / len(vals), 4)

print("%-14s " % "policy" + " ".join("%10s" % w["name"] for w in suite)
      + " %8s %11s" % ("wCR", "liveQ(main)"))
for nm in sorted(results):
    qs = results[nm]
    print("%-14s " % nm + " ".join("%10.4f" % qs[w["name"]] for w in suite)
          + " %8.4f %11.4f" % (weighted(qs), unweighted_main(qs)))

print()
print("CLAIM 1: adv_loop CR == 1.0 for every c4 policy?")
for nm in sorted(results):
    if nm.endswith("4"):
        print("  %-14s adv_loop CR = %s  hit_rate = %.6f" % (nm, results[nm]["adv_loop"], hitrates[nm]["adv_loop"]))

print()
print("CLAIM 2: ad_thrash_c4 identical to rec_logf_c4 on every suite trace?")
for w in suite:
    a, b = hitrates["ad_thrash_c4"][w["name"]], hitrates["rec_logf_c4"][w["name"]]
    print("    %-10s ad_thrash_c4=%.6f rec_logf_c4=%.6f diff=%+.2e" % (w["name"], a, b, a - b))

print()
print("CLAIM 3: live (unweighted main-suite) quality c2 vs c4 for ad_thrash:")
print("  ad_thrash_c2 liveQ =", unweighted_main(results["ad_thrash_c2"]), " wCR =", weighted(results["ad_thrash_c2"]))
print("  ad_thrash_c4 liveQ =", unweighted_main(results["ad_thrash_c4"]), " wCR =", weighted(results["ad_thrash_c4"]))

print()
print("CLAIM 4: margins vs human frontier (engine logic, eps=0.01).")
BYTES = 160
def cost(c): return c * K * BYTES
humans = [("%s%d" % (sc, c), weighted(results["%s%d" % (sc, c)]), cost(c))
          for sc in ce._SCORE_OPS for c in (2, 4)]
for h in humans: print("   human:", h)
front = [p for p in humans
         if not any(o is not p and o[1] >= p[1] and o[2] <= p[2]
                    and (o[1], o[2]) != (p[1], p[2]) for o in humans)]
print("  human Pareto front:", front)
for cand in ("ad_thrash_c4", "ad_thrash_c2", "rec_logf_c4", "rec_logf_c2"):
    cm = 4 if cand.endswith("4") else 2
    q = weighted(results[cand])
    reach = [hq for (_n, hq, hm) in front if hm <= cost(cm)]
    best_h = max(reach) if reach else 0.0
    margin = round(q - best_h, 4)
    flag = ("BEATS_HUMAN" if margin > 0.01 else
            "MATCHES_HUMAN" if margin >= -0.01 else "BELOW_HUMAN")
    print("  %-14s wCR=%.4f mem=%d bestHuman=%.4f margin=%+.4f -> %s" % (cand, q, cost(cm), best_h, margin, flag))

print()
print("CLAIM 5: min EWMA aux[0] / MRU-branch evictions per trace:")
def min_ewma(trace, k, cmul):
    cap = max(cmul * k, 4)
    cache = {}; aux = 0.5; mn = 0.5; trips = 0
    for t, key in enumerate(trace):
        st = cache.get(key)
        hit = st is not None
        aux = aux * 0.98 + 0.02 * (1 if hit else 0)
        mn = min(mn, aux)
        if hit:
            la, fr, ins = st
            cache[key] = [t, fr + 1, ins]
        else:
            if len(cache) >= cap:
                if aux < 0.08:
                    trips += 1
                    vk = min(cache, key=lambda _k: -cache[_k][0])
                else:
                    vk = min(cache, key=lambda _k: cache[_k][0] + 30 * cache[_k][1].bit_length())
                del cache[vk]
            cache[key] = [t, 1, t]
    return mn, trips
for w in suite:
    mn4, tr4 = min_ewma(traces[w["name"]], K, 4)
    mn2, tr2 = min_ewma(traces[w["name"]], K, 2)
    print("  %-10s c4: min_aux=%.4f mru_evicts=%6d | c2: min_aux=%.4f mru_evicts=%6d" % (w["name"], mn4, tr4, mn2, tr2))
