"""Independent reproduce script for the cache-eviction "crossing" question.

Scores ANY adaptive-score (ASCORE) cache-eviction policy on:
  (a) the DECLARED benchmark  -> benchmark margin vs the adaptive human tier, and
  (b) a FRESH holdout family   -> holdout margin vs the best-on-benchmark comparator,
using RDE's OWN primitives only (no invented scorer):
    _assemble_adaptive, measure_recall_full, BENCHMARK_SUITE, holdout_suite,
    adaptive_baselines, human_baseline_policies.

Engine-faithful details (rde/engine.py finalize()):
  * The op's REPRESENTATIVE policy = the capacity variant (cmul in {2,4}) with the
    best weighted-benchmark competitive ratio (engine lines 873-892).
  * benchmark margin = per-seed pairwise mean CR vs the frontier point (statics +
    ARC/LeCaR/2Q/W-TinyLFU) with the highest MEAN CR at <= the policy's cost
    (_classify_vs_ms, lines 690-724).
  * holdout margin = hq - cq where the comparator is the best-benchmark-CR baseline
    at <= the policy's cost, RE-MEASURED on the fresh holdout suite (finalize
    holdout block, lines 1095-1140).
  * cost = cmul * base_k(10) * 160 bytes (no ghost for an ASCORE policy).

VERIFIED FAITHFUL: with --hold-seed 3379786821 (a recorded run's holdout seed) the
two known pretenders reproduce their recorded margins to the digit:
    ad_thrash  bench +0.0349  holdout -0.1706  (vs wtlfu2)
    ad_guard15 bench +0.0591  holdout -0.1367  (vs wtlfu2)
    cf. runs/20260610_115502_cache_eviction_deterministic/result.json

USAGE (from repo root E:\recursive-discovery-engine):
  python scripts/repro_cache_crossing.py --pretenders --hold-seed 3379786821
  python scripts/repro_cache_crossing.py \
      --expr "(-la) if aux[0] < 0.10 else (la + 30 * fr.bit_length())" \
      --hold-seeds 3379786821,12345,777,2026,424242
Run with the project venv:  E:/_phase0_test_venv/Scripts/python.exe
"""
import sys, os, argparse, statistics, dataclasses, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rde.config import Config
from rde.domains import cache_eviction as CE
from rde.domains.cache_eviction import (
    BENCHMARK_SUITE, holdout_suite, adaptive_baselines,
    _assemble_adaptive, GROUNDING_SPEC,
    GHOST_ENTRY_COST_FRAC,
)
from rde.grounding import measure_recall_full

BYTES_PER_ENTRY = 160
BASE_K = 10
CFG = Config()
SPEC = GROUNDING_SPEC
PRETENDERS = {
    "ad_thrash":  "(-la) if aux[0] < 0.08 else (la + 30 * fr.bit_length())",
    "ad_guard15": "(-la) if aux[0] < 0.15 else (la + 30 * fr.bit_length())",
}


def wq(rec, suite):
    num = den = 0.0
    for w in suite:
        nm = w.get("name")
        if nm in rec:
            wt = float(w.get("weight", 1.0)); num += rec[nm] * wt; den += wt
    return round(num / den, 4) if den else 0.0


def wq_seeds(seeds_map, suite):
    n = max((len(v) for v in seeds_map.values()), default=1); out = []
    for i in range(n):
        num = den = 0.0
        for w in suite:
            nm = w.get("name")
            if nm in seeds_map:
                vals = seeds_map[nm]
                v = vals[i] if i < len(vals) else (sum(vals)/len(vals) if vals else 0.0)
                wt = float(w.get("weight", 1.0)); num += v*wt; den += wt
        out.append(round(num/den, 4) if den else 0.0)
    return out


def measure(code, suite, spec=None):
    full = measure_recall_full(code, suite, spec or SPEC, CFG)
    return dict(full["quality"]), dict(full["quality_seeds"])


def ref_cost(cmul, gr, sr):
    return int(round((cmul*(1.0+gr*GHOST_ENTRY_COST_FRAC)+cmul*sr/16.0)*BASE_K*BYTES_PER_ENTRY))


def policy_cost(cmul):
    return int(round(cmul*BASE_K*BYTES_PER_ENTRY))


def build_frontier():
    comps = []
    for label, code in CE.human_baseline_policies(None):
        rec, seeds = measure(code, BENCHMARK_SUITE)
        import re
        cap = int(re.search(r"CMUL\s*=\s*(\d+)", code).group(1))
        comps.append({"label": label, "kind": "static", "code": code,
                      "q": wq(rec, BENCHMARK_SUITE), "seeds": wq_seeds(seeds, BENCHMARK_SUITE),
                      "mem": policy_cost(cap)})
    for ref in adaptive_baselines():
        spec_ref = dataclasses.replace(SPEC, driver_src=ref["driver_src"])
        rec, seeds = measure(ref["stub"], BENCHMARK_SUITE, spec=spec_ref)
        comps.append({"label": ref["label"], "kind": "ref", "stub": ref["stub"],
                      "driver_src": ref["driver_src"],
                      "q": wq(rec, BENCHMARK_SUITE), "seeds": wq_seeds(seeds, BENCHMARK_SUITE),
                      "mem": ref_cost(ref["cmul"], ref["ghost_ratio"], ref["sketch_slots_ratio"])})
    return comps


def score(expr, hold_seed, frontier, caps=(2, 4)):
    # representative capacity = best benchmark CR
    best = None
    for cm in caps:
        code = _assemble_adaptive(expr, cm)
        rec, seeds = measure(code, BENCHMARK_SUITE)
        q = wq(rec, BENCHMARK_SUITE)
        if best is None or q > best[0]:
            best = (q, cm, code, seeds)
    bq, cmul, code, bseeds = best
    pmem = policy_cost(cmul)
    bqs = wq_seeds(bseeds, BENCHMARK_SUITE)
    cand = [c for c in frontier if c["mem"] <= pmem]
    comp = max(cand, key=lambda c: c["q"]) if cand else min(frontier, key=lambda c: c["mem"])
    cs = comp["seeds"] or [comp["q"]]*max(len(bqs), 1)
    S = min(len(bqs), len(cs)) or 1
    bench_margin = round(sum(bqs[i]-cs[i] for i in range(S))/S, 4) if bqs else round(bq-comp["q"], 4)
    # holdout
    hsuite = holdout_suite(random.Random(hold_seed))
    hrec, _ = measure(code, hsuite); hq = wq(hrec, hsuite)
    hcomp = max(cand, key=lambda c: c["q"]) if cand else None
    if hcomp["kind"] == "static":
        crec, _ = measure(hcomp["code"], hsuite)
    else:
        crec, _ = measure(hcomp["stub"], hsuite, spec=dataclasses.replace(SPEC, driver_src=hcomp["driver_src"]))
    hold_margin = round(hq - wq(crec, hsuite), 4)
    return dict(expr=expr, cmul=cmul, bench_q=bq, bench_margin=bench_margin,
                bench_comp=comp["label"], hold_q=hq, hold_margin=hold_margin,
                hold_comp=hcomp["label"], hold_seed=hold_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expr")
    ap.add_argument("--pretenders", action="store_true")
    ap.add_argument("--hold-seed", type=int, default=3379786821)
    ap.add_argument("--hold-seeds", default=None, help="comma list; overrides --hold-seed")
    args = ap.parse_args()
    seeds = [int(x) for x in args.hold_seeds.split(",")] if args.hold_seeds else [args.hold_seed]

    print("building benchmark frontier (statics + ARC/LeCaR/2Q/W-TinyLFU)...", file=sys.stderr)
    frontier = build_frontier()
    for c in sorted(frontier, key=lambda c: -c["q"]):
        print(f"  base {c['label']:18s} q={c['q']:.4f} mem={c['mem']}", file=sys.stderr)

    exprs = PRETENDERS if args.pretenders else {"policy": args.expr}
    for nm, expr in exprs.items():
        hs = [score(expr, s, frontier)["hold_margin"] for s in seeds]
        r = score(expr, seeds[0], frontier)
        tag = ""
        if len(seeds) > 1:
            tag = (f"  [holdout over {len(seeds)} seeds: mean={statistics.mean(hs):+.4f} "
                   f"min={min(hs):+.4f} max={max(hs):+.4f} pos={sum(1 for m in hs if m>0)}/{len(hs)}]")
        print(f"{nm:10s} cmul={r['cmul']} bench_margin={r['bench_margin']:+.4f} "
              f"(vs {r['bench_comp']})  hold_margin={r['hold_margin']:+.4f} "
              f"(vs {r['hold_comp']}, seed {seeds[0]}){tag}")


if __name__ == "__main__":
    main()
