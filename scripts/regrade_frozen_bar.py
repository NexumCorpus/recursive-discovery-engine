"""v11 A3: the FROZEN-BAR re-grade of every v9/v10 winning op, under the full
v11 standard: 5-seed benchmark, declared multi-seed flag rule with the
selection-penalised beats threshold, the broadened adaptive tier (ARC, LeCaR,
2Q, W-TinyLFU at three capacities), and the ghost-charge sweep. Output is the
freeze record for docs/V11_NOTES.md. Deterministic; run from the repo root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rde.config import Config                                   # noqa: E402
from rde.domains import cache_eviction as ce                    # noqa: E402
from rde.engine import Engine                                   # noqa: E402
from rde.roles import make_roles                                # noqa: E402

OPS = [
    # (name, fragment, kind, origin)
    ("ad_thrash", "(-la) if aux[0] < 0.08 else (la + 30 * fr.bit_length())",
     "ascore", "det-template"),
    ("ad_guard15", "(-la) if aux[0] < 0.15 else (la + 30 * fr.bit_length())",
     "ascore", "det-template"),
    ("rec_logf", "la + 30 * fr.bit_length()", "score", "det-template"),
    ("rec_plus_f", "la + 8 * fr", "score", "det-template"),
    ("adaptive_hotset", "(-la) if aux[0] < 0.1 else (la + (fr.bit_length() << 5))",
     "ascore", "v9-LLM"),
    ("adaptive_mixguard",
     "(-la) if aux[0] < 0.18 else (la + fr.bit_length() * 24 + (t - ins) // 8)",
     "ascore", "v9-LLM"),
    ("adaptive_freq_guard",
     "(-la) if aux[0] < 0.12 else (la + (fr.bit_length() << 5))",
     "ascore", "v9-LLM"),
    ("warm_freq_bias", "la + (fr << 4)", "score", "v10-LLM"),
    ("hybrid_hot_guard",
     "(-la) if aux[0] < 0.08 else (la + fr.bit_length() * 16)",
     "ascore", "v10-LLM"),
    ("adaptive_freq_stick",
     "(-la) if aux[0] < 0.07 else (la + (fr << 6) - ((t - ins) >> 2))",
     "ascore", "v10-LLM"),
]
FRACS = (0.0, 0.25, 0.5, 1.0)


def main() -> int:
    cfg = Config(bench_repeats=1)
    eng = Engine(cfg, ce.DOMAIN, make_roles(cfg))
    suite = eng._utility_workloads()
    static_front = eng._human_frontier_points()
    adaptive_front = eng._adaptive_frontier_points()
    n_graded = len(OPS)
    sel = eng._selection_factor(n_graded)
    base_k = 10
    print(f"FROZEN BAR (benchmark v11-multiseed, S=5; n_graded={n_graded}, "
          f"selection_factor={sel:.3f}; eps_beats={cfg.utility_eps * sel:.4f})")
    print("static front:", sorted({(round(q, 3), m) for q, m, *_ in static_front}))
    print("adaptive front:",
          sorted({(round(q, 3), m) for q, m, *_ in adaptive_front}))
    rows = []
    for name, frag, kind, origin in OPS:
        best = None
        for c in (2, 4):
            code = (ce._assemble_adaptive(frag, c) if kind == "ascore"
                    else ce._assemble_core(frag, ce._UPDATE_OPS["bump_both"], c))
            rec, missing, smap = eng._measure_suite_complete(code, suite)
            q = eng._weighted_quality(rec, suite)
            if best is None or q > best[0]:
                qseeds = eng._weighted_quality_seeds(smap, suite)
                best = (q, c * base_k * Engine._BYTES_PER_ENTRY, code, qseeds, c)
        q, mem, code, qseeds, cap = best
        sflag, smarg, sfrag = eng._classify_vs_ms(static_front, q, qseeds,
                                                  mem, n_graded)
        aflag, amarg, afrag = eng._classify_vs_ms(adaptive_front, q, qseeds,
                                                  mem, n_graded)
        # ghost-charge sweep on the adaptive tier
        per_frac = {}
        for f in FRACS:
            pts = [(m_["q"], m_["mem"], tuple(m_["seeds"]))
                   for m_ in eng._static_meas]
            for r in eng._ref_meas:
                cost_u = (r["cmul"] * (1.0 + r["ghost_ratio"] * f)
                          + r["cmul"] * r["sketch_slots_ratio"] / 16.0)
                pts.append((r["q"],
                            int(round(cost_u * base_k * Engine._BYTES_PER_ENTRY)),
                            tuple(r["seeds"])))
            front = [p for p in pts
                     if not any(o[0] >= p[0] and o[1] <= p[1]
                                and (o[0], o[1]) != (p[0], p[1])
                                for o in pts if o is not p)]
            fl, _m, _fr = eng._classify_vs_ms(front, q, qseeds, mem, n_graded)
            per_frac[f] = fl.value
        robust = all(v == "beats_human" for v in per_frac.values())
        fragile = sfrag or afrag
        rows.append((name, origin, cap, q,
                     sflag.value, smarg, aflag.value, amarg,
                     robust, fragile,
                     {k: v.replace("_human", "") for k, v in per_frac.items()}))
    print(f"\n{'op':<22}{'origin':<14}{'cap':<5}{'wCR':<8}"
          f"{'static':<16}{'adaptive':<16}{'robust':<8}{'fragile'}")
    for r in rows:
        print(f"{r[0]:<22}{r[1]:<14}c{r[2]:<4}{r[3]:<8.4f}"
              f"{r[4].replace('_human',''):>6} {r[5]:+.4f}  "
              f"{r[6].replace('_human',''):>6} {r[7]:+.4f}  "
              f"{str(r[8]):<8}{r[9]}")
        print(f"{'':<22}sweep: {r[10]}")
    n_beats = sum(1 for r in rows if r[6] == "beats_human")
    n_robust = sum(1 for r in rows if r[8])
    print(f"\nSUMMARY: {n_beats}/{len(rows)} beats_adaptive_human under the "
          f"frozen bar; {n_robust}/{len(rows)} robust_beats (all charge points); "
          f"fragile: {[r[0] for r in rows if r[9]]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
