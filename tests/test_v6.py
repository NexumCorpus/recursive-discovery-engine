"""v6 PROPOSAL QUALITY: causal guidance (the actual failing streams + missed
heavy hitters), Builder/Adversary self-play (win rate + margin), explicit utility
flags (beats_human / matches_human / below_human), and a cross-run utility history.

The honest headline v6 must earn: a system-PROPOSED op that BEATS the human Pareto
frontier (not just matches it), driven by a concrete diagnosed failure -- here the
threshold-prune op `declow`, which the Adversary's `decoy` attack motivates.
"""

import dataclasses

from rde.config import Config
from rde.domains import heavy_hitters as hh
from rde.engine import Engine
from rde.grounding import diagnose, measure_recall_on, ground
from rde.roles import make_roles
from rde.selfplay import (CausalGuidance, SelfPlayLedger, SelfPlayMatch,
                          UtilityFlag, append_utility_history,
                          summarize_utility_history)

CFG = Config(bench_repeats=1)


# --------------------------------------------------------------------------- #
# A. Causal guidance -- the concrete failure, not a keyword
# --------------------------------------------------------------------------- #

def test_diagnose_reports_missed_heavy_hitters():
    # A tiny-capacity policy on a hard late-arrival stream misses true hitters; the
    # diagnosis must surface WHICH ones (the causal signal fed to the proposer).
    code = hh._assemble_policy("min", "vhalf", "none", 8)
    wl = {"name": "param_l97_h400", "n": 40000, "k": 10,
          "dist": "param:late=0.97;heavy=400;coll=0.25"}
    d = diagnose(code, wl, hh.GROUNDING_SPEC, CFG)
    assert d and "expected" in d and "got" in d
    assert isinstance(d["missed"], list)
    # the oracle top-k and the policy top-k genuinely differ on this hard stream
    assert set(d["expected"]) != set(d["got"])


def test_causal_failure_tags_route_decoy_to_inflation_ops():
    # A decoy/spike failure must select the threshold-prune / no-inflation op family.
    g = CausalGuidance(workload="decoy", dist="prop:decoy", recall=0.10,
                       missed=[1, 2, 3])
    guidance = {"causal": g, "weak": set()}
    props = hh.propose_operations(hh.seed_vocabulary(), guidance)
    names = {p.name for p in props}
    assert "declow" in names, "a decoy failure should propose the prune-low op"


# --------------------------------------------------------------------------- #
# B. The headline: a PROPOSED op that BEATS the human frontier
# --------------------------------------------------------------------------- #

def _full_suite_with_decoy():
    block = hh.make_generator_block("decoy", hh._GEN_DECOY)
    spec = dataclasses.replace(
        hh.GROUNDING_SPEC,
        workload_gen_src=hh.GROUNDING_SPEC.workload_gen_src + "\n" + block)
    suite = (list(hh.WORKLOADS) + hh.adversary_param_attacks(10)
             + hh.adversary_generators(10)
             + [{"name": "decoy", "n": 40000, "k": 10, "dist": "prop:decoy"}])
    return spec, suite


def test_declow_op_beats_the_human_frontier():
    # declow (prune count<=2) is NOT in the human pre-vocab (none/dec_all/halve_all).
    # min/one/declow/32 must DOMINATE the best human-only policy: higher recall at
    # strictly LESS memory than min/one/16 (the strongest human combo at 18KB).
    spec, suite = _full_suite_with_decoy()
    names = [w["name"] for w in suite]

    def mean_mem(code):
        rec = measure_recall_on(code, suite, spec, CFG)
        ev = ground(code, spec, CFG)
        return (sum(rec.get(n, 0.0) for n in names) / len(names),
                ev.peak_memory_bytes // 1024)

    declow = hh._assemble_core(hh._VICTIM_OPS["min"], hh._INHERIT_OPS["one"],
                               hh._PRE_DECLOW, 32)
    human = hh._assemble_core(hh._VICTIM_OPS["min"], hh._INHERIT_OPS["one"],
                              hh._PRE_OPS["none"], 16)            # best human @ 18KB
    dq, dm = mean_mem(declow)
    hq, hm = mean_mem(human)
    assert dq > hq and dm < hm, (
        f"declow ({dq:.3f}@{dm}KB) must DOMINATE the human ({hq:.3f}@{hm}KB)")


# --------------------------------------------------------------------------- #
# C. Self-play ledger + utility flag + cross-run history (units)
# --------------------------------------------------------------------------- #

def test_selfplay_ledger_win_rate_and_margin():
    led = SelfPlayLedger()
    led.add(SelfPlayMatch(1, "declow", "decoy", op_recall=1.0, baseline_recall=0.7))
    led.add(SelfPlayMatch(1, "keep34", "comeback", op_recall=0.8, baseline_recall=0.8))
    assert led.win_rate() == 0.5
    assert abs(led.mean_margin() - 0.15) < 1e-9
    assert led.best_margin().op_name == "declow"
    assert "declow" in led.summary()["best"]


def test_utility_flag_ranks():
    assert UtilityFlag.BEATS_HUMAN.rank > UtilityFlag.MATCHES_HUMAN.rank
    assert UtilityFlag.MATCHES_HUMAN.rank > UtilityFlag.BELOW_HUMAN.rank


def test_utility_history_round_trips_and_trends(tmp_path):
    p = tmp_path / "hist.jsonl"
    append_utility_history(p, [{"run": "run1", "op": "a", "utility": "matches_human"},
                               {"run": "run1", "op": "b", "utility": "below_human"}])
    append_utility_history(p, [{"run": "run2", "op": "c", "utility": "beats_human"},
                               {"run": "run2", "op": "d", "utility": "beats_human"}])
    s = summarize_utility_history(p)
    assert s["runs"] == 2 and s["ops"] == 4
    assert s["beats_human"] == 2
    assert s["beats_rate_late"] > s["beats_rate_early"]   # quality trending up
    assert s["trend"] == "up"


# --------------------------------------------------------------------------- #
# D. Engine-level: one v6 run -- causal guidance + self-play + utility + history
# --------------------------------------------------------------------------- #

def test_v6_engine_produces_a_beats_human_op_via_self_play(
        hh_cfg, fast_hh_domain, tmp_path):
    cfg = dataclasses.replace(hh_cfg,
                              utility_history_path=str(tmp_path / "uh.jsonl"))
    eng = Engine(cfg, fast_hh_domain, make_roles(cfg))
    result = eng.run()
    assert result.status in {"accepted", "terminated", "exhausted"}
    v6 = result.v6_summary
    assert v6, "v6 summary must be produced"

    # (1) Every proposed op got an explicit utility label (no 'unknown' left over).
    util = v6["op_utility"]
    assert util and all(f != "unknown" for f in util.values()), util

    # (2) Proposal quality IMPROVED over v5: at least one proposed op BEATS the human
    #     Pareto frontier (v5 produced zero beats_human).
    assert v6["utility"]["beats_human"] >= 1, util

    # (3) Self-play RAN and is DESCRIPTIVE (not gating). v7 made the baseline the
    #     best human PER GENERATOR (not a fixed strawman default), so wins are no
    #     longer guaranteed -- we assert it ran + produced a margin, not a win.
    sp = v6["self_play"]
    assert sp["matches"] >= 1
    assert "win_rate" in sp and "mean_margin" in sp
    assert eng.selfplay.best_margin() is not None

    # (4) The cross-run history was appended.
    assert (tmp_path / "uh.jsonl").exists()
    assert v6["history"]["ops"] >= 1


def test_v6_higher_volume_proposes_more_than_v5(hh_cfg, fast_hh_domain):
    # v6 raises proposal volume + composes each op over more backbones.
    eng = Engine(hh_cfg, fast_hh_domain, make_roles(hh_cfg))
    eng.run()
    prov = eng.vocab.provenance_summary()
    assert prov["proposed_ops"] >= 3       # declow + min_or_one + ... admitted
    # and the protective invariants still hold (frontier members resolvable).
    for p in eng.frontier.points:
        assert eng.memory.get(p.approach_id) is not None
