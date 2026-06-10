"""v4: guided open-ended PRIMITIVE synthesis + generator-level Adversary + honest
frontier re-scoring.

Where v3 enumerated whole eviction POLICIES from a fixed named set, v4 supplies
only low-level OPERATIONS (victim choice / inherited count / global pre-step) and
SYNTHESISES the policy by composing them -- so most resulting policies were never
written by the implementer. These tests prove:

  A. the genome reaches structures OUTSIDE the v3 grammar, one of which the engine
     accepts and recommends (an invented primitive, not a recombination of named
     policies), and that invented primitive is more robust than the v3 policy it
     varies, at equal budget;
  B. the Adversary composes NEW attack TYPES (generator-DSL streams, not just
     re-parameterised templates) and surfaces weaknesses in the newly synthesised
     structures;
  C. hardening the suite RE-SCORES every existing frontier member, so all points
     stay comparable on the same (growing) metric -- closing v3's comparability gap.
"""

from rde.config import Config
from rde.domains import heavy_hitters as hh
from rde.engine import Engine
from rde.grounding import ground, measure_recall_on
from rde.roles import make_roles
from rde.types import AlgoClass

CFG = Config(bench_repeats=1)


# Distinctive source fragments that ONLY a v4 op can emit. The v3 grammar emits
# only `counts[x] = vv + 1` / `counts[x] = 1` with a min/first victim and no
# pre-step, so any of these proves a policy lies outside the v3 grammar.
_V4_ONLY_SIGNATURES = (
    "vv // 2",                                    # vhalf inheritance (decay)
    "max(1, vv - 1)",                             # vdec inheritance
    "_rest = [kk for kk in counts if kk != vk]",  # second_min victim
    "vk = max(counts, key=counts.get)",           # max victim
    "counts[_kk] -= 1",                           # dec_all pre-step
    "counts[_kk] = counts[_kk] // 2",             # halve_all pre-step
)


def _is_v4_only(code: str) -> bool:
    return any(sig in code for sig in _V4_ONLY_SIGNATURES)


def _full_suite(k: int = 10) -> list:
    """Base workloads + EVERY adversary attack (param + generator-DSL)."""
    return (list(hh.WORKLOADS)
            + hh.adversary_param_attacks(k)
            + hh.adversary_generators(k))


# --------------------------------------------------------------------------- #
# A. Genome reaches structures outside the v3 grammar
# --------------------------------------------------------------------------- #

def test_genome_synthesises_policies_outside_v3_grammar():
    # The genome must contain MORE than the three v3 named policies: the majority
    # of points should be tuples the v3 grammar never enumerated.
    v4_only = [(lbl, (v, i, p)) for (lbl, v, i, p, _m) in hh._GENOME
               if (v, i, p) not in hh.V3_POLICY_GENOMES]
    v3_pts = [(lbl, (v, i, p)) for (lbl, v, i, p, _m) in hh._GENOME
              if (v, i, p) in hh.V3_POLICY_GENOMES]
    assert len(v4_only) >= 5, "genome should be dominated by NEW (non-v3) policies"
    assert v3_pts, "keep a v3 baseline in the genome for the Adversary to collapse"

    # The op->source mapping is consistent: a v4-only genome emits a v4 signature;
    # a v3 genome does NOT (so the structural classifier is sound).
    for (lbl, v, i, p, m) in hh._GENOME:
        code = hh._assemble_policy(v, i, p, m)
        assert hh.classify(code) == AlgoClass.INVENTED   # all carry the genome marker
        assert "# v4-synth-policy" in code
        if (v, i, p) in hh.V3_POLICY_GENOMES:
            assert not _is_v4_only(code), f"{lbl} is a v3 policy yet looks v4-only"
        else:
            assert _is_v4_only(code), f"{lbl} is outside v3 grammar but unmarked"


def test_invented_halving_primitive_beats_v3_space_saving_at_equal_budget():
    # The headline primitive: min-evict with HALVED inheritance (vhalf) vs the v3
    # Space-Saving rule (vplus = victim+1), at the SAME capacity (mmul=32) so the
    # ONLY difference is the invented inheritance op. Measured on the full hostile
    # suite, the invented decay rule is strictly MORE ROBUST.
    invented = hh._assemble_policy("min", "vhalf", "none", 32)   # v4-only
    space_saving = hh._assemble_policy("min", "vplus", "none", 32)  # v3 grammar
    assert _is_v4_only(invented) and not _is_v4_only(space_saving)
    assert ("min", "vplus", "none") in hh.V3_POLICY_GENOMES
    assert ("min", "vhalf", "none") not in hh.V3_POLICY_GENOMES

    suite = _full_suite(10)
    inv_rec = measure_recall_on(invented, suite, hh.GROUNDING_SPEC, CFG)
    ss_rec = measure_recall_on(space_saving, suite, hh.GROUNDING_SPEC, CFG)
    names = [w["name"] for w in suite]
    inv_mean = sum(inv_rec.get(n, 0.0) for n in names) / len(names)
    ss_mean = sum(ss_rec.get(n, 0.0) for n in names) / len(names)
    assert inv_mean > 0.8, "invented primitive should reach high mean recall"
    assert inv_mean > ss_mean, "vhalf decay must beat v3 Space-Saving at equal cap"


def test_invented_policy_grounds_correct_and_bounded():
    # The winner must pass the REAL grounding harness (subprocess + oracle), recover
    # heavy hitters, and stay far under exact's O(distinct) memory.
    code = hh._assemble_policy("min", "vhalf", "none", 32)
    ev = ground(code, hh.GROUNDING_SPEC, CFG)
    assert ev.correct
    assert ev.quality > 0.6
    assert ev.peak_memory_bytes < 300 * 1024
    # And it is NOT byte-identical to any library strategy -- it was assembled.
    assert all(code.strip() != s.code.strip() for s in hh.STRATEGIES.values())


# --------------------------------------------------------------------------- #
# B. Generator-level Adversary: NEW attack TYPES, not re-parameterised templates
# --------------------------------------------------------------------------- #

def test_generator_dsl_composes_new_attack_types():
    gens = hh.adversary_generators(10)
    kinds = {g["dist"].split(":", 1)[1].split("|", 1)[0] for g in gens}
    assert {"phase", "spike", "mixture"} <= kinds      # three structurally new types
    for g in gens:
        assert g["dist"].startswith("gen:")            # generator DSL, not param:
        s = hh.gen_workload(2000, g["dist"], seed=3)
        assert len(s) == 2000                          # each actually generates


def test_phase_generator_shifts_the_heavy_set_midstream():
    # A 'phase' stream is STRUCTURALLY new: the heavy hitters in the first half are
    # a DIFFERENT set from those in the second half -- a non-stationarity no single
    # param template (which fixes one heavy set) can express.
    s = hh.gen_workload(8000, "gen:phase|ha=6|hb=6|split=0.5", seed=7)
    from collections import Counter
    first = Counter(s[:4000]).most_common(6)
    second = Counter(s[4000:]).most_common(6)
    top_first = {it for it, _ in first}
    top_second = {it for it, _ in second}
    assert top_first != top_second                     # the regime actually shifts


def test_generator_parser_is_bounded_and_safe():
    # Hostile / malformed generator specs must clamp or ignore, never crash or
    # exhaust memory (same guarantee the param parser gives).
    s = hh.gen_workload(500, "gen:spike|h=1e100|burst=nan", seed=1)
    assert len(s) == 500
    s2 = hh.gen_workload(500, "gen:phase|ha=abc|hb=-5|split=9", seed=1)
    assert len(s2) == 500
    s3 = hh.gen_workload(500, "gen:bogus_type|x=1", seed=1)
    assert len(s3) == 500                              # unknown type -> safe fallback


# --------------------------------------------------------------------------- #
# A+B+C. Engine-level: one run, three guarantees (kept to a single run for speed)
# --------------------------------------------------------------------------- #

def test_v4_engine_recommends_invented_primitive_and_keeps_frontier_honest(
        hh_cfg, fast_hh_domain):
    # Pinned to the v4 path (open_vocabulary=False): this asserts the v4 GENOME's
    # recommended policy is a v4-only INVENTED structure. v5/v6 open the vocabulary
    # and may recommend a proposed-op policy instead (covered by test_v5/test_v6).
    import dataclasses
    cfg = dataclasses.replace(hh_cfg, open_vocabulary=False)
    eng = Engine(cfg, fast_hh_domain, make_roles(cfg))
    result = eng.run()
    assert result.status in {"accepted", "terminated"}
    assert eng.frontier is not None

    # (A) The RECOMMENDED shippable point is an invented policy whose core logic is
    #     OUTSIDE the v3 grammar (a synthesised primitive, not a named recombination).
    rec = eng.frontier.recommended(eng.budget)
    assert rec is not None
    rec_a = eng.memory.get(rec.approach_id)
    assert rec_a.algo_class == AlgoClass.INVENTED
    assert rec_a.origin.value == "synthesis"
    assert "# v4-synth-policy" in rec_a.code
    assert _is_v4_only(rec_a.code), "recommended policy must use a v4-only op"
    assert rec.quality > 0.6 and rec.memory < 300 * 1024

    # (B) The Adversary surfaced a weakness in a newly SYNTHESISED structure.
    assert any("Adversary exposed" in i.text and "invented" in i.text
               for i in result.insights)

    # (C) Frontier RE-SCORING kept every member comparable: each frontier point has
    #     been measured on EVERY workload in the (hardened) suite -- including the
    #     seed, admitted before any hardening. This is the v4 honesty invariant.
    suite = [w["name"] for w in eng.spec.workloads]
    assert len(suite) > len(fast_hh_domain.grounding_spec.workloads)  # it hardened
    for p in eng.frontier.points:
        a = eng.memory.get(p.approach_id)
        assert a is not None, "frontier member was evicted from bounded memory"
        qbw = a.evidence.quality_by_workload or {}
        missing = [w for w in suite if w not in qbw]
        assert not missing, f"{a.id} not re-scored on {missing}"


def test_v4_frontier_survives_memory_pressure(hh_cfg, fast_hh_domain):
    """Regression for the eviction-shrink defect: under a memory cap small enough to
    force pruning, bounded memory must NOT evict a (recall, memory) frontier member
    (the pruner's perf/mem/simplicity Pareto set is a DIFFERENT objective). Every
    frontier point must stay resolvable AND fully covered, so hypervolume and the
    recommended point cannot be silently corrupted."""
    import dataclasses
    # Cap chosen to force at least one eviction while the (recall, memory) frontier
    # (~5 members) must still be fully retained: frontier_size <= cap < total_seen.
    # Pinned to the v4 path (open_vocabulary=False) so the frontier size is the v4
    # one this cap was calibrated against; v5's frontier-protection has its own test.
    tight = dataclasses.replace(hh_cfg, memory_max_approaches=5,
                                open_vocabulary=False)
    eng = Engine(tight, fast_hh_domain, make_roles(tight))
    result = eng.run()
    assert result.status in {"accepted", "terminated", "exhausted"}

    assert eng.memory.total_seen > len(eng.memory.approaches)        # pruning fired
    suite = [w["name"] for w in eng.spec.workloads]
    for p in eng.frontier.points:
        a = eng.memory.get(p.approach_id)
        assert a is not None, "a frontier member was pruned out of memory"
        missing = [w for w in suite if w not in (a.evidence.quality_by_workload or {})]
        assert not missing, f"{a.id} lost coverage on {missing}"
    # The recommended point is still resolvable (not a dangling id).
    rec = eng.frontier.recommended(eng.budget)
    assert rec is not None and eng.memory.get(rec.approach_id) is not None
