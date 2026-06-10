"""v5 OPEN VOCABULARY: the Builder proposes new primitive OPS as source and the
Adversary proposes new GENERATORS as source; both are AST-screened + grounded
before they can join the vocabulary / harden the suite; provenance (human vs
system-proposed) is tracked; and every v2-v4 protective invariant is preserved.

  A. the seed vocabulary is all human; admitting a proposed op extends assembly;
  B. an unsafe op/generator is rejected by the AST screen, a broken op by grounding;
  C. a proposed generator runs through the `prop:` dispatch and can collapse recall;
  D. an engine run proposes + grounds + ADMITS new ops/generators, at least one
     proposed op reaches the frontier, and at least one proposed generator exposes
     a candidate -- with comparability + frontier-protected memory intact.
"""

import dataclasses

from rde.config import Config
from rde.domains import heavy_hitters as hh
from rde.engine import Engine
from rde.grounding import (all_safety_violations, ground, measure_recall_on,
                           probe_generator)
from rde.roles import make_roles
from rde.vocab import (INHERIT, OpProposal, PRE, Provenance, VICTIM)

CFG = Config(bench_repeats=1)


def _probe_gen(body, name, n=40000):
    block = hh.make_generator_block(name, body)
    spec = dataclasses.replace(
        hh.GROUNDING_SPEC,
        workload_gen_src=hh.GROUNDING_SPEC.workload_gen_src + "\n" + block)
    return probe_generator(hh.PROBE_SOLVER,
                           {"name": name, "n": n, "k": 10, "dist": f"prop:{name}"},
                           spec, CFG)


# --------------------------------------------------------------------------- #
# A. Vocabulary registry + provenance
# --------------------------------------------------------------------------- #

def test_seed_vocabulary_is_all_human():
    v = hh.seed_vocabulary()
    assert v.has_op(VICTIM, "min") and v.has_op(INHERIT, "vhalf")
    assert v.has_op(PRE, "none")
    for e in (v.victim["min"], v.inherit["vhalf"], v.pre["none"]):
        assert e.origin is Provenance.HUMAN
    assert v.proposed_ops() == []                # nothing proposed yet
    prov = v.provenance_summary()
    assert prov["human_ops"] >= 9 and prov["proposed_ops"] == 0
    assert prov["human_generators"] >= 1


def test_admitting_a_proposed_op_extends_assembly():
    v = hh.seed_vocabulary()
    p = OpProposal(PRE, "prune_ones", hh._PRE_PRUNE_ONES, Provenance.PROPOSED_DET, "x")
    hh.admit_op(v, p, iteration=4)
    assert v.has_op(PRE, "prune_ones")
    assert "prune_ones" in v.proposed_op_names()
    assert v.op_entry("prune_ones").origin is Provenance.PROPOSED_DET
    assembled = hh.assemble_from_vocab(v)
    using = [lbl for (lbl, code, ops) in assembled if "prune_ones" in ops]
    assert using, "a proposed op must produce composed policies"
    # Every assembled policy is valid, marked, and byte-distinct from the library.
    lib = {s.code.strip() for s in hh.STRATEGIES.values()}
    for (lbl, code, ops) in assembled:
        compile(code, "<asm>", "exec")
        assert "# v4-synth-policy" in code
        assert code.strip() not in lib


# --------------------------------------------------------------------------- #
# B. Op proposal SAFETY + grounding gate
# --------------------------------------------------------------------------- #

def test_unsafe_op_is_rejected_by_ast_screen():
    # An inherit op smuggling a forbidden name: the probe policy USING it must trip
    # the AST screen, so the engine can never even ground (let alone admit) it.
    v = hh.seed_vocabulary()
    bad = OpProposal(INHERIT, "evilop", "__import__('os').getpid()",
                     Provenance.PROPOSED_LLM, "")
    probe = hh.op_probe(v, bad)
    assert all_safety_violations(probe)


def test_broken_op_is_rejected_by_grounding():
    # A victim op that never assigns `vk`: the probe is AST-safe but raises at
    # runtime on overflow -> grounding reports incorrect -> the engine drops it.
    v = hh.seed_vocabulary()
    broken = OpProposal(VICTIM, "novk", ["pass"], Provenance.PROPOSED_DET, "")
    probe = hh.op_probe(v, broken)
    assert not all_safety_violations(probe)       # passes the screen ...
    ev = ground(probe, hh.GROUNDING_SPEC, CFG)
    assert not ev.correct                          # ... but fails grounding


# --------------------------------------------------------------------------- #
# C. Generator proposal SAFETY + grounding via the `prop:` dispatch
# --------------------------------------------------------------------------- #

def _spec_with_generator(name, body):
    block = hh.make_generator_block(name, body)
    return block, dataclasses.replace(
        hh.GROUNDING_SPEC,
        workload_gen_src=hh.GROUNDING_SPEC.workload_gen_src + "\n" + block)


def test_proposed_generator_runs_through_prop_dispatch():
    block, spec = _spec_with_generator("comeback", hh._GEN_COMEBACK)
    assert not all_safety_violations(block)
    rec = measure_recall_on(hh.PROBE_SOLVER,
                            [hh.generator_probe_spec("comeback", 10)], spec, CFG)
    assert "comeback" in rec                       # produced a runnable stream


def test_unsafe_generator_is_rejected_by_ast_screen():
    block = hh.make_generator_block("evilgen", "import os\nreturn [os.getpid()] * n")
    assert all_safety_violations(block)            # `import os` tripped


def test_decoy_generator_collapses_a_low_memory_policy():
    # The proposed 'decoy' generator must drop a churning low-memory policy's recall
    # below the collapse floor -- the Adversary's invented attack does real work.
    _block, spec = _spec_with_generator("decoy", hh._GEN_DECOY)
    wl = {"name": "decoy", "n": 40000, "k": 10, "dist": "prop:decoy"}
    victim = hh._assemble_policy("min", "v", "dec_all", 16)   # the min_dec_keep policy
    r = measure_recall_on(victim, [wl], spec, CFG).get("decoy")
    assert r is not None and r < CFG.metric_collapse_floor


# --------------------------------------------------------------------------- #
# D. Engine-level: a full open-vocabulary run
# --------------------------------------------------------------------------- #

def test_v5_engine_proposes_grounds_and_uses_new_vocabulary(hh_cfg, fast_hh_domain):
    eng = Engine(hh_cfg, fast_hh_domain, make_roles(hh_cfg))
    result = eng.run()
    assert result.status in {"accepted", "terminated", "exhausted"}
    assert eng.vocab is not None
    prov = result.vocab_provenance

    # (A) the Builder proposed + the engine grounded + ADMITTED new ops ...
    assert prov["proposed_ops"] >= 1
    # ... and at least one PROPOSED op reached the frontier (advanced discovery).
    assert prov["proposed_ops_used"] >= 1
    assert any(e.used_on_frontier for e in eng.vocab.proposed_ops())
    frontier_ops = set()
    for p in eng.frontier.points:
        a = eng.memory.get(p.approach_id)
        frontier_ops.update(getattr(a, "ops_used", []) or [])
    assert frontier_ops & eng.vocab.proposed_op_names(), \
        "a proposed op must be used by a frontier policy"

    # (B) the Adversary proposed + grounded a new generator, and one exposed someone.
    assert prov["proposed_generators"] >= 1
    assert prov["proposed_generators_exposing"] >= 1
    assert any(g.exposed_a_candidate for g in eng.vocab.proposed_generators())


def test_v5_preserves_comparability_and_frontier_protection(hh_cfg, fast_hh_domain):
    eng = Engine(hh_cfg, fast_hh_domain, make_roles(hh_cfg))
    eng.run()
    suite = [w["name"] for w in eng.spec.workloads]
    # The suite hardened beyond the base (proposed/param/gen attacks were added).
    assert len(suite) > len(fast_hh_domain.grounding_spec.workloads)
    for p in eng.frontier.points:
        a = eng.memory.get(p.approach_id)
        assert a is not None, "frontier-protected memory: member must be resolvable"
        qbw = a.evidence.quality_by_workload or {}
        missing = [w for w in suite if w not in qbw]
        assert not missing, f"{a.id} not scored on the full live suite: {missing}"
        # hypervolume-primary fitness: a frontier member's fitness IS its HV gain.
        if a.fitness is not None and a.frontier_gain is not None:
            assert abs(a.fitness - a.frontier_gain) < 1e-9


def test_open_vocabulary_can_be_disabled(hh_cfg, fast_hh_domain):
    cfg = dataclasses.replace(hh_cfg, open_vocabulary=False)
    eng = Engine(cfg, fast_hh_domain, make_roles(cfg))
    assert eng.vocab is None and not eng.open_vocab
    result = eng.run()
    assert not result.vocab_provenance            # v4 path: nothing proposed


# --------------------------------------------------------------------------- #
# E. Review hardening: a broken/unbounded/non-stream generator must be REJECTED,
#    and a proposed op must be reported honestly (it MATCHES, not BEATS, human ops).
# --------------------------------------------------------------------------- #

def test_raising_generator_is_rejected_not_admitted_as_zero():
    # The old gate admitted a raising generator as "recall 0.0". The split runner +
    # probe_generator must report it DID NOT run -> rejectable.
    ok, _slen, err = _probe_gen("raise ValueError('boom')\nreturn []", "raisegen")
    assert not ok and "gen:" in (err or "")


def test_nonstream_generator_is_rejected():
    for body in ("return 42", "return 'hello'", "return [1.5] * n", "return []"):
        ok, _slen, _err = _probe_gen(body, "nsgen")
        assert not ok, f"non-stream body should be rejected: {body!r}"


def test_superlinear_generator_length_is_caught():
    # A generator that returns far more than n elements runs, but probe_generator
    # reports its length so the engine's boundedness gate (len > 4*n) rejects it.
    ok, slen, _err = _probe_gen("return list(range(n * 5))", "superlin", n=4000)
    assert ok and slen > 4 * 4000


def test_valid_generator_reports_bounded_length():
    ok, slen, _err = _probe_gen(hh._GEN_DECOY, "decoy2", n=40000)
    assert ok and slen == 40000


def test_proposed_op_matches_but_does_not_beat_human_at_equal_memory():
    # The honesty invariant the review demanded: a proposed op composed onto the
    # backbone does NOT strictly beat the human op at the same memory on the hardened
    # suite -- it ties. (min_or_one vs the human min victim, vhalf inherit, cap 16.)
    suite = (list(hh.WORKLOADS) + hh.adversary_param_attacks(10)
             + hh.adversary_generators(10))
    names = [w["name"] for w in suite]
    proposed = hh._assemble_core(hh._VIC_MIN_OR_ONE, hh._INHERIT_OPS["vhalf"],
                                 hh._PRE_OPS["none"], 16)
    human = hh._assemble_core(hh._VICTIM_OPS["min"], hh._INHERIT_OPS["vhalf"],
                              hh._PRE_OPS["none"], 16)
    pr = measure_recall_on(proposed, suite, hh.GROUNDING_SPEC, CFG)
    hr = measure_recall_on(human, suite, hh.GROUNDING_SPEC, CFG)
    pm = sum(pr.get(n, 0.0) for n in names) / len(names)
    hm = sum(hr.get(n, 0.0) for n in names) / len(names)
    assert pm <= hm + 0.01, "proposed op must not be claimed to BEAT the human op"


def test_v5_frontier_protected_under_memory_pressure(hh_cfg, fast_hh_domain):
    # v5's vocabulary search grows the frontier; with a cap BELOW the frontier size
    # the protected (recall, memory) members must still never be evicted (reorder
    # alone was insufficient -- protected members are exempt from cap truncation).
    cfg = dataclasses.replace(hh_cfg, memory_max_approaches=4)
    eng = Engine(cfg, fast_hh_domain, make_roles(cfg))
    eng.run()
    assert eng.memory.total_seen > len(eng.memory.approaches)   # pruning fired
    for p in eng.frontier.points:
        assert eng.memory.get(p.approach_id) is not None, \
            "a frontier member was evicted despite protection"
    rec = eng.frontier.recommended(eng.budget)
    assert rec is not None and eng.memory.get(rec.approach_id) is not None
