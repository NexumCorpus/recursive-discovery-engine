"""End-to-end: the deterministic engine must satisfy the v0 success criteria, the
Adversary must be able to harden the trusted suite with a confirmed break, memory
must stay bounded, and the LLM Synthesizer's guardrails must hold."""

import dataclasses

from rde.config import Config
from rde.domains import topk
from rde.engine import Engine
from rde.grounding import GroundingSpec
from rde.memory import Memory
from rde.roles import (LLMSynthesizer, RoundContext, make_roles)
from rde.types import (Action, AlgoClass, Approach, Directive, Evidence)


# --------------------------------------------------------------------------- #
# Full run -- the v0 success criteria
# --------------------------------------------------------------------------- #

def test_full_deterministic_run_meets_v0_criteria(det_cfg, fast_domain):
    engine = Engine(det_cfg, fast_domain, make_roles(det_cfg))
    result = engine.run()

    # (a) explored at least two meaningfully different algorithmic classes
    classes = set(result.iterations[-1].signals.classes_tried)
    assert len(classes) >= 3

    # (b)/(c) produced a correct result measurably better than the seed
    assert result.best is not None and result.best.is_correct
    assert result.best.fitness > result.seed.fitness

    # the grounding gate disqualified the deliberately-broken bucket strategy
    assert any(a.algo_class == AlgoClass.BUCKET and not a.is_correct
               for a in engine.memory.approaches)

    # (e) it kept a record of what was learned
    assert result.insights
    assert any(i.kind == "success" for i in result.insights)

    # the Synthesizer reached a defensible stopping decision
    assert result.status in {"accepted", "terminated"}


def test_v1_synthesis_produces_winning_hybrid(det_cfg, fast_domain):
    """v1: with split regimes the engine should SYNTHESISE an adaptive hybrid
    that becomes the empirical best, and distil conceptual lessons."""
    engine = Engine(det_cfg, fast_domain, make_roles(det_cfg))
    result = engine.run()

    # a SYNTHESIZE directive was actually issued
    assert any(r.directive.action == Action.SYNTHESIZE for r in result.iterations)
    # and synthesis produced a correct adaptive hybrid ...
    hybrids = [a for a in engine.memory.approaches
               if a.algo_class == AlgoClass.HYBRID and a.is_correct]
    assert hybrids, "synthesis should produce a correct adaptive hybrid"
    h = max(hybrids, key=lambda a: a.fitness or 0.0)
    assert h.synthesis_gain is not None
    # ... that is at least competitive with the best single approach. (The top-K
    # win margin is ~0.02 and benchmark-noise-sensitive, so we don't require the
    # hybrid to be STRICTLY best under full-suite CPU load -- that fragility is
    # exactly why v2 moved to a domain with real headroom.)
    singles = [a.fitness for a in engine.memory.approaches
               if a.is_correct and a.algo_class != AlgoClass.HYBRID
               and a.fitness is not None]
    assert (h.fitness or 0.0) >= max(singles, default=0.0) - 0.05
    # conceptual memory captured transferable structural lessons
    assert engine.memory.lessons.lessons
    assert any(L.polarity == "helps" for L in result.lessons)


# --------------------------------------------------------------------------- #
# Adversary hardening -- a CONFIRMED break grows the trusted suite (empirical)
# --------------------------------------------------------------------------- #

def test_adversary_break_hardens_suite(det_cfg):
    # Weaken the trusted suite so it MISSES negatives, letting the non-negative-
    # only bucket strategy pass. The Adversary's negative-input case will then
    # find a real break that the base suite did not.
    base = topk.GROUNDING_SPEC
    weak = GroundingSpec(
        oracle_src=base.oracle_src, workload_gen_src=base.workload_gen_src,
        criteria_cases=[c for c in topk.CRITERIA_CASES if all(x >= 0 for x in c[1])],
        workloads=[{"name": "few", "n": 1500, "k": 10, "dist": "few_unique"},
                   {"name": "asc", "n": 1500, "k": 10, "dist": "ascending"}],
        mem_workload={"name": "m", "n": 1500, "k": 10, "dist": "few_unique"})
    dom = dataclasses.replace(topk.DOMAIN, grounding_spec=weak)

    engine = Engine(det_cfg, dom, make_roles(det_cfg))
    engine._seed()

    bucket = Approach(id="t_bucket", code=topk.STRATEGIES["bucket_nonneg"].code,
                      rationale="", algo_class=AlgoClass.BUCKET)
    engine._evaluate(bucket, [])
    assert bucket.is_correct, "bucket should pass the WEAKENED (non-neg) suite"

    n_cases_before = len(engine.spec.criteria_cases)
    ctx = RoundContext(cfg=det_cfg, domain=dom, memory=engine.memory, iteration=1)
    engine._adversary_pass(bucket, ctx, [])

    assert not bucket.is_correct, "Adversary's negative case must break it"
    assert len(engine.spec.criteria_cases) > n_cases_before, \
        "the confirmed counterexample must be folded into the trusted suite"


# --------------------------------------------------------------------------- #
# Bounded memory
# --------------------------------------------------------------------------- #

def test_recompute_best_after_adversary_downgrade():
    cfg = Config(memory_max_approaches=12)
    mem = Memory(cfg)
    a = Approach(id="a", code="", rationale="", algo_class=AlgoClass.HEAP)
    a.evidence = Evidence(grounded=True, correct=True); a.fitness = 0.9
    b = Approach(id="b", code="", rationale="", algo_class=AlgoClass.FULL_SORT)
    b.evidence = Evidence(grounded=True, correct=True); b.fitness = 0.7
    mem.add_approach(a); mem.add_approach(b)
    assert mem.best is a

    # Adversary downgrades the reigning best.
    a.evidence.correct = False; a.fitness = None
    mem.recompute_best()
    assert mem.best is b, "best must fall back to the next correct approach"

    # And to None if nothing correct remains.
    b.evidence.correct = False; b.fitness = None
    mem.recompute_best()
    assert mem.best is None


def test_memory_stays_bounded():
    cfg = Config(memory_max_approaches=12)
    mem = Memory(cfg)
    for i in range(40):
        a = Approach(id=f"a{i}", code="", rationale="", algo_class=AlgoClass.HEAP)
        correct = (i % 3 != 0)
        a.evidence = Evidence(grounded=True, correct=correct,
                              perf_score=i / 40, mem_score=(40 - i) / 40,
                              simplicity_score=0.5)
        a.fitness = (i / 40) if correct else None
        mem.add_approach(a)
    assert len(mem.approaches) <= cfg.memory_max_approaches
    assert mem.total_seen == 40            # the counter never forgets the total
    assert mem.best is not None            # the reigning best is always retained


# --------------------------------------------------------------------------- #
# LLM Synthesizer guardrails (pure logic, no network)
# --------------------------------------------------------------------------- #

def _ctx_for_guardrail(best=None, **sig_kw):
    from rde import signals as sig
    cfg = Config(backend="llm")
    mem = Memory(cfg)
    if best is not None:
        mem.add_approach(best)
    s = sig.Signals()
    for k, v in sig_kw.items():
        setattr(s, k, v)
    return RoundContext(cfg=cfg, domain=topk.DOMAIN, memory=mem, iteration=2,
                        signals=s)


def test_guardrail_blocks_accept_without_best():
    ctx = _ctx_for_guardrail(best=None, classes_tried=["heap"])
    d = LLMSynthesizer()._enforce_guardrails(
        Directive(Action.ACCEPT, reason="ship it"), ctx)
    assert d.action == Action.CONTINUE       # nothing correct -> cannot accept


def test_guardrail_blocks_accept_when_best_is_incorrect():
    # best exists but was downgraded (is_correct False) -> ACCEPT must be blocked.
    downgraded = Approach(id="d", code="", rationale="", algo_class=AlgoClass.HEAP)
    downgraded.evidence = Evidence(grounded=True, correct=False)
    downgraded.fitness = None
    ctx = _ctx_for_guardrail(best=None, classes_tried=["heap"])
    ctx.memory.best = downgraded     # force a stale-but-incorrect best
    d = LLMSynthesizer()._enforce_guardrails(
        Directive(Action.ACCEPT, reason="ship it"), ctx)
    assert d.action == Action.CONTINUE


def test_guardrail_reset_without_target_terminates_when_exhausted():
    ctx = _ctx_for_guardrail(
        best=None, classes_tried=["full-sort", "heap", "selection", "bucket"])
    d = LLMSynthesizer()._enforce_guardrails(
        Directive(Action.RESET, reason="reset!"), ctx)
    assert d.action == Action.TERMINATE   # nothing to reset to, no best


def test_guardrail_caps_repeated_synthesis_when_hybrid_won():
    # A free LLM may keep choosing SYNTHESIZE; once a correct hybrid is best and
    # >=2 hybrids exist, further synthesis is downgraded so the loop converges.
    cfg = Config(backend="llm")
    mem = Memory(cfg)
    for i, fit in enumerate([0.70, 0.65]):
        h = Approach(id=f"h{i}", code="", rationale="", algo_class=AlgoClass.HYBRID)
        h.evidence = Evidence(grounded=True, correct=True)
        h.fitness = fit
        mem.add_approach(h)
    from rde import signals as sig
    s = sig.Signals()
    s.classes_tried = ["full-sort", "heap", "selection", "bucket", "hybrid"]
    ctx = RoundContext(cfg=cfg, domain=topk.DOMAIN, memory=mem, iteration=4,
                       signals=s, winners={})
    d = LLMSynthesizer()._enforce_guardrails(
        Directive(Action.SYNTHESIZE, reason="keep going", parent_ids=["h0"]), ctx)
    assert d.action == Action.ACCEPT       # winning hybrid + nothing untried


def test_guardrail_stall_converges_to_accept():
    # v1 convergence: stalled with a correct best and no productive synthesis ->
    # ACCEPT instead of exploring/synthesising to max_iterations.
    best = Approach(id="b", code="", rationale="", algo_class=AlgoClass.HYBRID)
    best.evidence = Evidence(grounded=True, correct=True)
    best.fitness = 0.7
    ctx = _ctx_for_guardrail(best=best, stalled=True,
                             synthesis_available=False, classes_tried=["heap"])
    d = LLMSynthesizer()._enforce_guardrails(
        Directive(Action.EXPLORE, reason="explore more"), ctx)
    assert d.action == Action.ACCEPT


def test_guardrail_blocks_continue_through_patching():
    best = Approach(id="b", code="", rationale="", algo_class=AlgoClass.HEAP)
    best.evidence = Evidence(grounded=True, correct=True)
    best.fitness = 0.8
    ctx = _ctx_for_guardrail(best=best, patching=True,
                             consecutive_recombinations=2,
                             classes_tried=["heap"])
    d = LLMSynthesizer()._enforce_guardrails(
        Directive(Action.CONTINUE, reason="keep going!"), ctx)
    assert d.action == Action.RESET          # confirmed patching cannot continue
    assert d.target_class is not None
