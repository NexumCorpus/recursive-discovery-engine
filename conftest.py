"""Pytest fixtures + path setup so `import rde` works from anywhere, and the
engine tests run on small, fast workloads instead of the full benchmark sizes.
"""

import dataclasses
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from rde.config import Config            # noqa: E402
from rde.domains import topk             # noqa: E402
from rde.domains import heavy_hitters as hh  # noqa: E402
from rde.grounding import GroundingSpec  # noqa: E402
from rde.types import AlgoClass          # noqa: E402


def _spec_with(workloads, mem_workload, criteria=None):
    base = topk.GROUNDING_SPEC
    return GroundingSpec(
        oracle_src=base.oracle_src,
        workload_gen_src=base.workload_gen_src,
        criteria_cases=list(base.criteria_cases if criteria is None else criteria),
        workloads=workloads,
        mem_workload=mem_workload)


@pytest.fixture
def det_cfg():
    # bench_repeats=1 keeps the subprocess fast; everything else default.
    return Config(backend="deterministic", bench_repeats=1, max_iterations=8)


@pytest.fixture
def fast_domain():
    """topk on small workloads -- correctness behaviour identical, just quick.
    Spans both a small-k (heap) and a large-k (sort) regime so the v1 synthesis
    path is exercised: an adaptive hybrid should win both."""
    spec = _spec_with(
        [{"name": "random_k10", "n": 20000, "k": 10, "dist": "random"},
         {"name": "random_k50", "n": 20000, "k": 50, "dist": "random"},
         {"name": "random_klarge", "n": 4000, "k": 2000, "dist": "random"},
         {"name": "random_kfull", "n": 4000, "k": 4000, "dist": "random"}],
        {"name": "mem", "n": 20000, "k": 50, "dist": "random"})
    return dataclasses.replace(topk.DOMAIN, grounding_spec=spec)


@pytest.fixture
def hh_cfg(tmp_path):
    # Point the v6 cross-run utility history at a throwaway file so the engine tests
    # never pollute the real runs/utility_history.jsonl demo artifact.
    return Config(backend="deterministic", bench_repeats=1, max_iterations=8,
                  utility_history_path=str(tmp_path / "test_utility_history.jsonl"))


@pytest.fixture
def fast_hh_domain():
    """Heavy-hitters trimmed for a quick run: 3 base workloads, 2 adversarial,
    and Count-Min dropped (its per-overflow eviction is slow). n stays at 40000 so
    distinct > the hybrid's budget and the hybrid genuinely overflows -> bounded."""
    base = hh.GROUNDING_SPEC
    spec = GroundingSpec(
        oracle_src=base.oracle_src, workload_gen_src=base.workload_gen_src,
        criteria_cases=list(base.criteria_cases),
        workloads=[{"name": "zipf", "n": 40000, "k": 10, "dist": "zipf"},
                   {"name": "front_loaded", "n": 40000, "k": 10,
                    "dist": "front_loaded"},
                   {"name": "competing_heavy", "n": 40000, "k": 10,
                    "dist": "competing_heavy"}],
        mem_workload={"name": "mem", "n": 40000, "k": 10, "dist": "uniform"},
        verdict_src=base.verdict_src)
    return dataclasses.replace(
        hh.DOMAIN, grounding_spec=spec,
        builder_priority=["misra_gries", "first_k_distinct"],
        meaningful_classes=[AlgoClass.EXACT, AlgoClass.MISRA_GRIES,
                            AlgoClass.SAMPLE],
        adversary_workloads=[
            {"name": "adv_freq_late", "n": 40000, "k": 10, "dist": "freq_late"},
            {"name": "adv_overflow", "n": 40000, "k": 10,
             "dist": "heavy_skew_overflow"}])
