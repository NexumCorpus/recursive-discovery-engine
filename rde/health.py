"""Bounded structural health rehearsal for the Recursive Discovery Engine.

This is deliberately not a performance benchmark. It drives one real grounded
discovery cycle, then checks the properties that make RDE an operating organ:
multiple classes were explored, the Synthesizer changed direction, a correct
hybrid was produced from compatible evidence, and a grounded best survived.
"""
from __future__ import annotations

import dataclasses
import json

from .config import Config
from .domains import topk
from .engine import Engine
from .grounding import GroundingSpec
from .roles import make_roles
from .types import Action, AlgoClass


class _EphemeralEngine(Engine):
    """Exercise the full loop without turning a health pulse into a run claim."""

    def _persist(self, result, started_at):  # noqa: ARG002
        return "health://ephemeral"


def _domain():
    base = topk.GROUNDING_SPEC
    spec = GroundingSpec(
        oracle_src=base.oracle_src,
        workload_gen_src=base.workload_gen_src,
        criteria_cases=list(base.criteria_cases),
        workloads=[
            {"name": "health_k10", "n": 20000, "k": 10, "dist": "random"},
            {"name": "health_k50", "n": 20000, "k": 50, "dist": "random"},
            {"name": "health_klarge", "n": 4000, "k": 2000, "dist": "random"},
            {"name": "health_kfull", "n": 4000, "k": 4000, "dist": "random"},
        ],
        mem_workload={"name": "health_mem", "n": 20000, "k": 50,
                      "dist": "random"},
    )
    return dataclasses.replace(topk.DOMAIN, grounding_spec=spec)


def rehearse() -> dict:
    cfg = Config(backend="deterministic", bench_repeats=1, max_iterations=8)
    engine = _EphemeralEngine(cfg, _domain(), make_roles(cfg))
    result = engine.run()
    approaches = list(engine.memory.approaches)
    classes = sorted({a.algo_class.value for a in approaches})
    hybrids = [a for a in approaches
               if a.algo_class == AlgoClass.HYBRID and a.is_correct]
    synthesized = any(row.directive.action == Action.SYNTHESIZE
                      for row in result.iterations)
    parent_classes = sorted({
        engine.memory.get(parent_id).algo_class.value
        for hybrid in hybrids for parent_id in hybrid.parent_ids
        if engine.memory.get(parent_id) is not None
    })
    checks = {
        "iterations": bool(result.iterations),
        "grounded_best": bool(result.best and result.best.is_correct),
        "class_diversity": len(classes) >= 3,
        "synthesizer_directed": synthesized,
        "correct_hybrid": bool(hybrids),
        "compatible_parents": {"heap", "full-sort"}.issubset(parent_classes),
    }
    return {
        "status": "HEALTHY" if all(checks.values()) else "ROT",
        "checks": checks,
        "iterations": len(result.iterations),
        "classes": classes,
        "hybrids": [a.id for a in hybrids],
        "hybridParentClasses": parent_classes,
        "best": result.best.id if result.best else None,
    }


def main() -> None:
    receipt = rehearse()
    print(json.dumps(receipt, sort_keys=True))
    if receipt["status"] != "HEALTHY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
