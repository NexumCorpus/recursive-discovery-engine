"""Frozen campaign manifests for live discovery attempts.

A model run is easy to narrate after the fact. This module makes the relevant
choices executable before a run starts: domain, backend, roster, budget, claim
mode, and the conditions that make the attempt a failure rather than a story.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config, assess_model_assignment


SCHEMA = "rde-campaign/v1"
REQUIRED = {
    "schema", "id", "domain", "backend", "models", "max_iterations",
    "builder_fanout", "claim_mode", "minimum_independence",
    "evidence_requirements", "kill_conditions",
}
GRADES = ("monoculture", "intra-family-diverse", "cross-provider")


def load_campaign(path: str | Path) -> dict:
    source = Path(path)
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load campaign manifest {source}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("campaign manifest must be a JSON object")
    missing = sorted(REQUIRED - set(manifest))
    if missing:
        raise ValueError("campaign manifest missing required fields: " + ", ".join(missing))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported campaign schema: {manifest.get('schema')!r}")
    if manifest.get("claim_mode") not in {"exploratory_only", "external_gate_required"}:
        raise ValueError("claim_mode must be exploratory_only or external_gate_required")
    if manifest.get("minimum_independence") not in GRADES:
        raise ValueError("minimum_independence must name a known model-independence grade")
    if not isinstance(manifest["models"], dict) or set(manifest["models"]) != {
        "builder", "adversary", "synthesizer"
    }:
        raise ValueError("models must contain exactly builder, adversary, synthesizer")
    if int(manifest["max_iterations"]) < 1 or int(manifest["builder_fanout"]) < 1:
        raise ValueError("max_iterations and builder_fanout must both be positive")
    if not manifest["kill_conditions"] or not manifest["evidence_requirements"]:
        raise ValueError("campaign must contain evidence requirements and kill conditions")
    return manifest


def validate_campaign(manifest: dict, cfg: Config, domain_name: str) -> list[str]:
    """Return every mismatch; callers must refuse a campaign with any mismatch."""
    problems = []
    if manifest["domain"] != domain_name:
        problems.append(f"domain is {domain_name!r}, manifest pins {manifest['domain']!r}")
    if manifest["backend"] != cfg.backend:
        problems.append(f"backend is {cfg.backend!r}, manifest pins {manifest['backend']!r}")
    if manifest["models"] != cfg.models:
        problems.append("configured role models differ from the frozen manifest")
    assessment = assess_model_assignment(cfg.models)
    if GRADES.index(assessment["grade"]) < GRADES.index(manifest["minimum_independence"]):
        problems.append(
            f"model independence is {assessment['grade']!r}, below required "
            f"{manifest['minimum_independence']!r}"
        )
    return problems
