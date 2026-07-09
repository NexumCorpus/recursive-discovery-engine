"""The model roster is evidence, not decoration: role diversity must be legible."""

import json
from pathlib import Path

from rde.config import Config, assess_model_assignment, model_vendor
from rde.domains import topk
from rde.engine import Engine
from rde.roles import make_roles


def test_codex_defaults_separate_builder_from_adversary(monkeypatch):
    for key in ("RDE_CODEX_MODEL", "RDE_BUILDER_MODEL", "RDE_ADVERSARY_MODEL",
                "RDE_SYNTHESIZER_MODEL"):
        monkeypatch.delenv(key, raising=False)
    cfg = Config(backend="codex-cli")

    assert cfg.models["builder"] == "gpt-5.6-terra"
    assert cfg.models["adversary"] == "gpt-5.5"
    assessment = assess_model_assignment(cfg.models)
    assert assessment["grade"] == "intra-family-diverse"
    assert not assessment["independent_corroboration"]


def test_global_codex_pin_is_reported_as_monoculture(monkeypatch):
    monkeypatch.setenv("RDE_CODEX_MODEL", "one-entitled-model")
    for key in ("RDE_BUILDER_MODEL", "RDE_ADVERSARY_MODEL", "RDE_SYNTHESIZER_MODEL"):
        monkeypatch.delenv(key, raising=False)

    assessment = assess_model_assignment(Config(backend="codex-cli").models)
    assert assessment["grade"] == "monoculture"
    assert not assessment["independent_corroboration"]


def test_cross_provider_roster_is_the_only_independent_grade():
    assessment = assess_model_assignment({
        "builder": "openai/gpt-5.6-terra",
        "adversary": "google/gemini-3.1-pro",
        "synthesizer": "anthropic/claude-opus-4.8",
    })
    assert assessment["grade"] == "cross-provider"
    assert assessment["independent_corroboration"]
    assert model_vendor("unfamiliar-model") == "unknown"


def test_run_artifacts_carry_model_assignment_provenance():
    cfg = Config(backend="deterministic", max_iterations=0)
    result = Engine(cfg, topk.DOMAIN, make_roles(cfg)).run()
    artifact = json.loads((Path(result.run_dir) / "audit" / "bundle.json").read_text())

    assert artifact["model_assignment"] == assess_model_assignment(cfg.models)
