import json

import pytest

from rde.campaign import load_campaign, validate_campaign
from rde.config import Config


def _manifest():
    return {
        "schema": "rde-campaign/v1", "id": "test-campaign",
        "domain": "cache_eviction", "backend": "codex-cli",
        "models": {"builder": "gpt-5.6-terra", "adversary": "gpt-5.5",
                   "synthesizer": "gpt-5.6-terra"},
        "minimum_independence": "intra-family-diverse",
        "max_iterations": 1, "builder_fanout": 1,
        "claim_mode": "external_gate_required",
        "evidence_requirements": ["preserve audit"],
        "kill_conditions": ["holdout failure is a negative result"],
    }


def test_campaign_manifest_accepts_only_the_frozen_configuration(tmp_path):
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    manifest = load_campaign(path)
    cfg = Config(backend="codex-cli", models=dict(manifest["models"]))

    assert validate_campaign(manifest, cfg, "cache_eviction") == []
    assert validate_campaign(manifest, cfg, "topk")
    cfg.models["adversary"] = "one-entitled-model"
    assert "configured role models differ" in " ".join(
        validate_campaign(manifest, cfg, "cache_eviction"))


def test_campaign_manifest_requires_kill_conditions(tmp_path):
    bad = _manifest()
    bad["kill_conditions"] = []
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")

    with pytest.raises(ValueError, match="kill conditions"):
        load_campaign(path)
