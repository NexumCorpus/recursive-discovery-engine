from rde.causal_xenosoma_checker import check
from rde.domains.causal_xenosoma import generate_family, preregistration, run


def test_preregistered_domain_is_balanced_and_surface_only_controls_are_null():
    prereg = preregistration()
    result = run({"surface": prereg["surface"], "toggle": prereg["toggle"]})
    assert result["ok"] is True
    assert result["falsifiers"] == []
    assert result["measurements"]["trial"]["informationGainBits"] == 1.0
    assert result["measurements"]["holdout"]["informationGainBits"] == 1.0
    assert all(value["informationGainBits"] == 0.0 for value in result["measurements"]["baselines"].values())
    assert len(generate_family("rde-owned-unseen-holdout-v1", 4)) == 4


def test_checker_recomputes_and_rejects_edited_measurements():
    prereg = preregistration()
    artifact = {"surface": prereg["surface"], "toggle": prereg["toggle"]}
    result = run(artifact)
    edited = {**result, "measurements": {**result["measurements"], "trial": {"accuracy": 0.0}}}
    checked = check(artifact, edited)
    assert checked["ok"] is False
    assert "measurements-mismatch" in checked["problems"]
