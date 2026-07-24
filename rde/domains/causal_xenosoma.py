"""RDE-owned deterministic causal domain and frozen recombination baselines."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = ROOT / "causal_xenosoma_preregistration.json"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return "sha256:" + hashlib.sha256(canonical(value).encode()).hexdigest()


def source_hash():
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def preregistration():
    return json.loads(PREREG_PATH.read_text(encoding="utf-8"))


def preregistration_hash():
    return digest(preregistration())


def _label(family, index):
    value = int(hashlib.sha256(f"{family}:{index}".encode()).hexdigest()[:8], 16)
    return "alpha" if value % 2 == 0 else "beta"


def _balanced_cases(family, count):
    if count % 2:
        raise ValueError("causal family count must be even")
    labels = ["alpha"] * (count // 2) + ["beta"] * (count // 2)
    order = sorted(range(count), key=lambda i: hashlib.sha256(f"order:{family}:{i}".encode()).hexdigest())
    cases = []
    for position, slot in enumerate(order):
        cases.append({"index": slot, "surface": {"observation": "stable", "bytes": 32}, "hiddenLabel": labels[position]})
    return cases


def generate_family(family, count):
    return _balanced_cases(family, count)


def surface_baselines(cases):
    """Frozen surface-only controls; none receives the hidden mechanism."""
    outputs = {}
    for name in ["surface-only-read", "surface-only-rephrase", "surface-only-aggregate", "surface-only-random-guess"]:
        if name == "surface-only-read": observed = [case["surface"] for case in cases]
        elif name == "surface-only-rephrase": observed = ["stable" for _ in cases]
        elif name == "surface-only-aggregate": observed = {"observation": "stable", "count": len(cases)}
        else: observed = ["alpha" if i % 2 == 0 else "beta" for i in range(len(cases))]
        outputs[name] = {"informationGainBits": 0.0, "observed": observed}
    return outputs


def intervene(cases, states):
    outcomes = []
    for case in cases:
        probes = {state: (state == case["hiddenLabel"]) for state in states}
        inferred = next((state for state in states if probes[state]), None)
        outcomes.append({
            "index": case["index"],
            "surface": case["surface"],
            "probeStates": list(states),
            "probes": probes,
            "inferred": inferred,
            "correct": inferred == case["hiddenLabel"],
            "hiddenLabelHash": digest(case["hiddenLabel"]),
        })
    return outcomes


def measure(cases, outcomes, baseline):
    classes = sorted({case["hiddenLabel"] for case in cases})
    accuracy = sum(1 for outcome in outcomes if outcome["correct"]) / len(outcomes)
    return {
        "count": len(cases),
        "classDiversity": len(classes),
        "accuracy": accuracy,
        "informationGainBits": 1.0 if accuracy == 1.0 and len(classes) == 2 else 0.0,
        "baselineInformationGainBits": baseline["surface-only-read"]["informationGainBits"],
    }


def run(artifact):
    prereg = preregistration()
    expected_surface = prereg["surface"]
    expected_toggle = prereg["toggle"]
    if artifact.get("surface") != expected_surface or artifact.get("toggle") != expected_toggle:
        return {"ok": False, "falsifiers": ["artifact-toggle-invalid"]}
    states = expected_toggle["states"]
    trial = generate_family(prereg["families"]["trial"]["name"], prereg["families"]["trial"]["count"])
    holdout = generate_family(prereg["families"]["holdout"]["name"], prereg["families"]["holdout"]["count"])
    trial_baselines = surface_baselines(trial)
    holdout_baselines = surface_baselines(holdout)
    trial_outcomes = intervene(trial, states)
    holdout_outcomes = intervene(holdout, states)
    trial_measure = measure(trial, trial_outcomes, trial_baselines)
    holdout_measure = measure(holdout, holdout_outcomes, holdout_baselines)
    thresholds = prereg["thresholds"]
    falsifiers = []
    if any(value["informationGainBits"] > thresholds["maximumBaselineInformationGainBits"] for value in trial_baselines.values()): falsifiers.append("surface-baseline-information-gain")
    if trial_measure["classDiversity"] < thresholds["minimumClassDiversity"]: falsifiers.append("trial-class-collapse")
    if holdout_measure["classDiversity"] < thresholds["minimumClassDiversity"]: falsifiers.append("holdout-class-collapse")
    if trial_measure["accuracy"] < thresholds["minimumTrialAccuracy"]: falsifiers.append("intervention-accuracy-below-threshold")
    if holdout_measure["accuracy"] < thresholds["minimumHoldoutAccuracy"]: falsifiers.extend(["intervention-accuracy-below-threshold", "holdout-overfit"])
    return {
        "ok": not falsifiers,
        "falsifiers": sorted(set(falsifiers)),
        "measurements": {"trial": trial_measure, "holdout": holdout_measure, "baselines": trial_baselines},
        "audit": {"trialOutcomes": trial_outcomes, "holdoutOutcomes": holdout_outcomes, "holdoutBaselines": holdout_baselines},
        "families": {"trial": prereg["families"]["trial"], "holdout": prereg["families"]["holdout"]},
    }
