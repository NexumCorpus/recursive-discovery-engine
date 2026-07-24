#!/usr/bin/env python3
"""RDE-owned causal receipt verifier.

The caller submits only a declarative toggle and provenance anchors.  RDE owns
the hidden mechanisms, balanced families, baselines, execution, checker, and
all verdict-bearing managed state.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "xenosoma-receipts"
VERIFIER = "rde-xenosoma-v1"
PROTOCOL = "rde-causal-xenosoma-v1"
PREREG_PATH = ROOT / "rde" / "causal_xenosoma_preregistration.json"
ADAPTER_PATH = ROOT / "rde" / "domains" / "causal_xenosoma.py"
FORBIDDEN = {"rdeEvidence", "claimsPath", "runId", "rdeRunId", "auditHash", "recommended_grade", "managedRunRoot", "hiddenLabels", "seeds", "verdict", "measurements"}
ALLOWED_INPUT = {"causalArtifact", "experimentHash", "commitmentRecordHash", "perturbationRecordHash", "evidenceAnchors", "holdoutAnchors", "generator", "grader"}

sys.path.insert(0, str(ROOT))
from rde.causal_xenosoma_checker import check as check_domain  # noqa: E402
from rde.domains.causal_xenosoma import canonical, digest, preregistration, preregistration_hash, run as run_domain, source_hash  # noqa: E402


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((canonical(value) + "\n").encode("utf-8"))


def head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def preregistration_commit():
    return subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(PREREG_PATH.relative_to(ROOT))], cwd=ROOT, text=True).strip()


def exact_hash(value):
    return isinstance(value, str) and len(value) == 71 and value.startswith("sha256:") and all(c in "0123456789abcdef" for c in value[7:])


def path_is_safe(path):
    path = Path(path)
    return not path.is_symlink() and path.resolve().is_relative_to(RUN_ROOT.resolve())


def state_files(run_dir):
    return {"artifact.json": run_dir / "artifact.json", "claims.json": run_dir / "claims.json", "audit/bundle.json": run_dir / "audit" / "bundle.json", "audit/claim-checker.json": run_dir / "audit" / "claim-checker.json"}


def frozen_bar():
    prereg_hash = preregistration_hash()
    adapter_hash = source_hash()
    payload = {"protocol": PROTOCOL, "preregistrationHash": prereg_hash, "adapterSourceHash": adapter_hash, "thresholds": preregistration()["thresholds"]}
    return {"version": "rde-causal-xenosoma-v1", "hash": digest(payload), "preregistrationHash": prereg_hash, "adapterSourceHash": adapter_hash}


def validate_input(bundle):
    problems = []
    unknown = sorted(set(bundle) - ALLOWED_INPUT)
    forbidden = sorted(k for k in FORBIDDEN if k in bundle)
    if unknown: problems.append("caller-fields-forbidden:" + ",".join(unknown))
    if forbidden: problems.append("caller-supplied-evidence-forbidden")
    artifact = bundle.get("causalArtifact")
    prereg = preregistration()
    if not isinstance(artifact, dict) or set(artifact) != {"surface", "toggle"}:
        problems.append("artifact-toggle-invalid")
    elif artifact.get("surface") != prereg["surface"] or artifact.get("toggle") != prereg["toggle"]:
        problems.append("artifact-toggle-invalid")
    for key in ("experimentHash", "commitmentRecordHash", "perturbationRecordHash"):
        if not exact_hash(bundle.get(key)): problems.append(key + "-invalid")
    for key in ("evidenceAnchors", "holdoutAnchors"):
        if not isinstance(bundle.get(key), list) or any(not exact_hash(v) for v in bundle[key]) or bundle[key] != sorted(set(bundle[key])):
            problems.append(key + "-invalid")
    if bundle.get("generator") in {VERIFIER, "atlas-station"}: problems.append("generator-verifier-overlap")
    if bundle.get("grader") == VERIFIER: problems.append("generator-verifier-overlap")
    return sorted(set(problems))


def managed_state(run_dir, artifact_value, artifact_hash, bar, holdout_family, report, checker):
    if not path_is_safe(run_dir): return None, ["run-root-escape"]
    run_dir.mkdir(parents=True, exist_ok=True)
    claims = {
        "runId": run_dir.name,
        "artifactHash": artifact_hash,
        "status": "pass" if checker["ok"] else "fail",
        "frozenBar": bar,
        "holdoutFamily": holdout_family,
        "measurements": report.get("measurements"),
        "falsifiers": report.get("falsifiers", []),
    }
    claims["claimsHash"] = digest(claims)
    audit = {
        "runId": run_dir.name,
        "artifactHash": artifact_hash,
        "verifier": VERIFIER,
        "sourceHead": head(),
        "adapterSourceHash": bar["adapterSourceHash"],
        "preregistrationCommit": preregistration_commit(),
        "report": report,
        "checker": checker,
    }
    audit_hash = digest(audit)
    claims["auditHash"] = audit_hash
    checker_record = {**checker, "auditHash": audit_hash, "claimsHash": claims["claimsHash"]}
    expected = {"artifact.json": artifact_value, "claims.json": claims, "audit/bundle.json": audit, "audit/claim-checker.json": checker_record}
    problems = []
    for rel, path in state_files(run_dir).items():
        if path.exists() and (path.is_symlink() or not path.is_file() or not path_is_safe(path)):
            problems.append("managed-state-path-invalid")
        elif path.exists() and read_json(path) != expected[rel]:
            problems.append("managed-run-state-tampered")
    if not problems:
        for rel, value in expected.items():
            if not state_files(run_dir)[rel].exists(): write_json(state_files(run_dir)[rel], value)
    return {"claims": claims, "audit": audit, "auditHash": audit_hash, "checker": checker_record}, sorted(set(problems))


def receipt(bundle):
    now = datetime.now(timezone.utc)
    validation = validate_input(bundle)
    artifact_value = dict(bundle)
    artifact_hash = digest(artifact_value)
    bar = frozen_bar()
    prereg = preregistration()
    holdout_family = {"name": prereg["families"]["holdout"]["name"], "count": prereg["families"]["holdout"]["count"], "hash": digest(prereg["families"]["holdout"])}
    run_id = "xenosoma_" + artifact_hash.split(":", 1)[1][:24]
    run_dir = RUN_ROOT / run_id
    if validation:
        report = {"ok": False, "falsifiers": validation, "measurements": {}, "families": {}, "audit": {}}
    else:
        report = run_domain(bundle["causalArtifact"])
    checker = check_domain(bundle["causalArtifact"], report) if isinstance(bundle.get("causalArtifact"), dict) else {"ok": False, "problems": ["artifact-toggle-invalid"], "recomputed": False}
    falsifiers = sorted(set(validation + report.get("falsifiers", []) + checker.get("problems", [])))
    state, state_problems = managed_state(run_dir, artifact_value, artifact_hash, bar, holdout_family, report, checker)
    falsifiers = sorted(set(falsifiers + state_problems))
    if not state:
        audit_hash = ""
        checker_result = False
    else:
        audit_hash = state["auditHash"]
        checker_result = state["checker"]["ok"] and not state_problems
    verdict = "pass" if checker_result and not falsifiers else "fail"
    measurements = report.get("measurements", {})
    payload = {
        "verifier": VERIFIER, "protocolVersion": PROTOCOL, "verifierSourceHead": head(),
        "preregistrationCommit": preregistration_commit(), "adapterSourceHash": bar["adapterSourceHash"],
        "experimentHash": bundle.get("experimentHash"), "commitmentRecordHash": bundle.get("commitmentRecordHash"), "perturbationRecordHash": bundle.get("perturbationRecordHash"),
        "evidenceAnchors": bundle.get("evidenceAnchors", []), "holdoutAnchors": bundle.get("holdoutAnchors", []), "artifactHash": artifact_hash,
        "rdeRunId": run_id, "frozenBarVersion": bar["version"], "frozenBarHash": bar["hash"], "holdoutFamilyHash": holdout_family["hash"],
        "claimCheckerResult": checker_result, "claimCheckerHash": digest(state["checker"] if state else checker), "auditBundleHash": audit_hash,
        "nonce": secrets.token_hex(32), "measurements": measurements, "verdict": verdict, "falsifiers": falsifiers,
        "issuedAt": now.isoformat(), "expiry": (now + timedelta(days=7)).isoformat(), "generator": bundle.get("generator"), "grader": bundle.get("grader"),
    }
    return {**payload, "recordHash": digest(payload)}


if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("usage: verify_xenosoma_receipt.py bundle.json")
    try: print(json.dumps(receipt(read_json(sys.argv[1])), sort_keys=True))
    except Exception as exc: raise SystemExit(f"RDE verifier error: {exc}")
