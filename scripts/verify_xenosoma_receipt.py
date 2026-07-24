#!/usr/bin/env python3
"""RDE-owned frozen-bar adapter for causal-xenosoma artifacts.

Atlas submits only the causal artifact.  All verdict-bearing state is created
under this repository's managed run root; caller-supplied claims, run IDs,
paths, and holdout verdicts are rejected before grading.
"""
import hashlib, json, secrets, subprocess, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

VERIFIER = "rde-xenosoma-v1"
PROTOCOL = "rde-frozen-bar-receipt-v2"
ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "xenosoma-receipts"
FORBIDDEN = {"rdeEvidence", "claimsPath", "runId", "auditHash", "recommended_grade"}

def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
def sha(value):
    return "sha256:" + hashlib.sha256(canon(value).encode()).hexdigest()
def head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canon(value) + "\n").encode("utf-8")
    path.write_bytes(encoded)
def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def artifact(bundle):
    out = dict(bundle)
    for key in ("rdeEvidence", "independentReceipts", "independentGates"):
        out.pop(key, None)
    return out

def frozen_bar():
    files = [ROOT / "scripts" / "regrade_frozen_bar.py", ROOT / "rde" / "domains" / "cache_eviction.py"]
    return {"version": "rde-v11-frozen-recombination-bar", "hash": sha({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})}

def check_claims(claims, artifact_hash, audit_hash):
    problems = []
    if claims.get("artifactHash") != artifact_hash: problems.append("artifact_hash_mismatch")
    if claims.get("status") != "unknown": problems.append("unsupported_claim_status")
    if claims.get("auditHash") != audit_hash: problems.append("audit_hash_mismatch")
    core = dict(claims); core.pop("auditHash", None); core.pop("claimsHash", None)
    if claims.get("claimsHash") != sha(core): problems.append("claims_hash_mismatch")
    if claims.get("holdout", {}).get("overfit") is True: problems.append("OVERFIT")
    return {"ok": not problems, "problems": problems, "claimsHash": sha(claims)}

def managed_state(run_dir, artifact_value, artifact_hash, bar, holdout_family):
    """Build and verify deterministic RDE-owned verdict state."""
    artifact_path = run_dir / "artifact.json"
    if artifact_path.exists() and read_json(artifact_path) != artifact_value:
        return None, ["run_id_collision"]
    claims = {
        "run_id": run_dir.name,
        "artifactHash": artifact_hash,
        "status": "unknown",
        "frozenBar": bar,
        "holdoutFamily": holdout_family,
        "benchmark": {"executed": True, "supportedDomain": False, "passed": False,
                      "reason": "causal artifact is not the frozen cache_eviction policy domain"},
        "holdout": {"executed": True, "supportedDomain": False, "passed": False,
                     "overfit": None, "reason": "no RDE-owned causal holdout adapter is registered"},
    }
    claims["claimsHash"] = sha(claims)
    audit = {"runId": run_dir.name, "artifactHash": artifact_hash,
             "claimsHash": claims["claimsHash"], "verifier": VERIFIER, "sourceHead": head()}
    audit_hash = sha(audit)
    claims["auditHash"] = audit_hash
    checker = check_claims(claims, artifact_hash, audit_hash)
    expected = {
        "artifact.json": artifact_value,
        "claims.json": claims,
        "audit/bundle.json": audit,
        "audit/claim-checker.json": checker,
    }
    state_problems = []
    for rel, value in expected.items():
        path = run_dir / rel
        if path.exists() and read_json(path) != value:
            state_problems.append("managed-run-state-tampered")
    for rel, value in expected.items():
        path = run_dir / rel
        if not path.exists() and not state_problems:
            write_json(path, value)
    return {"claims": claims, "audit": audit, "auditHash": audit_hash, "checker": checker}, sorted(set(state_problems))

def receipt(bundle):
    now = datetime.now(timezone.utc)
    forbidden = sorted(k for k in FORBIDDEN if k in bundle)
    artifact_value = artifact(bundle)
    artifact_hash = sha(artifact_value)
    bar = frozen_bar()
    holdout_family = {"version": "rde-xenosoma-holdout-v1", "seed": "rde-owned-unseen-family-20260724", "hash": sha({"version": "rde-xenosoma-holdout-v1", "seed": "rde-owned-unseen-family-20260724"})}
    run_id = "xenosoma_" + artifact_hash.split(":", 1)[1][:24]
    run_dir = RUN_ROOT / run_id
    falsifiers = ["caller-supplied-evidence-forbidden"] if forbidden else []
    state, state_falsifiers = managed_state(run_dir, artifact_value, artifact_hash, bar, holdout_family)
    if state is None:
        state_falsifiers = ["run_id_collision"]
    falsifiers.extend(state_falsifiers)
    claims = state["claims"] if state else {}
    audit_hash = state["auditHash"] if state else ""
    checker = state["checker"] if state else {"ok": False, "problems": ["managed-run-unavailable"]}
    benchmark = claims.get("benchmark", {})
    unseen_holdout = claims.get("holdout", {})
    if not checker["ok"]: falsifiers.extend(checker["problems"])
    if not forbidden and not run_dir.is_relative_to(RUN_ROOT): falsifiers.append("run_root_escape")
    payload = {
        "verifier": VERIFIER, "protocolVersion": PROTOCOL, "verifierSourceHead": head(),
        "experimentHash": bundle.get("experimentHash"), "commitmentRecordHash": bundle.get("commitmentRecordHash"), "perturbationRecordHash": bundle.get("perturbationRecordHash"),
        "evidenceAnchors": sorted(set(bundle.get("evidenceAnchors") or [])), "holdoutAnchors": sorted(set(bundle.get("holdoutAnchors") or [])),
        "artifactHash": artifact_hash, "rdeRunId": run_id, "frozenBarVersion": bar["version"], "frozenBarHash": bar["hash"],
        "claimCheckerResult": checker["ok"], "claimCheckerHash": sha(checker), "holdoutFamilyHash": holdout_family["hash"], "auditBundleHash": audit_hash,
        "nonce": secrets.token_hex(32), "measurements": {"benchmark": benchmark, "holdout": unseen_holdout, "managedRunRoot": str(RUN_ROOT)},
        "verdict": "unknown" if not falsifiers else "fail", "falsifiers": sorted(set(falsifiers)), "issuedAt": now.isoformat(), "expiry": (now + timedelta(days=7)).isoformat(),
        "generator": bundle.get("generator"), "grader": bundle.get("grader"),
    }
    return {**payload, "recordHash": sha(payload)}

if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("usage: verify_xenosoma_receipt.py bundle.json")
    try: print(json.dumps(receipt(read_json(Path(sys.argv[1]))), sort_keys=True))
    except Exception as exc: raise SystemExit(f"RDE verifier error: {exc}")
