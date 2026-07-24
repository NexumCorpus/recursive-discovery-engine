#!/usr/bin/env python3
"""Trusted RDE adapter for causal-xenosoma bundles.

RDE only says pass when a real local claims.json is paired with a frozen-bar
run and an unseen holdout that is not OVERFIT.  Missing evidence is unknown.
"""
import hashlib, json, secrets, subprocess, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

VERIFIER = "rde-xenosoma-v1"
PROTOCOL = "rde-frozen-bar-receipt-v1"
ROOT = Path(__file__).resolve().parents[1]

def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
def sha(value):
    return "sha256:" + hashlib.sha256(canon(value).encode()).hexdigest()
def head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

def receipt(bundle):
    now = datetime.now(timezone.utc)
    evidence = bundle.get("rdeEvidence") or {}
    falsifiers = []
    claims = None
    claims_path = Path(evidence.get("claimsPath", ""))
    if not evidence: falsifiers.append("rde_frozen_bar_artifact_missing")
    elif not claims_path.is_file(): falsifiers.append("claims_json_missing")
    else:
        try: claims = json.loads(claims_path.read_text(encoding="utf-8"))
        except Exception: falsifiers.append("claims_json_unparseable")
    if claims:
        if claims.get("run_id") != evidence.get("runId"): falsifiers.append("run_id_mismatch")
        if claims.get("recommended_grade", {}).get("adaptive") != "beats_human": falsifiers.append("frozen_bar_not_beaten")
        holdout = claims.get("holdout")
        if not holdout: falsifiers.append("unseen_holdout_missing")
        elif holdout.get("overfit") is True: falsifiers.append("OVERFIT")
        if evidence.get("auditHash") and evidence["auditHash"] != sha(claims): falsifiers.append("audit_hash_mismatch")
    verdict = "pass" if claims and not falsifiers else ("unknown" if "rde_frozen_bar_artifact_missing" in falsifiers or "unseen_holdout_missing" in falsifiers else "fail")
    payload = {
        "verifier": VERIFIER, "protocolVersion": PROTOCOL, "verifierSourceHead": head(),
        "experimentHash": bundle.get("experimentHash"),
        "commitmentRecordHash": bundle.get("commitmentRecordHash"),
        "perturbationRecordHash": bundle.get("perturbationRecordHash"),
        "evidenceAnchors": sorted(set(bundle.get("evidenceAnchors") or [])),
        "holdoutAnchors": sorted(set(bundle.get("holdoutAnchors") or [])),
        "nonce": secrets.token_hex(32),
        "measurements": {"runId": evidence.get("runId"), "claimsPath": str(claims_path) if evidence else None, "frozenBar": bool(claims and claims.get("recommended_grade", {}).get("adaptive") == "beats_human"), "holdout": (claims or {}).get("holdout")},
        "verdict": verdict, "falsifiers": falsifiers, "issuedAt": now.isoformat(),
        "expiry": (now + timedelta(days=7)).isoformat(), "generator": bundle.get("generator"),
        "grader": bundle.get("grader"),
    }
    return {**payload, "recordHash": sha(payload)}

if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("usage: verify_xenosoma_receipt.py bundle.json")
    print(json.dumps(receipt(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))), sort_keys=True))
