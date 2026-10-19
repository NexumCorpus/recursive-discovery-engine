"""The v11 template-library stamp must be stable across interpreter runs."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SNIPPET = (
    "import hashlib\n"
    "from rde.domains.cache_eviction import DOMAIN\n"
    "t = DOMAIN.audit_texts()['template_library']\n"
    "print(hashlib.sha256(t.encode('utf-8')).hexdigest())\n"
)


def _stamp(hashseed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    out = subprocess.run([sys.executable, "-c", SNIPPET], cwd=ROOT, env=env,
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_template_hash_independent_of_hash_seed():
    assert _stamp("1") == _stamp("2") == _stamp("12345")
