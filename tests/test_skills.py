"""Smoke-tests for the extracted RDE skills (skills/).

Each skill is self-contained with its own sys.path setup, so we run each as an
ISOLATED subprocess (avoiding cross-skill module-name collisions: every skill ships
its own grounding.py / frontier.py / safety_screen.py) and assert it prints
SMOKE_OK. This keeps the skills from rotting as the engine evolves.
"""
import pathlib
import subprocess
import sys

import pytest

_SKILLS = pathlib.Path(__file__).resolve().parent.parent / "skills"

_SMOKES = [
    "ast-safety-screen/safety_screen.py",
    "pareto-frontier/frontier.py",
    "grounded-code-execution/example.py",
    "multi-role-discovery/discovery_loop.py",
]


@pytest.mark.parametrize("rel", _SMOKES)
def test_skill_smoke(rel):
    script = _SKILLS / rel
    assert script.exists(), f"missing skill script: {script}"
    r = subprocess.run([sys.executable, str(script)],
                       capture_output=True, text=True, timeout=240)
    assert "SMOKE_OK" in r.stdout, (
        f"{rel} did not pass its smoke test\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")


def test_every_skill_has_a_skill_md():
    for d in _SKILLS.iterdir():
        if d.is_dir():
            md = d / "SKILL.md"
            assert md.exists(), f"{d.name} is missing SKILL.md"
            head = md.read_text(encoding="utf-8")[:400]
            assert head.startswith("---") and "name:" in head and "description:" in head, \
                f"{d.name}/SKILL.md needs YAML frontmatter with name + description"
