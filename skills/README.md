# RDE Skills

Reusable, well-documented building blocks extracted from the Recursive Discovery
Engine. These are the pieces that stayed **stable from v0 through v6** while the
engine churned around them — so they are the safest, highest-leverage things to
depend on (including for the engine to use/improve itself later).

Each skill is a self-contained directory: a `SKILL.md` (frontmatter + interface +
when/how-to-use + caveats) and stdlib-only Python, with a runnable smoke test.

| Skill | What it is | Depends on | Stable since |
|-------|-----------|------------|--------------|
| [`ast-safety-screen`](ast-safety-screen/) | Static screen for untrusted/LLM Python — disallowed imports, eval/exec/introspection builtins, dunder access. Defence-in-depth before you exec generated code. | — (stdlib) | v0 |
| [`grounded-code-execution`](grounded-code-execution/) | Run untrusted code in an isolated subprocess (`-E -s`, PYTHONPATH-stripped, timeout + RSS cap) and verify it against an oracle — correctness, benchmarks, peak memory, continuous quality (recall@K), completeness gate. The Empirical Grounding Rule. | `ast-safety-screen` (bundled) | v0 |
| [`pareto-frontier`](pareto-frontier/) | A 2D `(quality, cost)` Pareto frontier with **marginal-hypervolume gain** as the reward — 0 for a dominated/recombinant point, positive for one that opens a new region; plus a within-budget "recommended" pick. | — (stdlib) | v2 |
| [`multi-role-discovery`](multi-role-discovery/) | The methodology: a Builder/Adversary/Synthesizer loop with the protective mechanisms (grounding rule, marginal-HV reward, novelty/recombination, metric-attack, bounded memory). A runnable reference that composes the three skills above. | the three above | v0–v2 |

## How they compose

```
ast-safety-screen   ──┐
                       ├─► grounded-code-execution ──┐
(your oracle/gen) ────┘                              ├─► multi-role-discovery
pareto-frontier ─────────────────────────────────────┘   (the loop)
```

`grounded-code-execution` is the load-bearing one: it embeds the safety screen and
is what every candidate must pass. `pareto-frontier` turns a grounded `(quality,
cost)` result into a discovery *reward*. `multi-role-discovery` is the orchestration
pattern that wires them into a self-improving loop without drifting or collapsing.

## Run the smoke tests

```bash
python3 ast-safety-screen/safety_screen.py        # importable; see its SKILL.md
python3 grounded-code-execution/example.py         # full grounding example
python3 multi-role-discovery/discovery_loop.py     # the loop, end-to-end
```

Or via pytest from the repo root: `python3 -m pytest tests/test_skills.py`.

## Why these four

The selection criterion was **stability × leverage × reusability**, not recency. The
newer v5/v6 machinery (open vocabulary, self-play, utility flags) is powerful but
still moving; these four are the load-bearing primitives that have not needed to
change. Extracting them first gives v7 a cleaner, more modular foundation — and gives
*other* projects (and the engine itself) drop-in, dependency-light building blocks.

Canonical sources in the engine: `rde/grounding.py` (screen + harness),
`rde/frontier.py` (frontier), `rde/roles.py` + `rde/engine.py` (the loop).
