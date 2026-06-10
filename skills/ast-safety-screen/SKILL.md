---
name: ast-safety-screen
description: Statically screen untrusted or LLM-generated Python for unsafe constructs (disallowed imports, eval/exec/compile/getattr-style introspection builtins, and dunder attribute access) BEFORE executing it. Use whenever you are about to exec, import, or subprocess-run code that a model or an external source produced and you want a cheap, structural defence-in-depth check. Pairs with `grounded-code-execution` for the real OS-level boundary.
---

# AST Safety Screen

A tiny, zero-dependency static screen for untrusted Python. Extracted from the
Recursive Discovery Engine, where it has guarded every grounded execution since v0.

## When to use this

Reach for it the moment you are about to **run code you did not write** — an LLM
proposal, a plugin, a user snippet — and want to reject the obvious dangers before
paying for a subprocess. It is the first gate in a "screen → isolate → execute"
pipeline.

**Do NOT rely on it as a sandbox.** It is *defence in depth*. A static screen cannot
stop a determined attacker (forbidden names can be built from string fragments,
`compile`/`pathlib` chains exist, etc.). The real boundary is OS-level isolation: a
separate process with a hard timeout and a memory cap. Use the
`grounded-code-execution` skill for that. This screen exists to (a) cheaply reject
the obvious holes and (b) catch the mistakes models actually make (importing
`requests`, reaching for `__globals__`, calling the eval builtin).

## Interface

```python
from safety_screen import safety_violations, is_safe, defines_function

safety_violations(code, *, allowed_imports=None, forbidden_names=None) -> list[str]
    # Union of a regex denylist + an AST structural screen. Empty list == passed.
    # Each violation is a tag: "import:requests", "name:eval", "dunder:__globals__",
    # "syntax-error:...". Pass custom allow/forbid sets (iterables) to tune policy.

is_safe(code, **kw) -> bool          # True iff no violations
defines_function(code, name="solve") -> bool   # top-level def present & parses
simplicity_metrics(code) -> (loc, ast_nodes)   # cheap complexity proxy
```

What the AST screen enforces:
- **imports** must be on an **allow-list** (default: safe algorithmic stdlib —
  `heapq, math, collections, itertools, random, ...`). Anything else → `import:<m>`.
- a set of **forbidden builtins** by name (the eval/exec/compile/open/`__import__`/
  getattr/setattr/globals family) → `name:<id>`.
- any **dunder** name or attribute access (`__class__`, `__globals__`,
  `__subclasses__`, `__builtins__` ...) → `dunder:<attr>` / `dunder-name:<id>`.
- a **syntax error** is reported as a violation (so unparsable code never passes by
  leaving the AST screen empty).

## Usage

```python
from safety_screen import is_safe, safety_violations, defines_function

bad = "import requests\nx = ().__class__.__bases__\n"
assert not is_safe(bad)
# -> ['import:requests', 'dunder:__class__', 'dunder:__bases__']

ok = "import math\ndef solve(xs, k):\n    return sorted(xs)[:k]\n"
assert is_safe(ok) and defines_function(ok, "solve")

# Tighten the policy: forbid randomness too.
safety_violations(ok, allowed_imports={"collections"})   # -> ['import:math']
```

Typical gate before running generated code:

```python
v = safety_violations(generated)
if v:
    reject(reason="static-safety: " + ", ".join(v))
else:
    run_in_isolated_subprocess(generated)      # see grounded-code-execution
```

## Files

- `safety_screen.py` — the screen (stdlib-only: `ast`, `re`).

## Caveats / provenance

- The regex patterns are written with `\s*` gaps so this file never contains a raw
  forbidden call token (itself a defence-in-depth nicety).
- Allow-list rejection is conservative by design: better to reject a valid program
  (the proposer can try again) than to admit an unsafe one.
- Canonical source: `rde/grounding.py` in the Recursive Discovery Engine. Keep the
  two in sync, or have the engine import from here in a future refactor.
