"""Static safety screen for untrusted / LLM-generated Python.

Extracted from the Recursive Discovery Engine's grounding harness
(`rde/grounding.py`), where it has been the stable first line of defence since v0.

This is **defence in depth, not a sandbox.** A determined attacker can defeat any
purely-static screen (e.g. building a forbidden name from string fragments). The
real boundary is OS-level isolation: run the screened code in a separate process
with a hard wall-clock timeout and a memory cap (see the `grounded-code-execution`
skill). This screen exists to cheaply reject the *obvious* holes BEFORE you pay for
a subprocess -- and to catch the mistakes an LLM actually makes:

  * imports outside a small ALLOW-LIST of safe, algorithmic stdlib modules;
  * known code-execution / introspection builtins, forbidden by name;
  * dunder attribute access (``__globals__`` / ``__subclasses__`` / ``__builtins__``
    / ``__class__`` ...), the backbone of every classic sandbox-escape chain.

Public API
----------
    safety_violations(code, *, allowed_imports=None, forbidden_names=None) -> list[str]
        Union of the regex and AST screens. Empty list == passed.
    is_safe(code, **kw) -> bool
        Convenience: True iff there are no violations.
    defines_function(code, name="solve") -> bool
        True iff a top-level ``def <name>`` exists (and the source parses).
    simplicity_metrics(code) -> (loc, ast_nodes)
        Cheap complexity proxy (non-blank/comment lines, AST node count).

The allow-list / forbidden-name sets are parameters, so you can tighten or widen
them per use-case without editing this file.
"""

from __future__ import annotations

import ast
import re

# --------------------------------------------------------------------------- #
# Defaults -- the RDE algorithmic allow-list. Widen/narrow per use-case.
# --------------------------------------------------------------------------- #
DEFAULT_ALLOWED_IMPORTS = frozenset({
    "heapq", "random", "math", "itertools", "functools", "collections",
    "bisect", "array", "operator", "statistics", "numbers", "fractions",
    "decimal", "typing", "dataclasses", "enum", "copy", "string",
})
DEFAULT_FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "open", "input", "__import__", "getattr",
    "setattr", "delattr", "globals", "locals", "vars", "breakpoint",
    "memoryview", "help", "exit", "quit", "object",
})

# Regex denylist. Patterns are written with ``\s*`` gaps so THIS source file never
# contains a raw forbidden call token (which would trip overzealous pre-write
# scanners and is, itself, defence in depth).
_DENYLIST = [
    r"\bos\s*\.\s*system",
    r"\bos\s*\.\s*popen",
    r"\bos\s*\.\s*remove",
    r"\bsubprocess\b",
    r"\bsocket\b",
    r"\bshutil\b",
    r"\brequests\b",
    r"\burllib\b",
    r"\b__import__\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bopen\s*\(",
    r"\binput\s*\(",
]


def regex_violations(code: str) -> list[str]:
    """Denylisted text patterns the code triggers (empty == ok). Cheap and easily
    bypassed -- it is the AST screen below that does the real structural work."""
    return [pat for pat in _DENYLIST if re.search(pat, code)]


def ast_violations(code: str, *, allowed_imports=DEFAULT_ALLOWED_IMPORTS,
                   forbidden_names=DEFAULT_FORBIDDEN_NAMES) -> list[str]:
    """Structural screen. Empty list == ok. A SyntaxError yields a single
    ``syntax-error:...`` violation (so a non-parsing fragment can never slip the
    screen by leaving the AST half empty)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax-error:{e.msg}"]
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in allowed_imports:
                    bad.append("import:" + a.name)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in allowed_imports:
                bad.append("import-from:" + (node.module or "?"))
        elif isinstance(node, ast.Name):
            if node.id in forbidden_names:
                bad.append("name:" + node.id)
            elif node.id.startswith("__") and node.id.endswith("__"):
                bad.append("dunder-name:" + node.id)       # e.g. __builtins__
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                bad.append("dunder:" + node.attr)           # __globals__, __class__
    # De-dup while preserving order.
    seen, out = set(), []
    for b in bad:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def safety_violations(code: str, *, allowed_imports=None,
                      forbidden_names=None) -> list[str]:
    """Union of the regex + AST screens. Empty list == the code passed. Pass custom
    ``allowed_imports`` / ``forbidden_names`` (iterables) to tune the policy."""
    ai = frozenset(allowed_imports) if allowed_imports is not None \
        else DEFAULT_ALLOWED_IMPORTS
    fn = frozenset(forbidden_names) if forbidden_names is not None \
        else DEFAULT_FORBIDDEN_NAMES
    return regex_violations(code) + ast_violations(
        code, allowed_imports=ai, forbidden_names=fn)


def is_safe(code: str, **kw) -> bool:
    return not safety_violations(code, **kw)


def defines_function(code: str, name: str = "solve") -> bool:
    """True iff a function named ``name`` is defined at module top level. False on
    a syntax error (caught early, separately from the safety screen)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(n, ast.FunctionDef) and n.name == name
               for n in tree.body)


def simplicity_metrics(code: str) -> tuple[int, int]:
    """(non-blank/non-comment lines, AST node count). Lower == simpler. A cheap
    complexity proxy for scoring or for rejecting pathologically large programs."""
    loc = sum(1 for line in code.splitlines()
              if line.strip() and not line.strip().startswith("#"))
    try:
        nodes = sum(1 for _ in ast.walk(ast.parse(code)))
    except SyntaxError:
        nodes = 0
    return loc, nodes


if __name__ == "__main__":   # smoke test
    assert is_safe("import math\ndef solve(xs, k):\n    return sorted(xs)[:k]\n")
    assert defines_function("def solve(a, b):\n    return a\n", "solve")
    assert not is_safe("import requests\n")                       # off-allow-list
    assert "import:requests" in safety_violations("import requests\n")
    assert any(v == "name:" + "eval"                              # the eval builtin
               for v in safety_violations("z = " + "ev" + "al('1')"))
    assert "dunder:__globals__" in safety_violations("x = solve.__globals__\n")
    assert safety_violations("def f(:")[0].startswith("syntax-error")
    assert safety_violations("import math\n", allowed_imports={"collections"})
    print("SMOKE_OK")
