"""Structural fingerprinting -- the engine's eyes for the Novelty / Recombination
check.

The spec demands that the Synthesizer can tell *meaningful novelty* from *mostly
recombination of existing ideas*. You cannot do that over free text. So every
approach is reduced to a structural fingerprint (which library calls it makes,
which control structures it uses, the shape of its AST) and two fingerprints are
compared with a bounded similarity in [0, 1].

The rule the rest of the engine relies on:
    is_recombination(A) := there exists a prior approach P such that
        same_algo_class(A, P)  AND  similarity(A, P) >= cfg.recombination_similarity

That is intentionally conservative: a genuinely new *class* (e.g. switching from
a bounded heap to quickselect) is never flagged, no matter how the code is
shaped, while a cosmetic edit of an existing approach in the same class always is.
"""

from __future__ import annotations

import ast
import math
import re

from .types import AlgoClass, Approach


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #

# Ubiquitous structural builtins carry no *algorithmic* signal -- almost every
# solution calls list()/len()/range(). Counting them would let a cosmetic wrapper
# (e.g. `return list(heapq.nlargest(...))`) look meaningfully different from the
# bare call. We exclude them so similarity tracks algorithm, not boilerplate.
# Note: max/min/sorted are NOT here -- they ARE the signal for several classes.
_GENERIC_CALLS = {
    "list", "tuple", "dict", "set", "len", "range", "int", "float", "str",
    "bool", "enumerate", "reversed", "zip", "iter", "print", "abs",
}


def _call_names(tree: ast.AST) -> set[str]:
    """Names of every function/method called, e.g. {'sorted', 'heapq.nlargest',
    'list.sort'}. Method calls keep their attribute path so heapq.nlargest and
    list.sort are distinguishable -- which is exactly the algorithmic signal we
    want. Generic structural builtins are filtered out by the caller."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name):
            names.add(f.id)
        elif isinstance(f, ast.Attribute):
            # Build a dotted path: heapq.nlargest, arr.sort, ...
            parts = [f.attr]
            cur = f.value
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            names.add(".".join(reversed(parts)))
    return names


def _imports(tree: ast.AST) -> set[str]:
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def _node_histogram(tree: ast.AST) -> dict[str, int]:
    hist: dict[str, int] = {}
    for node in ast.walk(tree):
        name = type(node).__name__
        hist[name] = hist.get(name, 0) + 1
    return hist


def fingerprint(code: str, algo_class: AlgoClass) -> dict:
    """Reduce source to a comparable structural signature. Pure / deterministic.
    On a syntax error we still return a (degenerate) fingerprint so the caller
    never has to special-case it."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"algo_class": algo_class.value, "calls": [], "imports": [],
                "node_hist": {}, "loops": 0, "branches": 0,
                "comprehensions": 0, "syntax_ok": False}

    calls = _call_names(tree) - _GENERIC_CALLS
    loops = sum(isinstance(n, (ast.For, ast.While)) for n in ast.walk(tree))
    branches = sum(isinstance(n, ast.If) for n in ast.walk(tree))
    comps = sum(isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp,
                               ast.GeneratorExp)) for n in ast.walk(tree))
    return {
        "algo_class": algo_class.value,
        "calls": sorted(calls),
        "imports": sorted(_imports(tree)),
        "node_hist": _node_histogram(tree),
        "loops": loops,
        "branches": branches,
        "comprehensions": comps,
        "syntax_ok": True,
    }


# --------------------------------------------------------------------------- #
# Similarity
# --------------------------------------------------------------------------- #

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def _cosine_hist(a: dict[str, int], b: dict[str, int]) -> float:
    if not a and not b:
        return 1.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# The call-set carries the strongest algorithmic signal (heapq.nlargest vs
# sorted vs a partition loop), so it dominates the blend; the AST-shape cosine
# catches structural twins that happen to use the same primitives differently.
_W_CALLS, _W_IMPORTS, _W_SHAPE = 0.55, 0.10, 0.35


def similarity(fp1: dict, fp2: dict) -> float:
    """Structural similarity in [0, 1]. 1.0 == structurally identical."""
    calls = _jaccard(set(fp1.get("calls", [])), set(fp2.get("calls", [])))
    imps = _jaccard(set(fp1.get("imports", [])), set(fp2.get("imports", [])))
    shape = _cosine_hist(fp1.get("node_hist", {}), fp2.get("node_hist", {}))
    return _W_CALLS * calls + _W_IMPORTS * imps + _W_SHAPE * shape


# --------------------------------------------------------------------------- #
# Novelty assessment against prior work
# --------------------------------------------------------------------------- #

def assess_novelty(approach: Approach, priors: list[Approach],
                   recombination_threshold: float) -> tuple[float, bool, str]:
    """Compare `approach` (which must already carry a .fingerprint) to all priors.

    Returns (novelty, is_recombination, nearest_id):
      * novelty           = 1 - max similarity to any prior  (1.0 if no priors)
      * is_recombination  = the nearest same-class prior is >= threshold similar
      * nearest_id        = id of the most similar prior (or "")
    """
    if not priors:
        return 1.0, False, ""

    best_sim = 0.0
    nearest_id = ""
    is_recomb = False
    for p in priors:
        if p.id == approach.id or not p.fingerprint:
            continue
        sim = similarity(approach.fingerprint, p.fingerprint)
        if sim > best_sim:
            best_sim, nearest_id = sim, p.id
        # Recombination = high similarity *within the same algorithmic class*.
        if (sim >= recombination_threshold
                and p.algo_class == approach.algo_class):
            is_recomb = True
    return round(1.0 - best_sim, 4), is_recomb, nearest_id


# --------------------------------------------------------------------------- #
# Lightweight code helpers shared by the LLM backend
# --------------------------------------------------------------------------- #

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the first fenced python block from an LLM reply; fall back to the
    raw text if there is no fence (some models answer with bare code)."""
    m = _FENCE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()
