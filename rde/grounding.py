"""The grounding harness -- where claims go to be tested.

This module is the embodiment of the Empirical Grounding Rule: *no approach
advances without runnable evidence that can be independently verified and
potentially falsified*. Nothing an LLM (or a hand-written strategy) asserts about
correctness or speed is taken on faith. Every candidate is executed in an
ISOLATED subprocess with a hard timeout, checked against a brute-force oracle,
benchmarked, and memory-profiled. The numbers in the returned `Evidence` are the
only things downstream roles are allowed to reason about.

Key safety properties:
  * Candidate code runs in a separate `python` process, never in the engine's
    own interpreter -> no state leakage between candidates, and a runaway loop is
    killed by the OS-level subprocess timeout.
  * Results are streamed to a result.json file, so even if the subprocess is
    killed mid-benchmark we still recover the correctness verdict already written.
  * A static denylist rejects obviously dangerous candidate code (defense in
    depth for the LLM backend).
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .types import Evidence


# --------------------------------------------------------------------------- #
# Grounding specification (domain-supplied)
# --------------------------------------------------------------------------- #

@dataclass
class GroundingSpec:
    oracle_src: str            # source defining oracle(items, k)
    workload_gen_src: str      # source defining gen_workload(n, dist, seed)
    criteria_cases: list       # [(name, items, k)] -- the trusted suite
    workloads: list            # [{"name","n","k","dist"}]
    mem_workload: dict         # one workload spec used for the memory probe
    # v2: source of `verdict(got, expected, items, k) -> (ok: bool, quality: float)`.
    # Empty -> the harness uses exact equality (ok iff got == expected, quality
    # 1.0/0.0), preserving v0/v1 closed-oracle behaviour.
    verdict_src: str = ""
    # v8: optional TRUSTED driver source defining `_driver_solve(items, k)`. When set,
    # the runner binds the public `solve` to THIS (domain-owned) driver and discards
    # any candidate-authored solve -- so the candidate provides only the eviction
    # LOGIC (e.g. _evict_score / _on_hit / CMUL) and the metric is computed by trusted
    # code (verifiable by construction). Empty -> legacy candidate-solve path.
    driver_src: str = ""
    # v9: the driver contract's required candidate symbols, SPEC-DRIVEN so any domain
    # can adopt the trusted-driver pattern with its own symbol names:
    #   driver_funcs        -- required top-level function defs the driver imports;
    #   driver_literal_ints -- required assignments that must be EXACTLY ONE literal
    #                          int (the v8 CMUL soundness rule: the cost axis reads
    #                          these from SOURCE while the driver reads them at
    #                          RUNTIME, so a non-literal would let a candidate charge
    #                          one value and run another).
    # Defaults are the cache-eviction names; unused when driver_src is empty.
    driver_funcs: tuple = ("_evict_score", "_on_hit")
    driver_literal_ints: tuple = ("CMUL",)


# --------------------------------------------------------------------------- #
# Static safety screen for untrusted (LLM) candidate code
# --------------------------------------------------------------------------- #
# Patterns are written as regexes so this source file itself never contains the
# raw forbidden call tokens (which would trip overzealous pre-write scanners).
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


def static_safety_violations(code: str) -> list[str]:
    """Return the list of denylisted patterns the code triggers (empty == ok)."""
    hits = []
    for pat in _DENYLIST:
        if re.search(pat, code):
            hits.append(pat)
    return hits


# --------------------------------------------------------------------------- #
# AST-based safety screen -- the PRIMARY static defence.
# --------------------------------------------------------------------------- #
# A regex denylist is trivially bypassed (getattr(__builtins__, 'ev'+'al'),
# compile(), pathlib.Path().read_text(), import ctypes, ...). An AST screen is
# far stronger because it reasons about structure, not text:
#   * imports must be on an ALLOWLIST of safe, algorithmic stdlib modules;
#   * known code-execution / introspection builtins are forbidden by name;
#   * dunder attribute access (__globals__, __subclasses__, __builtins__, ...) --
#     the backbone of every classic sandbox-escape chain -- is rejected outright.
# This is still NOT a security boundary (see module docstring + README); it is
# defence-in-depth that closes the obvious holes.

_ALLOWED_IMPORTS = {
    "heapq", "random", "math", "itertools", "functools", "collections",
    "bisect", "array", "operator", "statistics", "numbers", "fractions",
    "decimal", "typing", "dataclasses", "enum", "copy", "string",
}
_FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "input", "__import__", "getattr",
    "setattr", "delattr", "globals", "locals", "vars", "breakpoint",
    "memoryview", "help", "exit", "quit", "object",
}


def ast_safety_violations(code: str) -> list[str]:
    """Structural safety check. Empty list == ok. A syntax error returns [] here
    (it is reported separately by `defines_solve`)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    bad.append("import:" + a.name)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in _ALLOWED_IMPORTS:
                bad.append("import-from:" + (node.module or "?"))
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                bad.append("name:" + node.id)
            elif node.id.startswith("__") and node.id.endswith("__"):
                bad.append("dunder-name:" + node.id)   # e.g. __builtins__
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                bad.append("dunder:" + node.attr)       # __globals__, __class__
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            # v9 review: `global CMUL` inside a candidate function would let it
            # rebind a gate-validated literal at RUNTIME (source != runtime).
            # Candidates have no legitimate use for global/nonlocal.
            bad.append("global-decl:" + ",".join(node.names))
    # De-dup while preserving order.
    seen, out = set(), []
    for b in bad:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def all_safety_violations(code: str) -> list[str]:
    """Union of the regex and AST screens."""
    return static_safety_violations(code) + ast_safety_violations(code)


# --------------------------------------------------------------------------- #
# Static structure checks (run in-process, no execution)
# --------------------------------------------------------------------------- #

def defines_solve(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(n, ast.FunctionDef) and n.name == "solve"
               for n in tree.body)


def driver_symbols_ok(code: str, funcs: tuple = ("_evict_score", "_on_hit"),
                      literal_ints: tuple = ("CMUL",)) -> bool:
    """v9 GENERIC trusted-driver gate: a candidate for a driver domain provides
    LOGIC symbols, not a solve. `funcs` are required top-level function defs;
    `literal_ints` are required assignments that must be EXACTLY ONE assignment to a
    LITERAL int each.

    The literal-int rule is a soundness gate, not pedantry: the cost axis reads
    these knobs from the SOURCE (regex) while the trusted driver reads them at
    RUNTIME, so any binding the regex cannot see (`CMUL += 7`, a nested
    `if ...: CMUL = 8`, `for CMUL in [8]`, a chained `CMUL = X = 8`, a `global
    CMUL` rebind inside a function) would let a free-form candidate charge a
    cheap value on the cost axis while the driver runs an expensive one. The v9
    review reproduced exactly that exploit, so the rule is now a WHOLE-TREE
    binding census: the name must be bound EXACTLY ONCE anywhere in the module,
    by a single-target top-level assignment (plain or annotated) to a literal
    int, with no deletes and no global/nonlocal declarations of it."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    fn_names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if any(f not in fn_names for f in funcs):
        return False

    def _is_literal_int(v):
        return (isinstance(v, ast.Constant) and isinstance(v.value, int)
                and not isinstance(v.value, bool))

    for lit in literal_ints:
        # (1) The single legal top-level binding: `LIT = <int>` with exactly one
        # target, or the annotated form `LIT: <ann> = <int>`.
        legal = 0
        for n in tree.body:
            if (isinstance(n, ast.Assign) and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id == lit and _is_literal_int(n.value)):
                legal += 1
            elif (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                    and n.target.id == lit and n.value is not None
                    and _is_literal_int(n.value)):
                legal += 1
        if legal != 1:
            return False
        # (2) Whole-tree census: that legal binding must be the ONLY binding of
        # the name anywhere -- no rebinds via Assign/AugAssign/AnnAssign/for/
        # with-as/walrus/comprehension (all Store-ctx Names), no deletes, and no
        # global/nonlocal declaration that would allow a function to rebind it.
        stores = dels = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == lit:
                if isinstance(node.ctx, ast.Store):
                    stores += 1
                elif isinstance(node.ctx, ast.Del):
                    dels += 1
            elif (isinstance(node, ast.AugAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == lit):
                stores += 1            # AugAssign target may not be Store-counted
            elif (isinstance(node, (ast.Global, ast.Nonlocal))
                    and lit in node.names):
                return False
        if stores != 1 or dels != 0:
            return False
    return True


def defines_driver_symbols(code: str) -> bool:
    """Back-compat alias: the cache-eviction symbol set (v8 name)."""
    return driver_symbols_ok(code)


def _aux0_literal_ok(node) -> bool:
    """AUX0 must be a LITERAL list/tuple of <= 8 numeric constants (unary minus
    allowed) -- statically auditable initial state, nothing computed."""
    if not isinstance(node, (ast.List, ast.Tuple)) or len(node.elts) > 8:
        return False
    for e in node.elts:
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
            e = e.operand
        if not (isinstance(e, ast.Constant) and isinstance(e.value, (int, float))
                and not isinstance(e.value, bool)):
            return False
    return True


def driver_shape_violations(code: str, spec) -> list:
    """v10 P2 STATELESSNESS screen for TRUSTED-DRIVER candidates only (returns []
    for legacy solve-style domains). The exact-cost axis (CMUL*k*bytes) is only
    sound if a candidate cannot hold state outside the driver-owned cache+aux, so
    the module shape is locked down:

      * top level may contain ONLY: imports (allow-list enforced separately by the
        AST safety screen), function defs, docstrings, the declared literal-int
        bindings (e.g. CMUL), and AUX0 as a literal list/tuple of <= 8 numbers.
        Any other module-level binding (e.g. `_history = []`) is a state stash.
      * mutable DEFAULT parameter values (`def f(..., _m={})`) are rejected --
        they persist across calls exactly like a module global.
      * attribute STORES (`_evict_score.cache = {}`, `obj.x = ...`) are rejected
        anywhere in the module -- function attributes are another cross-call stash.
      * global/nonlocal are already rejected by the AST safety screen; these rules
        compose with (never replace) that ban."""
    if not getattr(spec, "driver_src", ""):
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["syntax-error"]
    literal_ints = set(getattr(spec, "driver_literal_ints", ("CMUL",)))
    bad: list = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
            continue
        if (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, str)):
            continue                                    # docstring
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            if isinstance(n, ast.Assign):
                tgt = n.targets[0] if len(n.targets) == 1 else None
            else:
                tgt = n.target
            if isinstance(tgt, ast.Name) and tgt.id in literal_ints:
                continue                # single-literal rule enforced by the gate
            if (isinstance(tgt, ast.Name) and tgt.id == "AUX0"
                    and n.value is not None and _aux0_literal_ok(n.value)):
                continue
            nm = tgt.id if isinstance(tgt, ast.Name) else type(n).__name__
            bad.append(f"module-binding:{nm}")
            continue
        bad.append(f"module-stmt:{type(n).__name__}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.Lambda)):
            a = node.args
            for d in list(a.defaults) + [d for d in a.kw_defaults if d is not None]:
                if not isinstance(d, ast.Constant):
                    bad.append("mutable-default:"
                               + getattr(node, "name", "<lambda>"))
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in tgts:
                if isinstance(t, ast.Attribute):
                    bad.append("attribute-store:" + t.attr)
    seen, out = set(), []
    for b in bad:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def candidate_ok(code: str, spec: "GroundingSpec") -> bool:
    """A candidate is well-formed if it defines a top-level solve OR -- when the
    domain provides a trusted driver -- the spec's required logic symbols."""
    if defines_solve(code):
        return True
    if not getattr(spec, "driver_src", ""):
        return False
    return driver_symbols_ok(
        code,
        funcs=tuple(getattr(spec, "driver_funcs", ("_evict_score", "_on_hit"))),
        literal_ints=tuple(getattr(spec, "driver_literal_ints", ("CMUL",))))


def simplicity_metrics(code: str) -> tuple[int, int]:
    """(non-blank/non-comment lines, AST node count). Lower == simpler."""
    loc = 0
    for line in code.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            loc += 1
    try:
        tree = ast.parse(code)
        nodes = sum(1 for _ in ast.walk(tree))
    except SyntaxError:
        nodes = 0
    return loc, nodes


# --------------------------------------------------------------------------- #
# The subprocess runner (written to a temp file and executed)
# --------------------------------------------------------------------------- #
# This template is plain text; {oracle_src}/{gen_src} are spliced in. It writes
# its findings incrementally to result.json so a kill mid-benchmark still leaves
# the correctness verdict on disk.

_RUNNER_TEMPLATE = '''\
import json, sys, importlib, traceback, tracemalloc, random, math

JOB = json.loads(sys.argv[1])
RESULT_PATH = JOB["result_path"]

{oracle_src}

{gen_src}

# Default verdict: exact equality (closed-oracle domains). A domain may override
# `verdict` below to return a CONTINUOUS quality (e.g. recall@K) instead.
def verdict(got, expected, items, k):
    eq = (got == expected)
    return (eq, 1.0 if eq else 0.0)

{verdict_src}

# v8: optional TRUSTED driver. When the domain supplies driver_src it defines a
# module-level `_driver_solve(items, k)` here; the binding below then makes the
# public `solve` THIS driver and discards any candidate-authored solve.
{driver_src}

def _write(state):
    with open(RESULT_PATH, "w") as fh:
        json.dump(state, fh)

state = {{
    "phase": "start", "import_ok": False, "error": None, "traceback": None,
    "cases": {{}}, "all_correct": False, "any_mutated": False,
    "timings": {{}}, "timing_stddev": {{}}, "peak_memory_bytes": 0,
    "quality": {{}},
}}
_write(state)

# --- import the candidate (isolated module) ------------------------------- #
try:
    candidate = importlib.import_module("candidate")
    # v8: if a trusted driver was spliced in (driver_src non-empty), bind `solve`
    # to it UNCONDITIONALLY -- the candidate's own solve, if any, is discarded so it
    # has no channel to report the metric. Else the legacy candidate-solve path.
    if "_driver_solve" in globals():
        solve = _driver_solve
    else:
        solve = getattr(candidate, "solve")
    state["import_ok"] = True
except Exception as e:                       # noqa: BLE001
    state["error"] = "import: " + repr(e)
    state["traceback"] = traceback.format_exc()
    state["phase"] = "import_failed"
    _write(state)
    sys.exit(0)

def _trunc(seq, n=12):
    seq = list(seq)
    return seq[:n] + (["...(+%d)" % (len(seq) - n)] if len(seq) > n else [])

# --- correctness against the oracle, plus input-mutation detection -------- #
all_correct = True
any_mutated = False
for name, items, k in JOB["cases"]:
    rec = {{"ok": False, "mutated": False, "error": None, "quality": 0.0}}
    try:
        expected = oracle(items, k)
        passed_list = list(items)
        before = list(passed_list)
        got = solve(passed_list, k)
        rec["mutated"] = (passed_list != before)
        vok, vq = verdict(got, expected, items, k)
        rec["quality"] = float(vq)
        rec["ok"] = bool(vok) and not rec["mutated"]
        if not rec["ok"]:
            rec["expected"] = _trunc(expected)
            rec["got"] = _trunc(got) if isinstance(got, (list, tuple)) else repr(got)[:80]
            rec["k"] = k
            rec["n"] = len(items)
    except Exception as e:                   # noqa: BLE001
        rec["error"] = repr(e)
        rec["k"] = k; rec["n"] = len(items)
    if rec["mutated"]:
        any_mutated = True
    if not rec["ok"]:
        all_correct = False
    state["cases"][name] = rec

state["all_correct"] = all_correct
state["any_mutated"] = any_mutated
state["phase"] = "correctness_done"
_write(state)

# Only the empirically-correct earn a benchmark. (A broken candidate's speed is
# meaningless, and this avoids burning the timeout on pathological inputs.)
# Each phase is guarded: a failure here must be RECORDED, never silently swallowed
# -- a blind benchmark crash once masqueraded as a correct, 0ms result.
if all_correct and JOB.get("do_benchmarks"):
    import time
    seed = JOB["bench_seed"]
    reps = JOB["bench_repeats"]
    try:
        for wl in JOB["workloads"]:
            data = gen_workload(wl["n"], wl["dist"], seed)
            # v2: measure QUALITY (e.g. recall@K) on this workload, once.
            if JOB.get("do_quality"):
                try:
                    _exp = oracle(list(data), wl["k"])
                    _got = solve(list(data), wl["k"])
                    _ok, _q = verdict(_got, _exp, data, wl["k"])
                    state["quality"][wl["name"]] = float(_q)
                except Exception as _e:           # noqa: BLE001
                    state["quality"][wl["name"]] = 0.0
            samples = []
            for _ in range(reps):
                d = list(data)
                t0 = time.perf_counter()
                solve(d, wl["k"])
                samples.append(time.perf_counter() - t0)
            samples.sort()
            state["timings"][wl["name"]] = samples[len(samples) // 2]   # median
            # Sample stddev -> the harness reports its OWN measurement noise, so
            # downstream "is this a real speedup?" can be noise-aware.
            if len(samples) > 1:
                mean = sum(samples) / len(samples)
                var = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
                state["timing_stddev"][wl["name"]] = math.sqrt(var)
            else:
                state["timing_stddev"][wl["name"]] = 0.0
            state["phase"] = "bench:" + wl["name"]
            _write(state)
    except Exception as e:                       # noqa: BLE001
        state["bench_error"] = repr(e)
        state["bench_traceback"] = traceback.format_exc()
        _write(state)

# --- peak memory probe (separate from timing; tracemalloc adds overhead) -- #
if all_correct and JOB.get("do_memory"):
    try:
        mw = JOB["mem_workload"]
        data = gen_workload(mw["n"], mw["dist"], JOB["bench_seed"])
        d = list(data)                  # copy BEFORE start: don't count the input
        tracemalloc.start()
        solve(d, mw["k"])
        state["peak_memory_bytes"] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    except Exception as e:                        # noqa: BLE001
        state["mem_error"] = repr(e)
        _write(state)

# v2: the Adversary attacks the METRIC -- measure quality on extra, possibly
# hostile, workloads (no timing). Recorded into the same quality map.
if all_correct and JOB.get("quality_workloads"):
    for wl in JOB["quality_workloads"]:
        # v5: SEPARATE a broken GENERATOR (the workload itself is unusable -> record
        # NO recall, so the proposed-generator admission gate rejects it) from a
        # SOLVE crash on a valid stream (a genuine recall collapse -> record 0.0 so
        # the Adversary still exposes the candidate). The old single try/except
        # conflated them, letting a raising generator be admitted as "recall 0.0".
        # v11 A3: MULTI-SEED -- a workload may declare "seeds": [s1..sS]; every
        # seed runs INSIDE this one subprocess; quality = mean over seeds and the
        # per-seed values are recorded (quality_seeds) for spread/fragility
        # semantics. A generator failure on ANY seed leaves the workload
        # UNMEASURED (strict, rejectable); a solve failure scores that seed 0.0.
        _seeds = wl.get("seeds") or [JOB["bench_seed"]]
        try:
            _streams = []
            for _s in _seeds:
                data = gen_workload(wl["n"], wl["dist"], _s)
                if not isinstance(data, (list, tuple)) or not data:
                    raise TypeError("generator returned a non-sequence / empty stream")
                for _x in data:
                    if not isinstance(_x, int) or isinstance(_x, bool):
                        raise TypeError("generator stream is not all ints")
                _streams.append(data)
        except Exception as e:                    # noqa: BLE001
            state.setdefault("quality_errors", {{}})[wl["name"]] = "gen: " + repr(e)
            _write(state)                         # persist the error record too
            continue                              # NO quality recorded -> rejectable
        state.setdefault("quality_len", {{}})[wl["name"]] = len(_streams[0])
        _qs = []
        for data in _streams:
            try:
                _exp = oracle(list(data), wl["k"])
                _got = solve(list(data), wl["k"])
                _ok, _q = verdict(_got, _exp, data, wl["k"])
                _qs.append(float(_q))
            except Exception as e:                # noqa: BLE001
                state.setdefault("quality_errors", {{}})[wl["name"]] = \
                    "solve: " + repr(e)
                _qs.append(0.0)
        state["quality"][wl["name"]] = sum(_qs) / len(_qs)
        state.setdefault("quality_seeds", {{}})[wl["name"]] = _qs
        # v9 review: persist PER WORKLOAD (matching the benchmark phase), so a
        # timeout on workload N preserves the N-1 completed measurements instead
        # of discarding the whole quality map (which read as a spurious 0.0).
        _write(state)
    _write(state)

# v6: DIAGNOSE -- record the oracle top-k vs the policy top-k on specific workloads,
# so the engine can build CAUSAL guidance (which true heavy hitters were missed).
if all_correct and JOB.get("diagnose_workloads"):
    state["diagnose"] = {{}}
    for wl in JOB["diagnose_workloads"]:
        try:
            data = gen_workload(wl["n"], wl["dist"], JOB["bench_seed"])
            _exp = oracle(list(data), wl["k"])
            _got = solve(list(data), wl["k"])
            state["diagnose"][wl["name"]] = {{
                "expected": [int(x) for x in list(_exp)[:64]],
                "got": [int(x) for x in list(_got)[:64]],
            }}
        except Exception as e:               # noqa: BLE001
            state.setdefault("diagnose_errors", {{}})[wl["name"]] = repr(e)
    _write(state)

state["phase"] = "done"
_write(state)
'''


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def _run_with_limits(args: list, cwd: str, env: dict, timeout: float,
                     mem_cap_bytes: int) -> tuple:
    """Run a subprocess under a wall-clock timeout AND a best-effort RSS cap.

    The candidate/proposal runs in an isolated process; a wall-clock timeout alone
    lets an unbounded allocation thrash the host for the full timeout window before
    it is reaped. A psutil watchdog in the PARENT samples the child's RSS and kills
    it the moment it breaches the cap. `resource.setrlimit` is POSIX-only (absent on
    Windows, this project's platform), so the watchdog is the portable mechanism;
    if psutil is unavailable we fall back to a plain wall-clock kill.

    Returns (returncode, stderr, timed_out, mem_killed). stdout is discarded (the
    runner streams its findings to result.json, not stdout)."""
    try:
        import psutil
    except Exception:                              # noqa: BLE001
        psutil = None
    proc = subprocess.Popen(args, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ps = None
    if psutil is not None:
        try:
            ps = psutil.Process(proc.pid)
        except Exception:                          # noqa: BLE001
            ps = None
    start = time.monotonic()
    timed_out = mem_killed = False
    while proc.poll() is None:
        if time.monotonic() - start > timeout:
            timed_out = True
            proc.kill()
            break
        if ps is not None and mem_cap_bytes:
            try:
                rss = ps.memory_info().rss
                for ch in ps.children(recursive=True):
                    try:
                        rss += ch.memory_info().rss
                    except Exception:              # noqa: BLE001
                        pass
                if rss > mem_cap_bytes:
                    mem_killed = True
                    proc.kill()
                    break
            except Exception:                      # noqa: BLE001
                pass                               # process may have just exited
        time.sleep(0.05)
    try:
        _out, err = proc.communicate(timeout=5)
    except Exception:                              # noqa: BLE001
        try:
            proc.kill()
        except Exception:                          # noqa: BLE001
            pass
        _out, err = "", ""
    return (proc.returncode, err or "", timed_out, mem_killed)


def _run_subprocess(code: str, spec: GroundingSpec, cfg: Config, *,
                    cases: list, do_benchmarks: bool, do_memory: bool,
                    do_quality: bool = False,
                    quality_workloads: list | None = None,
                    diagnose_workloads: list | None = None) -> dict:
    """Execute `code` in an isolated process and return the parsed result dict.
    Always returns a dict (with an `error`/`timed_out` flag on failure)."""
    tmp = Path(tempfile.mkdtemp(prefix="rde_ground_"))
    try:
        (tmp / "candidate.py").write_text(code, encoding="utf-8")
        result_path = tmp / "result.json"
        runner = _RUNNER_TEMPLATE.format(
            oracle_src=spec.oracle_src, gen_src=spec.workload_gen_src,
            verdict_src=spec.verdict_src or "",
            driver_src=spec.driver_src or "")
        (tmp / "runner.py").write_text(runner, encoding="utf-8")

        # Serialise cases as plain lists (tuples -> lists) for JSON.
        job = {
            "result_path": str(result_path),
            "cases": [[n, list(it), k] for (n, it, k) in cases],
            "workloads": spec.workloads,
            "mem_workload": spec.mem_workload,
            "bench_repeats": cfg.bench_repeats,
            "bench_seed": cfg.bench_seed,
            "do_benchmarks": do_benchmarks,
            "do_memory": do_memory,
            "do_quality": do_quality,
            "quality_workloads": quality_workloads or [],
            "diagnose_workloads": diagnose_workloads or [],
        }
        # Strip PYTHONPATH so candidate code cannot import packages from the
        # parent's search path (e.g. the `rde` package itself); the subprocess
        # only needs stdlib + the candidate in its own cwd.
        child_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"

        # -E ignores PYTHON* env vars, -s skips user site-packages. (Not -I:
        # isolated mode also drops the script dir from sys.path, which would
        # break the runner's `import candidate`.)
        mem_cap = int(getattr(cfg, "subprocess_mem_cap_mb", 0) or 0) * 1024 * 1024
        rc, stderr, timed_out, mem_killed = _run_with_limits(
            [sys.executable, "-E", "-s", str(tmp / "runner.py"), json.dumps(job)],
            str(tmp), child_env, cfg.subprocess_timeout_s, mem_cap)
        proc_err = None
        if mem_killed:
            proc_err = (f"killed: RSS exceeded {cfg.subprocess_mem_cap_mb}MB cap "
                        f"(unbounded allocation)")
        elif rc != 0 and not timed_out:
            # Capture stderr on ANY nonzero exit, not only when output is missing --
            # a crash that leaves a partial result.json must still surface its
            # traceback (this is how the missing-import bug hid).
            proc_err = (stderr or "nonzero exit").strip()[-500:]

        no_output = {"phase": "no_output", "import_ok": False, "cases": {},
                     "all_correct": False, "any_mutated": False, "timings": {},
                     "peak_memory_bytes": 0, "error": proc_err}
        if result_path.exists():
            try:
                res = json.loads(result_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                # A kill mid-write can truncate result.json. Recover gracefully
                # instead of crashing the whole engine on one bad candidate.
                res = dict(no_output)
                res["error"] = proc_err or f"corrupted result.json: {e}"
        else:
            res = no_output
        res["timed_out"] = timed_out
        if timed_out and not res.get("error"):
            res["error"] = f"timeout after {cfg.subprocess_timeout_s}s " \
                           f"(phase={res.get('phase')})"
        if proc_err and not res.get("error"):
            res["error"] = proc_err
        return res
    finally:
        # Best-effort cleanup; keep no artifacts.
        for p in sorted(tmp.glob("*"), reverse=True):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            tmp.rmdir()
        except OSError:
            pass


def ground(code: str, spec: GroundingSpec, cfg: Config) -> Evidence:
    """Run the full grounding battery (trusted criteria + benchmarks + memory)
    on a candidate and return raw Evidence. Scores (perf/mem/simplicity) are
    left at 0 here; call `score()` once a baseline is known."""
    ev = Evidence(grounded=True)
    ev.loc, ev.ast_nodes = simplicity_metrics(code)

    violations = all_safety_violations(code)
    if violations:
        ev.error = "static-safety: " + ", ".join(violations)
        ev.correct = False
        return ev

    if not candidate_ok(code, spec):
        ev.error = ("no top-level solve(items, k) defined / syntax error"
                    if not spec.driver_src
                    else "driver-contract: candidate must define "
                         f"{spec.driver_funcs} and a single literal-int "
                         f"{spec.driver_literal_ints} (no other bindings of them)")
        ev.correct = False
        return ev

    shape = driver_shape_violations(code, spec)
    if shape:
        # v10 P2: precise, feedback-able contract error (also fed to the LLM
        # Builder's next prompt so it stops wasting rounds on interface failures).
        ev.error = "driver-contract: " + ", ".join(shape)
        ev.correct = False
        return ev

    do_quality = bool(spec.verdict_src)        # v2: continuous-quality domain?
    res = _run_subprocess(code, spec, cfg, cases=spec.criteria_cases,
                          do_benchmarks=True, do_memory=True,
                          do_quality=do_quality)

    ev.correct = bool(res.get("all_correct"))
    ev.mutated_input = bool(res.get("any_mutated"))
    ev.timings = res.get("timings", {}) or {}
    ev.timing_stddev = res.get("timing_stddev", {}) or {}
    ev.peak_memory_bytes = int(res.get("peak_memory_bytes", 0) or 0)
    # v2: per-workload quality (recall@K) and the mean over the MAIN workloads.
    ev.quality_by_workload = res.get("quality", {}) or {}
    if do_quality:
        main = [w["name"] for w in spec.workloads]
        qs = [ev.quality_by_workload[w] for w in main
              if w in ev.quality_by_workload]
        ev.quality = round(sum(qs) / len(qs), 4) if qs else 0.0
    # Collect every error that occurred -- benchmark and memory phases run
    # independently and can BOTH fail, so these are separate `if`s (an if/elif
    # chain would silently discard the second error).
    errs: list[str] = []
    if res.get("error"):
        errs.append(res["error"])
        ev.traceback = res.get("traceback")
    if res.get("bench_error"):
        errs.append("benchmark failed: " + res["bench_error"])
        ev.traceback = ev.traceback or res.get("bench_traceback")
    if res.get("mem_error"):
        errs.append("memory probe failed: " + res["mem_error"])

    # COMPLETENESS GATE: a candidate that passed correctness but whose benchmark
    # did not cover every workload (crash, timeout, or partial run) cannot be
    # scored or compared fairly -- a PARTIAL set of timings would let it win some
    # regimes while hiding a blow-up in another, corrupting winners and the
    # baseline. Such a candidate is NOT admitted as correct.
    expected_wls = {w["name"] for w in spec.workloads}
    incomplete = bool(expected_wls) and set(ev.timings) != expected_wls
    if ev.correct and incomplete:
        missing = sorted(expected_wls - set(ev.timings))
        if not any("benchmark" in e for e in errs):
            errs.append(f"benchmark incomplete (missing {missing})")
        ev.correct = False     # un-trustworthy evidence -> not a candidate winner

    if errs:
        ev.error = " | ".join(errs)

    # Record concrete failing cases so the lesson is specific, not "it failed".
    for name, rec in (res.get("cases") or {}).items():
        if not rec.get("ok"):
            ev.failures.append({
                "case": name,
                "mutated": rec.get("mutated", False),
                "error": rec.get("error"),
                "k": rec.get("k"), "n": rec.get("n"),
                "expected": rec.get("expected"), "got": rec.get("got"),
            })
    return ev


def check_cases(code: str, cases: list, spec: GroundingSpec, cfg: Config) -> dict:
    """Adversary helper: run a set of (name, items, k) cases against the oracle,
    NO benchmarks. Returns the raw per-case result dict."""
    if (all_safety_violations(code) or not candidate_ok(code, spec)
            or driver_shape_violations(code, spec)):
        return {"import_ok": False, "cases": {}, "all_correct": False,
                "error": "rejected by static checks"}
    return _run_subprocess(code, spec, cfg, cases=cases,
                           do_benchmarks=False, do_memory=False)


def measure_recall_on(code: str, workloads: list, spec: GroundingSpec,
                      cfg: Config) -> dict:
    """v2 Adversary helper: measure the candidate's quality (recall@K) on a set
    of (possibly hostile) WORKLOAD specs, with no timing/memory. Returns
    {workload_name: recall}. This is how the Adversary attacks the METRIC --
    finding streams on which a candidate's recall collapses."""
    if (all_safety_violations(code) or not candidate_ok(code, spec)
            or driver_shape_violations(code, spec)):
        return {}
    res = _run_subprocess(code, spec, cfg, cases=[], do_benchmarks=False,
                          do_memory=False, quality_workloads=workloads)
    return res.get("quality", {}) or {}


def measure_recall_full(code: str, workloads: list, spec: GroundingSpec,
                        cfg: Config) -> dict:
    """v11 A3: like measure_recall_on but returns the FULL result fields the
    multi-seed machinery needs: {"quality": {...}, "quality_seeds": {...}}."""
    if (all_safety_violations(code) or not candidate_ok(code, spec)
            or driver_shape_violations(code, spec)):
        return {"quality": {}, "quality_seeds": {}}
    res = _run_subprocess(code, spec, cfg, cases=[], do_benchmarks=False,
                          do_memory=False, quality_workloads=workloads)
    return {"quality": res.get("quality", {}) or {},
            "quality_seeds": res.get("quality_seeds", {}) or {}}


def probe_generator(probe_code: str, workload: dict, spec: GroundingSpec,
                    cfg: Config) -> tuple:
    """v5: validate a PROPOSED generator. Run a trusted probe solver on ONE
    workload the generator produces (the spec's workload_gen_src must already carry
    the generator's source block). Returns (ran_ok, stream_len, error):
      * ran_ok is True ONLY if the generator produced a well-formed, runnable stream
        (the runner records its length + a recall; a generator that raised or
        returned a bad shape leaves the name OUT of `quality`);
      * stream_len lets the caller reject a super-linear (unbounded) generator;
      * error carries the runner's diagnostic when it did not run.
    This is how a generator EARNS admission -- name-presence alone is not enough."""
    if (all_safety_violations(probe_code) or not candidate_ok(probe_code, spec)
            or driver_shape_violations(probe_code, spec)):
        return (False, 0, "probe solver rejected by static checks")
    res = _run_subprocess(probe_code, spec, cfg, cases=[], do_benchmarks=False,
                          do_memory=False, quality_workloads=[workload])
    name = workload["name"]
    q = res.get("quality", {}) or {}
    qlen = res.get("quality_len", {}) or {}
    qerr = res.get("quality_errors", {}) or {}
    if name not in q:
        return (False, int(qlen.get(name, 0)),
                qerr.get(name) or res.get("error") or "no runnable stream")
    return (True, int(qlen.get(name, 0)), qerr.get(name))


def diagnose(code: str, workload: dict, spec: GroundingSpec, cfg: Config) -> dict:
    """v6 CAUSAL guidance: run `code` on one workload and return WHICH true heavy
    hitters it missed -- {expected, got, missed}. This is the concrete failure the
    proposer is asked to target, instead of a keyword. Returns {} if it could not
    run (the diagnosis is best-effort, never blocking)."""
    if (all_safety_violations(code) or not candidate_ok(code, spec)
            or driver_shape_violations(code, spec)):
        return {}
    res = _run_subprocess(code, spec, cfg, cases=[], do_benchmarks=False,
                          do_memory=False, diagnose_workloads=[workload])
    d = (res.get("diagnose") or {}).get(workload["name"])
    if not d:
        return {}
    exp, got = d.get("expected", []), d.get("got", [])
    missed = [x for x in exp if x not in set(got)]
    return {"expected": exp, "got": got, "missed": missed}


# --------------------------------------------------------------------------- #
# Scoring -- relative to a fixed baseline so scores are stable across iterations
# --------------------------------------------------------------------------- #

def baseline_from(ev: Evidence) -> dict:
    """Capture the reference numbers (the seed's) used to normalise everyone.
    v1 keeps the PER-WORKLOAD baseline so perf can be scored regime-by-regime
    (a single large-n workload no longer dominates the aggregate)."""
    return {
        "perf_total": max(sum(ev.timings.values()), 1e-9),
        "perf_by_workload": {w: max(t, 1e-9) for w, t in ev.timings.items()},
        "peak_memory_bytes": max(ev.peak_memory_bytes, 1),
        "ast_nodes": max(ev.ast_nodes, 1),
    }


def _ratio_score(baseline_val: float, candidate_val: float) -> float:
    """Map a 'smaller is better' metric to [0,1]: 0.5 at parity with baseline,
    -> 1 as the candidate gets much better, -> 0 as it gets much worse."""
    candidate_val = max(candidate_val, 1e-9)
    ratio = baseline_val / candidate_val          # >1 == better than baseline
    return ratio / (1.0 + ratio)


def score(ev: Evidence, baseline: dict) -> Evidence:
    """Fill perf/mem/simplicity sub-scores in place, relative to `baseline`.

    v1 perf is the MEAN of per-workload ratio-scores, so every regime counts
    equally. This is what lets an adaptive hybrid (which wins each regime) score
    above a pure approach that only wins the largest-n regime -- the synthesis
    signal would otherwise be drowned out by raw-time summation."""
    by_wl = baseline.get("perf_by_workload")
    if ev.correct and ev.timings and by_wl:
        scores = [_ratio_score(by_wl[w], ev.timings[w])
                  for w in ev.timings if w in by_wl]
        ev.perf_score = sum(scores) / len(scores) if scores else 0.0
    elif ev.correct and ev.timings:                  # fallback: aggregate
        ev.perf_score = _ratio_score(baseline["perf_total"],
                                     sum(ev.timings.values()))
    else:
        ev.perf_score = 0.0
    if ev.correct and ev.peak_memory_bytes:
        ev.mem_score = _ratio_score(baseline["peak_memory_bytes"],
                                    ev.peak_memory_bytes)
    else:
        ev.mem_score = 0.0
    ev.simplicity_score = _ratio_score(baseline["ast_nodes"], ev.ast_nodes)
    return ev


def fitness(ev: Evidence, cfg: Config) -> float | None:
    """Composite fitness. Correctness is a HARD GATE: an incorrect approach is
    disqualified (None), never merely down-weighted -- this is the Empirical
    Grounding Rule made arithmetic."""
    if not (ev.grounded and ev.correct):
        return None
    return (cfg.w_perf * ev.perf_score
            + cfg.w_mem * ev.mem_score
            + cfg.w_simplicity * ev.simplicity_score)
