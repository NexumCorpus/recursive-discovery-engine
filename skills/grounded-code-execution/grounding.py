"""Grounded code execution -- run untrusted / generated Python in isolation and
verify it against an oracle.

Extracted from the Recursive Discovery Engine (`rde/grounding.py`), the load-bearing
"Empirical Grounding Rule" harness that has run every candidate since v0: *no claim
about a piece of code is believed until the code has been executed, in isolation,
and checked against a trusted reference.*

What it gives you, for a `solve(items, k)` style contract:
  1. a **static safety screen** (see `safety_screen.py`) -- reject obvious dangers;
  2. **isolated execution** in a fresh `python -E -s` subprocess with PYTHONPATH
     stripped, a hard **wall-clock timeout**, and a **memory cap** enforced by a
     psutil RSS watchdog (so an unbounded allocation can't thrash the host);
  3. **correctness** against a brute-force `oracle`, with input-mutation detection;
  4. **benchmarks** (median timing + measurement noise) and a **peak-memory** probe;
  5. optional **continuous quality** (e.g. recall@K) via a custom `verdict`, on the
     main workloads AND on extra/hostile ones;
  6. a **completeness gate** -- a candidate that passes correctness but whose
     benchmark crashed on some workload is NOT admitted (partial evidence is unfair);
  7. results streamed to disk so a mid-run kill still leaves a usable verdict.

The problem is described by a `GroundingSpec`: the source of `oracle(items, k)`, a
`gen_workload(n, dist, seed)` stream generator, trusted `criteria_cases`,
benchmark `workloads`, a memory-probe workload, and an optional `verdict`.

Public API
----------
    ground(code, spec, *, timeout_s, mem_cap_mb, bench_repeats, bench_seed) -> GroundingResult
    measure_quality_on(code, workloads, spec, **opts) -> {workload_name: quality}
    check_cases(code, cases, spec, **opts) -> raw dict   # correctness only
    probe_generator(probe_code, workload, spec, **opts) -> (ran_ok, stream_len, error)
    diagnose(code, workload, spec, **opts) -> {expected, got, missed}

`opts` are the same execution knobs (timeout_s, mem_cap_mb, bench_repeats, bench_seed).
Requires: Python 3.10+, stdlib only (`psutil` optional -- enables the memory cap).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from safety_screen import (defines_function, safety_violations,
                           simplicity_metrics)

# Default execution limits (override per call).
DEFAULT_TIMEOUT_S = 20.0
DEFAULT_MEM_CAP_MB = 1536
DEFAULT_BENCH_REPEATS = 5
DEFAULT_BENCH_SEED = 1234


@dataclass
class GroundingSpec:
    """Everything the harness needs to ground one problem. All `*_src` fields are
    Python SOURCE (e.g. via ``inspect.getsource``) spliced into the subprocess."""
    oracle_src: str            # source defining oracle(items, k) -> reference answer
    workload_gen_src: str      # source defining gen_workload(n, dist, seed) -> list
    criteria_cases: list       # [(name, items, k)] -- the trusted correctness suite
    workloads: list            # [{"name","n","k","dist"}] -- benchmark + main quality
    mem_workload: dict         # one {"name","n","k","dist"} for the memory probe
    # Optional source of verdict(got, expected, items, k) -> (ok: bool, quality: float).
    # Empty -> exact equality (closed oracle). Provide it for approximate/continuous
    # domains (e.g. recall@K), which also turns on per-workload quality measurement.
    verdict_src: str = ""


@dataclass
class GroundingResult:
    """The grounded evidence -- every field computed by execution, never asserted."""
    correct: bool = False
    mutated_input: bool = False
    timings: dict = field(default_factory=dict)            # workload -> median secs
    timing_stddev: dict = field(default_factory=dict)      # workload -> stddev
    peak_memory_bytes: int = 0
    quality: float = 1.0                                   # mean over main workloads
    quality_by_workload: dict = field(default_factory=dict)
    loc: int = 0
    ast_nodes: int = 0
    error: str | None = None
    traceback: str | None = None
    failures: list = field(default_factory=list)

    def total_time(self) -> float:
        return sum(self.timings.values())

    def summary(self) -> str:
        if self.error and not self.correct:
            return f"ERROR: {self.error}"
        return (f"ok | quality={self.quality:.2f} "
                f"mem={self.peak_memory_bytes // 1024}KB "
                f"| t={self.total_time() * 1e3:.1f}ms")


# --------------------------------------------------------------------------- #
# The subprocess runner (written to a temp file and executed). Plain text;
# {oracle_src}/{gen_src}/{verdict_src} are spliced in. Streams to result.json so a
# kill mid-benchmark still leaves the correctness verdict on disk.
# --------------------------------------------------------------------------- #
_RUNNER_TEMPLATE = '''\
import json, sys, importlib, traceback, tracemalloc, random, math

JOB = json.loads(sys.argv[1])
RESULT_PATH = JOB["result_path"]

{oracle_src}

{gen_src}

def verdict(got, expected, items, k):
    eq = (got == expected)
    return (eq, 1.0 if eq else 0.0)

{verdict_src}

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

try:
    candidate = importlib.import_module("candidate")
    solve = getattr(candidate, "solve")
    state["import_ok"] = True
except Exception as e:
    state["error"] = "import: " + repr(e)
    state["traceback"] = traceback.format_exc()
    state["phase"] = "import_failed"
    _write(state)
    sys.exit(0)

def _trunc(seq, n=12):
    seq = list(seq)
    return seq[:n] + (["...(+%d)" % (len(seq) - n)] if len(seq) > n else [])

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
    except Exception as e:
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

# Only the empirically-correct earn a benchmark. Each phase is guarded so a failure
# is RECORDED, never silently swallowed (a blind crash once read as "correct, 0ms").
if all_correct and JOB.get("do_benchmarks"):
    import time
    seed = JOB["bench_seed"]
    reps = JOB["bench_repeats"]
    try:
        for wl in JOB["workloads"]:
            data = gen_workload(wl["n"], wl["dist"], seed)
            if JOB.get("do_quality"):
                try:
                    _exp = oracle(list(data), wl["k"])
                    _got = solve(list(data), wl["k"])
                    _ok, _q = verdict(_got, _exp, data, wl["k"])
                    state["quality"][wl["name"]] = float(_q)
                except Exception as _e:
                    state["quality"][wl["name"]] = 0.0
            samples = []
            for _ in range(reps):
                d = list(data)
                t0 = time.perf_counter()
                solve(d, wl["k"])
                samples.append(time.perf_counter() - t0)
            samples.sort()
            state["timings"][wl["name"]] = samples[len(samples) // 2]
            if len(samples) > 1:
                mean = sum(samples) / len(samples)
                var = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
                state["timing_stddev"][wl["name"]] = math.sqrt(var)
            else:
                state["timing_stddev"][wl["name"]] = 0.0
            state["phase"] = "bench:" + wl["name"]
            _write(state)
    except Exception as e:
        state["bench_error"] = repr(e)
        state["bench_traceback"] = traceback.format_exc()
        _write(state)

if all_correct and JOB.get("do_memory"):
    try:
        mw = JOB["mem_workload"]
        data = gen_workload(mw["n"], mw["dist"], JOB["bench_seed"])
        d = list(data)                  # copy BEFORE start: don't count the input
        tracemalloc.start()
        solve(d, mw["k"])
        state["peak_memory_bytes"] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    except Exception as e:
        state["mem_error"] = repr(e)
        _write(state)

# Continuous quality on extra (possibly hostile) workloads. SEPARATE a broken
# GENERATOR (workload unusable -> record NO quality, so callers can reject it) from a
# SOLVE crash on a valid stream (a genuine collapse -> record 0.0).
if all_correct and JOB.get("quality_workloads"):
    for wl in JOB["quality_workloads"]:
        try:
            data = gen_workload(wl["n"], wl["dist"], JOB["bench_seed"])
            if not isinstance(data, (list, tuple)) or not data:
                raise TypeError("generator returned a non-sequence / empty stream")
            for _x in data:
                if not isinstance(_x, int) or isinstance(_x, bool):
                    raise TypeError("generator stream is not all ints")
        except Exception as e:
            state.setdefault("quality_errors", {{}})[wl["name"]] = "gen: " + repr(e)
            continue
        state.setdefault("quality_len", {{}})[wl["name"]] = len(data)
        try:
            _exp = oracle(list(data), wl["k"])
            _got = solve(list(data), wl["k"])
            _ok, _q = verdict(_got, _exp, data, wl["k"])
            state["quality"][wl["name"]] = float(_q)
        except Exception as e:
            state.setdefault("quality_errors", {{}})[wl["name"]] = "solve: " + repr(e)
            state["quality"][wl["name"]] = 0.0
    _write(state)

# DIAGNOSE: record the oracle answer vs the candidate's answer on specific
# workloads, so a caller can see WHAT the candidate got wrong (causal feedback).
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
        except Exception as e:
            state.setdefault("diagnose_errors", {{}})[wl["name"]] = repr(e)
    _write(state)

state["phase"] = "done"
_write(state)
'''


# --------------------------------------------------------------------------- #
# Isolated execution
# --------------------------------------------------------------------------- #
def _run_with_limits(args: list, cwd: str, env: dict, timeout: float,
                     mem_cap_bytes: int) -> tuple:
    """Run a subprocess under a wall-clock timeout AND a best-effort RSS cap. A
    wall-clock timeout alone lets an unbounded allocation thrash the host for the
    whole window before it is reaped; a psutil watchdog in the PARENT samples the
    child's RSS and kills it the instant it breaches the cap. `resource.setrlimit`
    is POSIX-only, so the watchdog is the portable mechanism; without psutil we fall
    back to a plain wall-clock kill. Returns (returncode, stderr, timed_out,
    mem_killed); stdout is discarded (the runner streams to result.json)."""
    try:
        import psutil
    except Exception:
        psutil = None
    proc = subprocess.Popen(args, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ps = None
    if psutil is not None:
        try:
            ps = psutil.Process(proc.pid)
        except Exception:
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
                    except Exception:
                        pass
                if rss > mem_cap_bytes:
                    mem_killed = True
                    proc.kill()
                    break
            except Exception:
                pass
        time.sleep(0.05)
    try:
        _out, err = proc.communicate(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        _out, err = "", ""
    return (proc.returncode, err or "", timed_out, mem_killed)


def _run_subprocess(code: str, spec: GroundingSpec, *, cases: list,
                    do_benchmarks: bool, do_memory: bool, do_quality: bool = False,
                    quality_workloads=None, diagnose_workloads=None,
                    timeout_s=DEFAULT_TIMEOUT_S, mem_cap_mb=DEFAULT_MEM_CAP_MB,
                    bench_repeats=DEFAULT_BENCH_REPEATS,
                    bench_seed=DEFAULT_BENCH_SEED) -> dict:
    """Execute `code` in an isolated process and return the parsed result dict
    (always a dict, with an error/timed_out flag on failure)."""
    tmp = Path(tempfile.mkdtemp(prefix="grounded_"))
    try:
        (tmp / "candidate.py").write_text(code, encoding="utf-8")
        result_path = tmp / "result.json"
        runner = _RUNNER_TEMPLATE.format(
            oracle_src=spec.oracle_src, gen_src=spec.workload_gen_src,
            verdict_src=spec.verdict_src or "")
        (tmp / "runner.py").write_text(runner, encoding="utf-8")

        job = {
            "result_path": str(result_path),
            "cases": [[n, list(it), k] for (n, it, k) in cases],
            "workloads": spec.workloads,
            "mem_workload": spec.mem_workload,
            "bench_repeats": bench_repeats,
            "bench_seed": bench_seed,
            "do_benchmarks": do_benchmarks,
            "do_memory": do_memory,
            "do_quality": do_quality,
            "quality_workloads": quality_workloads or [],
            "diagnose_workloads": diagnose_workloads or [],
        }
        # Strip PYTHONPATH so candidate code can't import the parent's packages; the
        # subprocess only needs stdlib + the candidate in its own cwd.
        child_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"

        # -E ignores PYTHON* env vars, -s skips user site-packages. (Not -I: that
        # also drops the script dir from sys.path, breaking `import candidate`.)
        mem_cap = int(mem_cap_mb or 0) * 1024 * 1024
        rc, stderr, timed_out, mem_killed = _run_with_limits(
            [sys.executable, "-E", "-s", str(tmp / "runner.py"), json.dumps(job)],
            str(tmp), child_env, timeout_s, mem_cap)
        proc_err = None
        if mem_killed:
            proc_err = f"killed: RSS exceeded {mem_cap_mb}MB cap (unbounded allocation)"
        elif rc != 0 and not timed_out:
            proc_err = (stderr or "nonzero exit").strip()[-500:]

        no_output = {"phase": "no_output", "import_ok": False, "cases": {},
                     "all_correct": False, "any_mutated": False, "timings": {},
                     "peak_memory_bytes": 0, "error": proc_err}
        if result_path.exists():
            try:
                res = json.loads(result_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                res = dict(no_output)
                res["error"] = proc_err or f"corrupted result.json: {e}"
        else:
            res = no_output
        res["timed_out"] = timed_out
        if timed_out and not res.get("error"):
            res["error"] = f"timeout after {timeout_s}s (phase={res.get('phase')})"
        if proc_err and not res.get("error"):
            res["error"] = proc_err
        return res
    finally:
        for p in sorted(tmp.glob("*"), reverse=True):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            tmp.rmdir()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def ground(code: str, spec: GroundingSpec, *, timeout_s=DEFAULT_TIMEOUT_S,
           mem_cap_mb=DEFAULT_MEM_CAP_MB, bench_repeats=DEFAULT_BENCH_REPEATS,
           bench_seed=DEFAULT_BENCH_SEED) -> GroundingResult:
    """Run the full battery (safety screen -> isolated correctness -> benchmarks +
    memory + quality -> completeness gate) and return grounded `GroundingResult`."""
    res_ev = GroundingResult()
    res_ev.loc, res_ev.ast_nodes = simplicity_metrics(code)

    violations = safety_violations(code)
    if violations:
        res_ev.error = "static-safety: " + ", ".join(violations)
        return res_ev
    if not defines_function(code, "solve"):
        res_ev.error = "no top-level solve(items, k) defined / syntax error"
        return res_ev

    do_quality = bool(spec.verdict_src)
    res = _run_subprocess(code, spec, cases=spec.criteria_cases,
                          do_benchmarks=True, do_memory=True, do_quality=do_quality,
                          timeout_s=timeout_s, mem_cap_mb=mem_cap_mb,
                          bench_repeats=bench_repeats, bench_seed=bench_seed)

    res_ev.correct = bool(res.get("all_correct"))
    res_ev.mutated_input = bool(res.get("any_mutated"))
    res_ev.timings = res.get("timings", {}) or {}
    res_ev.timing_stddev = res.get("timing_stddev", {}) or {}
    res_ev.peak_memory_bytes = int(res.get("peak_memory_bytes", 0) or 0)
    res_ev.quality_by_workload = res.get("quality", {}) or {}
    if do_quality:
        main = [w["name"] for w in spec.workloads]
        qs = [res_ev.quality_by_workload[w] for w in main
              if w in res_ev.quality_by_workload]
        res_ev.quality = round(sum(qs) / len(qs), 4) if qs else 0.0

    errs: list[str] = []
    if res.get("error"):
        errs.append(res["error"])
        res_ev.traceback = res.get("traceback")
    if res.get("bench_error"):
        errs.append("benchmark failed: " + res["bench_error"])
        res_ev.traceback = res_ev.traceback or res.get("bench_traceback")
    if res.get("mem_error"):
        errs.append("memory probe failed: " + res["mem_error"])

    # COMPLETENESS GATE: correctness-passed but benchmark didn't cover every workload
    # (crash/timeout/partial) -> un-trustworthy, NOT admitted as correct.
    expected_wls = {w["name"] for w in spec.workloads}
    if res_ev.correct and expected_wls and set(res_ev.timings) != expected_wls:
        missing = sorted(expected_wls - set(res_ev.timings))
        if not any("benchmark" in e for e in errs):
            errs.append(f"benchmark incomplete (missing {missing})")
        res_ev.correct = False
    if errs:
        res_ev.error = " | ".join(errs)

    for name, rec in (res.get("cases") or {}).items():
        if not rec.get("ok"):
            res_ev.failures.append({
                "case": name, "mutated": rec.get("mutated", False),
                "error": rec.get("error"), "k": rec.get("k"), "n": rec.get("n"),
                "expected": rec.get("expected"), "got": rec.get("got")})
    return res_ev


def check_cases(code: str, cases: list, spec: GroundingSpec, **opts) -> dict:
    """Correctness only (no benchmarks): run (name, items, k) cases vs the oracle."""
    if safety_violations(code) or not defines_function(code, "solve"):
        return {"import_ok": False, "cases": {}, "all_correct": False,
                "error": "rejected by static checks"}
    return _run_subprocess(code, spec, cases=cases, do_benchmarks=False,
                           do_memory=False, **opts)


def measure_quality_on(code: str, workloads: list, spec: GroundingSpec,
                       **opts) -> dict:
    """Measure the candidate's continuous quality (via `verdict`, e.g. recall@K) on
    a set of (possibly hostile) workload specs -- no timing/memory. Returns
    {workload_name: quality}. The way to find inputs where a candidate collapses."""
    if safety_violations(code) or not defines_function(code, "solve"):
        return {}
    res = _run_subprocess(code, spec, cases=[], do_benchmarks=False,
                          do_memory=False, quality_workloads=workloads, **opts)
    return res.get("quality", {}) or {}


def probe_generator(probe_code: str, workload: dict, spec: GroundingSpec,
                    **opts) -> tuple:
    """Validate a workload GENERATOR (whose source is already in
    spec.workload_gen_src): run a trusted probe solver on one workload it produces.
    Returns (ran_ok, stream_len, error). ran_ok is True only if it produced a
    well-formed runnable stream; stream_len lets you reject an unbounded generator."""
    if safety_violations(probe_code) or not defines_function(probe_code, "solve"):
        return (False, 0, "probe solver rejected by static checks")
    res = _run_subprocess(probe_code, spec, cases=[], do_benchmarks=False,
                          do_memory=False, quality_workloads=[workload], **opts)
    name = workload["name"]
    q = res.get("quality", {}) or {}
    qlen = res.get("quality_len", {}) or {}
    qerr = res.get("quality_errors", {}) or {}
    if name not in q:
        return (False, int(qlen.get(name, 0)),
                qerr.get(name) or res.get("error") or "no runnable stream")
    return (True, int(qlen.get(name, 0)), qerr.get(name))


def diagnose(code: str, workload: dict, spec: GroundingSpec, **opts) -> dict:
    """Run `code` on one workload and return what it got wrong vs the oracle:
    {expected, got, missed}. The concrete failure signal for causal feedback.
    Returns {} if it could not run (best-effort, never blocking)."""
    if safety_violations(code) or not defines_function(code, "solve"):
        return {}
    res = _run_subprocess(code, spec, cases=[], do_benchmarks=False,
                          do_memory=False, diagnose_workloads=[workload], **opts)
    d = (res.get("diagnose") or {}).get(workload["name"])
    if not d:
        return {}
    exp, got = d.get("expected", []), d.get("got", [])
    return {"expected": exp, "got": got,
            "missed": [x for x in exp if x not in set(got)]}
