---
name: grounded-code-execution
description: Safely execute untrusted or LLM-generated Python in an isolated subprocess and verify it against a trusted oracle — correctness, benchmarks, peak memory, and continuous quality (e.g. recall@K) — with a static safety screen, a hard timeout, and a memory cap. Use whenever you must BELIEVE NOTHING a model claims about a piece of code until it has been run and checked: grading generated solutions, validating candidate algorithms, benchmarking competing implementations, or building a discovery/search loop where every candidate must earn its evidence.
---

# Grounded Code Execution

The "Empirical Grounding Rule" harness from the Recursive Discovery Engine, extracted
as a standalone skill: **no claim about code is believed until the code has been
executed, in isolation, and checked against a trusted reference.**

## When to use this

Any time you have **code you must not trust on its word** and a way to check it:
- grade / validate LLM-generated solutions against expected behaviour;
- benchmark competing implementations fairly (timing + memory + correctness);
- run a **discovery or search loop** where candidates must earn grounded evidence
  before they influence anything (pairs with `pareto-frontier` for the reward and
  `multi-role-discovery` for the loop);
- measure a continuous quality (recall@K, accuracy, score) — not just pass/fail.

It is built for a `solve(items, k)` contract, but the *problem* is entirely yours:
you supply the oracle, the input generator, the test cases, and (optionally) a custom
verdict. Swap those and it grounds a different problem with zero harness changes.

## Safety model

Two layers, by design:
1. **Static screen** (`safety_screen.py`, the `ast-safety-screen` skill): rejects
   disallowed imports / introspection builtins / dunder access *before* execution.
2. **OS isolation** (the real boundary): a fresh `python -E -s` process with
   `PYTHONPATH` stripped, a hard **wall-clock timeout**, and an **RSS memory cap**
   enforced by a psutil watchdog (kills an unbounded allocation in ~50 ms instead of
   letting it thrash the host for the whole timeout). `psutil` is optional; without
   it you still get the timeout.

Results stream to a temp `result.json`, so even a mid-run kill leaves a usable
verdict. A **completeness gate** refuses to call a candidate "correct" if its
benchmark crashed on some workload (partial evidence is unfair evidence).

## Interface

```python
from grounding import GroundingSpec, ground, measure_quality_on, diagnose, probe_generator

spec = GroundingSpec(
    oracle_src=inspect.getsource(oracle),        # oracle(items, k) -> reference
    workload_gen_src=inspect.getsource(gen),     # gen_workload(n, dist, seed) -> list
    criteria_cases=[("tiny", [1,1,2], 1), ...],  # trusted correctness suite
    workloads=[{"name":"w1","n":8000,"k":10,"dist":"zipf"}, ...],  # benchmark/quality
    mem_workload={"name":"mem","n":8000,"k":10,"dist":"uniform"},  # memory probe
    verdict_src=inspect.getsource(verdict),      # verdict(got,exp,items,k)->(ok,quality)
)                                                # omit verdict_src -> exact equality

ev = ground(code, spec, *, timeout_s=20, mem_cap_mb=1536, bench_repeats=5, bench_seed=1234)
#   ev.correct, ev.quality, ev.quality_by_workload, ev.peak_memory_bytes,
#   ev.timings, ev.timing_stddev, ev.error, ev.failures, ev.total_time(), ev.summary()

measure_quality_on(code, workloads, spec, **opts)  -> {workload_name: quality}
diagnose(code, workload, spec, **opts)             -> {expected, got, missed}
probe_generator(probe_code, workload, spec, **opts)-> (ran_ok, stream_len, error)
check_cases(code, cases, spec, **opts)             -> raw correctness dict
```

## Usage

```python
ev = ground(candidate_code, spec)
if ev.correct:
    fitness = frontier.gain(FrontierPoint(id, ev.quality, ev.peak_memory_bytes))  # reward
else:
    learn_from(ev.error, ev.failures)            # specific, grounded failure
```

`measure_quality_on` finds inputs where a candidate *collapses* (the Adversary's
move). `diagnose` returns *what* it got wrong (the causal-feedback signal). See
`example.py` for a complete, runnable heavy-hitters problem.

## Files

- `grounding.py` — the harness (stdlib + optional `psutil`).
- `safety_screen.py` — the bundled static screen (same as the `ast-safety-screen` skill).
- `example.py` — runnable example + smoke test (`python3 example.py`).

## Caveats / provenance

- The static screen is **defence in depth, not a sandbox** — the subprocess
  isolation + limits are the real boundary. Do not relax them.
- `bench_seed` fixes the workload generation, so timings/quality are reproducible.
- The completeness gate means a correct-but-sometimes-crashing candidate reads as
  *incorrect* — intentional; partial evidence must not win.
- Canonical source: `rde/grounding.py` (this is decoupled from RDE's `Config`/`Evidence`).
