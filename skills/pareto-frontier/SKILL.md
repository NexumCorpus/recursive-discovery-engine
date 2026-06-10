---
name: pareto-frontier
description: Reward and track multi-objective discovery with a 2D Pareto frontier and marginal-hypervolume gain. Use when optimizing or searching over a (quality vs cost) trade-off — recall vs memory, accuracy vs latency, score vs money — and you want a candidate's fitness to be "how much it advanced the frontier" (0 for a dominated/recombinant point, positive for one that opens a new region), plus a principled "which point would I ship within budget" recommendation. Better than a scalar best when worse-on-one-axis-but-novel candidates matter.
---

# Pareto Frontier (marginal-hypervolume reward)

A small, dependency-free 2-objective Pareto frontier with a **marginal-hypervolume
reward**. Extracted from the Recursive Discovery Engine, stable since v2.

## When to use this

Use it whenever the objective is a **trade-off, not a single number**, and you are
*discovering / searching* rather than tuning one metric:

- recall vs memory, accuracy vs latency, quality vs $ / tokens, coverage vs runtime.
- you want "reaching a NEW region of the trade-off" to be rewarded even if the
  candidate is *worse on one axis* — which a scalar "beat the best" reward cannot do.
- you want an anti-collapse property for free: a candidate that merely re-derives an
  existing point scores **0**, so recombination/churn is automatically unrewarded.

If your objective is genuinely one scalar, you do not need this — use the scalar.

## The core idea

Track the non-dominated set over `(quality↑, cost↓)`. A candidate's **fitness** is
the *marginal hypervolume* it adds: the extra dominated area its point creates.

- on/under the frontier → **0.0** (dominated or duplicate);
- reaches an unreachable `(quality, cost)` combo → **positive**, scaled into `[0,1]`
  by normalising cost against a fixed reference (e.g. the baseline's cost).

## Interface

```python
from frontier import Frontier, FrontierPoint

f = Frontier(cost_ref=seed_cost)        # fixed worst-case cost (e.g. baseline)

p = FrontierPoint(label="cand-7", quality=0.84, cost=13_000)
reward = f.gain(p)                       # marginal HV contribution -> USE AS FITNESS
f.add(p)                                 # insert if non-dominated; prune dominated

f.hypervolume()                          # current frontier area (progress signal)
f.recommended(budget=64_000)             # the point to SHIP: best quality <= budget
f.quality_gap_above_budget(budget)       # how much quality sits just above budget
f.summary(budget)                        # one-line human-readable state
```

`FrontierPoint(label, quality, cost, extra={})`: `quality` higher-is-better
(typically `[0,1]`), `cost` lower-is-better (any unit), `extra` for secondary
metrics you want to carry but not optimize.

## Usage pattern (discovery loop)

```python
f = Frontier(cost_ref=baseline.cost)
for cand in candidates:
    pt = FrontierPoint(cand.id, quality=measure_quality(cand), cost=measure_cost(cand))
    cand.fitness = f.gain(pt)            # reward = advancing the frontier
    f.add(pt)
ship = f.recommended(budget)            # the within-budget winner
# A large quality_gap_above_budget(budget) is a concrete synthesis target:
# "there is high quality just over the line — get it within budget."
```

## Notes & extensions

- **2D is deliberate**: it keeps the hypervolume *exact* and cheap. Extending to 3D
  (e.g. add latency) is mechanical — sort + sweep — but loses the simple staircase.
- **Re-scoring on a moving metric**: if your quality measure hardens over time
  (e.g. adversarial workloads are added), re-measure every frontier point on the new
  metric and rebuild before comparing new candidates — otherwise points scored on
  different metrics are not comparable. (RDE does this in its engine; the frontier
  object itself is metric-agnostic.)
- `cost_ref` sets the "free" point: a candidate at `cost_ref` gets zero cost-saving
  credit; pick it as a stable worst case (the seed/baseline) so rewards are stable.

## Files

- `frontier.py` — the frontier (stdlib-only).

## Provenance

Canonical source: `rde/frontier.py`. RDE maps quality=recall@K, cost=peak-memory.
