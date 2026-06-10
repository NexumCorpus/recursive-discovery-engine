---
name: multi-role-discovery
description: Build a self-improving discovery/search loop with three adversarial roles — a Builder that proposes, an Adversary that disproves with executable evidence, and a Synthesizer that alone may change direction or stop — wired around the Empirical Grounding Rule and a marginal-hypervolume reward. Use when designing a system that iteratively GENERATES and IMPROVES solutions (code, configs, designs) and must avoid the failure modes of naive generate-and-check: drift, recombination/churn, shared blind spots, and collapse into low-value iteration. The methodology that ties together grounded-code-execution, pareto-frontier, and ast-safety-screen.
---

# Multi-Role Discovery Loop

The architecture and protective mechanisms behind the Recursive Discovery Engine,
packaged as a reusable pattern. It is the answer to: *how do you let a system (or an
LLM) keep proposing solutions and genuinely get better, instead of looping, gaming
its own metric, or fooling itself?*

## When to use this

Reach for it when you are building an **iterative generate → evaluate → improve**
system and a naive loop would rot:
- an LLM proposes code/configs/designs round after round;
- you need to avoid **drift** (wandering off-target), **recombination** (re-deriving
  what you already have), **shared blind spots** (the proposer and the checker miss
  the same thing), and **collapse** (endless low-value iteration);
- you want every "improvement" to be **earned by executable evidence**, and a single,
  auditable place where convergence is declared.

If you just need to run one piece of generated code once, use
`grounded-code-execution` alone. This skill is for the *loop around it*.

## The three roles (and why three)

| Role | Does | May NOT |
|------|------|---------|
| **Builder** | Proposes candidate solutions. | Decide it succeeded. |
| **Adversary** | Tries to *disprove* a candidate with executable evidence — hostile inputs, edge cases, metric attacks. | *Claim* a break — the grounding harness confirms it. |
| **Synthesizer** | The **only** role that may change direction, declare convergence, or terminate. Reasons over **objective signals it cannot fake**. | Override the measured evidence. |

Separating "propose" from "disprove" from "decide" is what prevents a single agent
from rationalising its own output. In the LLM flavour, the three roles run on
**different model vendors** so they do not share systematic blind spots.

## The protective mechanisms (non-negotiable)

1. **Empirical Grounding Rule** — no claim is believed until the code is *run, in
   isolation, and checked against a trusted oracle*. (→ `grounded-code-execution`.)
2. **Marginal-hypervolume reward** — fitness is *how much a candidate advances the
   (quality, cost) Pareto frontier*, which is **0 for a dominated or recombinant
   point**. Churn is unrewarded for free. (→ `pareto-frontier`.)
3. **Novelty / recombination check** — a structurally-near-identical proposal is
   flagged, not celebrated.
4. **Adversary attacks the metric** — a candidate that looks good on easy inputs but
   *collapses* on a hostile one (that a good method handles) is downgraded, and the
   exposing input is folded into the permanent eval suite (the metric only gets
   harder).
5. **Bounded memory** — keep the Pareto frontier + a little recency, not the full
   transcript, so context (and cost) do not grow without limit over a long run.
6. **Objective drift/stall signals** — computed by the engine, handed to the
   Synthesizer, *which cannot fake them*; they gate ACCEPT/EXPLORE/TERMINATE.
7. **Safety gate** — every piece of untrusted code is statically screened before it
   ever executes. (→ `ast-safety-screen`, inside grounding.)

## The loop (shape)

```
seed -> [ Builder.propose
          -> GROUND (run + verify; reject the unsafe/incorrect)
          -> novelty/recombination check
          -> Adversary.attack (downgrade a metric-gamer; harden the suite)
          -> frontier.gain(point)  == fitness  (0 for dominated/recombinant)
          -> remember (bounded)
          -> Synthesizer.decide (over objective signals)  ] * N
       -> ACCEPT (ship the recommended within-budget point) / TERMINATE
```

`discovery_loop.py` is a ~120-line **runnable** instance of exactly this, composing
the sibling skills. Run it:

```
python3 discovery_loop.py
```

It shows, on a tiny heavy-hitters problem: the grounding rule rejecting an unsafe
candidate, the Adversary exposing an under-capacity summary (recall 0.85 → 0.10 on a
hostile many-heavy stream), a byte-identical proposal flagged `[recomb]` with **zero**
reward, marginal-hypervolume as fitness, and the Synthesizer as the sole terminator.

## Files

- `discovery_loop.py` — the minimal runnable reference (imports the sibling skills).
- (depends on the `grounded-code-execution` and `pareto-frontier` skills next to it.)

## Going further

The full implementation — LLM roles on distinct vendors, a conceptual "why" memory,
behavioural-novelty detection, drift/recombination signals, generative adversaries,
an *open vocabulary* (the system proposes new primitive operations as source), causal
guidance, and Builder↔Adversary self-play — is the Recursive Discovery Engine
(`rde/`). This skill is the stable spine; treat the engine as the worked example of
how far the pattern goes.
