"""A MINIMAL, runnable reference for the multi-role discovery loop.

This is the *pattern*, in ~120 lines, composing the sibling skills:
  * grounded-code-execution -> the Empirical Grounding Rule (run + verify);
  * pareto-frontier         -> the discovery reward (marginal hypervolume);
  * ast-safety-screen       -> the safety gate (used inside grounding).

It is deliberately tiny. The full implementation (LLM roles, bounded conceptual
memory, behavioural-novelty, drift/recombination detection, self-play, open
vocabulary, ...) is the Recursive Discovery Engine. This file exists to make the
ROLES and the PROTECTIVE MECHANISMS concrete and testable.

The three roles (the spec):
  * BUILDER     proposes candidate solutions.
  * ADVERSARY   tries to DISPROVE them with executable evidence (never claims a
                break -- the grounding harness decides).
  * SYNTHESIZER is the ONLY role allowed to change direction / declare convergence
                / terminate. It reasons over OBJECTIVE signals it cannot fake.
"""
import sys
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "grounded-code-execution"))
sys.path.insert(0, str(_HERE.parent / "pareto-frontier"))

from grounding import ground, measure_quality_on            # noqa: E402
from frontier import Frontier, FrontierPoint                # noqa: E402
from example import SPEC, EXACT, APPROX                      # reuse the example domain


# --------------------------------------------------------------------------- #
# BUILDER -- proposes candidates. Here: a fixed repertoire + cosmetic variants.
# --------------------------------------------------------------------------- #
_SPACE_SAVING = lambda m: (
    "def solve(stream, k):\n"
    f"    m = max({m} * k, 64); counts = {{}}\n"
    "    for x in stream:\n"
    "        if x in counts: counts[x] += 1\n"
    "        elif len(counts) < m: counts[x] = 1\n"
    "        else:\n"
    "            vk = min(counts, key=counts.get); vv = counts.pop(vk)\n"
    "            counts[x] = vv + 1\n"
    "    return [it for it,_ in sorted(counts.items(), key=lambda kv:(-kv[1],-kv[0]))[:k]]\n")

_REPERTOIRE = [("exact", EXACT), ("ss8", _SPACE_SAVING(8)),
               ("ss32", _SPACE_SAVING(32)), ("ss8_dup", _SPACE_SAVING(8))]


class Builder:
    def __init__(self):
        self._i = 0

    def propose(self):
        if self._i >= len(_REPERTOIRE):
            return None
        label, code = _REPERTOIRE[self._i]
        self._i += 1
        return label, code


# --------------------------------------------------------------------------- #
# ADVERSARY -- attacks the metric. Measures quality on a hostile workload (one a
# GOOD method handles, so a collapse means the candidate is gaming/under-provisioned,
# not that the stream is inherently hard). A RELATIVE collapse -- much worse than the
# candidate's own headline quality -- is what counts. The harness decides, not the
# Adversary; here `many_heavy` overflows a small bounded summary.
# --------------------------------------------------------------------------- #
_HOSTILE = {"name": "adv_many_heavy", "n": 8000, "k": 10, "dist": "many_heavy"}


class Adversary:
    def attack(self, code, ev):
        recalls = measure_quality_on(code, [_HOSTILE], SPEC, bench_repeats=1)
        r = recalls.get(_HOSTILE["name"])
        if r is not None and (ev.quality - r) > 0.25 and r < 0.5:
            return min(ev.quality, r)            # confirmed collapse -> downgrade
        return ev.quality


# --------------------------------------------------------------------------- #
# SYNTHESIZER -- the ONLY terminator. Reasons over signals it cannot fake.
# --------------------------------------------------------------------------- #
class Synthesizer:
    def decide(self, advanced, exhausted):
        if exhausted:
            return "ACCEPT: repertoire exhausted; ship the best grounded point."
        if not advanced:
            return "CONTINUE: last round did not advance the frontier."
        return "CONTINUE: frontier advanced; keep exploring."


def run(max_rounds=8):
    builder, adversary, synth = Builder(), Adversary(), Synthesizer()
    # Seed fixes the frontier's cost reference (worst-case memory).
    seed = ground(EXACT, SPEC, bench_repeats=1)
    frontier = Frontier(cost_ref=seed.peak_memory_bytes * 1.2)
    frontier.add(FrontierPoint("exact", seed.quality, seed.peak_memory_bytes))
    seen_fps = set()                             # novelty / recombination guard

    for rnd in range(1, max_rounds + 1):
        proposed = builder.propose()
        if proposed is None:
            print(synth.decide(advanced=False, exhausted=True))
            break
        label, code = proposed

        # EMPIRICAL GROUNDING RULE: believe nothing until it runs + verifies.
        ev = ground(code, SPEC, bench_repeats=1)
        if not ev.correct:
            print(f"  round {rnd}: {label:8} REJECTED ({ev.error})")
            continue

        # NOVELTY / RECOMBINATION: a byte-identical proposal is not a discovery.
        fp = code.strip()
        is_recomb = fp in seen_fps
        seen_fps.add(fp)

        # ADVERSARY downgrades a metric-gamer; then the FRONTIER scores the HONEST
        # point. fitness = marginal hypervolume (0 for dominated/recombinant -- free).
        quality = adversary.attack(code, ev)
        point = FrontierPoint(label, quality, ev.peak_memory_bytes)
        gain = frontier.gain(point)
        advanced = gain > 1e-6
        frontier.add(point)

        flag = " [recomb]" if is_recomb else ""
        print(f"  round {rnd}: {label:8} q={quality:.2f} mem={ev.peak_memory_bytes//1024}KB"
              f" fitness(HV-gain)={gain:.3f}{flag}")
        print("           " + synth.decide(advanced, exhausted=False))

    rec = frontier.recommended(budget=seed.peak_memory_bytes * 0.5)
    print(f"\nFRONTIER: {frontier.summary(seed.peak_memory_bytes * 0.5)}")
    print(f"RECOMMENDED within budget: {rec.label} "
          f"(q={rec.quality:.2f}, {rec.cost // 1024}KB)")
    return frontier


if __name__ == "__main__":
    run()
    print("SMOKE_OK")
