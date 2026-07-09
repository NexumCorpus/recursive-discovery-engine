"""Command-line entry point.

    python -m rde --backend deterministic              # zero-cost reference run
    python -m rde --backend llm --profile cheap        # live, three vendors
    python -m rde --backend llm --interactive          # pause at each checkpoint

The CLI wires the chosen backend's roles into the Engine and streams progress.
With --interactive it installs the human-override hook the spec calls for.
"""

from __future__ import annotations

import argparse
import sys

from .config import Config, assess_model_assignment, load_dotenv
from .domains import cache_eviction, heavy_hitters, topk
from .engine import Engine, format_report
from .roles import make_roles
from .types import Action, AlgoClass, Directive


DOMAINS = {"topk": topk.DOMAIN, "heavy_hitters": heavy_hitters.DOMAIN,
           "cache_eviction": cache_eviction.DOMAIN}


def _interactive_checkpoint(state: dict, directive: Directive) -> Directive:
    """Human-override at a synthesis checkpoint. The operator may accept the
    Synthesizer's call, force a different action, or abort the run."""
    print("\n" + "-" * 64)
    print(f"HUMAN CHECKPOINT @ iteration {state['iteration']}")
    print(f"  signals : {state['signals']}")
    print(f"  best    : {state['best']}")
    print(f"  proposed: {directive.action.value.upper()} -- {directive.reason}")
    print("  [Enter]=accept the Synthesizer | a=accept-best | t=terminate | "
          "c=continue | r=reset")
    try:
        choice = input("  > ").strip().lower()
    except EOFError:
        choice = ""
    override = {"a": Action.ACCEPT, "t": Action.TERMINATE,
               "c": Action.CONTINUE, "r": Action.RESET}.get(choice)
    if override:
        directive.action = override
        directive.reason = f"[human override -> {override.value}] " + directive.reason
        # A human RESET still needs a concrete target, else it silently degrades
        # to the Builder's default priority order.
        if override == Action.RESET and state.get("untried"):
            directive.target_class = AlgoClass(state["untried"][0])
    return directive


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rde", description="Recursive Discovery Engine (v0)")
    p.add_argument("--backend", choices=["deterministic", "llm", "codex-cli"],
                   default="deterministic",
                   help="deterministic = reference policy, no API; "
                        "llm = live models via OpenRouter; codex-cli = local Codex subscription")
    p.add_argument("--profile", choices=["default", "cheap"], default="cheap",
                   help="LLM model tier (llm backend only)")
    p.add_argument("--domain", choices=list(DOMAINS), default="topk")
    p.add_argument("--iterations", type=int, default=None,
                   help="max iterations (overrides config default of 8)")
    p.add_argument("--fanout", type=int, default=None,
                   help="approaches proposed per round (default 1)")
    p.add_argument("--interactive", action="store_true",
                   help="pause for human override at each synthesis checkpoint")
    p.add_argument("--quiet", action="store_true",
                   help="suppress the per-step progress stream")
    return p


def main(argv: list[str] | None = None) -> int:
    # Models emit arbitrary unicode (em-dashes, etc.); a Windows cp1252 console
    # would raise UnicodeEncodeError on print. Force UTF-8 where supported.
    try:
        sys.stdout.reconfigure(encoding="utf-8")        # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

    args = build_parser().parse_args(argv)
    load_dotenv()

    cfg = Config(backend=args.backend, profile=args.profile)
    if args.iterations is not None:
        cfg.max_iterations = args.iterations
    if args.fanout is not None:
        cfg.builder_fanout = args.fanout

    domain = DOMAINS[args.domain]
    roles = make_roles(cfg)

    llm = None
    if cfg.backend in {"llm", "codex-cli"}:
        from .llm import LLMClient, LLMError
        try:
            llm = LLMClient(cfg)
        except LLMError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"backend={cfg.backend}  models: {cfg.models}")
        if cfg.backend == "codex-cli":
            assignment = assess_model_assignment(cfg.models)
            print(f"model-independence={assignment['grade']} "
                  f"vendors={','.join(assignment['vendors']) or 'unknown'}")
            if not assignment["independent_corroboration"]:
                print("warning: Codex roles lack cross-provider corroboration; "
                      "this run is exploratory until an external grader and holdout agree")

    logger = (lambda m: None) if args.quiet else print
    engine = Engine(cfg, domain, roles, llm=llm, logger=logger,
                    on_checkpoint=_interactive_checkpoint if args.interactive
                    else None)
    result = engine.run()

    print("\n")
    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
