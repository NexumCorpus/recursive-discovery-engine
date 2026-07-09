# Codex CLI backend

RDE now accepts `python -m rde --backend codex-cli`. This is a transport change,
not a claim that Codex has produced a discovery.

- Each Builder, Adversary, and Synthesizer call is an ephemeral `codex exec`
  session with a fresh role prompt.
- The CLI runs `read-only`, ignores unrelated user plugins/MCP configuration by
  default, and returns JSONL that RDE parses for the final agent message and
  token usage.
- RDE continues to treat every generated source fragment as untrusted: AST
  screen, isolated grounding subprocess, oracle, frontier, adversarial pressure,
  and holdout discipline remain unchanged.
- `RDE_CODEX_BIN` selects the executable; `RDE_CODEX_MODEL` is optional;
  `RDE_CODEX_USE_USER_CONFIG=1` opts into local Codex plugin/MCP configuration.
- The default assigns `gpt-5.6-terra` to Builder/Synthesizer and `gpt-5.5` to
  the Adversary. That creates useful intra-family friction, but it is still not
  the original cross-vendor blind-spot defense. RDE prints this distinction as
  `intra-family-diverse`, never as independent corroboration. Configure
  `RDE_BUILDER_MODEL`, `RDE_ADVERSARY_MODEL`, and `RDE_SYNTHESIZER_MODEL` to
  the available role models; `RDE_CODEX_MODEL` is a deliberate global pin that
  collapses the roster to a labelled monoculture.

This backend is suitable for a fresh, pre-registered campaign only after the
domain, baselines, metric, seeds, and holdout plan are frozen. A green RDE run is
not a certified estate claim until it survives the independent claim gate.
