"""OpenRouter chat client + robust JSON extraction.

Deliberately stdlib-only (urllib): the whole project's premise is that the
empirical harness runs anywhere with zero install friction, and the LLM transport
should not be the thing that breaks that. OpenRouter exposes an OpenAI-compatible
/chat/completions endpoint, so a thin POST is all we need.

Two responsibilities live here and nowhere else:
  * talking to the network (with retries + a hard timeout), and
  * coercing a model's free-text reply into the strict JSON / code the roles
    require. LLMs are chronically chatty; the roles must never have to parse prose.
"""

from __future__ import annotations

import http.client
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .config import Config


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    """Running tally so a run can report what it actually cost in tokens/calls."""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, pt: int, ct: int) -> None:
        self.calls += 1
        self.prompt_tokens += pt
        self.completion_tokens += ct
        m = self.by_model.setdefault(model, {"calls": 0, "pt": 0, "ct": 0})
        m["calls"] += 1; m["pt"] += pt; m["ct"] += ct

    def summary(self) -> str:
        return (f"{self.calls} LLM calls, "
                f"{self.prompt_tokens} prompt + {self.completion_tokens} "
                f"completion tokens")


class LLMClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.usage = Usage()
        self.codex_bin = _resolve_codex_bin(cfg)
        if cfg.backend == "codex-cli":
            return
        if not cfg.api_key:
            raise LLMError(
                f"No API key in ${cfg.api_key_env}. Put it in .env "
                f"(see .env.example) or run with --backend deterministic.")

    def chat(self, model: str, system: str, user: str, *,
             temperature: float = 0.4, max_tokens: int = 4096) -> str:
        """Single-turn completion. Returns the assistant message text. Retries
        transient failures with exponential backoff; raises LLMError on give-up."""
        if self.cfg.backend == "codex-cli":
            return self._chat_codex_cli(model, system, user, temperature=temperature)
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
            # Optional OpenRouter attribution headers; harmless if ignored.
            "HTTP-Referer": "https://localhost/rde",
            "X-Title": "Recursive Discovery Engine",
        }
        url = f"{self.cfg.api_base}/chat/completions"

        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                req = urllib.request.Request(url, data=body, headers=headers,
                                             method="POST")
                with urllib.request.urlopen(
                        req, timeout=self.cfg.request_timeout_s) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                return self._parse_response(model, payload)
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                # 4xx (except 429) are not worth retrying -- the request is bad.
                if e.code != 429 and 400 <= e.code < 500:
                    raise LLMError(f"{model}: HTTP {e.code}: {detail}") from e
                last_err = LLMError(f"HTTP {e.code}: {detail}")
            except (urllib.error.URLError, TimeoutError, OSError,
                    http.client.HTTPException, json.JSONDecodeError) as e:
                # OSError is load-bearing: urllib wraps request-phase socket
                # errors in URLError but getresponse()-phase errors arrive RAW
                # (a server dying mid-call raises ConnectionResetError straight
                # through -- crashed the 2026-07-01 campaign at iteration 2).
                # HTTPException too: a server dying mid-BODY raises
                # IncompleteRead, which subclasses HTTPException, not OSError.
                last_err = e
            # backoff: 1s, 2s, 4s, ...
            time.sleep(2 ** attempt)
        raise LLMError(f"{model}: gave up after {self.cfg.max_retries} "
                       f"attempts: {last_err}")

    def _chat_codex_cli(self, model: str, system: str, user: str, *,
                         temperature: float) -> str:
        """Run one isolated Codex CLI turn and read its JSONL result stream.

        The RDE roles are intentionally single-turn and already place all needed
        context in `system` + `user`, so `--ephemeral` prevents a role's prior
        answer from becoming hidden context for its next proposal. Candidate code
        is still treated as untrusted by the normal grounding path.
        """
        prompt = (
            "You are one role in a Recursive Discovery Engine. Follow the system "
            "contract and user request exactly. Return only the requested final "
            "artifact; do not use tools, edit files, or claim empirical results.\n\n"
            f"SYSTEM CONTRACT:\n{system}\n\nUSER REQUEST:\n{user}"
        )
        command = [self.codex_bin, "exec", "--ephemeral", "--json", "--color", "never"]
        if not self.cfg.codex_use_user_config:
            command.append("--ignore-user-config")
        requested_model = model if model != "codex-cli" else self.cfg.codex_model
        if requested_model:
            command.extend(["--model", requested_model])
        command.extend(["-C", str(Path(__file__).resolve().parent.parent),
                        "-s", "read-only", prompt])
        try:
            proc = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=self.cfg.request_timeout_s,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired as e:
            raise LLMError(f"{model}: Codex CLI timed out after {self.cfg.request_timeout_s}s") from e
        except OSError as e:
            raise LLMError(f"{model}: could not start Codex CLI ({e})") from e
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "Codex CLI failed").strip()[-500:]
            raise LLMError(f"{model}: Codex CLI exit {proc.returncode}: {detail}")

        reply = ""
        usage = {}
        for line in proc.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "item.completed":
                item = event.get("item") or {}
                if item.get("type") == "agent_message" and item.get("text"):
                    reply = str(item["text"])
            elif event.get("type") == "turn.completed":
                usage = event.get("usage") or {}
            elif event.get("type") in {"turn.failed", "error"}:
                detail = event.get("error") or event.get("message") or "Codex turn failed"
                raise LLMError(f"{model}: {detail}")
        if not reply:
            raise LLMError(f"{model}: Codex CLI emitted no agent message")
        self.usage.add(model, int(usage.get("input_tokens", 0)),
                       int(usage.get("output_tokens", 0)))
        return reply

    def _parse_response(self, model: str, payload: dict) -> str:
        if "error" in payload:
            raise LLMError(f"{model}: {payload['error']}")
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"{model}: malformed response: "
                           f"{json.dumps(payload)[:300]}") from e
        u = payload.get("usage", {}) or {}
        self.usage.add(model, int(u.get("prompt_tokens", 0)),
                       int(u.get("completion_tokens", 0)))
        return text or ""


def _resolve_codex_bin(cfg: Config) -> str:
    if cfg.codex_bin:
        return cfg.codex_bin
    local = os.environ.get("LOCALAPPDATA")
    if local:
        root = Path(local) / "OpenAI" / "Codex" / "bin"
        try:
            candidates = sorted(root.glob("*/codex.exe"), reverse=True)
            if candidates:
                return str(candidates[0])
        except OSError:
            pass
    return "codex.exe" if os.name == "nt" else "codex"


# --------------------------------------------------------------------------- #
# JSON extraction -- models wrap JSON in prose / fences / stray commentary.
# --------------------------------------------------------------------------- #

def extract_json(text: str):
    """Best-effort: pull the first well-formed JSON object/array from a reply.

    Strategy: try a straight parse; if that fails, scan for balanced {...} or
    [...] spans (respecting strings/escapes) and parse the first that loads.
    Raises ValueError if nothing parses.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip a ```json ... ``` fence if present.
    if "```" in text:
        fenced = text.split("```")
        for chunk in fenced:
            c = chunk.strip()
            if c.startswith("json"):
                c = c[4:].strip()
            if c[:1] in "{[":
                try:
                    return json.loads(c)
                except json.JSONDecodeError:
                    continue

    # Try whichever structure appears FIRST in the text. An array of objects has
    # its '[' before any '{', so preferring the earliest start avoids grabbing an
    # inner object out of a top-level array.
    candidates = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start != -1:
            candidates.append((start, opener, closer))
    for _, opener, closer in sorted(candidates):
        span = _balanced_span(text, opener, closer)
        if span is not None:
            try:
                return json.loads(span)
            except json.JSONDecodeError:
                continue
    raise ValueError("no parseable JSON found in model reply")


def _balanced_span(text: str, opener: str, closer: str) -> str | None:
    """Return the first balanced opener..closer span, ignoring braces inside
    string literals. None if there is no balanced span."""
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    quote = ""
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
        else:
            if ch in "\"'":
                in_str = True
                quote = ch
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None
