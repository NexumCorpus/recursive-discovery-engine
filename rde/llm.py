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

import json
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
        if not cfg.api_key:
            raise LLMError(
                f"No API key in ${cfg.api_key_env}. Put it in .env "
                f"(see .env.example) or run with --backend deterministic.")

    def chat(self, model: str, system: str, user: str, *,
             temperature: float = 0.4, max_tokens: int = 4096) -> str:
        """Single-turn completion. Returns the assistant message text. Retries
        transient failures with exponential backoff; raises LLMError on give-up."""
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
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_err = e
            # backoff: 1s, 2s, 4s, ...
            time.sleep(2 ** attempt)
        raise LLMError(f"{model}: gave up after {self.cfg.max_retries} "
                       f"attempts: {last_err}")

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
