"""LLM-backend plumbing that must work WITHOUT a network: robust JSON extraction
and the Adversary's input sanitiser (the harness must never run untrusted, ill-
typed 'cases' a model hallucinated)."""

import json
import subprocess

import pytest

from rde.config import Config
from rde.llm import LLMClient, extract_json
from rde.roles import LLMAdversary


def test_extract_plain_object():
    assert extract_json('{"action": "accept"}') == {"action": "accept"}


def test_extract_from_json_fence():
    reply = 'Sure!\n```json\n{"action": "reset", "target_class": "heap"}\n```\n'
    assert extract_json(reply) == {"action": "reset", "target_class": "heap"}


def test_extract_object_amid_prose():
    reply = ('My analysis is that we should stop. Here it is: '
             '{"action": "terminate", "reason": "done {really}"} -- thanks!')
    out = extract_json(reply)
    assert out["action"] == "terminate"
    assert out["reason"] == "done {really}"   # braces inside strings respected


def test_extract_array():
    reply = 'Cases:\n[{"items": [1,2,3], "k": 2}]'
    assert extract_json(reply) == [{"items": [1, 2, 3], "k": 2}]


def test_extract_raises_without_json():
    with pytest.raises(ValueError):
        extract_json("no json here at all")


def test_codex_cli_backend_parses_jsonl_and_records_usage(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        stream = "\n".join([
            json.dumps({"type": "thread.started", "thread_id": "t-1"}),
            json.dumps({"type": "item.completed", "item": {
                "type": "agent_message", "text": '{"action":"continue"}'}}),
            json.dumps({"type": "turn.completed", "usage": {
                "input_tokens": 12, "output_tokens": 7}}),
        ])
        return subprocess.CompletedProcess(command, 0, stdout=stream, stderr="")

    monkeypatch.setattr("rde.llm.subprocess.run", fake_run)
    cfg = Config(backend="codex-cli", codex_bin="codex-test")
    client = LLMClient(cfg)
    assert client.chat("codex-cli", "system rule", "user request") == '{"action":"continue"}'
    assert captured["command"][:4] == ["codex-test", "exec", "--ephemeral", "--json"]
    assert "--ignore-user-config" in captured["command"]
    assert captured["command"][-2:] == ["read-only", captured["command"][-1]]
    assert client.usage.calls == 1
    assert client.usage.prompt_tokens == 12
    assert client.usage.completion_tokens == 7


def test_codex_cli_backend_honors_per_role_model(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="\n".join([
            json.dumps({"type": "item.completed", "item": {
                "type": "agent_message", "text": "ok"}}),
            json.dumps({"type": "turn.completed", "usage": {}}),
        ]), stderr="")

    monkeypatch.setattr("rde.llm.subprocess.run", fake_run)
    cfg = Config(backend="codex-cli", codex_bin="codex-test",
                 models={"builder": "gpt-5-codex", "adversary": "other", "synthesizer": "other"})
    assert LLMClient(cfg).chat("gpt-5-codex", "system", "user") == "ok"
    assert captured["command"][captured["command"].index("--model") + 1] == "gpt-5-codex"


def test_adversary_sanitiser_filters_bad_cases():
    adv = LLMAdversary()
    proposals = [
        {"items": [3, 1, 2], "k": 2, "hypothesis": "ok"},     # valid
        {"items": "not a list", "k": 1},                       # bad items
        {"items": [1, 2], "k": "two"},                         # bad k type
        {"items": [1, True, 3], "k": 1},                       # bool sneaking in
        {"items": [1, 2, 3], "k": -1},                         # negative k
        {"k": 1},                                              # missing items
    ]
    cases = adv._sanitize(proposals)
    assert len(cases) == 1
    name, items, k, hyp = cases[0]
    assert items == [3, 1, 2] and k == 2
