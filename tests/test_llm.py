"""LLM-backend plumbing that must work WITHOUT a network: robust JSON extraction
and the Adversary's input sanitiser (the harness must never run untrusted, ill-
typed 'cases' a model hallucinated)."""

import pytest

from rde.llm import extract_json
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
