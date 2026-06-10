"""AST-level mutation -- v3's second open-ended generation mechanism.

Where the grammar (in the domain) COMPOSES new structures from primitives, this
module MUTATES an existing program's structure: it perturbs numeric literals and
flips comparison operators in the AST. That is enough to discover, by search, that
(say) a different capacity multiplier or threshold yields a better point on the
frontier -- a candidate whose *behaviour* is new even if its skeleton is shared.

It is deliberately small and safe: it only edits constants/operators of a parsed
tree and re-emits source with `ast.unparse`. Every mutant still goes through the
full grounding + Adversary + frontier pipeline, so a broken or useless mutation
is simply disqualified or earns zero frontier gain.
"""

from __future__ import annotations

import ast


_INT_MUTATORS = [
    lambda v: v * 2,
    lambda v: max(1, v // 2),
    lambda v: v + 1,
    lambda v: max(1, v - 1),
]

_CMP_FLIP = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt,
    ast.Gt: ast.GtE, ast.GtE: ast.Gt,
}


def _mutable_int_nodes(tree: ast.AST) -> list[ast.Constant]:
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, int)
                and not isinstance(node.value, bool) and node.value not in (0,)):
            out.append(node)
    return out


def mutate(code: str, rng) -> tuple[str, str] | None:
    """Return (mutated_source, description) or None if nothing mutable was found.
    `rng` is a random.Random for determinism."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    ints = _mutable_int_nodes(tree)
    cmps = [n for n in ast.walk(tree)
            if isinstance(n, ast.Compare) and len(n.ops) == 1
            and type(n.ops[0]) in _CMP_FLIP]

    choices = []
    if ints:
        choices.append("int")
    if cmps:
        choices.append("cmp")
    if not choices:
        return None

    kind = rng.choice(choices)
    if kind == "int":
        node = rng.choice(ints)
        old = node.value
        node.value = rng.choice(_INT_MUTATORS)(old)
        desc = f"int {old}->{node.value}"
    else:
        node = rng.choice(cmps)
        old_op = type(node.ops[0])
        node.ops[0] = _CMP_FLIP[old_op]()
        desc = f"cmp {old_op.__name__}->{type(node.ops[0]).__name__}"

    try:
        return ast.unparse(tree), desc
    except Exception:                                # noqa: BLE001
        return None
