"""Model-table structure refusals.

See README.md for what
belongs here (kept behaviour + its ground) versus in
../test_open_divergences.py (behaviour we intend to change).
"""

from __future__ import annotations

import pytest
from _helpers import (
    _model_fn,
    _node,
)

# ------------------------------------------ model-table structure checks --
#
# A node with two parents is NOT a cycle. Children are forced to strictly
# follow their parent, which rules out cycles by construction and is what makes
# traversal terminate without a depth counter. Under that rule a shared child
# is a decision DAG — the walk from the root still takes exactly one path, and
# it scores exactly what the table names. It is not a wrong-answer bug.
#
# It is refused anyway, because the validator checks full tree-ness. Given
# children that strictly follow their parent, a table is a tree exactly when
# every non-root node has ONE parent: one parent each makes the parent
# function total, and the ordering makes walking parents strictly decrease, so
# every node has a unique path back to node 0. The validator keeps a
# saturating parent count (`reachable`) and rejects both the ZERO and the
# two-or-more case from the same array, with no extra pass.


def test_a_shared_child_is_refused_as_not_a_tree():
    """Scores fine — one path, terminating — and is refused anyway, because
    the table is not a tree and nothing we target emits one."""
    nodes = [
        _node(0, 0, 0.5, 1, 2),
        _node(1, 0, 0.25, 3, 4),
        _node(2, 0, 0.75, 3, 5),  # <- node 3 has two parents
        _node(3, -1, 0.0, -1, -1, value=10.0),
        _node(4, -1, 0.0, -1, -1, value=20.0),
        _node(5, -1, 0.0, -1, -1, value=30.0),
    ]
    with pytest.raises(ValueError, match="child 3 already has a parent"):
        _model_fn(nodes)
    # ... and the same shape with node 3 duplicated into a genuine tree builds,
    # so what is refused is the SHARING, not the shape around it.
    nodes[2] = _node(2, 0, 0.75, 6, 5)
    nodes.append(_node(6, -1, 0.0, -1, -1, value=10.0))
    fn = _model_fn(nodes)
    rows = [{"id": 0, "x": v} for v in (0.1, 0.4, 0.6, 0.9)]
    assert [r["p"] for r in fn.infer_rows(rows)] == [10.0, 20.0, 10.0, 30.0]


# Every OTHER refusal the spec claims, checked by construction rather than
# assumed. All nine hold.
@pytest.mark.parametrize(
    ("what", "nodes", "kw", "match"),
    [
        (
            "child index out of range",
            [_node(0, 0, 0.5, 1, 9), _node(1, -1, 0.0, -1, -1, value=1.0)],
            {},
            "child 9 out of range",
        ),
        (
            "child precedes its parent (how a cycle would have to be spelled)",
            [
                _node(0, 0, 0.5, 1, 2),
                _node(1, 0, 0.5, 0, 2),
                _node(2, -1, 0.0, -1, -1, value=1.0),
            ],
            {},
            "must follow its parent",
        ),
        (
            "node unreachable from its tree's root",
            [
                _node(0, 0, 0.5, 1, 2),
                _node(1, -1, 0.0, -1, -1, value=1.0),
                _node(2, -1, 0.0, -1, -1, value=2.0),
                _node(3, -1, 0.0, -1, -1, value=3.0),
            ],
            {},
            "unreachable from its tree's root",
        ),
        (
            "leaf with children",
            [_node(0, -1, 0.0, 1, 1), _node(1, -1, 0.0, -1, -1)],
            {},
            "leaf .* with children",
        ),
        (
            "split node missing a child",
            [_node(0, 0, 0.5, 1, -1), _node(1, -1, 0.0, -1, -1, value=1.0)],
            {},
            "split node missing a child",
        ),
        (
            "feature beyond the declared width",
            [
                _node(0, 5, 0.5, 1, 2),
                _node(1, -1, 0.0, -1, -1),
                _node(2, -1, 0.0, -1, -1),
            ],
            {},
            "beyond the declared width",
        ),
        (
            "node id out of dense order",
            [
                _node(0, 0, 0.5, 1, 2),
                _node(7, -1, 0.0, -1, -1),
                _node(2, -1, 0.0, -1, -1),
            ],
            {},
            "out of dense order",
        ),
        (
            "unknown agg",
            [_node(0, -1, 0.0, -1, -1, value=1.0)],
            {"agg": "median"},
            "unknown agg",
        ),
        (
            "unknown link",
            [_node(0, -1, 0.0, -1, -1, value=1.0)],
            {"link": "probit"},
            "unknown link",
        ),
    ],
)
def test_every_claimed_model_table_refusal_holds(what, nodes, kw, match):
    with pytest.raises(ValueError, match=match):
        _model_fn(nodes, **kw)
