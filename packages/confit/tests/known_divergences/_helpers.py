"""Shared probes for the divergence record.

`probe` runs a snippet in a FRESH interpreter - several of these findings only
reproduce on a clean process.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pyarrow as pa
from confit import DuckDBInferFn

PROBE_PRELUDE = """
import pyarrow as pa
from confit import DuckDBInferFn
"""


def probe(body: str) -> subprocess.CompletedProcess[str]:
    """Run a snippet in its own process and hand back the result.

    A build that dies of stack overflow returns 0xC00000FD on Windows and a
    signal elsewhere; either way it is not catchable in-process.
    """
    return subprocess.run(  # noqa: S603 — the source is this file's own literal
        [sys.executable, "-c", PROBE_PRELUDE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=180,
    )


# ---- the tree-UDF fixture -------------------------------------------
# Shared: the model-table checks build refusals out of it, and the WHERE
# short-circuit test uses it as a trap that must not be reached.

NODE_SCHEMA = pa.schema(
    [
        pa.field("model_id", pa.int64(), nullable=False),
        pa.field("tree_id", pa.int64(), nullable=False),
        pa.field("node_id", pa.int64(), nullable=False),
        pa.field("feature", pa.int32(), nullable=False),
        pa.field("threshold", pa.float64(), nullable=False),
        pa.field("left", pa.int32(), nullable=False),
        pa.field("right", pa.int32(), nullable=False),
        pa.field("missing_left", pa.bool_(), nullable=False),
        pa.field("value", pa.float64(), nullable=False),
    ]
)
MODEL_SCHEMA = pa.schema(
    [
        pa.field("model_id", pa.int64(), nullable=False),
        pa.field("base", pa.float64(), nullable=False),
        pa.field("agg", pa.string(), nullable=False),
        pa.field("link", pa.string(), nullable=False),
    ]
)


def _node(nid, feature, threshold, left, right, value=0.0):
    return {
        "model_id": 0,
        "tree_id": 0,
        "node_id": nid,
        "feature": feature,
        "threshold": threshold,
        "left": left,
        "right": right,
        "missing_left": True,
        "value": value,
    }


MODEL_ROW_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("x", pa.float64(), nullable=False),
    ]
)


class _TreeUDF:
    """A tree transform straight from Arrow — the engine protocol without a
    packer behind it."""

    def __init__(self, nodes, headers, n_features):
        self.name = "m"
        self.takes = pa.schema([(f"f{i}", pa.float64()) for i in range(n_features)])
        self.returns = pa.float64()
        self.instances = {0: None}
        self._t = (nodes, headers, "float32")

    def tree_tables(self):
        return self._t


def _tree_udf(nodes, agg="sum", link="identity", n_features=1):
    return _TreeUDF(
        pa.Table.from_pylist(nodes, schema=NODE_SCHEMA),
        pa.Table.from_pylist(
            [{"model_id": 0, "base": 0.0, "agg": agg, "link": link}],
            schema=MODEL_SCHEMA,
        ),
        n_features,
    )


def _model_fn(nodes, agg="sum", link="identity", features=("x",)):
    return DuckDBInferFn(
        "SELECT m(id, x) AS p FROM __THIS__",
        row_tables={"__THIS__": MODEL_ROW_SCHEMA},
        static_tables={},
        udfs=[_tree_udf(nodes, agg, link, len(features))],
    )
