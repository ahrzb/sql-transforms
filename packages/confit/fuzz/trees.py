"""Random tree ensembles, built straight into the engine's tree protocol.

The campaign used to fit sklearn estimators and wrap them in
sql_transform's TreeBasedTransform. Confit does not depend on sql_transform
(the arrow runs the other way), so the fuzzer owns its models: random trees
generated directly as the `(nodes, models, compare_grid)` tables the
protocol takes, plus `__call__`, a pure-Python walk of those same tables
that DuckDB calls as the UDF. The native kernel is then checked against that
walk through the ordinary differential — it is the protocol's reference
semantics, spelled independently of the Rust:

* a split sends `x <= threshold` left, a NaN (a NULL feature) the way
  `missing_left` says;
* leaf values accumulate left to right in tree order; `sum` seeds the
  accumulator with `base`, `mean` adds `base` to the average;
* `sigmoid` is `1 / (1 + exp(-score))`;
* on a float32 grid an integer feature narrows to float32 in ONE rounding
  before the compare, a double feature compares as it is; on a float64 grid
  both compare as doubles;
* an id with no model raises, as the kernel traps.

Parity with real sklearn models is sql-transform's gate, not this one.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pyarrow as pa

from . import gen as G

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

# Integer thresholds straddle the same boundary pool the generator draws
# integer features from, so the one-rounding narrowing is actually exercised.
_INT_POOL = [
    0,
    1,
    -1,
    7,
    -13,
    100,
    2**31 - 1,
    2**53 - 1,
    2**53,
    2**53 + 1,
    -(2**53) - 1,
]


class TreeModel:
    """A tree transform: the engine-facing protocol plus its reference walk."""

    returns = pa.float64()

    def __init__(self, name, n_features, int_features, trees, models, grid):
        self.name = name
        self.takes = pa.schema(
            [
                (f"f{i}", pa.int64() if i in int_features else pa.float64())
                for i in range(n_features)
            ]
        )
        self.instances = dict.fromkeys(range(len(models)))
        self._int = set(int_features)
        self._trees = trees  # model -> [tree -> [node dicts, tree-local ids]]
        self._models = models  # model -> {"base", "agg", "link"}
        self._grid = grid

    def tree_tables(self):
        rows = [
            {"model_id": m, "tree_id": t, "node_id": i, **node}
            for m, trees in enumerate(self._trees)
            for t, nodes in enumerate(trees)
            for i, node in enumerate(nodes)
        ]
        headers = [{"model_id": m, **h} for m, h in enumerate(self._models)]
        return (
            pa.Table.from_pylist(rows, schema=NODE_SCHEMA),
            pa.Table.from_pylist(headers, schema=MODEL_SCHEMA),
            self._grid,
        )

    def _feature(self, i, v) -> float:
        if v is None:
            return math.nan
        if i in self._int and self._grid == "float32":
            return float(np.float32(np.int64(v)))  # one rounding, as the kernel
        return float(v)

    def __call__(self, iid, *feats):
        if iid is None:
            return None
        if not 0 <= iid < len(self._models):
            raise ValueError(f"predict: no model with id {iid}")
        x = [self._feature(i, v) for i, v in enumerate(feats)]
        m = self._models[iid]
        acc = m["base"] if m["agg"] == "sum" else 0.0
        for nodes in self._trees[iid]:
            n = nodes[0]
            while n["feature"] >= 0:
                v = x[n["feature"]]
                go_left = n["missing_left"] if math.isnan(v) else v <= n["threshold"]
                n = nodes[n["left"] if go_left else n["right"]]
            acc += n["value"]
        if m["agg"] == "mean":
            acc = m["base"] + acc / len(self._trees[iid])
        if m["link"] == "sigmoid":
            acc = 1.0 / (1.0 + math.exp(-acc))
        return (acc,)


def _tree(rng: random.Random, spec: G.TreeSpec) -> list[dict]:
    """One tree, depth <= spec.depth, nodes in pre-order with tree-local ids."""
    nodes: list[dict] = []

    def grow(depth: int) -> int:
        me = len(nodes)
        nodes.append({})
        if depth >= spec.depth or rng.random() < 0.25:
            nodes[me] = {
                "feature": -1,
                "threshold": 0.0,
                "left": -1,
                "right": -1,
                "missing_left": True,
                "value": rng.choice(G.QUANT) + rng.choice([0.0, 0.1, 1e-3]),
            }
            return me
        f = rng.randrange(spec.n_features)
        if f in spec.int_features:
            thr = float(rng.choice(_INT_POOL)) + rng.choice([0.0, 0.5, -0.5])
        else:
            thr = rng.choice(G.QUANT) + rng.choice([0.0, 0.005])
        left = grow(depth + 1)
        right = grow(depth + 1)
        nodes[me] = {
            "feature": f,
            "threshold": thr,
            "left": left,
            "right": right,
            "missing_left": rng.random() < 0.5,
            "value": 0.0,
        }
        return me

    grow(0)
    return nodes


def make_tree(spec: G.TreeSpec, seed: int) -> TreeModel:
    """The case's model set: `spec.kind` picks the layout a packer would
    emit (one tree summed; a mean of three; a boosted sum of three over a
    base, half of them through a sigmoid), `seed` everything else."""
    rng = random.Random(seed)  # noqa: S311 — a reproducible campaign, not crypto
    trees, models = [], []
    for _ in range(spec.instances):
        k = 1 if spec.kind == "dtr" else 3
        trees.append([_tree(rng, spec) for _ in range(k)])
        if spec.kind == "rf":
            models.append({"base": 0.0, "agg": "mean", "link": "identity"})
        elif spec.kind == "gbr":
            link = "sigmoid" if rng.random() < 0.5 else "identity"
            models.append({"base": rng.choice(G.QUANT), "agg": "sum", "link": link})
        else:
            models.append({"base": 0.0, "agg": "sum", "link": "identity"})
    grid = rng.choice(["float32", "float64"])
    return TreeModel("trees", spec.n_features, spec.int_features, trees, models, grid)
