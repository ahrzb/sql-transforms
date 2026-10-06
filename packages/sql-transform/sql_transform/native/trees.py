"""Tree embeddings: each tree's leaf, one-hot over its leaf node ids
(sklearn 1.9, `sklearn/ensemble/_forest.py` and `sklearn/tree/_tree.pyx`).

`RandomTreesEmbedding.transform(X)` is
`one_hot_encoder_.transform(self.apply(X))`, in this order:

1. `BaseForest._validate_X_predict` narrows X to float32 (`dtype=DTYPE`),
   allowing NaN (the trees' `allow_nan` tag is True) and raising on a value
   that is infinite as a float32: ±inf, or a double past float32's range.
2. `BaseForest.apply` runs each tree's `Tree._apply_dense`: from node 0,
   while the node has children, a NaN feature goes left when the node's
   `missing_go_to_left` is set and right otherwise, else
   `float32(x) <= threshold` (the threshold a float64) goes left; the leaf
   is the node id it stops at.
3. The fitted `OneHotEncoder` (`sparse_output=self.sparse_output`, its
   other parameters the defaults) maps tree t's leaf onto its block:
   `categories_[t]`, tree t's leaf ids sorted, a lane each, 1.0 at the
   leaf's lane and 0.0 elsewhere, as float64. Every leaf holds a training
   sample, so every leaf is a category.

Lane k of tree t is 1.0 when the row takes the path from the root to leaf
k: the conjunction of the branch tests on that path, built root first, so
two leaves under one node share that node's conjunction, a pure
subexpression confit computes once per row (#363, #377, #387). No float32
cast: rounding to float32 is monotone, so `float32(x) <= t` is `x <= t'`
for the one double cutpoint `t'` that `sql_transform._trees.
_f32_grid_threshold` computes (its docstring has the proof). NaN is
routed by an explicit test, since DuckDB orders NaN above every number
(`NaN <= t'` is false). Comparisons and constants only, so the entry is
bit-exact. Where the twin raises (±inf, or past float32's range) the
entry answers a leaf (goal.md, "Tolerated differences").

One nested CASE per tree answering its leaf id, each lane then `leaf =
k`, serves faster (0.11 against 0.22 us a lane per row at 30 trees of
depth 5), but a function body is substituted as text, so the CASE is
spelled again in every lane of its tree: quadratic in a tree's leaves,
and the default `n_estimators=100, max_depth=5` expands past confit's
4,000,000-token cap (PLANS, "Needs from confit", "A value bound once").
The paths are linear in the leaves times their depth, and the build
follows them at 0.14 to 0.28 ms a path step (release build, 8 features,
fits of 2,000 rows, 2026-10-06):

    forest                 lanes   path steps   build    serve a row
    30 trees, depth 5        693        3,307    0.6 s       115 us
    100 trees, depth 5     2,286       10,894    2.7 s       460 us
    100 trees, depth 6     3,839       21,753    4.9 s       866 us
    250 trees, depth 5     5,758       27,398    7.8 s     1,501 us  (refused)
    1 tree, unbounded      2,000       33,688    4.7 s       417 us  (refused)

against the twin's 5.5 ms a row at 30 trees. `MAX_PATH_STEPS` stops an
estimator at about 7 s of build at the slowest rate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
from confit import sql as S
from sklearn.ensemble import RandomTreesEmbedding

from sql_transform._trees import _f32_grid_threshold
from sql_transform.native._helpers import f64, isnan
from sql_transform.native._registry import NotNative, translates

# The most path steps (a leaf's depth, summed over an estimator's leaves)
# an estimator translates to: about 7 s of build at the slowest measured
# rate (module docstring).
MAX_PATH_STEPS = 25_000
STEP_SECONDS = 0.28e-3


def _paths(tree: Any, x: list[S.Expr]) -> dict[int, S.Expr | None]:
    """Each leaf's node id, and the test a row passes to land there: the
    branches from the root, `_apply_dense`'s on the float32 grid, ANDed
    root first (None for a tree that is one leaf)."""
    t = tree.tree_
    left, right = t.children_left, t.children_right
    feature, missing_left = t.feature, t.missing_go_to_left
    # Leaves keep their -2.0 sentinel threshold, never read.
    cut = _f32_grid_threshold(np.asarray(t.threshold, dtype=np.float64))
    out: dict[int, S.Expr | None] = {}
    stack: list[tuple[int, S.Expr | None]] = [(0, None)]
    while stack:
        i, path = stack.pop()
        if left[i] == -1:
            out[i] = path
            continue
        xf = x[int(feature[i])]
        go_left = xf <= f64(cut[i])
        if missing_left[i]:
            go_left = go_left | isnan(xf)
        for child, test in ((left[i], go_left), (right[i], ~go_left)):
            stack.append((int(child), test if path is None else path & test))
    return out


def _steps(est: Any) -> int:
    """The path steps an estimator translates to: its leaves' depths."""
    total = 0
    for tree in est.estimators_:
        t = tree.tree_
        depth = np.zeros(t.node_count, dtype=np.int64)
        for i in range(t.node_count):  # a child's id is above its parent's
            if t.children_left[i] != -1:
                depth[t.children_left[i]] = depth[t.children_right[i]] = depth[i] + 1
        total += int(depth[t.children_left == -1].sum())
    return total


@translates(RandomTreesEmbedding)
def _embed(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    if est.sparse_output:
        raise NotNative(
            "RandomTreesEmbedding(sparse_output=True): the output is sparse"
        )
    cats = est.one_hot_encoder_.categories_
    if len(cats) != len(est.estimators_):
        raise NotNative(
            f"RandomTreesEmbedding: {len(cats)} encoded blocks for"
            f" {len(est.estimators_)} trees"
        )
    steps = _steps(est)
    if steps > MAX_PATH_STEPS:
        raise NotNative(
            f"RandomTreesEmbedding: {steps:,} path steps, past"
            f" {MAX_PATH_STEPS:,} (an estimated {steps * STEP_SECONDS:.0f} s build)"
        )
    out: list[S.Expr] = []
    for tree, leaves in zip(est.estimators_, cats, strict=True):
        paths = _paths(tree, x)
        for k in leaves:
            path = paths[int(k)]
            if path is None:  # a tree that is one leaf: every row lands there
                out.append(f64(1.0))
            else:
                out.append(S.case(path, f64(1.0)).otherwise(f64(0.0)))
    return out
