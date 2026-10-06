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

Lane k of tree t is 1.0 when the row lands on leaf k of tree t, else
0.0: the test cast to DOUBLE (`CAST(test AS DOUBLE)`, which builds faster
than a CASE on the two constants). The test has one of two spellings:

- One CASE per tree: a nested CASE over the tree's branches answers the
  leaf id, and lane k tests `leaf = k`. The CASE is one object that every
  lane of its tree reads, so confit binds it once (#412) and a row runs
  one path of each tree.
- The paths: the conjunction of the branch tests on the path from the
  root to leaf k, built root first, so two leaves under one node share
  that node's conjunction, a pure subexpression confit computes once per
  row (#363, #377, #387).

No float32 cast: rounding to float32 is monotone, so `float32(x) <= t` is
`x <= t'` for the one double cutpoint `t'` that `sql_transform._trees.
_f32_grid_threshold` computes (its docstring has the proof). NaN is
routed by an explicit test, since DuckDB orders NaN above every number
(`NaN <= t'` is false). Comparisons and constants only, so the entry is
bit-exact. Where the twin raises (±inf, or past float32's range) the
entry traps: its input guard (`_registry.rejects`).

One CASE per tree serves 1.5-2x as fast as the paths (100 trees of
depth 5: 318 us a row against 585), but its build grows faster with the
lanes. The entry spells a forest with one CASE per tree where that build
is estimated within MAX_BUILD_S, else with the paths where theirs is,
else refuses it. The estimates (`_case_seconds`, `_paths_seconds`) are
fitted on warm builds of 36 forests, each the slower of a struct and a
list return (release build, 8 features and, for five forests, 16 to 64;
fits of 500 to 4,000 rows; master 6aea15e, 2026-10-06). The struct return
is the slower, by 3.5 to 17 times: confit builds each field read of a
struct on its own, the input guard's tests again with each
(loops/native/PLANS.md, "Needs from confit"). The guard has two tests a
feature here, so each estimate has a term in the lanes times the
features. Each build over 0.3 s is within 0.70 to 1.33 times the CASE
estimate, past two deep forests that it refuses either way, and within
0.94 to 1.07 times the paths estimate. A struct return, one instance,
8 features and fits of 2,000 rows unless the row says, 10,000-row
batches; a refused forest shows the build of its faster spelling, with
the cap lifted:

    forest                           lanes   spelling   build   serve a row
    30 trees, depth 5                  693   CASE       1.2 s         70 us
    100 trees, depth 5               2,286   CASE       6.5 s        318 us
    100 trees, depth 5, 32 features  2,242   paths      6.9 s        803 us
    130 trees, depth 5               3,021   paths      6.4 s        804 us
    1 tree, unbounded                2,000   CASE       4.2 s        232 us
    160 trees, depth 5               3,745   paths      8.5 s    (refused)
    100 trees, depth 6               3,839   paths      8.7 s    (refused)
    1 tree, unbounded, 4,000 rows    4,000   paths     11.8 s    (refused)

against the twin's 6.6 ms a row at 30 trees. A list return builds in
0.65 to 0.78 s at 100 trees of depth 5, over 8 to 64 features, so the
estimates may fall when confit builds a struct read field by field as
it builds a list.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
from confit import sql as S
from sklearn.ensemble import RandomTreesEmbedding

from sql_transform._trees import _f32_grid_threshold
from sql_transform.native._helpers import f64, isnan, narrows_to_infinity
from sql_transform.native._registry import NotNative, rejects, translates

# The longest build the entry takes on, in seconds, by the estimates below
# (module docstring).
MAX_BUILD_S = 7.0


def _case_seconds(lanes: int, features: int) -> float:
    """The estimated build of one CASE per tree, from the lanes and the
    features (the input guard's tests, two a feature)."""
    return 5.32e-4 * lanes + 7.67e-7 * lanes**2 + 5.32e-5 * lanes * features


def _paths_seconds(lanes: int, steps: int, features: int) -> float:
    """The estimated build of the paths, from the lanes, the path steps
    and the features (the input guard's tests, two a feature)."""
    return (
        4.43e-4 * lanes
        + 2.9e-7 * lanes**2
        + 4.59e-5 * steps
        + 5.45e-5 * lanes * features
    )


def _leaf(tree: Any, x: list[S.Expr]) -> S.Expr | None:
    """The node id of the leaf a row lands on: one nested CASE over the
    branches, `_apply_dense`'s on the float32 grid (None for a tree that
    is one leaf). Every lane of the tree reads this one object."""
    t = tree.tree_
    left, right = t.children_left, t.children_right
    if left[0] == -1:
        return None
    feature, missing_left = t.feature, t.missing_go_to_left
    # Leaves keep their -2.0 sentinel threshold, never read.
    cut = _f32_grid_threshold(np.asarray(t.threshold, dtype=np.float64))
    node: dict[int, S.Expr] = {}
    for i in range(t.node_count - 1, -1, -1):  # a child's id is above its parent's
        if left[i] == -1:
            node[i] = S.lit(i)
            continue
        xf = x[int(feature[i])]
        go_left = xf <= f64(cut[i])
        if missing_left[i]:
            go_left = go_left | isnan(xf)
        node[i] = S.case(go_left, node[int(left[i])]).otherwise(node[int(right[i])])
    return node[0]


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


def _spelling(est: Any) -> str:
    """The spelling, "case" (one CASE per tree) or "paths": the first
    whose estimated build is within MAX_BUILD_S. Raises `NotNative` past
    both."""
    lanes = sum(len(c) for c in est.one_hot_encoder_.categories_)
    features = int(est.n_features_in_)
    case_s = _case_seconds(lanes, features)
    if case_s <= MAX_BUILD_S:
        return "case"
    paths_s = _paths_seconds(lanes, _steps(est), features)
    if paths_s <= MAX_BUILD_S:
        return "paths"
    raise NotNative(
        f"RandomTreesEmbedding: an estimated {min(case_s, paths_s):.0f} s build,"
        f" past {MAX_BUILD_S:.0f} s ({lanes:,} lanes)"
    )


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
    for xj in x:
        rejects(narrows_to_infinity(xj))
    case = _spelling(est) == "case"
    out: list[S.Expr] = []
    for tree, leaves in zip(est.estimators_, cats, strict=True):
        if case:
            leaf = _leaf(tree, x)
            tests = {
                int(k): None if leaf is None else leaf == S.lit(int(k)) for k in leaves
            }
        else:
            tests = _paths(tree, x)
        for k in leaves:
            test = tests[int(k)]
            # None: a tree that is one leaf, where every row lands.
            out.append(f64(1.0) if test is None else test.cast(pa.float64()))
    return out
