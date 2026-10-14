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
   `categories_[t]`, tree t's leaf ids sorted, an output field each, 1.0 at
   the leaf's output field and 0.0 elsewhere, as float64. Every leaf holds a
   training sample, so every leaf is a category.

Output field k of tree t is 1.0 when the row lands on leaf k of tree t, else
0.0: the test cast to DOUBLE (`CAST(test AS DOUBLE)`, which builds faster
than a CASE on the two constants). The test is one CASE per tree: a
nested CASE over the tree's branches answers the leaf id, and output field k
tests `leaf = k`. The CASE is one object that every output field of its tree
reads, so confit binds it once (#412) and a row runs one path of each
tree.

No float32 cast: rounding to float32 is monotone, so `float32(x) <= t` is
`x <= t'` for the one double cutpoint `t'` that `sql_transform._trees.
_f32_grid_threshold` computes (its docstring has the proof). NaN is
routed by an explicit test, since DuckDB orders NaN above every number
(`NaN <= t'` is false). Comparisons and constants only, so the entry is
bit-exact. Where the twin raises (±inf, or past float32's range) the
entry traps: its input guard (`_registry.rejects`).

One CASE per tree serves a row about twice as fast as the conjunctions of
the branch tests on each leaf's path (100 trees of depth 5: 222 us against
459), and since confit #430 it builds about as fast, so it is the entry's
one spelling. The entry takes a forest whose build is estimated within
MAX_BUILD_S (`_build_estimate`), about 5,990 output fields at 8 features,
and refuses it past that. The estimate is fitted on warm builds of 43
forests, each the slower of a struct and a list return (release build, 8
features and, for five forests, 16 to 64; fits of 500 to 5,000 rows; master
b4b8d03 with confit #430, 2026-10-14). The struct return is the slower, by
1.2 to 3.7 times: confit builds each field read of a struct on its own
(loops/native/PLANS.md, "Needs from confit"). Since #430 it no longer builds
the input guard's tests again at each, but the estimate keeps a term in the
output fields times the features: 100 trees of depth 5 build in 1.6 s over 8
features and 2.3 s over 64. Each build over 0.3 s is within 0.76 to 1.13
times the estimate, except forests of 4 to 10 unbounded trees, which build
1.2 to 1.5 times it. The table below was measured an hour after the fit, and
its builds took 0.9 to 1.2 times the estimate (1.65 times at 5 unbounded
trees). A struct return, one instance, 8 features and fits of 2,000 rows
unless the row says, 10,000-row batches; a refused forest shows its build
with the cap lifted:

    forest                           output fields   build   serve a row
    30 trees, depth 5                          693   0.4 s         51 us
    100 trees, depth 5                       2,286   1.7 s        222 us
    100 trees, depth 5, 32 features          2,242   2.0 s        205 us
    160 trees, depth 5                       3,745   3.9 s        397 us
    100 trees, depth 6                       3,839   4.2 s        357 us
    250 trees, depth 5                       5,758   7.2 s        741 us
    1 tree, unbounded                        2,000   1.4 s        136 us
    1 tree, unbounded, 4,000 rows            4,000   3.3 s        333 us
    5 trees, unbounded, 1,000 rows           5,000   8.6 s        326 us
    3 trees, unbounded                       6,000   8.4 s    (refused)

against the twin's 8.9 ms a row at 30 trees. A list return builds in
0.7 to 0.8 s at 100 trees of depth 5, over 8 to 64 features, so the
estimates may fall again when confit builds a struct read field by
field as it builds a list.
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

# The longest build the entry takes on, in seconds, by the estimate below
# (module docstring).
MAX_BUILD_S = 7.0


def _build_estimate(est: Any) -> float:
    """The seconds confit takes to build the entry (warm, release build),
    from its output fields and features (module docstring)."""
    fields = sum(len(c) for c in est.one_hot_encoder_.categories_)
    features = int(est.n_features_in_)
    return 3.69e-4 * fields + 1.27e-7 * fields**2 + 4.86e-6 * fields * features


def _leaf(tree: Any, x: list[S.Expr]) -> S.Expr | None:
    """The node id of the leaf a row lands on: one nested CASE over the
    branches, `_apply_dense`'s on the float32 grid (None for a tree that
    is one leaf). Every output field of the tree reads this one object."""
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
    estimate = _build_estimate(est)
    if estimate > MAX_BUILD_S:
        fields = sum(len(c) for c in cats)
        raise NotNative(
            f"RandomTreesEmbedding: an estimated {estimate:.0f} s build,"
            f" past {MAX_BUILD_S:.0f} s ({fields:,} output fields)"
        )
    for xj in x:
        rejects(narrows_to_infinity(xj))
    out: list[S.Expr] = []
    for tree, leaves in zip(est.estimators_, cats, strict=True):
        leaf = _leaf(tree, x)
        for k in leaves:
            # None: a tree that is one leaf, where every row lands.
            out.append(
                f64(1.0) if leaf is None else (leaf == S.lit(int(k))).cast(pa.float64())
            )
    return out
