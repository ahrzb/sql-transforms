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
   training sample, so every leaf is a category. Under `sparse_output=True`
   (the default) the block is sparse, and the step makes it dense
   (`_udf._dense`): the same values.

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
the branch tests on each leaf's path (100 trees of depth 5 at T30: 222 us
against 459), and since confit #430 it builds about as fast, so it is the
entry's one spelling. The entry takes a forest whose build is estimated
within MAX_BUILD_S (`_build_estimate`), about 10,000 output fields at 8
features, 8,600 at 32, 7,200 at 64 and 5,200 at 128, and refuses it past
that. The estimate is fitted on 35 warm trial builds (`_registry._builds`,
without `to_native`'s translation and probe) over 0.3 s, each the slower
of a struct and a list return (release build, 8 features and, for nine
forests, 32 to 128; fits of 1,000 to 8,000 rows; T46 on master 12aca98,
2026-10-16). The struct return is the slower, by 1.7 to 5.8 times, more as
the output fields grow: confit builds each field read of a struct on its own
(loops/native/PLANS.md, "Needs from confit"). The estimate keeps a term in
the output fields times the features, larger since T44's reads
(`_registry._apart`): 100 trees of depth 5 build in 0.81 s over 8 features
and 2.7 s over 128. Before T44's reads, in one process, they built in 2.2 s
over 128 features, and served a row in 1.08 to 1.24 times the time over 8 to
128. Each of the 35 builds over 0.3 s (0.61 to 25.8 s) is within 0.88 to
1.23 times the estimate, the most at 2 unbounded trees over 5,000 rows. The
cap took on none slower than 6.3 s (400 trees of depth 5) and refused none
faster than 7.5 s (800 trees of depth 4). Fitted before T44's reads (master
318575b with confit #459 and #461), the estimate reached 7 s at about 10,300
output fields at 8 features and 6,800 at 128. The table below was measured
on the code of the fit. A struct return, one instance, 8 features and fits
of 2,000 rows unless the row says, 10,000-row batches; a refused forest
shows its build with the cap lifted:

    forest                           output fields   build   serve a row
    30 trees, depth 5                          693   0.2 s         24 us
    100 trees, depth 5                       2,286   0.8 s        108 us
    100 trees, depth 5, 32 features          2,242   1.1 s         95 us
    160 trees, depth 5                       3,745   1.6 s        193 us
    100 trees, depth 6                       3,839   1.5 s        194 us
    250 trees, depth 5                       5,758   2.9 s        379 us
    400 trees, depth 5                       9,210   6.3 s        913 us
    300 trees, depth 6                      11,402   8.1 s    (refused)
    1 tree, unbounded                        2,000   0.7 s         54 us
    1 tree, unbounded, 4,000 rows            4,000   1.6 s        159 us
    5 trees, unbounded, 1,000 rows           5,000   2.5 s        221 us
    3 trees, unbounded                       6,000   3.6 s        309 us
    2 trees, unbounded, 5,000 rows          10,000   8.7 s    (refused)

On master 318575b (T40), the twin served 30 trees in 7.1 ms a row. A list
return builds in 0.44 to 1.2 s at 100 trees of depth 5, over 8 to 128
features, and in 4.1 s at 800 trees (23.7 s as a struct, timed once), so
the estimate may fall again when confit builds a struct read field by
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
    """The seconds of confit's trial build of the entry alone
    (`_registry._builds`, warm, release build), without the translation
    and the probe that `to_native` also runs, from its output fields and
    features (module docstring)."""
    fields = sum(len(c) for c in est.one_hot_encoder_.categories_)
    features = int(est.n_features_in_)
    return 1.89e-4 * fields + 4.56e-8 * fields**2 + 7.19e-6 * fields * features


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
