# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|
| T16 | `RandomTreesEmbedding(sparse_output=False)` | `claude/native-random-trees` | — | a new module, `__init__.py`, `catalog_test.py`, PLANS; reuses `_trees._f32_grid_threshold` | `session_017biqGYBE4kdVHVe2HSMVzg` | | in progress |
| T17 | `KBinsDiscretizer(dtype=np.float32)` | `claude/native-kbins-f32` | — | `discretize.py`, `catalog_test.py`, PLANS; builds on `_trees._f32_grid_threshold` | `session_01KQRoacPtnLuhre7a2WZR4K` | | in progress |

Neither ticket changes `_f32_grid_threshold`, which both build on and
`TreeBasedTransform` depends on. After them, PLANS "Next" has nothing
unticketed. Every other "not yet" row waits on the owner (the matvec and
`AdditiveChi2Sampler` records in `decisions/open/`) or needs a design
first (PLANS "Later"). The next tickets come from the partly native
classes' "Left Python" lines. Inline: reading 4 at the end of this wave,
covering waves 4 to 6.

## T16: `RandomTreesEmbedding(sparse_output=False)`

**Why.** `sklearn.ensemble.RandomTreesEmbedding` is a "not yet" row. PLANS
files it under "Later" with the transformers that read their fit samples,
but it reads only its fitted trees. Its `transform` is
`one_hot_encoder_.transform(self.apply(X))`: each tree's leaf, one-hot over
the leaf node ids. Checked against sklearn 1.9 on 2026-10-06:
- The internal encoder's `categories_[t]` are tree t's leaf node ids,
  sorted. Every leaf is there, since each leaf holds a training sample.
- The forest narrows X to float32 (`_validate_X_predict`, `dtype=DTYPE`)
  and keeps thresholds in float64. So a split is
  `float32(x) <= threshold`.
- The trees take missing values (the `allow_nan` tag is True). A NaN goes
  left or right by `tree_.missing_go_to_left`.
- A value past float32's range, or ±inf, makes the twin raise ("Input X
  contains infinity or a value too large for dtype('float32')").
- `sparse_output=False` returns a dense float64 array. The default,
  `sparse_output=True`, is sparse.

No FLOAT cast is needed. Rounding to float32 is monotone, so
`float32(x) <= t` is `x <= t'` for one double cutpoint `t'`.
`sql_transform._trees._f32_grid_threshold` already computes that cutpoint
for the tree predictors (`TreeBasedTransform`), with its proof in the
docstring.

**Do.**
- A new module, `@translates(RandomTreesEmbedding)`.
  - Read the twin in the installed sklearn: `RandomTreesEmbedding.transform`,
    `BaseForest.apply` and `_validate_X_predict`, and the tree's dense
    apply. State its operation order in the PR.
  - Reuse `_f32_grid_threshold` rather than copying it, and do not change
    it: `TreeBasedTransform` depends on it, and T17 builds on it beside
    this ticket. If it must change, tell the supervisor first.
- Each tree's leaf is one nested CASE over the rewritten cutpoints, with
  NaN routed by `missing_go_to_left`. Each lane is `leaf_t = k` as 1.0 or
  0.0, in `categories_` order.
  - confit computes a repeated pure subexpression once per row (#363,
    #377, #387), so each tree's CASE should be evaluated once. Measure it:
    serve time against the number of trees.
- `sparse_output=True` stays `NotNative`, named, as for `OneHotEncoder`
  (`decisions/open/sparse-outputs.md`).
- Fixtures:
  - small forests: `n_estimators` 1 to 30, `max_depth` 1 to 5,
    `min_samples_leaf` above 1, `max_leaf_nodes`;
  - NaN in the fit data where allowed;
  - `sparse_output=False`;
  - within `MAX_LANES` (1,000).
- A test at and beside each cutpoint, as `test_spline_at_the_knots` does
  for knots: the threshold, the float32 values on both sides of it, the
  doubles beside the cutpoint, ±0, the subnormals and NaN.
- Where the twin raises (past float32's range, ±inf), the entry may
  answer (goal.md, "Tolerated differences"). It must not raise where the
  twin answers. For example, the largest double that still rounds to a
  finite float32 must be answered.
- Timings: build and serve at the widest fixture, and at the default
  `n_estimators=100, max_depth=5` (up to 3,200 lanes) if it builds.
- PLANS: take it out of "Later". Add "Left Python" lines for what stays
  Python, and a "Needs from confit" entry if a width needs one.

**Acceptance.**
- Gate green.
- `NATIVE_SEEDS=200` over its fixtures, bit-exact.
- Coverage regenerated: 31 of 68 native.

**Beside this ticket:** T17 (`KBinsDiscretizer(dtype=np.float32)`,
`claude/native-kbins-f32`) runs at the same time. Both append a `FIXTURES`
block to `catalog_test.py` (keep both sides on a conflict) and edit their
own PLANS lines.

**Environment.** The container's uv 0.8.17 may offer only CPython
3.14.0rc2, on which the locked pydantic fails to import (reading 3,
finding 50). If so, install a current uv into `~/.local/bin`
(`curl -LsSf https://astral.sh/uv/install.sh | sh`), run
`uv python install 3.14.8`, then run the brief's install command.

**Branch:** `claude/native-random-trees` (create it from `origin/master`).
This is native T16 on the board, `loops/native/tickets.md`.

## T17: `KBinsDiscretizer(dtype=np.float32)`

**Why.** PLANS "Left Python" has `KBinsDiscretizer(dtype=np.float32)`: the
twin narrows x to float32 before it bins it, "which the entry does not
spell (a cast to FLOAT would have to round as numpy does, unproven)".
That needs no FLOAT cast:
- `transform` validates X with the estimator's `dtype`, then runs
  `np.searchsorted(bin_edges[jj][1:-1], Xt[:, jj], side="right")`. The
  edges stay float64 (checked, sklearn 1.9).
- So each comparison is `edge <= float32(x)`. Rounding to float32 is
  monotone, so that is one comparison of the double x against a moved
  cutpoint.
- `sql_transform._trees._f32_grid_threshold` computes such a cutpoint for
  `float32(x) <= t`, with its proof. The ticket needs the other side,
  `t <= float32(x)`.

**Do.**
- In `discretize.py`, serve `dtype=np.float32`: move each inner edge to its
  double cutpoint at build, and keep the lanes as today. Codes and one-hot
  0/1 in float32 are exact as doubles.
  - Build on `_f32_grid_threshold` without changing it:
    `TreeBasedTransform` depends on it, and T16 reuses it beside this
    ticket. The other side follows from it. `t <= float32(x)` fails
    exactly when `float32(x) <= g`, where g is the largest float32 below
    t, and that is `x <= _f32_grid_threshold(g)`. Put the helper in
    `discretize.py`, with a test against numpy's rounding.
- The edge cases, each in a test:
  - an edge at a float32 tie, where ties-to-even decides;
  - edges equal to each other (searchsorted's count);
  - ±0;
  - the subnormals;
  - x past float32's range. The twin raises ("Input X contains infinity
    or a value too large for dtype('float32')"), so the entry may answer.
    But the largest double that still rounds to a finite float32 must be
    answered as the twin does.
  - NaN (the twin raises).
- Fixtures: `dtype=np.float32` for each strategy and each dense encoding.
  Add a test at and beside every edge's cutpoint.
- PLANS: update the `KBinsDiscretizer` line in "Left Python".

**Acceptance.**
- Gate green.
- `NATIVE_SEEDS=200` over the `KBinsDiscretizer` fixtures, bit-exact.
- Coverage regenerated (the row stays native; its note may change).

**Beside this ticket:** T16 (`RandomTreesEmbedding`,
`claude/native-random-trees`) runs at the same time. Both append a
`FIXTURES` block to `catalog_test.py` (keep both sides on a conflict) and
edit their own PLANS lines.

**Environment.** The container's uv 0.8.17 may offer only CPython
3.14.0rc2, on which the locked pydantic fails to import (reading 3,
finding 50). If so, install a current uv into `~/.local/bin`
(`curl -LsSf https://astral.sh/uv/install.sh | sh`), run
`uv python install 3.14.8`, then run the brief's install command.

**Branch:** `claude/native-kbins-f32` (create it from `origin/master`).
This is native T17 on the board, `loops/native/tickets.md`.
