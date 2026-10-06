# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|
| T15 | Boolean features in the fixture generator | `claude/native-bool-features` | — | `catalog_test.py` (every class's draws), `encode.py`, `compose.py`, `function.py`, PLANS | `session_01Ra1e4qVNTBjoUhom83gGE7` | | in progress |

T15 runs alone: it changes every class's draws. After it, PLANS "Next" is
empty. Every "not yet" row either waits on the owner (the matvec and
`AdditiveChi2Sampler` records in `decisions/open/`) or needs a design
first (PLANS "Later"). The supervisor will propose the next tickets from
two places:
- the partly native classes' "Left Python" lines, where the float32
  configurations would all be served by one proof that confit's
  `CAST(x AS FLOAT)` rounds as numpy does;
- `RandomTreesEmbedding(sparse_output=False)`, whose trees compare
  float32 values and need no matvec.

## T15: boolean features in the fixture generator

**Why.** A step can declare a feature boolean (`pa.bool_()`, the step's
`"i1"`). `_as_feature` hands it to `transform` as a Python `bool`, and a
NULL as NaN. The entries read it as DOUBLE 0/1, with NULL as NaN
(`_registry._feature`). No fixture draws a boolean feature, so no entry is
tested on one:
- the encoders refuse every boolean feature ("which the catalog's
  fixtures do not make yet", `encode.py`);
- `compose.py` refuses a `Pipeline` over boolean features only, and an
  integer weight beside a boolean feature;
- `FunctionTransformer` refuses any `func` but the identity over a
  boolean feature;
- every other entry serves boolean features untested.

What the twin sees depends on the row (sklearn 1.9, numpy 2.5.1, checked
2026-10-06):
- Every feature a non-NULL boolean: the row is a `bool` array. Most
  entries' validation casts it to float64. `Binarizer` and the selectors
  hand it back as booleans, which the step reads as 0/1.
  `SimpleImputer(strategy="most_frequent")` raises ("does not support
  data with dtype bool"). `strategy="mean"` answers.
- Any number or NULL in the row: a float64 array.
- A string in the row: an object array holding Python `bool`s.

The fit data follows the same rules. An `OrdinalEncoder` fitted on a
boolean column has `categories_` of dtype `bool` (`[False, True]`) when
the matrix is all boolean. Beside numbers they are float64
(`[0.0, 1.0]`), and beside strings they are objects (`[False, True]`).

**Do.**
- `catalog_test.py`, `_draw` and `_fit_matrix`: draw boolean features.
  - Give `types` a third choice, `pa.bool_()`, at a share that keeps the
    gate's time (about 15%).
  - For a categorical estimator, let a boolean join the strings and the
    few-valued numbers.
  - The fit column is booleans. Holes, where the estimator takes missing
    values, are NaN, which makes the column float, as a NULL boolean
    reaches `transform` as NaN.
  - Make sure some steps are all boolean, so the `bool`-array row is
    drawn, not only rows mixed with numbers.
- `_rows`: a boolean feature draws `True` or `False`, and NULL in the
  "nulls" regime.
- Run every class at `NATIVE_SEEDS=200` with the new draws. Wherever an
  entry parts from its twin, either fix it, exactly and locally, or refuse
  the configuration with a named reason and a test, as for any other
  configuration. Look at these in particular:
  - The encoders' blanket refusal. Serve booleans where the translation
    is exact, for each of the three `categories_` dtypes above, or refuse
    narrower than today and say why.
  - All-boolean rows: `SimpleImputer` (the twin's validation rejects the
    dtype, so the entry may answer: goal.md, "Tolerated differences"), and
    `Binarizer`, the selectors and `FeatureAgglomeration` (bool in, bool
    out or a mean of bools).
  - The `Pipeline` and weight refusals in `compose.py`: keep, narrow or
    lift them, with evidence either way.
  - `FunctionTransformer` over a boolean feature: keep the refusal unless
    the sweep shows a function the twin answers in float64.
- Gate time: report the catalog test's wall time before and after (8
  seeds, 4 workers). If it grows past 10%, lower the boolean share.
- PLANS: remove "Boolean features in the fixture generator" from "Next".
  In "Left Python", update "An encoder over a boolean feature", plus each
  refusal you add, narrow or lift.
- Regenerate coverage if any row changes.

**Acceptance.**
- Gate green.
- `NATIVE_SEEDS=200` over every class with boolean features drawn: 0
  breaches. Report how many steps had at least one boolean feature, and
  how many were all boolean, per class.
- Every refusal is named in its `NotNative` message and tested.
- The PR gives a table: for each class, what it does over a boolean
  feature. Either served bit-exact, refused (why), or the twin raises and
  the entry answers.

**Run alone.** Every class's draws change, so no other ticket that touches
`catalog_test.py` or fixtures runs beside this one.

**Environment.** The container's uv 0.8.17 may offer only CPython
3.14.0rc2, on which the locked pydantic fails to import (reading 3,
finding 50). If so, install a current uv into `~/.local/bin`
(`curl -LsSf https://astral.sh/uv/install.sh | sh`), run
`uv python install 3.14.8`, then run the brief's install command.

**Branch:** `claude/native-bool-features` (create it from `origin/master`).
This is native T15 on the board, `loops/native/tickets.md`.
