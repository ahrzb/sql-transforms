# Mixed native + fallback pipeline (serve-time) — Design

> **STATUS: design drafted — pending user review, then writing-plans.** Re-cuts
> the near-term M1 sklearn work: this fallback-execution slice lands *before* the
> compose-in / estimator-interface surface (that direction is backlogged — see
> "Deferred"). PM to re-sequence Phase A around this.

**Goal:** Run a pipeline that **mixes native `SQLTransform` steps with opaque
(non-native) fallback steps** — fitted sklearn objects we have no engine
expression for — serving `.transform` (batch) and `.infer` (row) by threading
data through and handing off to the object's `.transform()` at each opaque
barrier. This is what makes **partial coverage shippable**: mix native where we
have it, fallback where we don't, in one pipeline, and swap each fallback for a
native `SQLTransform` later without changing the pipeline surface.

## The problem this slice solves

Two step kinds:

| | Native step | Opaque / fallback step |
|---|---|---|
| **What** | a `SQLTransform` (engine-expressible) | a fitted sklearn-like object |
| **Run** | our engine — `.infer` (row) / DataFusion (batch) | we can only call `.transform(X)` |
| **Row path** | compiles into `InferFn`, fuses | Python call — the barrier |

A pure-native pipeline is already covered by composition (`{a}({b}(x))`); a
pure-opaque one is just "call sklearn." **The mix is the only new problem** — how
the runner stitches a fused engine segment to an opaque Python call and back. The
`infer` row path is where it has teeth (Python call + array materialization on the
serving hot path — the accepted "inefficient but works" cost); **batch is a
near-free byproduct** (calling `sklearn.transform` on a whole Arrow batch is
already performant).

## Settled: scope (this slice)

- **Serve-time `.transform` / `.infer` only, over already-fitted steps** — fitted
  sklearn objects + frozen `SQLTransform`s. The pipeline **does not fit**.
- **At least one opaque step between native steps** (`native → opaque → native`) —
  the mixed hand-off is the thing under test; a lone opaque step proves nothing
  structural.
- **Both engines**, `infer` is the target; batch falls out for free.
- **Constraint (decided): no unfit `SQLTransform` feeds into or nests inside an
  opaque step** — nothing that would need *fitting through* the opaque barrier.
  Fitting (the fit-cascade) stays entirely inside native/SQL segments; opaque
  steps only ever consume concrete, already-produced columns. This excludes the
  one hard case (below).

## Settled: execution model

**A thin new `Pipeline`: an ordered list of steps**, each either a native
`SQLTransform` or an opaque fitted object. Not an extension of `SQLTransform`
(that is one expression); **not** the sklearn `Pipeline` API (that is the
compose-in compliance surface — backlogged). Just enough to hold a sequence and
run it.

**Sequential whole-feature-set threading (sklearn-`Pipeline` semantics, not
`ColumnTransformer` routing — routing is deferred to the assembly slice).** The
running value is a set of **named columns** (a dict per row for `infer`; an Arrow
table for `transform`). Each step maps the current named columns → new named
columns; its output is the next step's full input.

- **Native step** (`SQLTransform`): current columns become `__THIS__`; output = its
  `SELECT` aliases. To carry a column forward it must be in the `SELECT`.
- **Opaque step** (fitted object): order the current columns → build a 2-D array →
  `obj.transform(X)` → take the output array → name the output columns via
  `obj.get_feature_names_out(input_features)` → those become the new named columns.
  - **Input column order:** align to `obj.feature_names_in_` when the object has it
    (fit on named data); otherwise the current named-column order. Deterministic
    either way.
  - **Output names:** `get_feature_names_out()` on the fitted object — used *as a
    plain function on their object*, needing **zero compliance work on our side**
    (consistent with backlogging our-transformer compliance).

**Steps are duck-typed, no wrapper class:** a step is a `SQLTransform` (native) or
anything else exposing `.transform` + `.get_feature_names_out` (opaque). A thin
`Fallback(obj, input_features=…)` wrapper is added *only if* an estimator lacking
`feature_names_in_` needs an explicit input order — not built preemptively.

**The runner:**
- `infer(row)` / `infer_batch(rows)`: thread the named-column dict(s); native steps
  via their existing `InferFn`; at each opaque barrier, materialize the ordered
  input columns to a small numpy array, call `obj.transform`, merge the named
  outputs back, continue.
- `transform(table)`: identical threading over Arrow batches; `obj.transform(whole
  batch)`.

**Fork decided — defer fusing contiguous native steps.** Adjacent native steps run
as *separate* `InferFn` calls in sequence; we do **not** fuse `native_A`+`native_B`
into one pass. Fusing contiguous natives *is* the shipped `{a}({b}(x))` composition
machinery — layer it on later, no surface change. This slice proves the opaque
barrier, not native fusion.

## Settled: parity oracle

**`transform` (batch) == `infer` (row)** for the same mixed pipeline — the
differential philosophy extended *across* an opaque barrier. For native steps this
is the existing DataFusion-vs-`InferFn` parity, now exercised inside a pipeline;
for the opaque step **both paths call the identical `obj.transform`**, so the test
proves the *stitching* is consistent between engines (opaque output fed into the
next native step the same way in both). Correctness of the pieces is already
guaranteed — DataFusion for natives, sklearn for opaque — so consistency + one
hand-checked value for a single row is sufficient.

## Deferred (backlog / later slices)

- **`.fit_transform` across an opaque barrier** — the fit-cascade would have to
  materialize training data *forward* through the sklearn call and stage the fit on
  the far side (the "Approach C" materialize-forward path). The scope constraint
  above excludes it from this slice. **This is the main deferred item.**
- **Fusing contiguous native steps** into one `InferFn` — reuse composition.
- **Our-transformer sklearn compliance** (`check_estimator`, compose into a *stock*
  `Pipeline`/`ColumnTransformer`) — already backlogged (the compose-in / hook-1
  direction).
- **`ColumnTransformer`-style column routing + assembly parity** — PM's later
  assembly slice; this slice is sequential whole-set threading only.
- **Multi-output specifics** beyond what `get_feature_names_out` already returns.

## Components

- **`sql_transform/_pipeline.py`** (new) — the `Pipeline` class (ordered steps +
  the `transform`/`infer`/`infer_batch` runner), step-kind dispatch (native
  `SQLTransform` vs opaque object), and the numpy↔named-column marshalling at the
  barrier. Optional `Fallback` wrapper only if an explicit input order is needed.
- **`sql_transform/__init__.py`** — export `Pipeline` (and `Fallback` if added).
- **`tests/test_diff_pipeline.py`** (new) — the mixed-pipeline differential tests.

## Testing

Differential parity (`transform` == `infer`) for a `native → opaque → native`
pipeline over already-fitted steps:
- **native_A**: a frozen scaler `SQLTransform("SELECT (x - AVG(x) OVER()) /
  STDDEV(x) OVER() AS a FROM __THIS__").fit(train)`.
- **opaque_S**: a fitted, genuinely native-less sklearn transformer —
  `PowerTransformer()` fit on `a` (stateful, out-of-scope-for-native, deterministic;
  `get_feature_names_out(['a'])` → `['a']`, name passes through 1:1).
- **native_B**: `SQLTransform("SELECT a * 2 AS out FROM __THIS__")` consuming the
  opaque output column `a`.
- Assert `Pipeline([native_A, opaque_S, native_B]).transform(test)` equals
  `.infer_batch(test_rows)` row-by-row, and equals a hand-computed reference for one
  row. Plus a two-column variant (native_A emits two columns, opaque transforms
  both) to prove multi-column ordering at the barrier. Plus the error case: an
  **unfit** `SQLTransform` placed as/into an opaque step raises a clear `ValueError`
  (the scope constraint).

## Next

User review → **writing-plans**. Task sequence: the `Pipeline` step model +
native/opaque dispatch → the opaque-barrier marshalling (numpy ↔ named columns,
input order + output names) → the `infer`/`transform` runners → the differential
tests + the constraint error. Loop PM in to re-sequence Phase A around this slice
before the plan hardens.
