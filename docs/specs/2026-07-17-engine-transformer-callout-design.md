# Engine transformer callout (Part 1) — Design

> **STATUS: design complete — ready for writing-plans.** Part 1 of the opaque-transform
> split (AmirHossein, 2026-07-17). Part 2 (the SQL/authoring surface — `{ref}` row→row
> composition, multi-output native refs, output-column naming, `SQLTransform.fit`
> wiring) is deferred and tracked separately. Supersedes the bundled
> `2026-07-17-opaque-transform-refs-design.md`.

**Goal:** Give both engines the capability to invoke an **opaque, already-fitted
Python transformer** (an sklearn transformer, or a whole fitted sklearn `Pipeline`)
during evaluation of a single query — struct in, struct out. `transform`
(DataFusion) registers it as a UDF; `infer` (Rust `InferFn`) calls it row-at-a-time.
Nothing about *how it gets authored* is in scope — this is the raw engine capability,
exercised with hand-written SQL.

## The capability, by example

```python
import pyarrow as pa
from sklearn.preprocessing import StandardScaler
from sql_transform._interpreter import InferFn

sc = StandardScaler().fit(train_df[["age", "income"]])   # fit on NAMED data
out_schema = pa.schema([("age", pa.float64()), ("income", pa.float64())])

sql = "SELECT __tfm_0__(named_struct('age', age, 'income', income)) AS s FROM __THIS__"
fn = InferFn(sql, row_tables={"__THIS__": ThisModel},
             static_tables={}, transformers={"__tfm_0__": (sc, out_schema)})
fn.infer({"__THIS__": [row]})   # -> [{"s": {"age": <z>, "income": <z>}}]
```

The output column `s` is a **struct**. Expanding it into flat named columns is a
Part 2 concern — Part 1 stops at producing the struct.

## Settled: the registry boundary

`InferFn.__new__` gains one parameter, mirroring the existing `row_tables` /
`static_tables`:

```python
transformers: dict[str, tuple[object, pa.Schema]]   # reserved-name -> (fitted object, output schema)
```

- **Name** — a reserved identifier (e.g. `__tfm_0__`) the SQL calls as a function.
  Part 2 will generate these; Part 1 writes them by hand.
- **Object** — anything exposing the sklearn calling convention (`.transform`,
  `feature_names_in_`). The engine knows *that* convention and nothing else — no
  `get_feature_names_out` (output shape is **declared**, below).
- **Output schema** — a `pyarrow.Schema`, declared by the caller (chosen over
  introspection: keeps the engine from guessing dtypes, works for any object). It is
  used **two ways**: as the DataFusion UDF's return type (a struct of these fields),
  and as the row engine's output-marshalling target. Parsed to our internal `Schema`
  by reusing the field-extraction already in `schema::from_arrow_table`.

## Settled: the sklearn calling convention (both engines identical)

- **Input alignment — by name to `feature_names_in_`.** The single argument evaluates
  to a `Value::Struct`; its fields are **reordered to the object's `feature_names_in_`
  order** before building the array. `feature_names_in_` is **required**: an object fit
  on bare numpy (or a `Pipeline` whose first step never saw names) lacks it → a clear
  **build-time** error telling the caller to fit on named data. Field-name *set* must
  equal `feature_names_in_` (checked at build) — extra/missing fields error.
- **Call** — `obj.transform(X)` where `X` is 2-D array-like of one or more rows.
- **Output** — take the transformed row(s), marshal each position back **through the
  declared output schema** into a `Value::Struct`. `transform` returns numpy, so output
  elements are numpy scalars — converted via the declared field type, not extracted
  blindly.

Both engines align and marshal the **same way**, so the differential oracle is exact
by construction. (Batch `transform` on N rows and row-at-a-time `infer` give identical
per-row output — a fitted transformer's `.transform` is per-row independent.)

## Settled: `transform` side (DataFusion UDF) — the oracle

Register the object as a DataFusion scalar UDF named after its registry key, struct
input type / declared struct return type, **vectorized** (receives the whole batch's
`StructArray`). The wrapper: extract each field array, reorder to `feature_names_in_`,
`np.column_stack` → `X`, `obj.transform(X)` → `Y`, build the output `StructArray` from
`Y`'s columns named by the output schema. **Verified empirically (2026-07-17):**
DataFusion 54 accepts a struct-in/struct-out Python UDF via `register_udf` and runs it
vectorized — one `.transform` call per batch.

`register_udf` requires the UDF's **input** struct type, and it isn't guessable (a
categorical encoder takes strings, not floats), so the helper takes an input schema
too: `_transformer_udf(obj, in_schema, out_schema)`. `in_schema` is a `pa.Schema`
whose fields match the authored `named_struct` (names + types of the fed columns) — the
caller knows them because it wrote the query. This is **oracle-side only**: the row
engine never needs a declared input type (it marshals whatever `Value::Struct` it
receives and derives the arg's field types from the existing row schema at
`infer_type`), so `in_schema` stays out of the `transformers` registry. Part 1 uses the
helper to register the transformer into the DataFusion context for the parity test;
Part 2 threads it into `SQLTransform.transform`/`run_batch`.

## Settled: `infer` side (Rust) — the callout node

- **New `Expr::Transform` node** (`src/expr.rs`): carries the object (`Py<PyAny>`), the
  input feature order (`Vec<String>`, read from `feature_names_in_` once at build), the
  declared output `Schema`, and the boxed argument expression.
- **Build-time resolution** (in `InferFn::new`, which holds `py`): a pass over the
  plan's projection expressions rewrites every `Expr::Function { name, args }` whose
  `name` is in `transformers` into `Expr::Transform`. Here (with `py`) it reads
  `feature_names_in_` (erroring clearly if absent) and requires exactly one argument.
  Runs **after** `build_plan`, **before** `validate_columns`.
- **Type inference** (`src/types.rs`): `infer_type(Expr::Transform)` infers the arg's
  type, requires a `Base::Struct` whose field-name set equals the stored input features
  (else error), and returns `Base::Struct(output_schema)`. So `synthesize_output_model`
  and output-model validation work unchanged.
- **Eval** (`src/expr.rs`): `expr::eval` is pure Rust with no `py` token. Rather than
  thread `py` through the whole recursive eval/execute path, the `Expr::Transform` arm
  uses `Python::with_gil(|py| …)` — cheap and re-entrant, since `infer()` already holds
  the GIL. Inside: eval the arg → `Value::Struct`; reorder fields to the stored order;
  build `[[…]]` as a Python list-of-lists (sklearn's `check_array` coerces it — **no
  numpy import in Rust**); `obj.transform(…)`; take row 0; marshal each position through
  the declared schema into a `Value::Struct`.

## Reused, already shipped (no new work)

- `Expr::Struct` + `named_struct(…)` → `Expr::Struct` (`expr_build.rs:124-149`).
- `Value::Struct`, `Value::to_pyobject`, `Value::from_pyobject_typed` (rich-type layer).
- `schema::from_arrow_table` field extraction (reused to parse the declared schema).
- The differential harness `_val_equal` already recurses into dicts/structs, so a
  struct output column compares directly.

## Testing — standard differential parity (`transform == infer`)

For a struct-returning query `SELECT __tfm_0__(named_struct('age', age, 'income',
income)) AS s FROM __THIS__`, run the **same** SQL through DataFusion (UDF registered
via `_transformer_udf`) and through `InferFn` (with `transformers=…`), and assert the
struct column `s` is equal row-by-row. No `unnest`, no derived table, no field-name
aliasing — the struct is compared directly (that expansion is Part 2).

Cases:
- A fitted `StandardScaler` over two float columns (the motivating case).
- A whole fitted `Pipeline` (e.g. `[("sc", StandardScaler()), ("pt", PowerTransformer())]`)
  as a single opaque object — proves "whole pipeline as one node".
- A categorical `OrdinalEncoder` — exercises non-float in **and** out, and string
  marshalling both ways.
- One hand-computed value for a single row (the marshalling is not self-evident).
- Errors (build-time): object with no `feature_names_in_`; the transformer argument
  does not infer to a struct; struct field-name set ≠ `feature_names_in_`.

A tiny `test_diff_transformer_callout.py` following the existing differential-test
pattern; `numpy` + `scikit-learn` join the dev/test dependency set (sklearn pulls numpy).

## Scope

- **In:** the raw engine capability on both engines — one registered transformer per
  node, struct in / struct out, aligned by `feature_names_in_`, declared output schema,
  differential parity. Transformer calls resolve in **projection** expressions.
- **Out (Part 2):** the `{ref}` authoring surface; `SQLTransform` / t-string
  integration; the derived-table lowering and flat output-column naming (`unnest`);
  multi-output *native* refs; fitting anything (opaque objects arrive pre-fitted; the
  no-fit-through-a-transformer rule lives with the fit-cascade surface).

## Components

- `src/expr.rs` — `Expr::Transform` variant + `with_gil` eval arm.
- `src/types.rs` — `infer_type` arm (arg struct-field check → `Struct(output_schema)`).
- `src/lib.rs` — `transformers` param on `InferFn::new`; the build-time resolution pass;
  parse each declared `pa.Schema` → `Schema`.
- `sql_transform/_transformer_udf.py` (new) — `_transformer_udf(obj, in_schema, out_schema)`
  building the vectorized DataFusion UDF (the oracle side).
- `tests/test_diff_transformer_callout.py` (new) — the differential tests.

## Next

**writing-plans.** Rough task order: the DataFusion UDF helper + its standalone test
(oracle first) → `Expr::Transform` + `infer_type` + declared-schema parsing (build path,
type-checks green) → the `with_gil` eval callout (`infer` produces the struct) → the
differential parity tests + build-time error cases.
