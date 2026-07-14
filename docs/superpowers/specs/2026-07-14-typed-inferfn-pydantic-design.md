# Typed InferFn (Pydantic v2) — Design Spec

## Goal

Replace `InferFn`'s dict-based row API with a strongly-typed one built on Pydantic v2: row tables are declared as Pydantic model classes at construction time, and the output is either a model the caller supplies or one `InferFn` synthesizes from the query itself. Unknown columns, provably-wrong output types, and unsupported patterns are rejected at construction time (`ValueError`), not deep inside a row at inference time.

This is a follow-up to the (already merged) [Phase 2 interpreter spec](2026-07-14-phase2-interpreter.md), which built the dict-based `InferFn` this spec replaces. It is a breaking change to that just-shipped API — the whole existing test suite is rewritten, not extended.

## Motivation

The dict-based `InferFn` has no schema for row tables until a row actually arrives at `.infer()` — that's *why* it parses SQL itself instead of using DataFusion's schema-requiring logical planner (see the Phase 2 spec's Architecture section). But it means a typo'd column name, or a value of the wrong type, only surfaces deep inside a specific row, at inference time, in production. Pydantic models close that gap: a model class is a real schema, known before any row exists, so `InferFn` can validate the query against it once, at construction.

## Public API

```python
from pydantic import BaseModel
from sql_transform import InferFn

class DataRow(BaseModel):
    id: int
    x: int
    name: str | None

# Output model omitted — InferFn synthesizes one from the query
fn = InferFn(
    "SELECT data.x, UPPER(data.name) AS name_upper FROM data",
    row_tables={"data": DataRow},
    static_tables={},
)
fn.output_model  # dynamically synthesized: fields x: int, name_upper: str | None

results = fn.infer({"data": [DataRow(id=1, x=5, name="alice")]})
# results: list[fn.output_model instances]

# Output model explicitly supplied — validated against the query at build time
class Result(BaseModel):
    x: int
    name_upper: str | None

fn2 = InferFn(sql, row_tables={"data": DataRow}, static_tables={}, output_model=Result)
```

- `InferFn(sql, row_tables, static_tables, output_model=None)`
  - `row_tables: dict[str, type[BaseModel]]` — table name → Pydantic v2 model class. Replaces the old `list[str]`.
  - `static_tables: dict[str, pyarrow.Table]` — **unchanged** from the dict-based API. Arrow already carries a schema (`pyarrow.Table.schema`), so there's nothing for Pydantic to add here.
  - `output_model: type[BaseModel] | None = None` — if provided, validated against the query's inferred output shape at build time (see Output Model Validation below). If omitted, `InferFn` synthesizes one via `pydantic.create_model()` from the query's own projection.
- `fn.output_model` — the model class in effect (the one you passed, or the synthesized one). Always available after construction, whichever path was taken.
- `fn.infer(tables: dict[str, list[BaseModel instances]]) -> list[BaseModel instances]` — same shape as before, just typed.

## Architecture

Three new/changed pieces sit on top of the existing dict-based interpreter (`src/expr.rs`, `src/plan.rs`, `src/expr_build.rs`, `src/lookup.rs`, `src/lib.rs`), whose internals — `Value`, the nested `Row`, `Expr`, `eval()`, `execute()` — are **unchanged**. Typing only touches the boundary (how a row gets in, how a row gets out) and adds one build-time validation pass.

```
src/schema.rs   (new)  — extract {column: (base_type, nullable)} from a Pydantic
                          model class's model_fields, or from a pyarrow.Table's
                          .schema. Both feed the same Schema map shape.

src/types.rs    (new)  — FieldType { Int | Float | Str | Bool | Other }, the
                          static type-inference pass (mirrors eval()'s structure,
                          computes types instead of values), and the
                          compatible(inferred, declared) relation used for
                          output-model validation. Kept separate from schema.rs
                          (which only reads schemas) so it has room to grow into
                          a real type system later without another reorg.

src/plan.rs    (extend) — build_plan/optimize gain one tree walk that: validates
                          every Expr::Column against the right table's Schema
                          (unknown column -> ValueError, build time); collects,
                          per row table, the set of columns actually referenced
                          (-> RelNode::TableScan.columns: Vec<String>, replacing
                          "pull the whole row"); and computes
                          (FieldType, nullable) per projection expression via
                          types.rs, used either to synthesize output_model or to
                          validate a user-supplied one.

src/lib.rs     (extend) — InferFn::new: build Schema for every row_tables model
                          + every static_tables Arrow table, run the validation
                          walk, then either synthesize output_model
                          (pydantic.create_model) or validate + accept the
                          caller's. InferFn::infer: row-table rows are now
                          Py<PyAny> model instances -> per-referenced-column
                          getattr(instance, col) (see Row Conversion) instead of
                          dict-key lookup; output rows go through
                          output_model.model_validate(dict) instead of being
                          returned as raw PyDict.
```

No new Cargo dependency. Pydantic and Arrow schemas are read purely via PyO3 attribute/method calls on the Python objects passed in (`model_fields`, `.schema`, `create_model`, `model_validate`) — the same pattern the merged code already uses for `pyarrow.Table.to_pylist()` in `src/lookup.rs`.

## Schema Extraction

`Schema = HashMap<String, FieldType>`, `FieldType { base: Base, nullable: bool }`, `Base = Int | Float | Str | Bool | Other`.

- **Row table (Pydantic model class):** for each `(name, field_info)` in `model_class.model_fields`, map `field_info.annotation` to a `Base` (`int`→Int, `float`→Float, `str`→Str, `bool`→Bool, anything else→Other); `nullable = true` iff the annotation is `Optional[T]`/`T | None` (i.e. `type(None)` appears in the annotation's `Union` args).
- **Static table (`pyarrow.Table`):** for each field in `table.schema`, map the Arrow type to a `Base` the same way, `nullable = field.nullable`.

## Column Validation + Column Collection (one tree walk)

Walking the already-built `Plan` (projection exprs, WHERE predicate, JOIN ON exprs), for every `Expr::Column { table, name }`:
- Resolve `table` (or, if unqualified, the single table in scope that has `name` — same ambiguity rule as runtime `resolve_column`) against that table's `Schema`.
- Missing → `InterpError::Build` → `ValueError` at `InferFn()` construction, quoting the table and column name.
- Found → record `name` into that row table's referenced-columns set (static tables don't need this — their lookup index is already built eagerly from the whole `pyarrow.Table`, unchanged from the merged code).

`RelNode::TableScan` gains `columns: Vec<String>` (the referenced-columns set for that table), populated by this walk. Static-table-only `RelNode` variants (`LookupJoin`'s target) are untouched — this only affects row tables, which are the only place a Pydantic model is involved.

## Static Type Inference

`types::infer_type(expr: &Expr, schemas: &HashMap<String, Schema>) -> Result<FieldType, InterpError>`, mirroring `eval()`'s recursion structure but returning a `FieldType` instead of evaluating a `Value`:

| `Expr` | Base | Nullable |
|---|---|---|
| `Column` | the table's declared `Base` for that column | the table's declared `nullable` |
| `Literal` | the literal's own type | `false`, except a bare `NULL` literal → `Other`, nullable |
| `+ - * / %` | `Int` if both operands `Int`, else `Float` | either operand nullable |
| `= != < > <= >= AND OR` | `Bool` | either operand nullable |
| `NOT` | `Bool` | inner nullable |
| `CAST(x AS T)` | `T` exactly | inner nullable |
| `UPPER LOWER TRIM SUBSTR SUBSTRING` | `Str` | any argument nullable |
| `ABS ROUND` | same numeric `Base` as the argument | argument nullable |
| `CONCAT` | `Str` | `false`, always (verified empirically: `CONCAT` skips NULL args and never itself produces NULL, even with all-NULL inputs) |
| `COALESCE NULLIF` | type of the first argument | `true` (conservative — not attempting to prove tighter) |
| anything unresolvable (e.g. a `FieldType::Other` column feeding an expression with no rule above) | `Other` | `true` |

**This table is deliberately sound but not tight.** "Nullable" here means "we cannot prove this can't be NULL," not "this will be NULL." A tighter analysis (e.g. `x AND FALSE` provably being non-null regardless of `x`) is out of scope — see Non-Goals.

## Output Model: Synthesized or Validated

**If `output_model` is omitted:** for each `(alias, expr)` in the projection, compute `(base, nullable)` via `infer_type`, map to a Python type (`Int`→`int`, `Float`→`float`, `Str`→`str`, `Bool`→`bool`, `Other`→`typing.Any`; wrap in `Optional[...]` if `nullable`), and call `pydantic.create_model("OutputRow", **{alias: (python_type, ...) for each alias})`. Store the result as `self.output_model` (accessible from Python as `fn.output_model` after construction). This always "validates" trivially since it's derived from the same rules used at `infer()` time to produce the actual row dict.

**If `output_model` is supplied:** for each `(alias, expr)` in the projection:
- The alias must exist as a field on `output_model` — missing or extra → `InterpError::Build` → `ValueError`, listing the mismatched names.
- `compatible(inferred_base, declared_base)` must hold, or → `InterpError::Build` → `ValueError`, naming the alias and both types:
  - equal bases → compatible
  - inferred `Int`, declared `float` → compatible (every valid `int` is a valid `float`; Pydantic v2's default lax mode coerces this without loss)
  - inferred `Other` (unresolvable) → always compatible — we have no basis to reject it
  - anything else (inferred `Float`→declared `int`, inferred `Str`→declared `int`, inferred `Bool`→declared `str`, etc.) → provably incompatible → build-time error
- **Nullability is never a build-time error, in either direction.** Declaring a field `Optional` when we inferred non-nullable is fine (just looser than necessary). Declaring a field non-`Optional` when we inferred (possibly-over-cautiously) nullable is *also* not a build-time error — our nullability inference is sound but not tight, so "we couldn't prove it's non-null" isn't the same as "it will be null." If a `None` genuinely reaches that field at `infer()` time, `output_model.model_validate(dict)` raises Pydantic's own `ValidationError` right there, which is the correct place for that failure to surface: we didn't have enough static information to rule it out, so the runtime is the actual authority.

## Row Conversion (at `infer()`)

For each row table, instead of `HashMap<String, Value>` built from a full dict, Rust does `getattr(instance, col_name)` for exactly the columns in that table's `RelNode::TableScan.columns` (populated during the build-time walk) — not `model_dump()`. This is cheaper (skips fields the query doesn't use) and naturally duck-types: a structurally-compatible-but-differently-classed instance still works (no `isinstance` check), and a genuinely missing/misnamed attribute raises `AttributeError`, mapped to a clear `PyErr`. Everything downstream of this point (`Value::from_pyobject` per attribute, the nested `Row`, `execute()`) is byte-for-byte the same code that exists today.

## Output Conversion

Unchanged up through building the per-row `HashMap<String, Value>` → `PyDict`. The one change: instead of returning that `PyDict` directly, call `self.output_model.call_method1("model_validate", (dict,))?` and collect the resulting `Vec<Py<PyAny>>` of validated model instances.

## Error Taxonomy

Same `InterpError` enum and `PyErr` mapping as the dict-based API (`Build`→`ValueError`, `MissingKey`→`KeyError`, `Eval`→`ValueError`) — this spec adds new *sources* of `Build` errors (unknown column against a Pydantic/Arrow schema, output-model field mismatch, output-model provably-wrong base type) but no new error variants.

## Non-Goals

- **Tight nullability inference.** The table above is sound (never wrongly claims non-nullable) but not tight (may over-mark something nullable that a deeper analysis could prove isn't). Tightening this is a possible future enhancement, not required here.
- **Pydantic v1 support.** v2 (`model_fields`, `model_validate`, `create_model`) only.
- **Typing `static_tables`.** Stays `pyarrow.Table` — Arrow's own schema already provides everything the build-time validation walk needs from it.
- **A general-purpose type system.** `src/types.rs` is scoped to exactly the `FieldType` lattice and rules above; it is *not* meant to become a full SQL type system in this pass (see Architecture note on why it's a separate module from `schema.rs`, positioned to grow later).

## Testing Strategy

Every test in `tests/test_interpreter.py` that constructs `InferFn` gets rewritten: `row_tables` becomes a `dict[str, type[BaseModel]]` with small per-test Pydantic models, and `infer()` calls pass model instances instead of dicts. New coverage, one test per row in the rules above plus:

| Test | Covers |
|---|---|
| Unknown column on a row table | Build-time `ValueError` |
| Unknown column on a static table | Build-time `ValueError` |
| Synthesized `output_model` field set + types | Matches the projection exactly, one test per type-inference rule |
| Synthesized `output_model` nullability | `Optional[T]` exactly where the rule table says nullable |
| User-supplied `output_model`, compatible | Build succeeds, `.infer()` returns instances of the supplied model |
| User-supplied `output_model`, missing/extra field | Build-time `ValueError` |
| User-supplied `output_model`, provably wrong base type | Build-time `ValueError` |
| User-supplied `output_model`, declared non-nullable but inferred nullable | Build succeeds; if a row actually produces `None` for that field, `.infer()` raises Pydantic's `ValidationError` |
| Duck-typed row instance (structurally compatible, different class) | `.infer()` succeeds |
| Row instance missing a referenced attribute | Clear error, not a raw `AttributeError` leaking through |
| `getattr` only pulls referenced columns | A row model with extra fields not used by the query still works, and those fields are never touched |
