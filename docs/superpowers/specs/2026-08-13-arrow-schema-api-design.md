# Arrow schema API (2026-08-13)

Every schema the engine speaks is `pyarrow`. Row-table schemas become
`pa.Schema` (statics and UDF signatures already are). Row data stays
dict-or-object in, becomes dict out. The pydantic surface is deleted.

Decided with AmirHossein 2026-08-13: strict totality, no coercion, one row
method, clean break (v0, no compat shim), lands inside the m-8 epic.

## Why

- pydantic `int` is width-less — the entire `_INEXPRESSIBLE_INPUTS` class in
  the corpus replay exists because `round(a, b)` needs an INTEGER `b` that a
  pydantic model cannot declare. Arrow types carry widths.
- The engine already speaks arrow at every other boundary: `static_tables`,
  UDF `takes`/`returns`, `infer_arrow`, the emit path. Pydantic was the one
  foreign vocabulary left, and the measured serving cost sits at exactly that
  boundary (see `project_rust_vs_codegen_benchmark`).
- The synthesized output model and the `output_model` refusal on
  `infer_arrow` (known-limitations §3a) both exist only to serve pydantic
  out. Dict-out deletes the machinery and the limitation.

## Surface

```python
# BEFORE
class Row(BaseModel):
    a: float | None = None
    b: int | None = None                     # width inexpressible -> BIGINT
fn = DuckDBInferFn(sql, row_tables={"__THIS__": Row}, static_tables={},
                   output_model=..., output="model", shape="map")
out = fn.infer_rows([Row(a=1.0, b=2)]); out[0].o

# AFTER
fn = DuckDBInferFn(
    sql,
    row_tables={"__THIS__": pa.schema([("a", pa.float64()),
                                       ("b", pa.int32())])},
    static_tables={},
    shape="map",                             # udfs= unchanged too
)
out = fn.infer_rows([{"a": 1.0, "b": 2}]); out[0]["o"]
out = fn.infer_rows([my_obj])                # attribute access also accepted
```

Deleted: `output_model=`, `output=`, the `output_model` attribute, the
`output` property, and `infer(tables=..., **kwargs)`. Kept unchanged:
`static_tables`, `udfs`, `shape`, `infer_arrow`, and the `shape` /
`backend` / `boundary` properties.

```python
fn.output_schema   # -> pa.Schema: names, arrow types, nullability, exactly
                   #    what infer_arrow's output table carries
```

`row_tables` stays a dict: the key is the table name the SQL references, and
more than one row table remains *expressible*. The engine refuses >1 at build
today ("the specializer takes exactly one row table", duckdb/mod.rs:1280);
when that lifts, the row path grows a tables form — not now.

Static-tables-only queries (`Engine::Constant`) are served by
`infer_rows([])` — the constant path already ignores its input rows and emits
the fixed rows; this replaces the old `infer()` no-args call.

## Row schema vocabulary (v0)

| arrow type | binds as | Python value accepted |
|---|---|---|
| `pa.bool_()` | BOOLEAN | `bool` |
| `pa.int8()` / `int16()` / `int32()` / `int64()` | TINYINT / SMALLINT / INTEGER / BIGINT | `int` in range (never `bool`) |
| `pa.float64()` | DOUBLE | `float` |
| `pa.string()` | VARCHAR | `str` |

Anything else — `float32` (engine computes in f64), `uint*`, `decimal`,
date/time, nested, `large_string` — refuses at build naming the field and
type. No third mode: a schema either binds fully or construction raises.

Nullability is the arrow field flag: `pa.field("k", pa.int64(),
nullable=False)` binds NOT NULL (what pydantic's non-`| None` field meant);
arrow's default is nullable, matching `| None = None`.

## Marshalling rules (`infer_rows`)

- **Strict totality.** Every schema field must be present — a dict key or an
  object attribute. Missing refuses by name (catches typos; a typo'd key
  always leaves a schema field missing). NULL is an explicit `None`.
- **Extra dict keys are ignored.** The totality check already catches every
  typo, and object input can't enumerate extras anyway — symmetric.
- **No coercion.** Exact Python type per the table above or refuse by name.
  `"1"` is not 1; `1` is not `1.0`; and `True` is not an `int` value even
  though Python subclasses it — the bool/int split is checked explicitly.
- **Range.** An `int` outside the declared width refuses naming the column,
  the value, and the SQL type — the input mirror of the narrow-output check.
- **Nullability.** `None` into a `nullable=False` field refuses by name.

`infer_arrow` ingest becomes schema-driven: input columns must match the
declared row schema's types exactly (today it pins int64/double/string/bool;
a declared `int32` column now expects an int32 array — cast first otherwise).

## Output

`infer_rows` returns `list[dict]`, keys = `output_schema` names, values plain
Python (`None` for NULL). Shape semantics unchanged — only the element type
migrates (model -> dict). Fresh dicts per call; callers may mutate.

## Non-goals

- The dialect catalog wire form (`dialect/py.rs`: DuckDB type strings from a
  DESCRIBE walk) is a deliberately different surface and keeps its shape.
- The f32-static silent-widen hole closes with TASK-96 (narrow statics), not
  here — this migration touches the row and output surfaces only.
- Multi-row-table serving; a `confit.Int32`-style annotation layer (dead —
  arrow types carry widths natively).

## Blast radius and landing

Measured on master 217799d: 19 confit test files + `fuzz/oracle.py` +
`tests/test_corpus_replay.py` reference the pydantic surface; 13
sql-transform files (`_projection.py`, `model/_projection.py`, tests). The
corpus replay gets *simpler*: it passes DuckDB's own arrow schema straight
through and `_INEXPRESSIBLE_INPUTS` is deleted — its rows must now serve.

Single clean-break PR, built in this order:

1. Rust core, TDD red-first (Fable): `pa.Schema` ingestion for `row_tables`,
   dict/object marshaller with the rules above, dict emit, `output_schema`
   property, deletions.
2. Mechanical migration of confit tests + fuzzer + corpus (cheap-model
   subagents — sonnet, worktree-isolated, TDD skill named in dispatch).
3. sql-transform projection: synthesize `pa.Schema` instead of models
   (Fable on `_projection.py` core, cheap models on tests).
4. Gates: full pytest from root, cargo test, serving bench before/after
   (latency is a control KPI — dict-out must not lose), 20k fuzz campaign.

`pydantic` should drop out of `packages/confit/pyproject.toml` entirely at
the end; if anything still imports it, that's a missed migration site.
