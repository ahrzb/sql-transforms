# Rich (recursive) type system + `UNNEST` — Design

**Goal:** Replace the interpreter's closed scalar type layer with a **recursive,
extensible, schema-driven** one so the `InferFn` engine can carry the full pyarrow
type surface — starting with the **container types (`struct`, `list`)** and the
**`UNNEST`** expansion function, with temporal / decimal / map / etc. as
fast-follows the new spine makes localized. Supersedes the narrower "Rust
struct-support" ticket.

**Why now (decided 2026-07-16):** the composition output model needs structs;
`struct.*` isn't a thing in DataFusion (`UNNEST` is); and more broadly the engine
should carry real feature-data types (this is also feature-contract groundwork).
So we build the type *layer* properly rather than bolt on one type.

**Reference is DataFusion** (differential-harness parity). Confirmed live:
`unnest(struct)` → the fields as **columns** (same cardinality); `unnest(list)` →
one **row per element** (cardinality expansion); `named_struct('x',1,…)` builds a
struct; `s.x` field access works only on an aliased struct **column**
(`(expr).x` is rejected); struct deep-equality + struct-as-join-key work.

## Architecture — the spine

### 1. `Value` becomes recursive (`src/expr.rs:4`)
Add container variants alongside the scalars (keep `Object` for genuinely-opaque):
```
Value::Struct(Vec<(String, Value)>)   // ordered named fields (order matters for unnest)
Value::List(Vec<Value>)
```
Extend the hand-written `Clone` / `Hash` / `PartialEq` (`src/expr.rs:16-74`)
structurally — `PartialEq`/`Hash` are load-bearing: struct/list equality + join-key
use rely on them, and DataFusion has struct deep-equality.

### 2. `Base` becomes recursive (`src/types.rs:4`)
```
Base::Struct(Vec<(String, FieldType)>)   // ordered, nested
Base::List(Box<FieldType>)
```
This is a **structural** change: every `match Base` gets new arms — `compatible()`
(`types.rs:154`), `field_type_to_python` (`schema.rs:174`), `arrow_type_to_base` /
`python_type_to_base`, `annotation_to_field_type` (`schema.rs:100`). `Schema`
(`types.rs:20`) stays the flat top-level `HashMap`, but field types may now nest.

### 3. Schema-driven Python↔`Value` marshalling ("dynamic in and out")
- **In** (`src/lib.rs` input read): read each Python input into the right `Value`
  per its **declared** field type — a Python `list` → `Value::List`, a `dict` /
  nested pydantic model → `Value::Struct`, recursively; scalars as today.
- **Out** (`src/lib.rs:161-210`, `synthesize_output_model` `lib.rs:27`): `Value` →
  Python per the output type — `Value::Struct` → a **nested** pydantic model
  (recursive `create_model`), `Value::List` → `list[T]`. `field_type_to_python`
  recurses.
- **pyarrow schema reading** (`schema.rs:37` `from_arrow_table`): walk
  `pyarrow.StructType` / `ListType` children instead of prefix-matching a type
  string.

### 4. Per-type operation dispatch
- **Equality / compare** (`expr.rs:313` `compare_values`): structural for
  struct/list (needed for join-key + DataFusion parity). New code, not a
  fall-through.
- **Arithmetic / CAST on containers**: clean error — already the behavior
  (`as_f64` `expr.rs:286`, `eval_cast` `expr.rs:455` reject non-scalars), matching
  DataFusion's "Cannot coerce arithmetic … Struct" / "Unsupported CAST from Struct".
  Verify a `Value::Struct`/`List` hits those paths.

## Expressions — construction + access (`src/expr_build.rs`)

- **Struct construct:** `named_struct('x', e1, 'y', e2)` (and DataFusion's
  `struct(e1,e2)` → auto-names `c0,c1`) → `Expr::Struct(Vec<(String, Expr)>)`.
- **List construct:** `[e1, e2, …]` (sqlparser `Expr::Array`) → `Expr::List(Vec<Expr>)`.
- **Field access `s.x`:** the current `parts.len()==2` guard (`expr_build.rs:15`)
  assumes `table.column`. Loosen to: resolve `parts[0]` against known relation
  aliases first; if it's not a relation, treat as struct field access
  (`Expr::FieldAccess{base, field}`), recursively for `s.a.b`. Scope to the
  `CompoundIdentifier` dotted form (matches what DataFusion accepts); do **not**
  support `(expr).field` (DataFusion rejects it). The table-vs-struct-name
  ambiguity resolution lives in `validate_expr` / `resolve_tables`
  (`plan.rs:743`).
- `infer_type` (`types.rs:29`) gets matching arms: struct-construct → a
  `Base::Struct`; field-access → look the field up in the base's struct schema;
  list-construct → `Base::List(elem)`.

## `UNNEST` — the two shapes

sqlparser's exact `UNNEST` AST node is **an open item to verify** at build time
(function vs. dedicated node); the semantics below are the contract regardless.

### `unnest(struct_expr)` — projection expansion (same cardinality)
A **build-time rewrite** in `build_projection` (`plan.rs:247`): replace the
`unnest(struct)` select-item with one `(field_name, FieldAccess)` entry **per
struct field** — field names known statically from the expr's `Base::Struct` type.
Output shape stays fixed-at-build (the engine requires that). Column naming follows
DataFusion (`<expr>.field`) unless aliased. (Note: `SELECT *` / `QualifiedWildcard`
is unsupported today, `plan.rs:258`; this is *not* built on it — it's a targeted
`unnest`-only expansion.)

### `unnest(list_expr)` — row expansion (**cardinality change — the hard part**)
The interpreter is strictly **1 input row → 1 output row** today (`execute`
`plan.rs:430` maps each row to one output). `unnest(list)` breaks that: one row
with an N-element list becomes N rows. Model it as a **new relational operator**
`RelNode::Unnest { input, list_expr, output_col }` (sibling to the joins), applied
before projection: for each input row, evaluate the list and emit one row per
element (binding `output_col`), preserving the other columns. Match DataFusion on
the edge cases — **empty list → zero rows, NULL list → zero rows** (verify both
against DataFusion; the live check hit a harness-name bug, re-verify).

**Sequencing within the slice:** land the spine + struct + list + marshalling +
`unnest(struct)` first (all cardinality-preserving), then `RelNode::Unnest` for
`unnest(list)` as the final, separately-tested piece — split to an immediate
fast-follow if it proves large.

## Testing (differential harness, per case)
`transform` (DataFusion) == `infer` (`InferFn`) for: struct construct + marshalling
round-trip (Python dict/nested-model in and out); list construct + round-trip;
`s.x` field access; `unnest(struct)` → columns; `unnest(list)` → rows incl. empty +
NULL; struct/list equality + struct-as-join-key. Plus non-container regressions
stay green (the `match Base` churn is the risk).

## Non-goals / deferred (fast-follows once the spine lands)
- Other pyarrow types — **temporal (timestamp/date), decimal, map, dictionary,
  binary** — localized additions on the recursive spine; not this slice.
- `(expr).field` inline dot-access (DataFusion rejects it).
- `SELECT *` / general wildcard (only targeted `unnest` expansion here).
- Multi-`unnest` in one SELECT / `unnest` cross-product semantics — verify DataFusion
  behavior and scope later if needed.

## Open items
1. **sqlparser `UNNEST` AST shape** — verify how `unnest(x)` parses (function vs.
   dedicated node) before wiring `expr_build`.
2. **`unnest(list)` empty/NULL cardinality** — re-verify against DataFusion (the
   live check errored on a harness table-name reuse, not semantics).
3. **struct field ordering** through marshalling — ensure `Value::Struct` field
   order round-trips (drives `unnest(struct)` column order).
4. **table-alias vs struct-field name collision** in `s.x` resolution — pick the
   precedence rule (relation alias wins, else struct field).
