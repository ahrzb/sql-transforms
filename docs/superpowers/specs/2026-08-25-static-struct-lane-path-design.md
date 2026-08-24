# Static struct lanes carry a structured path (TASK-132)

Decision ground: docs/rfcs/2026-08-19-static-struct-lane-encoding.md
(accepted, alternative A). Surface map: full-code survey 2026-08-25, the
three riskiest sites re-verified by hand. DuckDB reference model
source-verified in the RFC.

## Goal

A static table's struct leaves stop being identified by their dotted
spelling. The lane's PATH becomes data; the dotted string survives only
where a human reads it (error text). No resolution or data path ever
builds or splits a dotted name again.

## Non-goals (land later, on top)

- Unqualified `w.mean` resolution (TASK-127's unqualified-reference
  item; DuckDB serves it, we refuse with a wrong reason today).
- Collision *detection messages* (TASK-127's collision-detection item)
  beyond what the encoding makes structurally unreachable.
- Lifting the `fname.contains('.')` skip that silently DROPS a struct
  field whose own name contains a dot (duckdb/mod.rs:180, :1277). The
  new encoding removes the reason for the skip; removing the skip
  itself is a behavior change ticketed separately once this lands.

## The design: statics get the tree rows already have

The row side is already alternative A: `StructCol { pos, name, fields }`
with `StructNode::{Leaf(lane), Opaque, Nested}` (plan.rs:111-137), and
resolution walks the tree (`walk_struct`, frontend.rs:3531). The static
side flattens to dotted names and greps them. The refactor gives
`StaticTable` the same tree and shares the walk:

```rust
pub(crate) struct StaticTable {
    pub cols: Vec<Col>,          // unchanged shape; leaf Col.name becomes DISPLAY-ONLY
    pub structs: Vec<StructCol>, // NEW: one per struct column, leaves -> lane indices
    pub opaque: Vec<(String, String)>, // loses its ("w", "struct") entries
    ...
}
```

- `flatten_static` still interleaves leaf lanes into `cols` in declared
  order (indices stay the data-path currency; star order unchanged) but
  ALSO builds the `StructCol` tree. `Col.name` keeps the dotted
  spelling purely for display.
- Struct heads move out of the `opaque` `"struct"` string-sniff: head
  lookup is `structs.iter().find(name)`. `opaque` keeps only genuinely
  unservable column types.
- Resolution (`qualified_path`) becomes: resolve head among plain cols
  and struct heads, then walk remaining parts through the SAME
  `walk_struct` used for rows. The `join(".")` + byte-slice prefix scan
  dies; "is a struct" vs "no such key" comes from the tree node kind
  (Nested vs missing child), which is what the scan was approximating.
- Data paths carry segments as data, not by splitting a string:
  `materialize_map`'s get-closure, `Marshaller::build`, and `run_rows`
  receive the path `Vec` (or walk the tree) instead of `name.split('.')`.
  `prepare_opaque` serializes the path vector into `StaticSpec`, dotted
  only in its display field.
- `arrow.rs::ingest` detects structs by `!structs.is_empty()`, not by
  sniffing a dot in a name.

## Deliberate behavior flips (each is a bug fix, live-oracle pinned)

1. The RFC's collision table serves BOTH spellings like DuckDB:
   `s.w.mean` (3 parts) resolves head `w` + field `mean`;
   `s."w.mean"` (2 parts, quoted) exact-matches the literal column.
   Today both refuse.
2. A quoted `"w.mean"` with NO such literal column stops binding the
   struct leaf (today it does — the namespaces are one string space;
   DuckDB says the column does not exist).
3. A plain static column literally named `a.b` is served correctly.
   Today `materialize_map` walks `row["a"]["b"]` — it reads a DIFFERENT
   value or errors "missing column" for a column that exists.
4. A non-ASCII struct field name stops being able to panic the binder
   (the byte-slice prefix scan dies with the encoding).

## Consumer map (27 functions, from the survey)

- Creation (3): `flatten_static`, `build_fields` (row side, shares the
  tree types — audit only), `DuckDBInferFn::new`.
- frontend.rs resolution (19): `frontend`, `bind_from` (USING/NATURAL
  arms feed lane names into `column()` today — they iterate plain cols
  + tree leaves explicitly instead), `resolve_static`,
  `opaque_static_refusal`, `bind_on`, `promote_key`, `static_col_of`,
  `apply_column_alias`, `decode_star_filter`, `expand_star_lanes`,
  `static_lane`, `key_lane`, `compound`, `qualified_path`,
  `this_col_with_fields`, `bare_col_with_fields`, `walk_struct`
  (shared), `column`, `qualified`.
- specializer/mod.rs (1): `prepare_opaque`.
- duckdb/mod.rs data path (3): `materialize_map`, `Marshaller::build`,
  `run_rows`.
- duckdb/arrow.rs (1): `ingest`.
- Untouched by design: all of lower.rs (index-keyed),
  `dedup_output_names`, `arrow::output_schema`, output naming
  (`default_name` takes the LAST part — already structural).

## Display rule

One helper renders a path dotted (`path.join(".")`), used only in:
error text (`key_lane`, `promote_key`, `materialize_map` missing/
DECIMAL messages, `run_rows` missing-attribute), and `Col.name` for
debug/star display. The survey's display table is the checklist; every
message keeps its current spelling so existing pins survive.

## Test plan (red first, per class)

- NEW Rust unit constructor: `tests.rs` today has no `StaticTable` with
  struct lanes at all (`stat()` is all_scalar) — add one; unit-test the
  tree resolution and the collision table at Rust level.
- Live-oracle pins (Python): the RFC collision table both spellings;
  quoted-vs-unquoted matrix; a plain static column named `a.b` served
  end-to-end (data-path flip #3); mirror-path invariant
  (`w.x.y.z.a` vs `w.z.y.x.a`) survives; non-ASCII field name builds
  and serves.
- Survive list (must stay green untouched): TASK-116 lane serving,
  left-miss NULL, refuse-by-name trio, TASK-125 star + EXCLUDE order,
  TASK-121 ambiguity trio, TASK-126 aliases, row-struct star pins in
  tests.rs.
- Fuzzer: gen.py's `Struct.leaves()` keeps emitting dotted REFERENCE
  spellings (that is SQL, still valid); its EXCLUDE filter and CTE
  alias dot-workarounds are re-checked against the new behavior.
- Gate: full suite release + debug, then a 4k campaign with residue
  attributed against the pre-refactor baseline.

## Sequencing

One branch (task-132), commits by class: (1) tree construction +
display-only names with everything still green, (2) resolution moved to
the tree, (3) data paths moved off split('.'), (4) flips pinned. Each
class red-first. Estimated a day-plus, as the RFC priced.
