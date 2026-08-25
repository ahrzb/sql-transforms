# Single-evaluation field access: STRUCT_EXTRACT over ecall (TASK-63, DRAFT-24 loop 4)

Ticket: `backlog/tasks/task-63 - Single-evaluation-field-access-STRUCT_EXTRACT-over-ecall-in-the-engine.md`.
Draft: draft-24 (loops 1-3 landed; this is loop 4's thorough half — the
one-slot row-path memo was the cheap half and is superseded, see below).
Design approved in-session 2026-08-04.

## The change

k addressed fields of one fitted transformer serve as ONE call plus k lane
reads, on both serving paths.

```sql
-- before: two lane UDFs, each a FULL transform() per row
__cf_tf0_g0(__cf_p0.__cf_est, __cf_t.a, __cf_t.b) AS pca0,
__cf_tf0_g1(__cf_p0.__cf_est, __cf_t.a, __cf_t.b) AS pca1
-- after: one whole-value call, two field reads
(__cf_tf0(__cf_p0.__cf_est, __cf_t.a, __cf_t.b)).pca0 AS pca0,
(__cf_tf0(__cf_p0.__cf_est, __cf_t.a, __cf_t.b)).pca1 AS pca1
```

- Field access **survives into the serving AST** as STRUCT_EXTRACT over the
  whole-value call (today `_LevelRewriter` consumes it and mints a
  `__cf_tf{j}_g{m}` lane UDF per field).
- Width-1 emits the bare call — today's `tf_width1` shape; the field name is
  still validated at fit (P7 carve-out).
- Bare width-k items (loop 3) expand to the same shape: k field reads over
  one call, flat aliases unchanged (`AS e` → `e_pca0`, `e_pca1`).
- `TransformLane` and the `__cf_tf{j}_g{m}` lane UDFs are deleted (v0, no
  compat).

## Why one evaluation holds, per path

**DuckDB (batch, `transform`)**: width-k fitted UDFs register as
`STRUCT(name TYPE, ...)` from `return_names` (replacing the dead `DOUBLE[]`
arm in `UDF._duck_signature`). DuckDB CSEs identical pure scalar-UDF calls —
measured 2026-08-04 on duckdb in-repo: four textual mentions of `f(x)` cost
exactly 100 calls on 100 rows, in every spelling (dot, `struct_extract`,
subquery, CTE); `NULL` in → `NULL` struct → `NULL` fields (P14 intact).
Purity is already the contract (P15); `side_effects` stays unset.

**confit (row, `infer`/`infer_batch`)**: the frontend binds `call.field` to
`ExternCall { site, ext, args, ret: i, whole: false }` — the lane-select the
plan layer already models (`plan.rs`), lowering already dedupes per site
(`PB.externs`, `lower.rs:1547`), and both backends already consume arbitrary
`ret` through the shared `call_extern`. All k field reads get **one shared
site** via a frontend cache keyed on (extern index, syntactic args) — the
same collapse `wide_extern_lanes` uses today for bare items. Execution,
verifier, IR, and both backends are untouched.

## Engine diff (packages/confit)

- `ExternSpec { name, params, rets }` grows `ret_names: Vec<String>` (v0
  break; empty = unnamed, and field access over an unnamed extern refuses by
  name). The pyo3 layer reads `return_names` off the UDF protocol object.
- `frontend.rs`: `CompoundFieldAccess` with a single `Dot` over a width-k
  extern call binds the named lane (works mid-expression — a field read is
  width-1 and composes in arithmetic). Unknown field name → named refusal
  listing declared names. Field access over a **width-1** extern refuses:
  its DuckDB registration is scalar, so `.f0` errors there, and C2 forbids
  confit accepting what DuckDB won't (sql-transform never emits it — width-1
  field access resolves to the bare call at rewrite time).
- The width-k-as-scalar refusal (`frontend.rs:4444`, "width-{k} udf ... used
  as a scalar expression") keeps its exact text for calls **without** a
  field access. No third behavior (C5).

## sql-transform diff

- `_marginalize.py`: `_tf_call_node` emits the whole-value call, wrapped in
  a STRUCT_EXTRACT node when a field is addressed on width-k;
  `step["fields"]` survives only for fit-time validation (unknown field →
  fit error naming the fitted outputs). `expand_wide_items` emits field
  reads over one call instead of `{target}_w{i}` lane UDFs.
- `_udf.py`: width>1 `_duck_signature` returns
  `duckdb.struct_type(dict(zip(return_names, field_types)))`; `_scalar`
  returns `dict(zip(return_names, out))` (None stays None → NULL struct).
  `TransformLane` deleted.
- `_projection.py`: `_lane_udfs` deleted; the whole-value UDF is the served
  UDF (no post-expansion deletion of it).

## Law amendment — P16

The clause "no STRUCT or LIST value ever flows at serving time" narrows: a
struct value exists **transiently inside DuckDB's expression evaluation**
for fitted width-k calls, but never crosses the output boundary (every
output column stays scalar), and confit never materializes one at all (a
field read is one SSA lane of one ecall). Field access reads a lane off the
one call — no per-lane UDFs exist anymore. The engine's `list | None`
boundary for direct `DuckDBInferFn(udfs=...)` users is unchanged.

## Verification

- **New counting test** (none exists in-repo): a duck-typed estimator whose
  `transform()` increments a counter; asserts 1 call per row (not k) on the
  row path AND n calls total (not k·n) on the DuckDB batch path, for both
  addressed fields and a bare width-k item; plus an interleaved two-group
  case (distinct instance ids stay distinct).
- Gates C1-C5 unchanged in meaning. Pins that move: `_named_outputs_test`
  lane-UDF spellings (`__cf_tf0_g0` …), `udf_check`'s width-k DuckDB
  registration (struct variant for field-addressed parity SQL). Pins that
  must NOT move: `test_wide_mid_expression_refuses`, the corpus buckets and
  D1 totals, `test_wide_udf_bare_item_is_list_field` (engine list boundary).
- Full gate: `uv run pytest` from the repo root + `cargo test` in
  packages/confit.

## D2 / bench

Re-run `uv run python -m benchmarks.bench_transforms` (row path) and add a
batch-path (`p.transform`) reading; record before/after for `tf_fields2`
and `tf_bare2` in docs/kpis.md D2. Expected: tf_fields2 ~138.6µs → ~87µs
per row, converging on tf_width1's single-call cost; batch path k→1
likewise.

## Deliberately out

- DRAFT-23 native UDF families — the ~100x lever, awaiting discussion.
- Composition (DRAFT-24 loop 5) — this loop builds its prerequisite (lane
  reads off an ecall); no extern → extern wiring here.
- The one-slot row-path memo (loop 4's cheap half, written by a parallel
  session, lost uncommitted): superseded by this engine fix, which covers
  both paths — not reintroduced. DRAFT-24 gets a note.
- Vectorized `apply_batch` / rayon-over-rows (mid-band lever, separate).
