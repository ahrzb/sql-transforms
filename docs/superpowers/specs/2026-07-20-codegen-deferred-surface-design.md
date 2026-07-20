# Codegen deferred SQL surface — design (TASK-29)

**Author:** Ritchie (codegen dev)
**Date:** 2026-07-20
**Status:** approved for planning

## Summary

Make the codegen serving engine (`sql_transform/_codegen/`) feature-complete for
the SQL surface it currently defers, so the differential suite's **16 codegen
skips drop to zero**. Every deferred shape is already supported by the native
`InferFn` engine and by the DataFusion oracle; codegen raises `UnsupportedInCodegen`
for it. This closes that gap.

decision-7 is ruled (native is the default engine, codegen is opt-in), which
satisfies the framing precondition — the goal is codegen feature-completeness for
opt-in users.

The work is delivered in **three sequenced phases**, each its own implementation
plan, review, and merge, so the skip count drops visibly and each review stays
tractable. All three land 16 → 0.

## Principles

- **Native is the reference; DataFusion is the oracle.** Each shape mirrors the
  native engine's semantics, and is proven by differential parity against
  DataFusion (decision-1). Where native and DataFusion disagree, DataFusion wins
  and it's a native-parity bug (xfail + PM ticket), not a codegen concern.
- **Test bar (no test = not done):** each implemented shape (a) flips from
  skipped to green on the codegen backend in the differential suite, and (b)
  moves from `_DEFERRED` to `_COMMITTED` in `tests/test_codegen_coverage.py`,
  which pins the exact skip set and catches anything missed or silently dropped.
- **v0, no backward-compat.** Direct changes, no shims.
- **Scope stays in `sql_transform/_codegen/` and `tests/`.** No native-engine or
  fit/state/rewrite changes.

## The 16-skip inventory (Fermi-confirmed on master)

| Phase | Count | Shapes |
|---|---|---|
| **A — scalar operators** | 2 | unary-minus-on-a-non-literal ×1; `\|\|` operator ×1 |
| **B — container scalars** | 9 | struct field access ×3; struct/list-typed column passthrough ×3; struct/list construction (`named_struct`/`array`) ×1; `named_struct()` ×1; struct/list comparison ×1 |
| **C — UNNEST** | 5 | `unnest(l)` row-expansion ×5 |

The `UnsupportedInCodegen` raises in `sql_transform/_codegen/plan.py` (and the one
in `engine.py`'s container-output guard) are the exact map of what to implement.

---

## Phase A — scalar operators (retire 2 skips)

Both are scalar expressions codegen already has the machinery for; they're
currently early-`raise`d in `_convert_expr`.

### Unary minus on a non-literal
`_convert_expr`'s `exp.Neg` branch folds a literal (`-1`) but defers `-x`. Mirror
native (`expr_build.rs`: unary minus lowers to `0 - x`, reusing Sub's numeric
promotion): convert `exp.Neg` of a non-literal to `BinaryOp("sub", Literal(0),
inner)`. No new runtime helper — `rt.sub` already gives int→int / float→float
promotion and NULL propagation. `infer_type` then types it via the existing
`sub` arithmetic rule.

### The `||` operator
`exp.DPipe` is deferred because its NULL-propagating semantics differ from
`CONCAT` (which skips NULLs). Mirror native (`expr.rs` `concat_op`: any NULL
operand → NULL, else string-concat via `display`). Add a NULL-propagating binary
concat runtime helper (`None if l is None or r is None else display(l) +
display(r)`; exact name chosen in the plan). Convert `exp.DPipe` to a dedicated
IR op (a `BinaryOp` concat variant or a `Func`), emit it to that helper, and type
it as `STR` (nullable if either side is nullable).

**Testing:** the two currently-skipped `||` / unary-minus differential cases flip
to green; add both shapes to `_COMMITTED`.

---

## Phase B — container scalars (retire 9 skips)

Codegen today has **no container-value support** — every runtime value is a
scalar, and `engine.py` raises `UnsupportedInCodegen` when an output column's
inferred type is a struct/list (`is_container`). Native represents these as
`Value::Struct(Vec<(String, Value)>)` / `Value::List(Vec<Value>)`. In codegen the
natural runtime representation is a **Python dict** (struct, field order
significant) and a **Python list** (list) — which is exactly what `_to_native`
already produces when unwrapping a pydantic struct/list column, and what
`field_type_to_python` already builds output models for (`StructBase`/`ListBase`
are already handled). So the type/output-model machinery is largely ready; the
gap is the expression surface and value semantics.

Sub-parts (each its own set of differential cases + `_COMMITTED` entries):

1. **Struct/list-typed column passthrough (3):** relax the `is_container` output
   guard in `engine.py` so a struct/list column can be projected to output. The
   value flows through as a dict/list; the output model already accepts it via
   `field_type_to_python`. Verify `_to_native` unwrapping + `infer_type` column
   typing already cover this (they mirror native's schema-driven read).

2. **Construction (2):** `named_struct(k, v, ...)` → build an ordered dict;
   `struct(...)` → dict with positional names (`c0`, `c1`, …, mirroring
   `expr_build.rs`); `array(...)` / `make_array(...)` → a list. New IR nodes
   (`Struct`, `List`) + `infer_type` (Struct base from field types; List base
   from unified element type, mirroring native's `unify_list_element_types`) +
   emission to runtime constructors.

3. **Field access (3):** `s.field` (`exp.Dot`), and the 2-part / 3-part
   `exp.Column` forms that parse as field access. Mirror native's
   `Expr::FieldAccess` (NULL base → NULL; missing field → error) — subscript the
   dict at runtime; `infer_type` resolves the field's type from the struct base.

4. **Comparison (1):** struct/list `=` / `!=` → deep structural equality.
   DataFusion/native compare structs/lists structurally (element-wise), and
   type-strictly (a struct's `Int(1)` field ≠ `Float(1.0)`), unlike bare Python
   `==` where `1 == 1.0`. A type-aware recursive equality helper is required so
   codegen matches native/oracle; ordering (`<`,`>`) on containers stays
   unsupported (as in native). Only `=`/`!=` on same-typed containers is
   committed.

**Design note:** Phase B is the bulk. Its plan will decompose into the four
sub-parts above as separate tasks, each independently testable, sharing the new
dict/list runtime value convention.

---

## Phase C — UNNEST (retire 5 skips)

`unnest(list)` is **relational**, not scalar: it turns one input row into N
output rows (one per list element). Codegen's engine emits a per-input-row
projection loop (`_emit_rel`), so this needs a new relational node + emission,
mirroring native's `RelNode::Unnest` (`plan.rs`):

- A `Unnest` plan node wrapping the input, carrying the list expression.
- Build-time: the projection's `unnest(l)` item is rewritten to reference a
  synthetic emitted column (native uses a reserved key like `\0unnest`); the
  `Unnest` node evaluates the list per input row and emits one row per element
  binding the element to that column.
- **Constraint (mirror native):** at most one `unnest(list)` per query — a second
  is a build error.
- Emission: `_emit_rel` gains an `Unnest` case that wraps the body in a
  `for <elem> in <list>:` loop, binding the synthetic column, so `project`
  emits one output row per element. NULL/non-list handling mirrors native
  (`unnest(NULL)` and a non-list value error/skip per native's behavior — the
  plan pins the exact rule against the oracle).
- `infer_type`: the emitted column types as the list's element type.

**Note:** `unnest(struct)` (per-field expansion) is a separate native path
(`expand_unnest_struct`) and is **not** in the 5-skip list-unnest inventory; it
stays deferred unless a skip for it exists (the plan verifies against the actual
skip set).

**Design note:** Phase C is architecturally distinct and higher-risk. If, during
its plan, the relational-emission change proves larger than a single tractable
plan, I'll recommend to the PM splitting it into its own ticket rather than
forcing it under TASK-29 — but the default is to complete it here.

---

## Testing (all phases)

- **Differential parity:** the shapes currently skipped on the codegen backend in
  `tests/test_diff_*.py` flip to green (codegen output == DataFusion oracle). No
  new bespoke harness — the existing parametrized `check()` already runs codegen
  vs oracle; these queries stop raising `UnsupportedInCodegen` and start
  asserting.
- **Coverage guard:** each implemented shape moves from `_DEFERRED` to
  `_COMMITTED` in `tests/test_codegen_coverage.py`. When the last shape lands,
  `_DEFERRED` is empty (or contains only genuinely-still-deferred surface like
  `unnest(struct)` if it's out of scope), and the codegen skip count in the full
  suite is zero for this surface.
- **Skip delta reported per phase** at merge (Iris's ask): the differential skip
  count before/after.

## Out of scope

- Native-engine changes; fit/state/rewrite changes.
- `unnest(struct)` per-field expansion, unless a current skip covers it.
- Codegen transformer support — that's TASK-34, queued after TASK-29.

## Phasing summary

| Phase | Skips retired | Plan | Merge |
|---|---|---|---|
| A — operators | 2 | own plan | own merge, report skip delta |
| B — container scalars | 9 | own plan (4 sub-tasks) | own merge, report skip delta |
| C — UNNEST | 5 | own plan | own merge, report skip delta |

Each phase: worktree, rebase on current master first (natives land frequently),
rebuild native after rebasing (the `.pyd` goes stale), full suite green + skip
count dropped before merge.
