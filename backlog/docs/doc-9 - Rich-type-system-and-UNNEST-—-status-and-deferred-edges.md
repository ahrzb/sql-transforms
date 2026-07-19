---
id: doc-9
title: Rich type system and UNNEST — status and deferred edges
type: other
created_date: '2026-07-19 01:07'
---
**✅ First slice shipped** — on master (`4809470`). Recursive `Value`/`Base` spine, struct + list types, schema-driven Python↔`Value` marshalling (nested output models), `s.x` field access, `unnest(struct)`→columns, `unnest(list)`→rows (`RelNode::Unnest`), struct equality + join-keys. +17 differential parity tests (159→176), no regressions. **Live remaining work = the fast-follow types and the deferred edges below — none block anything.**

Foundation for the composition output model, fan-out transformers, and the feature contract. Supersedes the narrower "Rust struct-support" ticket. Rather than bolt structs onto the closed scalar type layer, replace that layer with a **recursive, extensible, schema-driven** one so `InferFn` can carry the full pyarrow type surface. Spec: [rich type system design](superpowers/specs/2026-07-16-rich-type-system-design.md).

**Why the pivot (2026-07-16):** composition needs structs; DataFusion has no `struct.*` — it uses **`UNNEST`** (`unnest(struct)`→columns, `unnest(list)`→rows), so we match that; and the engine should carry real feature-data types (also feature-contract groundwork). Build the type *layer* properly, not one type.

**First slice** (reference semantics are DataFusion's throughout, differential-harness enforced):
- Recursive `Value` (`Struct`/`List`, `src/expr.rs`) + recursive `Base` (`src/types.rs`) — a **structural** change touching every `match Base` arm (`compatible`, `field_type_to_python`, `arrow_type_to_base`, …); non-container regressions staying green is the main risk.
- Struct/list construction (`named_struct` / `[…]`) + `s.x` field access on aliased struct **columns** (not `(expr).field` — DataFusion rejects that) (`src/expr_build.rs`).
- `unnest(struct)` → columns: build-time projection expansion (cardinality-preserving).
- `unnest(list)` → rows: **the hard novel piece** — a cardinality change (1 row → N), modeled as a new `RelNode::Unnest` relational operator; empty/NULL list → 0 rows.
- Schema-driven Python↔`Value` marshalling both boundaries (dynamic in/out) + pyarrow struct/list schema reading.

**Fast-follows the spine enables (deferred, not this slice):** temporal (timestamp/date), decimal, map, dictionary, binary — localized additions once the recursive spine lands.

**Open items (in spec):** sqlparser `UNNEST` AST shape (function vs. dedicated node); `unnest(list)` empty/NULL cardinality re-verify; struct field-order round-trip; table-alias vs struct-field-name precedence in `s.x`.

## Deferred edges — all fail loud, none block
- **Ordered comparison (`<`,`>`,`<=`,`>=`) on structs/lists.** `=`/`!=` are implemented (structural equality, `src/expr.rs` `comparison`); DataFusion does **lexicographic ordering** for `<`/`>` on structs and lists (`named_struct('x',1) < named_struct('x',2)` → `true`; `[1,2] < [1,3]` → `true`), which we don't — a full fix needs real lexicographic `Ord` on `Value::Struct`/`List`. Clean runtime error today (`compare_values`' scalar-only `as_f64` fallback). Pick up if a real query needs ordered struct/list comparison.
- **Static-table struct/list values stay `Value::Object` (`src/lookup.rs`).** A struct/list column in a static lookup table isn't marshalled into the recursive `Value`, so a struct join-key against a static table **never matches**. Scalar keys unaffected.
- **Null-struct field access — accepted divergence.** Field access on a NULL struct yields InferFn `NULL` vs DataFusion's quirky `0`. Values-only parity is the bar and DataFusion's `0` is the odd one, so this is accepted, not a fix.
- **`unnest` naming edges.** `unnest(x) AS <existing col>` raises a spurious ambiguity error; an unaliased `unnest(list)` column is named `"unnest"`. Cosmetic/edge; revisit if it bites real queries.
