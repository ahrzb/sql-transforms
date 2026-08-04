---
id: DRAFT-24
title: "Named outputs: struct→struct closure and the three naming decisions"
status: Draft
type: spike
created: 2026-08-04
---

## Where this came from

Design dialogue with AmirHossein, 2026-08-04, starting from "can projections
be composable?" (`q(age)`), through the criticism of call-position, to his
three corrections that turned the design around. Recorded because the
conclusion is small but the reasoning is not re-derivable from the code.

## The frame

```
fit : (Transform, Data[S]) → (S → T)
```

**S is fixed by the bundle** you fit on (a named struct — sklearn's
`feature_names_in_`/`n_features_in_`, but checked at build time and matched
name-keyed, per confit's existing rule
`struct_compatibility_is_name_keyed_not_positional`). **T is learned**
(PCA(2) → 2 fields; OneHotEncoder → the training vocabulary). Before fit a
transform is shape-polymorphic; after fit it is monomorphic.

Consequences that were latent and are now stated:

- **Two kinds of unknown.** Unknown *shape* (missing field, wrong width,
  wrong type) → build error by name (P7). Unknown *value* (a group never
  seen) → NULL (P14). Unseen-group-is-NULL is not a weasel: it is a
  value-level fact inside a well-typed domain.
- **The fit DAG is the type-inference order.** `sc(pca(x))` cannot be
  type-checked until `pca` is fitted, because T is learned. So for composed
  fitted transforms, codomain mismatches are caught **at fit**, not at
  construction — everything syntactic stays construction-time, nothing
  moves to serve time. **P7 needs this carve-out written in when composition
  lands.**
- **Why the round-trip control is provable:** fit yields a function whose
  domain is the training struct type; the round-trip applies it inside its
  own domain.

## Why call-position wins (the criticism that was wrong)

The first reading — "compose projections in FROM position, `q(age)` is a
scalar costume on a table function" — was wrong on three counts, all
AmirHossein's:

1. **Struct → Struct is a real type.** Names are preserved on both sides, so
   nothing is thrown away and there is no macro-hygiene problem that named
   fields don't answer.
2. **A fitted projection is a transform** (fit/transform), so it belongs in
   the transformer vocabulary rather than needing its own mechanism —
   `q(...) OVER (PARTITION BY country)` fitting one q per country falls out
   free.
3. **A transform is a partial application; a CTE is a full one.** Unfitted
   is `λtrain. λrow. f(θ(train), row)`; fitted is `f(θ, ·)`. A CTE binds its
   input at authoring time and has no free argument, so it can never be
   applied to a *different* argument — which is exactly what `f∘g` needs
   when the caller's schema uses different names (`q(struct_pack(age :=
   years))`). FROM-position is application-to-ambient-scope; it has nowhere
   to write the binding.

Parse pins (2026-08-04, DuckDB): `q(struct_pack(a := x))` → FUNCTION;
`q(...) OVER (...)` → WINDOW_AGGREGATE; `q(...).z` → OPERATOR/STRUCT_EXTRACT
(parens optional); `q(COLUMNS(* EXCLUDE y))` → child is a STAR node;
`q({'a': x})` desugars to struct_pack; **`q(*)` erases to zero children**
(stays refused); **`(expr).*` is a syntax error** — so "bare item means all
fields" is a rule we own, not one we inherit.

## The defect this exposes

The protocol is asymmetric: **named struct in, anonymous tuple out.** That
is what breaks closure — a consumer wants fields, a producer emits
positions. Fix the return side and the vocabulary closes: sklearn
transforms, author UDFs, and projections all become `Struct → Struct`, and
`sc(pca(q(...)))` composes.

Closure costs nothing at runtime, by the law we already hold (P16): bundles
are destructured at construction, so composition is **lane wiring at
construction** — no struct materializes. The engine already chains `ecall`
outputs (SSA values) into `ecall` arguments; only the frontend refusal
"width-k must be a bare SELECT item" needs relaxing to "…or be consumed
lane-wise by another extern".

## The three decisions (all approved 2026-08-04)

### 1. Generated names are part of the type identity

Measured, `OneHotEncoder`, one new category that sorts first:

```
fit #1 lanes: ['color_blue', 'color_red']   row 'red' -> f1 = 1.0
refit  lanes: ['color_aqua', 'color_blue', 'color_red']   row 'red' -> f1 = 0.0
```

Positional identity builds and serves *wrong* after a refit — the C5
failure mode. Name identity turns the same refit into a loud build error
("no field `color_red` in {color_aqua, color_blue}"). For data-dependent
widths — the transforms that get refit most — names are the only safe
identity.

Name sources, in order: a projection → its select-item output names; an
sklearn object → `get_feature_names_out(S names)`; a declared UDF → its
declared names; canonical `f0..fk-1` only when nothing else exists
(duck-typed objects). Names are **compile-time keys** (field access resolves
during the rewrite), so unquotable names like `k_a."b` are harmless until
they reach the output boundary — a boundary problem, not a composition one.

### 2. Author override at registration

`Named(PCA(2), returns=("size", "cost"))` — a second, authoritative naming
source, checked against the learned width at fit (a fixed override cannot
serve a learned-width transform, and says so). Not required for
correctness: SQL is already a rename operator
(`struct_pack(x := p.pca0, ...)`, or a wrapper level).

### 3. Boundary naming for *bare* width-k items

Only bare items are in question — an addressed field is already a scalar
column. Chosen: **flat, alias-prefixed** — `... AS e` emits `e_pca0`,
`e_pca1` instead of one `array[n]`. Collision-proof by construction, no
nested-model path at the marshaller.

Deliberate consequence: flattening loses the "NULL list vs list of NULLs"
distinction (P16) — an unseen group and an all-NULL result both become k
NULL columns. Accepted; note it when P16 is amended.

## The split (sequenced loops, one PR each)

**Loop 1 — named outputs + field access (pure Python, no engine change).**
`take_names`/`return_names` on the UDF protocol; names derived at fit
(sklearn → canonical fallback); `t(...).field` resolves at marginalize time
to a **per-field width-1 UDF** (`__cf_tf{j}_g{m}` — positional so the
minted SQL name is always a legal identifier, mapped to a field name at
fit); a requested field absent from the fitted output refuses at fit,
naming the available fields; per-group shape disagreement (one group's
fit yields a different width/names than another's) refuses at fit.
*Cost accepted:* k accessed fields = k evaluations of the same opaque
transform. Fixed later by loop 4, not by weakening anything.

**Loop 2 — `Named(...)` registration override.** Small; depends on loop 1.

**Loop 3 — flat alias-prefixed boundary.** Bare width-k items emit named
columns; touches confit's `WideOut` (names travel on the udf object,
boundary only). Independent of loop 2.

**Loop 4 (not yet requested) — single-evaluation field access.** Teach the
engine STRUCT_EXTRACT-over-`ecall` (or lane-select), so k field accesses
share one call. Removes loop 1's k× cost; also the prerequisite for
composition proper (extern → extern lane wiring).

**Loop 5 (not yet requested) — composition.** Projections as vocabulary
members: `q(struct_pack(...))` with symbolic inlining as the fast path,
partition composition (call-site keys prepend to q's internal windows),
`__cf_` α-renaming for hygiene, recursion refusal, and the P7 carve-out.
