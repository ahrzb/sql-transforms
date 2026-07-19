---
id: doc-8
title: 'Composition — {transform}(col) references, mechanics and deferred slices'
type: other
created_date: '2026-07-19 01:07'
---
Let one `SQLTransform` reference **another `SQLTransform` object** inside its SQL, applied to a column, and combine the two correctly. Target syntax — a t-string where an embedded transform is invoked on a column: `SQLTransform(t"SELECT {scaler}(age) AS age_scaled FROM __THIS__")`, with `scaler` a `SQLTransform` interpolated in. `{scaler}(age)` = apply `scaler`'s transform to column `age`. The first implementable step of the execution model ([[doc-7]]), and the primitive our `Pipeline` / sklearn composition builds on.

## Shipped
- **First slice (frozen path)** — master (through `bb22526`). `{a.transform}(col)` inlines a fitted transform's frozen scalar expression, fused into one per-row expression with exact `transform`/`infer` parity; the outer taking its own window aggregate over the inlined column works; a bare `{a}` on a fitted object and `{a.transform}` on an unfit object both error explicitly. Identifier handling locked to DataFusion-faithful verbatim quoting (the earlier quoting gap in the inline + PARTITION BY paths is fixed, `c056ec3`).
- **Second slice (fit-cascade)** — master (`5ac613e`, suite 188). An outer `SQLTransform` can reference an **unfit** `SQLTransform` via `{a}(col)`; the ref's window-aggregate state is fit **into the composite** during the composite's `.fit()` (sklearn-staged), with arbitrary nesting/chaining (`{a}({b}(x))`), outer aggregates over the cascade, and free mixing with the frozen path (`{a}({b.transform}(x))`). Single-output `{a}(col)` auto-unwraps to a scalar; the ref is never mutated (clone contract). Design/decisions in the [fit-cascade spec](superpowers/specs/2026-07-16-fit-cascade-composition-design.md).

**Live remaining work = the "Deferred to follow-up slices" list below** — multi-output fan-out, multi-input refs, and unfit-*composite* refs (all error explicitly today). They re-enter with the sklearn transformers that need them (OneHot fan-out, multi-input encoders).

## Reference forms encode fit intent (the API's key decision)
- **`{a}(col)`** — composes `a` as a *fittable* step; `a` participates in the outer's `fit_transform` cascade. **Errors if `a` is already fitted** — a bare reference to a fitted object is ambiguous (reuse its state, or re-fit it?), so force the user to be explicit. The fit-cascade path.
- **`{a.transform}(col)`** — uses `a`'s **frozen** transform; **no fitting happens** (errors if `a` is *not* fitted). The `.transform` at the call site makes "no fitting" unmissable. The frozen-reuse path.

## Frozen-path mechanics (the first slice)
Dramatically cheaper: a frozen inner's window aggregates are already `__STATE__` constants, so `{a.transform}(col)` inlines to a **plain scalar** expression (no live window function). The outer then fits + rewrites as a normal `SQLTransform` in **one flat pass** — even the outer's own aggregates over the inner output (`AVG({a.transform}(age)) OVER ()`) are legal flat SQL, because there's no nested window aggregate.
- **Arity — single-input, single-output referenced transforms only.** `{a.transform}(col) AS name` maps one input column to one output column (scaler / imputer shape). Multi-output *fan-out* and multi-input transforms are deferred (below).
- **Input remapping:** the referenced transform reads exactly one `__THIS__` column; `(age)` substitutes the outer's `age` for that input column throughout `a`'s frozen expression.
- **Inline:** substitute `a`'s frozen rewritten scalar expression for the reference, remapped to `col`.
- **State merge:** union `a`'s `__STATE__` tables into the outer's, **name-scoped** per referenced transform so they don't collide (e.g. `__STATE__@a`).

**Referenced transformers are definitions, never mutated (both forms).** The composite owns *all* fitted state; a reference is always **read-only on `a`**: `{a.transform}` reads `a`'s existing frozen state; `{a}` reads `a`'s *definition* and fits it fresh **into the composite's own name-scoped state** (`__STATE__@a`), leaving `a` untouched. This is sklearn's clone contract — `Pipeline.fit` clones each step and fits the clone, never the original — so the same `a` can be referenced by many composites without interference.

## Open (design)
- **API surface — t-string (gate RESOLVED):** Python floor is now **3.14**, so PEP 750 t-strings are native. A t-string produces a `Template` exposing literal parts and interpolations *separately*, so an embedded `SQLTransform` arrives as the **real object**, not a stringified repr — making `{scaler}(age)` a genuine structural hand-off. Residual: the concrete `SQLTransform(t"…")` constructor shape (accept a `Template`, walk its interpolations to bind each embedded transform).
- **Reference mechanism:** embed by Python object (t-string interpolation, intended) vs by name in a registry — confirm object-embedding is the surface.
- **`__STATE__` name-scoping token:** the concrete collision-safe naming for merged state tables.

## Deferred to follow-up slices (error explicitly today, not built)
- **Multi-output fan-out** referenced transforms (OneHot) — output naming/placement + column-count-from-state; unpacks via `unnest({a}(col))` on the shipped struct type.
- **Multi-input** referenced transforms — positional/named binding for `{transform}(a, b)`.
- **Unfit-*composite* references** — a `{a}(col)` where `a` is itself a composite (deeper recursion). Frozen-composite refs already work; unfit-composite is the deferred deeper case.
- **Minor, guarded** (fit-cascade review, `5ac613e`): a referenced transform whose *inner definition* has >1 distinct `PARTITION BY` set would collide on `fit_into_scope`'s single-scope state key — raised as a loud `NotImplementedError`. Unreachable for single-in/out refs; revisit when partitioned or multi-input refs land.
