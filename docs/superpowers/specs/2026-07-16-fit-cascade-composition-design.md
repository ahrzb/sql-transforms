# Recursive (fit-cascade) composition — Design **[PARKED]**

> **STATUS: UNBLOCKED — parked pending go-ahead to start.** The mechanism and
> semantics below are settled. The one prerequisite (a type layer that can carry a
> struct output) is **satisfied**: the rich (recursive)
> [type system + `UNNEST`](2026-07-16-rich-type-system-design.md) first slice
> shipped to master (`4809470`), so struct/list + `unnest` now exist on the engine.
> No technical blocker remains — this spec is held only until the fit-cascade slice
> is chosen to start.
>
> **On pickup, one reconciliation:** the output-type model is now **struct +
> `UNNEST`** on the new type layer, *not* the earlier **struct + `.*`** (DataFusion
> has no `struct.*`). The "Output-type model" § below still says `.*` — left as-is
> on purpose; reconcile it to `UNNEST` when picking this up, then proceed to
> writing-plans.

**Goal:** Let an outer `SQLTransform` reference an **unfit** `SQLTransform` via
`{a}(col)`, fitting `a` into the composite during `fit` (staged, sklearn-style),
with arbitrary **nesting/chaining** (`{a}({b}(x))`). Extends the shipped frozen
path (`{a.transform}(col)`).

## Settled: fit mechanism — recursive fused cross-join fit (Approach A)

`inline_references` generalizes to walk nested placeholder calls **bottom-up**.
For each `__COMPOSE_i__(arg)`:
1. Recursively inline any placeholders inside `arg` first → `arg`'s inlined
   expression `E`, accumulating the deeper refs' scoped states.
2. **Frozen ref** (`{a.transform}`): reuse `a`'s existing state; inline its frozen
   expr over `E` (today's behavior).
3. **Unfit ref** (`{a}`): **fit `a` into a fresh scope** — remap `a`'s single
   input column → `E`, extract `a`'s window aggregates *over `E`* from the training
   data, **cross-joining the deeper scopes' states** (reusing
   `build_state_tables(join_tables=…)` + agg-over-expression). Yields `a`'s scoped
   state + its now-frozen expr over `E`; inline it.
4. **Fitted `a` referenced bare `{a}`**: error (ambiguous — use `{a.transform}`).

After all refs inline, the outer is plain → fit its own windows cross-joining all
scoped states (existing path), merge, rewrite, build `InferFn`. Fit is
topologically ordered **by the AST nesting itself**; nothing is materialized;
**inference is the same fused inline as the frozen path**.

**One new helper:** `fit_into_scope(ref, input_expr, deeper_states, ctx, training)
→ (frozen_expr, scope, scoped_state)` — `SQLTransform`'s own fit pipeline
(`find_window_aggregates` → `build_state_tables` → rewrite) applied to the ref's
*definition* with its input remapped to `input_expr` and its state name-scoped.
`inline_references` now needs the `ctx` + training table (which `fit` already has).

Chosen over the alternative (Approach C: clone each ref + materialize its input by
`transform`-ing training forward + `clone.fit()`). Both yield the **identical**
fused composite; A keeps everything in the one inline+cross-join mechanism with no
materialized intermediates. (C is the more sklearn-literal staging; recorded as a
fallback if A's fused fit proves awkward.)

## Settled: semantics

- `{a}` on **unfit** `a` → fits into the composite's own name-scoped state; **`a`
  is never mutated** (`.fit()` is not called on it — sklearn clone contract). The
  composite owns all state.
- `{a}` on **fitted** `a` → error (ambiguous: reuse or re-fit?; use
  `{a.transform}` to reuse, or a fresh unfit instance to re-fit).
- `{a.transform}` unchanged (frozen reuse); mixes freely with unfit refs
  (`{a}({b.transform}(x))`).

## Settled: state name-scoping

Reuse the frozen path's `__STATE_R{i}__` (one scope per placeholder index).
Collision-safe for nesting/chaining for free: distinct placeholders →
`__STATE_R0__`/`__STATE_R1__`/…; single-in/out refs have only global state, so no
partition-key collisions. (The backlog's `__STATE__@a` was illustrative; this is
the concrete token. Flagged for a quick PM confirm.)

## Settled: scope (this slice)

- **Single-input / single-output** referenced transforms (unchanged from the frozen
  slice). Fan-out (multi-output) + multi-input still deferred.
- **Nesting/chaining of unfit refs: yes**, arbitrary depth.
- **Unfit refs must be plain (non-composite) `SQLTransform`s** this slice; frozen
  composite refs already work. (Unfit-*composite* references = deeper recursion,
  deferred.)

## Output-type model — **DECIDED: struct + `.*` — BLOCKING**

A transformer's per-row output is its **output row** — a value with named output
fields. Decided model: **`{a}(col)` is a struct**; the user unpacks it —
`{a}(col).*` expands to the output columns in a projection, `{a}(col).field`
selects one. Single-output is a 1-field struct (or auto-unwraps to scalar — to be
resolved).

**This is not supported by the Rust engine today** and is the reason this spec is
parked. Required Rust work (its own ticket, scope below):
- a real **struct `Value`** with named fields (distinct from the opaque `Object`);
- **field projection** (`expr.field`) in eval + static type inference;
- **wildcard `.*` expansion** at the plan/rewrite layer (a struct-valued projection
  → its columns), incl. how it interacts with `SELECT`-list placement + aliasing;
- **struct-aware output-model synthesis** (`synthesize_output_model` /
  `field_type_to_python`) — nested Pydantic model or column expansion;
- **DataFusion (transform-path) parity** — DataFusion has native `STRUCT`; the Rust
  side must agree, enforced by the differential harness.

Until that lands, the *single-output* cascade could in principle ship scalar-first
(single-output is a scalar in either model), but per this decision the output type
is foundational and should be built on struct support, not retrofitted — hence the
park rather than a scalar-only partial ship.

## Testing (when unparked)

Differential parity (`transform` == `infer`) for: single `{a}(x)`, nested
`{a}({b}(x))`, mixed `{a}({b.transform}(x))`, outer-aggregate-over-cascade; the
clone contract (`a` still unfit after `composite.fit`); fitted-`{a}` errors; and —
once struct support lands — struct output + `.*` unpacking parity.

## Next

1. **Rust struct-support ticket** (scoped separately, BACKLOG) — the prerequisite.
2. Resume this spec once struct support is understood: finalize the output-type
   semantics (auto-unwrap single-output? `.*` placement rules?), then writing-plans.
