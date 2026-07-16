# Recursive (fit-cascade) composition — Design

> **STATUS: design complete — ready for writing-plans.** Prerequisite satisfied
> (the rich recursive [type system + `UNNEST`](2026-07-16-rich-type-system-design.md)
> shipped to master, `4809470`). Output-type model reconciled to **struct +
> `UNNEST`** (§ below). **Scope (decided 2026-07-16): single-input / single-output
> this slice**; multi-output fan-out is designed here for coherence but deferred to
> the sklearn slice that needs it (OneHot).

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

**Provenance — a build constraint on this inline pipeline (readiness, not the
feature).** Keep *all* inlining centralized in `inline_references` (one choke point)
so origin tags — referenced transformer / ref scope (`__STATE_R{i}__`) + authored
SQL span — can later thread through a single place, targeting a rendered failure
like *"div-by-zero in `{scaler}` applied to `age`, from `x / STDDEV(x) OVER ()`."*
Full **runtime** attribution is the separate error-attribution BACKLOG item, and it
needs Rust work: the composite's rewritten SQL reaches `InferFn` as a **string**, so
a build-time tag on the sqlglot AST does not survive to the interpreter — the tag
must be propagated *through* the Rust engine and back out on error. This slice only
owes the **centralization** (cheap now; scattering inlining would make the later
threading far harder), not the tagging or the render.

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

## Output-type model — struct + `UNNEST` (single-output auto-unwraps)

A transformer's per-row output is its **output row**. On the shipped rich-type
layer the model is:
- **Single-output `{a}(col)` → a scalar** (a 1-field output row, auto-unwrapped) —
  usable directly in expressions (`{scaler}(age) / 2`, `AVG({scaler}(age)) OVER ()`).
  This is exactly what the shipped frozen path (`{a.transform}(col)`) already does,
  so the two paths stay consistent. **This slice implements this case.**
- **Multi-output `{a}(col)` → a `struct`** (the N-field output row); expand it to
  columns with **`unnest({a}(col))`** — the shipped `unnest(struct)`→columns. Using
  a multi-output ref in a scalar position (arithmetic) errors, like any struct.
  **Designed here for coherence; deferred** with fan-out (below), landing with the
  sklearn transformer that needs it (OneHot).

This supersedes the earlier **struct + `.*`** framing: DataFusion has no `struct.*`;
`UNNEST` is its expansion mechanism, already matched by the engine. **No new Rust
work** — single-output is a scalar (works today via the frozen-path machinery), and
multi-output uses struct + `unnest`, both already shipped.

## Testing

Differential parity (`transform` == `infer`) for: single `{a}(x)`, nested
`{a}({b}(x))`, mixed `{a}({b.transform}(x))`, outer-aggregate-over-cascade; the
clone contract (`a` still unfit after `composite.fit`); fitted-`{a}` and
unfit-`{a.transform}` errors; and **provenance rendered on a forced failure inside a
nested ref** (locks the error-attribution threading). (Multi-output struct +
`unnest` unpacking is deferred with fan-out.)

## Next

Design complete → **writing-plans**. Task sequence: the recursive `inline_references`
fit pipeline (`fit_into_scope` helper, with provenance threading) → nesting/chaining
→ outer-aggregate-over-cascade → the error/clone-contract tests.
