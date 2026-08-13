# A function signature registry — Design (TASK-92)

**Why now (2026-08-13).** `Binder::function` is one ~1200-line
`match name.as_str()` where every arm hand-rolls arity, per-argument type
checks, bare-NULL binding, and the result type. The m-8 phase-2 width work
made the cost concrete: typing `ascii` INTEGER took edits in ~12 unrelated
arms (`substr`, slices, subscripts, pads, `repeat`, `round2`, math1/2, tree
ids, UDF args), each a `!= Ty::I64` that had silently meant "any integer".
Three shipped bug classes were signature facts hand-coded inconsistently:
TASK-82 (pad counts are INTEGER), TASK-86 (bare NULL picks the overload),
and TASK-79 itself. Dispatch runs once at prepare — everything compiles to
IR after — so this costs nothing at serve time.

## The design

What a function ACCEPTS and RETURNS becomes one declarative table; HOW its
node is built stays a small builder per arm. rustc still forces totality:
the builder match is over the same names, and a table row without a builder
(or vice versa) fails a unit test that walks both.

```rust
/// One overload of one function name.
struct Sig {
    /// Accepted type per parameter; `Repeat` marks the last as variadic.
    params: &'static [ArgTy],
    /// Result type rule.
    ret: Ret,
}

enum ArgTy {
    Exact(Ty),   // exactly this type (Str, I1)
    Int,         // any integer width; implicit upcast into the i64 lane
    Int32,       // INTEGER-or-narrower — the TASK-82 pad-count boundary
    Num,         // any int width or f64; f64 promotion inserted when mixed
}

enum Ret {
    Fixed(Ty),   // length -> I64, ascii -> I32, sin -> F64
    Arg(usize),  // abs / round / trunc / nullif -> that argument's own type
    Widen,       // integer-width promotion across the int args (xor, add)
    Unify,       // full numeric unification incl. f64 (coalesce, greatest)
}
```

```rust
// The table IS the measured width catalogue (spec 2026-08-11 + probes):
("ascii",   Sig { params: &[Exact(Str)],                    ret: Fixed(I32) }),
("length",  Sig { params: &[Exact(Str)],                    ret: Fixed(I64) }),
("lpad",    Sig { params: &[Exact(Str), Int32, Exact(Str)], ret: Fixed(Str) }),
("abs",     Sig { params: &[Num],                           ret: Arg(0)    }),
("nullif",  Sig { params: &[Num, Num],                      ret: Arg(0)    }),
```

Resolution, once, in `Binder::function`'s head:

1. bind each argument with `expr_or_null`;
2. a bare NULL adopts its parameter's declared type — TASK-86's rule as
   DATA (`nullif`'s first param says what DuckDB types it, and the repeat
   BLOB face stays a named refusal in repeat's builder until the Blob
   phase);
3. check each arg against `ArgTy`, inserting `promote_f64`/width upcasts;
4. compute the result type from `Ret`;
5. call the arm's builder with typed args + result type. Quirks stay in
   builders: the 1 GiB budget refusal, `bit_length = 8*strlen` desugar,
   `coalesce`/`greatest` lazy CASE construction (they use the same `Unify`
   helper for their type), regex constant-pattern gates.

## What does NOT migrate

- Operators (`binary`/`cmp`/`arith`) — already centralized through
  `numeric_promote`/`int_width_promote`; a second home would drift.
- CASE / COALESCE binding order and `in_guarded` — control flow, not
  signatures; they share the unification helpers only.
- The UDF path (`bind_udf_args`) — its vocabulary is the user-facing API
  (RED LINE, its own approval round; unchanged here).
- No new functions (`sign()` stays absent until its own decision), no
  behavior change of any kind.

## The gate

This refactor is behavior-IDENTICAL on its base. Acceptance:

1. full pytest from root green with ZERO test edits;
2. `cargo test` green;
3. fuzz smoke + a 5k differential campaign: no new verdict classes vs the
   same campaign on the base commit;
4. a table/builder totality unit test (every name in exactly one table row
   and one builder).

Lands as its own PR from master, BEFORE the phase-2 PR; task-79 then
rebases and re-expresses its function typing as table rows — which is the
demonstration that the registry does its job.

## Migration mechanics

An audit fleet (workflow) extracts per-arm facts from the CURRENT code —
names/aliases, arity, accepted types per arg, bare-NULL behavior, result
type, quirks — and a DuckDB prober re-verifies each row live. The table is
written from that audit; the single-file rewrite happens inline (parallel
agents editing one file would only merge-conflict).
