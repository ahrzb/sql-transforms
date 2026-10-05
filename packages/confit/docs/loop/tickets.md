# Ticket board

The live split of open work into tickets (see [`README.md`](README.md) §2).
Each row is one branch and one PR. A ticket's full text below is what the
worker receives after [`worker-brief.md`](worker-brief.md). Remove a ticket
when its PR merges, and move what it learned into `PLANS.md`.

| id | ticket | branch | depends on | overlaps | worker session | PR | state |
|---|---|---|---|---|---|---|---|
| T1 | compute a repeated pure subexpression once (Normalizer) | `claude/cse-call-body` | #341 | `plan.rs` (can_trap), `lower.rs` | | | ready |
| T2 | refuse an oversized program before the expensive compile | `claude/early-size-refusal` | — | `exec/cranelift.rs`, `duckdb/mod.rs` | | | ready |
| T3 | UBIGINT and HUGEINT on a 128-bit lane | `claude/i128-hugeint` | #341 | `frontend/typing.rs`, `ir`, both backends | | | ready |

Next up, once a slot frees: struct-valued outputs (ruled class 3), then
superlinear build time of long AND/OR chains (20,000 terms: about 23 s).

---

## Your ticket: T1 — compute a repeated pure subexpression once (branch `claude/cse-call-body`)

The native catalog's Normalizer is a struct SqlFunction whose lane j is `x_j / g(norm(x_0..x_{n-1}))`: the SAME row-norm subexpression appears verbatim in all n lanes, so the bound body is O(n²). Measured on master (through `sql_transform.native.to_native`, which reads every lane): Normalizer l1/l2 at 32 features hits Cranelift's size limit ("compiled query is past the code generator's size limit") after 8 s (l1) / 19 s (l2); l1 at 48 after 36 s; the max norm is capped at 8 features for the same reason. Reading every lane once per call is already linear (`frontend/calls.rs`, PR #339); the remaining blow-up is the repeated subexpression itself.

Goal: identical PURE, trap-free subexpressions evaluated once per row, so a Normalizer-shaped body is O(n) to build and serve at any width.

Pointers:
- `src/specializer/plan.rs` (SExpr/SKind, `can_trap`, `trap_skeleton`, `bind_foldable`), `src/specializer/lower.rs` (SExpr -> IR, `emit`), `src/specializer/ir/` (IR, `canonicalize`), `src/specializer/frontend/calls.rs`.
- SExpr derives PartialEq (structural); there is no hashing yet.
- Two plausible designs; pick by measurement, say why in the PR: (a) CSE during lowering (a memo from structurally-equal pure SExpr to its emitted lane, scoped to the block/dominance region where reuse is valid), or (b) an IR-level CSE/GVN pass. Mind the CASE/branch structure: a value computed in one arm is not available in another; and NEVER share or hoist anything that can trap (`can_trap`) or is impure (extern calls with side effects, trees) — moving a trap changes which rows trap.
- Reproduce without the catalog: a struct SqlFunction with n lanes `x_j / sqrt(x_0*x_0 + ... + x_{n-1}*x_{n-1})` (and an l1 variant with abs, and a max variant with greatest), every lane read by field, n = 16/32/64/128. Record build time and serving time per 64-row infer_arrow before/after.

Acceptance: those shapes build in roughly linear time (state the numbers), parity tests for them (assert_parity with udfs=[fn]) including NULL and NaN inputs, no change in trap behaviour (add a test where a shared subexpression sits next to a trapping one), gate green, 10k+ campaign clean.

## Your ticket: T2 — refuse an oversized program before the expensive compile (branch `claude/early-size-refusal`)

When a query is too large for Cranelift, confit now refuses by name ("unsupported: the compiled query is past the code generator's size limit ..."; `CompileError::TooLarge`, `src/specializer/exec/cranelift.rs::define_error`), but only after Cranelift has spent its time: the native catalog measured 10 s (Normalizer l1, 32 features) and 26 s (l2) before the refusal arrives. Its fallback (serve in Python) would rather get that answer in well under a second.

Goal: decide early, cheaply, and conservatively that a program will not fit, and refuse with the same named message; programs that DO fit must not be refused.

Pointers:
- Cranelift raises CodeTooLarge when the function needs more than `VReg::MAX` (2^21) virtual registers (`cranelift-codegen-0.126/src/machinst/vcode.rs`), i.e. roughly when the lowered IR has too many values.
- The IR program is available before codegen (`src/specializer/ir/`, the program `lower` produces; `src/duckdb/mod.rs` calls `cranelift::compile_ext`). Measure the relation between IR size (instructions / values / blocks) and Cranelift vreg usage on real large programs (the catalog-like shape: a struct SqlFunction with many lanes all read by field; see tests/test_sql_functions.py::test_a_wide_struct_function_builds_once_per_call for one way to generate big programs), and pick a threshold with a clear safety margin. Also time where the 10–26 s goes (it may be confit's own passes, e.g. verify or the interpreter compile that runs first) and say so.
- Prefer refusing on a bound you can justify; if no bound is safe, an alternative is a budget on the interpreter-first compile, or compiling in a way that fails fast — measure before choosing.

Acceptance: a program that previously refused after >5 s refuses in <1 s with the same named message; a test pins it (with a generous time bound); no program that compiled before is refused now (run the gate and a 10k+ campaign; report max program size seen); numbers in the PR.

## Your ticket: T3 — UBIGINT and HUGEINT on a 128-bit lane (branch `claude/i128-hugeint`)

The ruled query class "HUGEINT / unsigned (i128 lane)" (packages/confit/PLANS.md, docs/decisions/closed/next-query-classes.md). Slice 1 (PR #341, merged before you start) served UTINYINT/USMALLINT/UINTEGER as narrow widths of the i64 lane: read it first (`Ty::U8/U16/U32`, `int_width_promote` / `int_family_promote` in `frontend/typing.rs`, `tests/test_unsigned.py`, the probes it cites). Your slice: UBIGINT (u64) and HUGEINT (i128), which do not fit the i64 lane. UHUGEINT may stay refused by name if it costs much more; say so.

What exists: DECIMAL already has an i128 representation end to end (`Ty::Dec`, `ColData::Dec`, `KeyBits::Dec`, i128 lanes in both backends — interp and Cranelift `h_load_dec`/iconcat, kernels in `exec/kernels.rs`); reuse that machinery rather than inventing a second one. DuckDB exports HUGEINT to Arrow as decimal128(38,0) — measure what it does for UBIGINT (uint64) and for values beyond 38 digits, and what pyarrow/infer_arrow must emit.

Measure first (Oracle, optimizer off) and model exactly: result types of every operator and of CASE/COALESCE/greatest between HUGEINT/UBIGINT and every other integer width (extend the tables slice 1 measured), overflow traps at the type's bounds, literals (9223372036854775808 is HUGEINT; `-9223372036854775808` is BIGINT; today the bare one refuses by name in `frontend/expr.rs`), casts to/from every type (including VARCHAR parsing and DOUBLE rounding), comparisons, division/modulo by zero, `abs`/`round`/`sign`-style functions, join keys, Arrow in/out and the Python row boundary. What DuckDB computes in a way you cannot reproduce exactly refuses by name.

Generator: extend slice 1's seed gate (`_unsigned` in fuzz/gen.py, seeds 6 mod 11) or add a separate gate, so the campaign reaches these widths; add the storage names to `fuzz/oracle.py::_ARROW` if needed.

Acceptance: parity tests (new file, e.g. tests/test_hugeint.py), named refusals pinned, old refusal pins updated, generator reach with a gated-seed run reported (counts by verdict), gate green, 10k+ campaign clean, PLANS.md and known-limitations.md updated.
