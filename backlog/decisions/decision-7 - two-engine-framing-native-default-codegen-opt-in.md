---
id: decision-7
title: 'Two-engine framing: native default, codegen opt-in (near-term)'
date: '2026-07-19'
status: accepted
---
## Context

The native `InferFn` interpreter is the current serving engine and the parity baseline
(oracle = DataFusion, [[decision-1]]). A **codegen engine** was subsequently built
(`worktree-codegen-inferfn`, `sql_transform/_codegen/`, plan
`superpowers/plans/2026-07-17-codegen-inferfn.md`, 11 tasks): suite 397 passed / 14
skipped (containers) / 3 xfailed (pinned native divergences), parity target = the
DataFusion oracle. The old "codegen / compiled inference path" was parked as *likely
won't do, revisit only with a benchmark in hand* — that condition is now satisfied (the
n=1 boundary-bound benchmark) and an engine actually exists.

## Decision

**RULED (2026-07-19, AmirHossein): the native `InferFn` is the default / maintained
serving engine; the codegen engine is OPT-IN for now.** Near-term call — revisitable if a
benchmark or serving need later pushes codegen toward default — not a permanent close.

Downstream consequences of "opt-in":
- TASK-29 (codegen deferred SQL surface) and TASK-34 (codegen transformer support) stay
  **Low / not actively prioritized**. Their "precondition: framing decided" is now
  *satisfied* (decided → opt-in → low), no longer an open blocker: codegen completeness is
  a fast-follow only for someone who opts in, not milestone work.
- The multi-language inference runtimes design ([[doc-4]]) still holds — two engines exist
  and native-as-one-of-N is intact; that epic stays parked on its own merits.
- Native remains the parity baseline (oracle = DataFusion, [[decision-1]]); an opted-in
  codegen path is still held to that oracle.

## Consequences / notes

- Codegen's adversarial review surfaced **2 codegen-only parity divergences** (native
  already matched the oracle — no native ticket): float→string for |x| < 1e-4
  (`CAST(1e-5 AS VARCHAR)` → DF `'0.00001'`, codegen `'1e-05'`); integer arithmetic
  overflow (`9223372036854775807 * 2` → DF/native wrap to `-2`, codegen Python-bigint
  `18446744073709551614`). **Both fixed & merged into codegen (`131fa0b`); TASK-4 Done.**
- If oracle parity on the codegen path is wanted, codegen also needs the shared native
  edge-case fixes (the historical `SUBSTR`/`NaN`/`CAST` set); it already fixed `ROUND(int)`
  and `COALESCE(int,float)` typing.
- Ratified **opt-in**: codegen is a maintained-but-opt-in engine, not the default; native
  is the default serving path. Revisit the default question only with a serving need or
  benchmark that argues for promoting codegen.
