---
id: m-8
title: "duckdb's type lattice"
---

## Description

Adopt DuckDB's type lattice in the frontend so no query is refused for its
TYPES, while row-time compute stays on the two machine widths this engine
serves from. Decided with AmirHossein 2026-08-11, out of three fuzz campaigns
(55k seeds) whose entire remaining finding family is this one gap.

**The rule:** semantic typing is unconditional — bit-for-bit means typing
exactly as DuckDB types. Compute machinery is bought per type, by what its
values can actually reach.

| type | strategy | why |
|---|---|---|
| `INT32` | erase to bounded i64 | DuckDB's INT32 is checked-never-wrapping, which is observably identical to i64 + range trap. Buys the schema face AND the trap face with no new lane. |
| `HUGEINT` / i128 | real IR lane | one new integer WIDTH on existing int instructions, not a new op set. Hard dependency: cranelift i128 legalization — verify before committing the phase. |
| `DECIMAL(p,s)` | scaled integer over i64/i128 | `(p,s)` are STATIC, so every rescale factor is a build-time constant and decimal arithmetic IS integer arithmetic. Not a second arithmetic tower — that framing was wrong and was what justified refusing. |
| `BLOB` | byte string over the existing arena | own compare/display, reuses the string machinery. |

**What this closes:** TASK-79 (int32 output schema), TASK-84's data-dependent
residual, TASK-86's refusals (bare NULL typing int32/BLOB), static-only
`sum()` emitting DECIMAL(38,0), bare decimal literals, and the folded-NULL
int32 lanes — i.e. every class the campaigns still find on the measured
surface.

**What stays refused, and is NOT a type refusal:**

```text
specialization  non-constant regex pattern    inherent to build-once/serve-many
scope           aggregation at inference      founding product decision
resource        pad count > builder budget    TASK-88; revisit once typed
```

## Phases

Each phase ends with a fuzz campaign certifying it before the next starts —
the campaign is the acceptance test, not a formality.

1. **Params-table measurement (phase 0).** What types do fitted statics
   actually carry today (fit-time `sum()` produces DECIMAL)? Decides whether
   decimal STORE+COMPARE is needed early, which is far short of arithmetic.
2. **INT32 typing + output schema.** `Ty::I32` real in the frontend
   (inference, promotion, overload rules, NULL typing, CASE unification),
   erased at lowering. Certify.
3. **INT32 bounds traps.** Trap where DuckDB traps, same named errors.
   Certify.
4. **i128 lane + exact `sum`.** Aggregation is build-time-only by design, so
   this starts as one accumulator plus a decimal128 boundary type; the lane
   lands when a row-time entry point demands it. Certify.
5. **DECIMAL scale propagation.** Measure DuckDB's per-operator result-scale
   rules first; then constant rescales over integer lanes. Certify.
6. **BLOB.** Last, demand-driven — the entry points are narrow.

## Gate before any of it

The UDF protocol's declared vocabulary (`takes` / `returns` as
`bool/int64/double/string`) grows under this design. That is user-facing API
and needs its OWN approval round with concrete before/after usage cases —
never folded in silently as part of an engine change.

Spec to be written before phase 2 starts (brainstorm → spec, per the
project's own rule); this milestone is the shape, not the spec.
