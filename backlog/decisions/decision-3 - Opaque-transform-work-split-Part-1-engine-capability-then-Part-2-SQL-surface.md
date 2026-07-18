---
id: decision-3
title: 'Opaque-transform work split: Part 1 engine capability then Part 2 SQL surface'
date: '2026-07-18 14:00'
status: accepted
---
## Context

The bundled opaque-transform-refs spec (`f213d6c`) mixed two layers: (1) the engine
capability to invoke an opaque fitted Python transformer, and (2) the SQL authoring
surface for it. The *surface* half started dragging real engine complexity in for
cosmetic reasons — the lowering wanted a **derived table** (to bind a struct once
and project its fields under clean names, since DataFusion won't alias `unnest`
output or do inline field access), which in the row engine means **adding a
projection node inside the `RelNode` plan tree** (today `Plan { projection, input }`
projects only at the top level). Engine surgery bought by a naming limitation in the
*other* engine.

## Decision

**Split the work in two (2026-07-17), Part 1 first.**
- **Part 1 — engine capability:** the Rust engine can invoke an opaque already-fitted
  Python transformer (marshal out → `.transform()` → marshal back). Independent of
  how it's expressed in SQL.
- **Part 2 — SQL/authoring surface:** the `{ref}` row→row model, lowering + output
  naming, DataFusion-side UDF, cross-engine parity.

## Consequences

- Part 1 shipped `4d5c85c`; Part 2 (authoring `{ref}` in t-strings, nested
  threading) shipped `6a270d4`. Both differentially green.
- Part 2 shipped by **avoiding** the derived-table wall (whole-struct passthrough,
  no field access) and **deferring** the cases that hit it —
  aggregate-over-transformer-output needs inline struct-field access DataFusion
  doesn't do. The split's thesis held.
- Follow-ups tracked in TASK-2 (Part 1) and TASK-3 (Part 2).
