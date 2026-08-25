---
id: m-9
title: "the dialect logical plan"
---

## Description

Query ⇄ Plan for DuckDB, Spark, BigQuery: author in DuckDB SQL, parse to a
bound/typed logical plan, print to the target dialect — bit-exact where
representable, ε-bounded where float accumulation forbids it, refused by
name otherwise. Spec: `packages/confit/docs/specs/2026-08-13-dialect-logical-plan-design.md`
(decisions D1–D7, laws L1–L4). Lives at `packages/confit/src/dialect/`;
`specializer/` and the JSON-AST marginalizer are untouched by this epic.

## State at milestone creation (2026-08-13)

Merged: phase-0 pins (`pins-dialect/`), plan core (Scan/Filter/Project,
verifier, canonical text), DuckDB frontend+printer (L2 gate, floor 235/678),
Spark printer + live L3 gate (floor 213), BigQuery printer (refuses all
calls), 17 scalar functions + auto-naming, CI Spark leg.

Open: the rest of the v0 node set (Join), the fit surface (Window,
Aggregate, Distinct, scalar subqueries), fuzzer invisibility routing, Spark
signature growth, the BigQuery scheduled gate, reverse frontends.

## Standing rules

- Every printer-table row is pinned by probe before it ships; unprobed = refusal.
- Corpus accounting is three-outcome (match / clean-unsupported / fail);
  gate floors only ratchet up.
- ε tolerance stays provisional until Spark-phase measurements exist;
  `quantize` is the documented opt-out.
