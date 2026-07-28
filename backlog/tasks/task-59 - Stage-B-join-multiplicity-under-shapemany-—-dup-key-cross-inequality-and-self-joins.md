---
id: TASK-59
title: >-
  Stage B: join multiplicity under shape='many' — dup-key, cross/inequality, and
  self-joins
status: Done
assignee: []
created_date: '2026-07-28 01:55'
updated_date: '2026-07-28 03:07'
labels:
  - specializer
  - stage-b
dependencies: []
type: feature
ordinal: 53000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User go received 2026-07-28 ("feel free to implement the feature gap"), sequenced behind the TASK-58 shape flag: every stage-B construct builds ONLY under shape='many' (0..N rows out per row in) — the serving fence the user asked for.

Constituency (census, pins-waveA/census-all-nonmatches.json): 27 cases = 17 'duplicate map key' (true 1:N equi-joins vs statics with repeated keys, PLUS keyless shapes that reduce to one bucket: comma/cross joins with range residuals, inequality ON predicates) + 10 self-joins (the batch as both probe and build side; mostly EXCLUDE/USING star semantics). Target: corpus 529 -> ~556 of 678 (82%). After this ships, everything left is out-of-scope-by-decision.

Pins-first (fleet): 1:N emission ORDER and determinism, LEFT-miss null-extension under multiplicity, cross/inequality semantics, self-join star expansion + dup-name renames + USING merges, NULL keys among duplicate keys. Design per TASK-50 notes: per-key row lists in the frozen maps, inner emit loop per probe hit, bucket scan + residual predicates for keyless joins, per-call batch-side map build for self-joins. Backend decision (interpreter-first vs cranelift emit loops) goes in the spec, justified.

Hard stop after shipping: columnar-API discussion with the user before any next wave.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pins spec for multiplicity semantics (emission order/determinism, LEFT-miss, cross/inequality, self-join star forms, NULL keys among dups) with raw JSONs
- [x] #2 All stage-B constructs REJECT under shape='filter'/'map' exactly as today; they build only under shape='many'
- [x] #3 1:N equi-joins, cross/inequality joins, and self-joins serve bit-exact (order-insensitive multiset vs DuckDB) under shape='many'
- [x] #4 Corpus replay strictly above 529 with zero FAILs (replay builds stage-B cases with shape='many')
- [x] #5 Limitations doc + twin updated: stage-B rows move from 'designed, not built' to the shape='many' contract
- [x] #6 Full gates green on release build; PR opened (stacked on #44)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Emit-machinery study (pre-spec, 2026-07-28):

TODAY: both backends are strictly one-emit-per-row. Interpreter: per row, block walk ends at CTerm::Emit (emitted += 1, output values already written to st.out by instructions) or CTerm::Skip. Cranelift: compiled row_fn returns 0=emitted / 1=skip / 2+=trap — the RETURN CODE is the emit signal, so N-per-row cannot be expressed by return codes.

DESIGN DIRECTION for multiplicity (validate in spec): (1) emission becomes an explicit operation instead of a terminal side effect — either Inst::EmitRow (appends the current out-lane values) with CTerm::Emit demoted to plain end-of-row, or loop-structured blocks using the EXISTING Brif/Jump machinery (loop header + body + back-edge) with an EmitRow inst in the body. Verify/canonicalize must accept back-edges (check: today's block graph is probably a DAG — verify.rs may assume forward-only; if so that is the first thing to extend). (2) New probe ops: ProbeStart {join} -> cursor, ProbeNext {join, cursor} -> (has: i1, cursor', value lanes) over per-key row LISTS in StaticTy::Map (values become Vec-of-rows per key — flat arena + (off, len) per key is the lean layout). (3) Cranelift: emission via a helper call h_emit_row(cx) that appends to st.out (helpers already write through Cx), loop via normal CLIF blocks — feasible without new architecture; the return code stays for trap/end only. If this turns out heavy, interpreter-first for shape='many' with cranelift fallback is the documented fallback plan (bench guard exempts 'many' from backend-identity tests). (4) Self-join: per-call build of the batch-side map before the row loop (new PreparedStatic variant built in run(), or a batch-map handed via RunState); marshaller unaffected. (5) Cross/inequality: a join with ZERO key columns = single bucket = ProbeStart/Next over the whole static (or batch) + residual predicate in the loop body. (6) LEFT: emit null-extension when the loop body emitted nothing for this row (a per-row 'any_emitted' flag register). (7) shape gate: prepare learns the target shape (prepare_opaque param or Prepared metadata + duckdb/mod.rs check): multiplicity constructs produce a NAMED reject unless shape='many'; corpus replay builds stage-B-shaped cases with shape='many' (detect by retrying on the named errors, or always pass shape='many'? NO — corpus must keep proving the default rejects; retry-with-many on the named needles is the honest harness).
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage B shipped as two stacked PRs: #45 (part 1: dup-key equi-joins, cross/inequality/constant-ON joins) and #46 (part 2: self-joins via the batch as build side). Corpus 529 -> 550 match / 0 FAIL of 678 (81%); the 27-case stage-B pool is fully resolved (21 served + 3 USING-self-join named rejections + 3 rowid self-joins routed to the documented rowid descope).

Everything builds ONLY under shape='many' (TASK-58's fence): default shapes are byte-identical, dup keys and self-joins still reject. Machinery: Term::EmitTo emit-and-continue back-edges (verifier now allows terminating cycles only), StaticTy::MultiMap (stable-sorted equal-key runs; insertion order = the engine's documented 1:N emission order) + ProbeRange/ProbeRead, StaticTy::BatchMap (per-call batch flattening for self-joins, whole ON as per-pair residual — cross-then-filter, pins-proved identical), loop-structured lowering riding state on the live stack with per-block probe-cache reseeds (block splits from CASE machinery), residual-vs-WHERE gate split (residual decides match-ness and the LEFT null-extension; WHERE only gates emission). Interpreter-only; cranelift pre-rejects into the existing fallback.

Central pins finding: DuckDB's join output ORDER is a hash-join accident (LIFO chains, lockstep vector passes, run-to-run divergence at scale) — parity is MULTISET (corpus sorts; duck oracle tests sort), and the engine's own deterministic order (probe outer, insertion inner, null-extension in place) is pinned as a test. Pins: 2026-07-28-stageB-multiplicity-pins.md + pins-stageB/*.json (5 agents, 57 pins). Follow-up left open: USING/NATURAL self-joins (named rejection).
<!-- SECTION:FINAL_SUMMARY:END -->
