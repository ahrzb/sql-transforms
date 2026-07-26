---
id: TASK-43
title: >-
  Specializer M-lower: frontend + BTA + produce/consume lowering for the v0
  subset
status: In Progress
assignee: []
created_date: '2026-07-25 02:31'
updated_date: '2026-07-26 05:14'
labels: []
milestone: m-7
dependencies:
  - TASK-42
documentation:
  - docs/superpowers/specs/2026-07-25-sql-specializer-design.md
  - docs/superpowers/specs/2026-07-25-sql-specializer-loop-execution-design.md
type: feature
ordinal: 37000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The load-bearing milestone (design doc §5): sqlparser(DuckDB dialect) frontend to relational IR; binding-time analysis that taints __THIS__ and evaluates every all-static subtree in DuckDB at prepare time, materializing Const structures (scalar / dense array / perfect hash / inline); produce/consume lowering of the dynamic frontier to imperative IR. v0 subset per design doc §4. Differential oracle is DuckDB (python pkg); the mined corpus at tests/corpus/duckdb_mined.jsonl replays under the three-outcome contract (match / clean-unsupported / FAIL) documented in scripts/mine_duckdb_corpus.py. Wide-mechanical: run per-operator fan-out via workflows per the loop-execution doc §2.3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 v0-subset queries prepare end-to-end: sql + static tables -> verified imperative IR running on the interpreter backend
- [x] #2 All-static subtrees are evaluated at prepare time: a static-tables-only query lowers to a constant emitter with no probe/filter ops in its IR
- [x] #3 Differential suite vs DuckDB green on hand-written v0 cases; engine-vs-oracle disagreement follows the xfail-strict + ticket protocol
- [x] #4 Corpus replay reports match / clean-unsupported / FAIL counts; zero FAILs; every unsupported rejection is a clean build-time error naming the construct
- [x] #5 mise gate-specializer green (corpus replay wired into it)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stretch plan (~5-6 stretches estimated, recorded 2026-07-26):
1. Frontend spine: sqlparser (DuckDB dialect) -> relational IR (scan/project/filter over __THIS__), reusing shape lessons from src/datafusion; end-to-end sql -> imperative IR -> interpreter for arithmetic projections.
2. 3VL lowering: SQL NULL semantics compiled to flag algebra (comparisons, AND/OR Kleene logic, CASE, IS NULL); type + nullability derivation; the correctness core.
3. BTA + statics: taint __THIS__, equi-joins to static tables lower to probes, scalar folding; Python materialization through the DuckDBInferFn shell (pa.Table -> StaticData).
4. IR builtin extensions (workflow fan-out per op): upper/lower/trim/substr/abs/round/concat + :: casts + COALESCE/NULLIF as lowerings; each new instruction lands across verifier/printer/parser/interp with pin tests.
5.-6. Differential suite vs duckdb-python (pytest backend id "specialized") + corpus replay under the three-outcome contract; grind divergences to zero FAILs; xfail-strict + ticket for genuine oracle disagreements.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stretch 3 landed (commit 49362ee on claude/specializer-m-lower, PR #26): INNER/LEFT equi-joins to static tables lower to prologue probes (ordered JoinSpec list, no join tree nodes in v0); valid_hit = map hit AND every nullable key flag; f64 probe keys canonicalize -0.0/NaN at compile AND probe (DuckDB DOUBLE `=` pins: NaN=NaN, 0.0=-0.0); join key promotion both directions (i64 dyn promoted via IntToFloat; f64-vs-i64-col converts the build side at materialization per StaticSpec). fold.rs: conservative const folder, both-sides-const only, never drops a potentially-trapping dynamic operand (`FALSE AND a%0` keeps the rem). DuckDBInferFn replaces the stub: one row table (pydantic, ordered fields), pyarrow statics -> StaticData::Map per StaticSpec (NULL-key rows dropped, NULL value col errors, dup keys rejected at compile), output model synthesized via create_model. Differential pytest vs duckdb-python: 13 cases (joins, promotion, error contracts). Deliberate ceilings: per-block probe re-emission (pure, cached per block); all non-key cols become map values (prune later); key cols referenced outside their ON clause -> clean unsupported; supplied output_model trusted as-is.

AC #2 (static-only queries fold to a constant emitter) is NOT yet satisfied: needs DuckDB evaluation at prepare time (Python side) and a constant-emitter program shape (emits N fixed rows regardless of input) — parked for the stretch 5-6 grind. Stretch 4 started: builtin catalogue; `::` casts already landed in stretch 2 (CastKind::DoubleColon). Fan-out adaptation: per-op implement-in-worktree would conflict on the same 7 coupled files (ir/mod, verify, print, parse, interp, frontend, fold), so the workflow fans out DuckDB semantics PINNING (8 parallel agents, duckdb-python 1.5.5) and post-integration adversarial verify; implementation is integrated centrally (one integrator, many measurers).

Stretch 4 landed (commit ac3786a): builtin catalogue implemented strictly from measured pins (8-agent workflow fan-out measured DuckDB 1.5.5; spec at docs/superpowers/specs/2026-07-26-stretch4-builtin-pins.md). New IR insts across all six files + fuzz gen: supper/slower, strim.{both,lead,trail}, ssubstr (virtual-window codepoint arithmetic), iabs/fabs/fround, BinOp::Frem. Frontend catalogue: upper/lower/ltrim/rtrim/abs/round(1-arg)/concat/coalesce/nullif + TRIM/SUBSTRING dedicated AST forms; || is ALWAYS concat (even 1 || 2) with implicit VARCHAR casts; CONCAT skips NULLs (all-NULL -> ''); COALESCE per-row lazy via CASE desugار; NULLIF compares at promoted type, keeps first arg's type. TWO DIVERGENCES IN PREVIOUSLY-LANDED CODE were found by pinning and FIXED in-branch with pin tests (PM: flagging per the disagreement protocol — these were fixes toward the oracle, not semantics patches to pass tests): (1) integer % by zero returns NULL in DuckDB, we trapped -> lowering now CASE-guards the divisor, MIN % -1 still traps like DuckDB; (2) DOUBLE comparisons: DuckDB order is NaN=NaN / NaN above everything / zeros equal, we had IEEE partial -> shared exec::duck_fcmp used by interp AND fold. Known deliberate divergence (strict-xfail + needs a ticket): Rust std lacks Unicode SIMPLE case maps, so upper('ß') gives 'ß' vs DuckDB 'ẞ' and lower('İ') differs; ASCII exact. Deliberate ceilings: round(x, digits) unsupported; decimal literals stay f64 (stretch-1 ceiling). 114 cargo tests + 21 differential pytest + 1 strict xfail; gate green. Adversarial verify fan-out (6 probe agents vs duckdb) running; findings will be triaged fix-vs-xfail before stretch 5 (differential suite backend id + corpus replay in gate).

Stretch 6 landed (commit 0782bf1): AC #2 satisfied — static-only queries become constant emitters, evaluated once at build time by DuckDB itself (Python boundary; no IR is built, so trivially no probe/filter ops). The fallback fires on Unsupported/Parse prepare errors and self-validates: dynamic queries reference the row table, unknown to DuckDB, so evaluation fails and the original clean error surfaces. Bonus: aggregation/ORDER BY/dialect-beyond-sqlparser all work on the static-only path. Final corpus: 53 match / 625 clean-unsupported / 0 FAIL of 678, wired into the gate via pytest. MILESTONE COMPLETE pending review — hard stop before M-cranelift (TASK-44). Two items for PM: (1) deviation note — AC #3's 'backend id specialized' wording: the DataFusion-oracle differential harness (test_diff_*) can't host the specializer (different oracle, e.g. `/` semantics differ by design), so the specializer has its own duck_check differential suite (32 cases) instead; (2) ticket request per the disagreement protocol — Unicode SIMPLE case mapping (upper('ß')/'İ'/'ᾀ' class): Rust std only exposes full maps; strict-xfail in place; candidate fixes are a small static table of the ~100 divergent codepoints or a unicode-data crate dependency.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
M-lower delivered on branch claude/specializer-m-lower (PR #26), six stretches: (1) frontend spine sql->IR->interpreter; (2) 3VL flag algebra + CASE/CAST/IS NULL; (3) BTA + statics — equi-joins lower to map probes with canonical f64 key bits, DuckDBInferFn Python boundary, 13-case differential suite; (4) builtin catalogue measured by an 8-agent pin fan-out (upper/lower/trim/substr/abs/round/concat/coalesce/nullif, ||, float %, DuckDB DOUBLE comparison order) — 7 new IR instructions across verify/print/parse/gen/interp; (5) adversarial 6-agent probe fleet (~1,400 probes) found and fixed: NULL-divisor % trap, the trap-under-false-flag payload class bug, Zs trim set, vectorized-path substr semantics (backwards negative lengths, ±2^32 guards), DuckDB float->VARCHAR rendering; plus corpus replay wired into the gate under the three-outcome contract; (6) static-only constant emitters (AC #2). Final: gate green (cargo + 603 pytest, 13 xfail), corpus 53 match / 625 clean-unsupported / 0 FAIL of 678. Known strict-xfail: Unicode simple case maps (ticket requested). DuckDB dual-path substr inconsistency documented; oracle arrow-pushdown artifact harness-fixed.
<!-- SECTION:FINAL_SUMMARY:END -->
