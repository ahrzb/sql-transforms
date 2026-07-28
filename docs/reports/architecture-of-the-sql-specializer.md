# The architecture of the SQL specializer: partial evaluation for row serving

**Abstract.** The SQL specializer (`DuckDBInferFn`) is a partial evaluator: it takes fixed SQL and frozen static tables at build time and emits a specialized native function whose only remaining input is the request batch. Its contract is deliberately narrow — for any SQL it either serves bit-for-bit identical to DuckDB 1.5.5 or refuses loudly at build time with a named error; there is no third mode (docs/known-limitations.md). This report describes the pipeline from token stream to native code, the IR design decisions that make whole bug classes unrepresentable rather than merely tested for, the shape contract that turns row-multiplicity into a static proof, and the boundary work that the measurements say is where serving time actually goes. It also records what was deliberately not built, and why the engine model justifies each omission. All numbers are measured (release builds, p50) on this project; the corpus arc ran 53 → 395 → 505 → 511 → 529 → 546 → 550 of 678 DuckDB-mined statements bit-exact, with zero FAILs at every point.

## 1. The problem shape

A fitted transform in this library is three things: a SQL string that never changes, a set of static tables frozen at `fit()` time (join tables, statistics), and a stream of small row batches at inference time — typically 1 to a few hundred rows per call. A per-query engine treats every call as a fresh query: parse, bind, optimize, instantiate a plan, execute, tear down. Those per-call costs are irrelevant for analytics and fatal for serving.

Measured concretely (docs/proposals/2026-07-28-columnar-path.md, titanic scenario, DuckDB handed a pre-built Arrow table each call):

| n rows/call | DuckDB | us (row path) | ratio |
|---|---|---|---|
| 1 | 6.58 ms | 3.3 µs | 2055× faster |
| 64 | 6.75 ms | 206 µs | 33× |
| 1024 | 7.75 ms | 3.42 ms | 2.3× |
| 16384–262144 | — | — | DuckDB wins 3–5× |

DuckDB pays roughly 5.5–7 ms of per-query cost on every call regardless of batch size; the specializer pays it once at build. The crossover sits around 2–3k rows/call. Below it — the serving regime — the specializer wins by one to three orders of magnitude; above it, DuckDB's parallel columnar execution wins, and that is fine, because that is not serving (§8).

The design (docs/superpowers/specs/2026-07-25-sql-specializer-design.md) frames this as the first Futamura projection: `prepare(sql, static_tables) -> f`, where binding-time analysis collapses everything static — a hash join against a frozen table becomes a probe of a prepare-time map; with no pipeline breakers left, the query is a straight-line function over baked-in data.

The correctness side of the contract is enforced by three standing mechanisms: the corpus replay of 678 statements mined from DuckDB's own test suite (tests/test_corpus_replay.py — every case must match bit-for-bit, reject cleanly at build time, or be a named, measured divergence), the executable twin of the limitations document (tests/test_known_limitations.py — lifting a documented limitation breaks a test), and a standing differential regex fuzzer (tests/test_duckdb_regexp_fuzz.py). Every wave of new SQL surface was preceded by a "pins" spec recording DuckDB's measured behavior verbatim (docs/superpowers/specs/2026-07-26-wave5-structural-pins.md, 2026-07-27-waveB-regexp-pins.md, 2026-07-28-waveA-structural-tails.md, 2026-07-28-stageB-multiplicity-pins.md, plus their pins-*/ JSON directories) — measure first, then implement against the pins.

| corpus point | statements bit-exact (of 678) |
|---|---|
| v0 subset (M-lower) | 53 |
| after the builtin/join waves (wave-5 baseline) | 395 |
| structural + regexp waves | 505, then 511 |
| wave A structural tails | 529 |
| stage B (parts 1, 2) | 546, 550 |

Zero FAILs at every point: growth only ever converted clean rejections into matches, never a wrong answer into a right one.

## 2. The pipeline

```
sql text
  │  tokenize (sqlparser 0.62, GenericDialect)
  ▼
token pre-rewrites ............ src/specializer/rewrite.rs
  ▼  parse
frontend / binder ............. src/specializer/frontend.rs
  ▼  bound relational tree + join specs (plan.rs); statics -> frozen maps
lowering (produce/consume) .... src/specializer/lower.rs
  ▼  imperative IR
VERIFIER ...................... src/specializer/ir/verify.rs   <- the airtight boundary
  ▼
backends:  interpreter (oracle) ... src/specializer/exec/interp.rs
           cranelift JIT .......... src/specializer/exec/cranelift.rs
           [columnar core: built, closed unmerged — §8]
```

### Token pre-rewrites

sqlparser 0.62 cannot represent several DuckDB surface forms, and one of them fails in the worst possible way: `SELECT k: expr` (DuckDB's prefix-alias syntax) parses **silently wrong** as a Snowflake JsonAccess under every dialect — no parse error will ever fire (src/specializer/rewrite.rs, pins-wave5/sqlparser-spike.json). Silent misparse is exactly the failure mode the contract forbids, so these forms are fixed on the token stream before sqlparser sees them: colon aliases in select-item and FROM position, `GLOB` rewritten to `LIKE` with a `__glob_pat` marker call, star name filters (`* LIKE 'pat'`, which sqlparser cannot parse at all) encoded into the one form it can parse with the real operator carried in a `\u{1}`-prefixed marker string, and paren-less `* REPLACE` items wrapped. The frontend rejects any user SQL that already contains a marker (frontend.rs), and the binder's JsonAccess rejection stays as the backstop for anything the rewrite misses. The dialect choice itself was measured, not assumed: GenericDialect is a verified superset of DuckDbDialect for the served forms, with the 678-case corpus as the regression net.

### Frontend and binder

frontend.rs (the largest single component, ~4.7k lines) binds against a deliberately small scope model: exactly one dynamic row table, a catalog of static tables, case-insensitive matching with spelling preserved. Three decisions are worth naming:

- **Opaque columns.** A row-model column whose type has no scalar lane (timestamp, list) becomes an opaque entry — position and name only — instead of a construction error. Referencing it, by any route including star expansion, raises the named rejection; `EXCLUDE` and name filters can remove it. A non-scalar column no longer blocks a scalar-only query (wave A).
- **Structs as lanes.** A `STRUCT(i INT, j INT)` column contributes flat scalar lanes for its leaf paths; the binder alone knows the tree shape — IR, lowering, and executors are untouched. A NULL struct materializes as NULL in every leaf lane at ingest, which makes DuckDB's null-propagation pins fall out for free.
- **n-part resolution.** `a.b.c` resolves longest-qualifier-first with backtracking — try `(schema.table).column`, then `(table|alias).column`, then bare column; commit to the longest prefix whose *column* binds, remaining parts become field extractions. This reproduces DuckDB's measured behavior including its trap: a table alias with the same name silently beats the struct column.

Errors are classified (`parse error:` / `unsupported:` / `bind error:` / internal), and the corpus's three-outcome contract depends on that classification being honest: `Unsupported` is only ever real SQL deliberately not served.

### Lowering

lower.rs is a produce/consume lowering to the imperative IR, built around an env-threading function builder: `CASE` and guarded CASTs split the CFG, and because values may not cross blocks except as branch arguments (§4), every live value rides the branch at each split. Kleene AND/OR lower branchless from flag algebra; CAST failure is a conditional trap block; TRY_CAST folds the failure into the null lane. Blocks re-load columns rather than threading them — pure loads, identical semantics, simpler lowering (a marked, deliberate simplification).

### The verifier

ir/verify.rs is the airtight boundary: everything upstream is allowed to assume a verified program, and nothing may execute or compile an unverified one. Its six rules, as documented in the module:

> 1. **Structure**: at least one block; entry has no params and is never a branch target; batch column names unique per side; the function name is an identifier and map statics have ≥ 1 key and ≥ 1 value column (a verified program must print to parseable canonical text).
> 2. **SSA**: every value defined exactly once function-wide; every use sees a definition earlier in the same block or a param of the same block (strict block-param form — cross-block uses are illegal, which is what lets the verifier skip dominance analysis entirely).
> 3. **Types**: every operand matches its instruction's signature; `.opt` forms are mandatory for nullable columns/statics and illegal on non-nullable ones (the null lane can be neither skipped nor invented).
> 4. **Statics**: every `@N` resolves; probe/sload match the static's kind, arity, and types.
> 5. **CFG**: all blocks reachable from entry; branch args match target params in count and type (cycles are legal since stage-B multiplicity loops).
> 6. **Stores**: no path stores a column twice, whatever its terminator (including `trap` — a double store is always a lowering bug); paths to `emit` store every column exactly once; paths to `skip` store nothing; store states must agree at joins.

The verifier is not hygiene; per the design doc it is the review substitute that let an unattended build loop make safe progress between human checkpoints. Every rule has a rejecting test, and reachability uses an iterative DFS because recursion stack-overflowed at ~8k blocks under adversarial fuzzing.

### Backends

The **interpreter** (exec/interp.rs) is the oracle: one pre-traversal of a verified program builds closures per block; execution is plain dispatch. It is never optimized and stays that way — correctness and coverage over speed. The **Cranelift JIT** (exec/cranelift.rs) maps trivially-safe ops inline to CLIF and routes everything with nontrivial semantics — checked integer arithmetic, string ops, loads/stores, statics — through `extern "C"` helpers that call the *same* functions the interpreter uses, so the two backends cannot drift where they share code; a random-IR differential guards the rest, and every prepared query is cross-checked interpreter-vs-JIT. The third backend, a columnar core, was built and deliberately not merged (§8).

## 3. The null lane: no nullable SSA type

The IR has no nullable value type (src/specializer/ir/mod.rs, the normative spec). SSA values are always bare scalars — `i1`, `i64`, `f64`, `str` — so a nullable value cannot reach an arithmetic instruction *by construction*: the type system has no way to express it. Nullability exists only at the edges. A nullable column or static is accessed through the `.opt` instruction forms, which split it into an `i1` validity flag plus a bare payload; NULL logic is then ordinary boolean algebra (`and`, `or`, `select`); `store.opt` reassembles flag and payload. On a false flag the payload is the type's default — defined, deterministic, never poison.

This is the difference between checking for three-valued-logic bugs and making them unrepresentable. There is no instruction sequence that forgets a null check, because there is no nullable value to forget it on — and rule 3 makes the `.opt` forms mandatory on nullable columns and illegal on non-nullable ones, so the null lane can be neither skipped nor invented.

## 4. Strict block-param SSA, and why the verifier needs no dominance analysis

A value may be used only in the block that defines it, after its definition, or received as a block parameter; everything flowing between blocks rides on branch arguments. A conventional SSA verifier must compute dominance to check that every use is dominated by its definition. Here the check degenerates to a per-block linear scan plus branch-argument type checking — no dominator tree, no global analysis, a materially smaller trusted core for the one component everything else assumes. The CFG was acyclic in v0; stage B relaxed exactly that, and only that (§6).

## 5. The shape contract: multiplicity as API design

`DuckDBInferFn(..., shape=...)` declares how many output rows each input row may produce, and the declaration is **proved at build time**, not checked at runtime (docs/known-limitations.md §2):

- `"filter"` (default): 0..1 rows per input row — the engine's native shape.
- `"map"`: exactly one, `out[i] ↔ in[i]`. Proved by rejecting anything that can drop a row — a WHERE clause, an INNER join (key misses drop rows), a static-only constant query. In the IR this is statically visible: `|out| == |in|` holds exactly when `skip` is unreachable.
- `"many"`: 0..N — the multiplicity opt-in. Duplicate-key joins, cross joins, inequality/constant `ON`, and self-joins build *only* under it.

The point is that multiplicity can never sneak into a serving path by default. A model serving stack that assumes row alignment gets a build-time error, not a silently misaligned batch.

## 6. Stage-B multiplicity: loops, multimaps, and the order finding

Stage B (TASK-59, docs/superpowers/specs/2026-07-28-stageB-multiplicity-pins.md) taught the engine 1:N joins under `shape='many'`, and required the only structural change the IR has had since M-ir: loops.

- `StaticTy::MultiMap` stores per-key row *lists* (flat arena, equal-key runs); `probe.range` yields a `[start, end)` index range, `probe.read` reads one row.
- The new terminator `emit.to` is emit-and-continue: the output row is complete, and control jumps back to the loop header for the next match. Illustratively:

```
b_probe:   %s, %e = probe.range @0, %k
           jump b_loop(%s)
b_loop(%i): %done = icmp.ge %i, %e
           brif %done, b_after, b_body
b_body:    %v = probe.read @0, %i
           ... residual predicate, stores ...
           %i2 = iadd %i, 1
           emit.to b_loop(%i2)        # row emitted; keep looping
```

- Self-joins use `StaticTy::BatchMap`: the batch itself becomes the build side, flattened per call before the row loop, with the whole `ON` clause as a per-pair residual — cross-then-filter, pins-proved identical to DuckDB.
- Cycles are now legal **but must terminate**: the old acyclicity rule was relaxed exactly enough for multiplicity loops. Every reachable block must reach a row-ending terminator (`emit`/`skip`/`trap` — `emit.to` continues, so it does not count); a cycle with no exit still fails verification. The store dataflow stays sound without fixpoint iteration because a back-edge's store state must match the header's known entry state (ir/verify.rs).

The central pins finding deserves its own sentence: DuckDB's join output *order* is a hash-join accident — the optimizer picks the streamed side by cost, duplicate-key matches emit in reverse build-insertion order in per-2048-row lockstep passes, and at scale the order differs run-to-run on the same connection. A row-at-a-time engine cannot reproduce it and must not try. Parity for `'many'` is therefore **multiset**, and the engine defines its own deterministic order (probe rows in input order, matches contiguous in build insertion order, LEFT null-extension in place) — a deliberate, documented divergence from behavior that SQL never promised. Stage B is interpreter-only for now; Cranelift pre-rejects into its existing fallback.

## 7. The boundary layers

The founding measurement of this project is that inference cost lives at the FFI/pydantic boundary, not in compute. TASK-57's decomposition (titanic: 10 input columns, 31 output columns):

| component | cost |
|---|---|
| whole boundary floor (trivial query, incl. 10-col ingest) | ~262 ns/row |
| output emission | ~37 ns per output column (~1.15 µs at 31 cols) |
| compute (the compiled program) | ~1.7 µs/row |
| handcrafted Python twin — everything | ~2.2 µs/row |

Two boundary layers exist, both specialized at prepare time:

**The generated row marshaller** (src/duckdb/mod.rs, `Marshaller`): everything knowable at prepare time is done at prepare time — attribute names interned once in fixed field order, input buffers and run state owned and reused (cleared, not dropped, per call), and output rows for the synthesized model built by filling pydantic v2's instance slots directly (measured on pydantic 2.13: `model_construct` 1432 ns > `model_validate` 882 ns > slot fill 491 ns per row). The slot fill is only sound for the plain synthesized model, so a user-supplied output model goes through `validate` and keeps full pydantic semantics — an adversarial-review finding, kept as a comment at the decision site.

**The Arrow boundary** (src/duckdb/arrow.rs, TASK-60): `infer_arrow(pa.Table) -> pa.Table`. Ingest walks pyarrow's raw buffers directly — address and size via the buffer API, bit-unpacked validity — into the engine's `ColData` lanes with no arrow-rs dependency; output builds one Arrow array per *column* from Rust-built buffers. Zero per-value Python objects on either side, which is precisely the ~1.4 µs/row of boxing the decomposition identified. Measured: at n ≥ 1024 it beats the row-object path on every scenario, and beats the handcrafted twin on most (house_prices 1.8× faster than the twin) — the compute core is unchanged; only the boundary moved.

## 8. What was deliberately not built

Each omission follows from the engine model, not from difficulty, and each is a named build-time rejection with an executable twin.

**Aggregation, ORDER BY, DISTINCT, LIMIT, CTEs, set operations.** The engine serves row-at-a-time feature transforms; whole-relation constructs have no per-row output shape. `FULL OUTER JOIN` is rejected on the same ground stated more sharply: it emits rows that no input row produced. ORDER BY additionally conflicts with §6's order finding — the engine defines its own order precisely because SQL without ORDER BY promises none.

**Per-row generality.** Non-constant regex patterns are rejected because regexes compile at prepare; per-row compilation is the opposite of specialization. The same logic rejects non-constant replacement strings and group indexes.

**Constructs that would parse silently wrong.** `^` (power in DuckDB, but sqlparser's precedence differs, so `2*x^y` would compute the wrong tree silently), prefix `~`, `#`, and a 20-plus-entry regex reject list assembled by the differential battery and the standing fuzzer — each entry a measured silent-wrong-answer risk (docs/known-limitations.md §4).

**The v1 columnar core.** Built to completion on its own branches and **closed unmerged, by decision**. The measurement: its kernels called the same scalar helpers per row, so it ran at row-core compute parity — the columnar layout alone bought nothing without vectorized kernels, and the regime where vectorization pays (large batches) is the regime DuckDB already wins at 3–5×. The decision (docs/proposals/2026-07-28-columnar-path.md and the post-stage-B discussion): large-batch columnar is ceded to DuckDB; the sub-5k-rows-per-call serving regime is the product, and there the boundary — not compute — was the lever, which `infer_arrow` already pulled. The work is preserved, not deleted: branches `task-61-columnar-core` / `task-61-columnar-exec` (with `benchmarks/scaling_results.json` on the former; `git fetch origin` then `git show origin/task-61-columnar-core:benchmarks/scaling_results.json`), available if a real consumer ever moves the goalposts.

The pattern across all four is the same one that runs through the whole project: decide the engine model first, measure against the oracle before implementing, and when a construct cannot be served exactly, say so at build time — loudly, by name, with a test that notices if the claim ever stops being true.
