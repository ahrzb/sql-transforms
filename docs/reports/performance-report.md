# Performance report: the serving regime, measured

**Scope.** This report collects the serving-path performance measurements of the SQL specializer (`DuckDBInferFn`) as of 2026-07-28: where a call's time actually goes, how the engine compares per call with DuckDB itself, what the Arrow boundary bought, and why the columnar execution core was built, measured, and then deliberately closed unmerged. Every number is a p50 from a release build on this project's benchmark harness (`benchmarks/bench_serving.py`, `scripts/bench_scaling.py` on branch `task-61-columnar-core`), taken after a three-way parity gate (specializer == DuckDB == handcrafted Python) confirmed the outputs agree — a scenario that disagrees aborts the bench before any timing. The headline, stated plainly: for calls of 1–1,000 rows the specializer is one to three orders of magnitude faster per call than invoking DuckDB, the crossover sits at a few thousand rows per call, and above that DuckDB wins by 3–5× — a boundary we have measured and chosen not to contest.

## 1. The serving problem

The product is per-call latency at small batches. An inference service receives one request — one row, or a few dozen, occasionally a thousand — and must answer now. Throughput over a quarter-million-row table is an analytics problem; DuckDB already solves it well. The specializer exists for the other regime: `prepare(sql, static_tables) -> f` once at fit time (seconds are fine), then `f(batch)` millions of times with a nanosecond budget (`docs/superpowers/specs/2026-07-25-sql-specializer-design.md`).

The performance numbers below are only meaningful against the correctness backdrop, so it is stated once: the engine's contract is *serve bit-exact with DuckDB 1.5.5 or refuse loudly at build time* (`docs/known-limitations.md`). The mined-corpus replay (`packages/confit/tests/test_corpus_replay.py`) enforces a three-outcome contract — match, clean-unsupported, or FAIL — with zero FAILs required. The match ladder over the waves ran 53 → 395 → 505 → 511 → 529 → 546 → 550 of 678 statements, zero FAILs throughout (wave census anchors in `docs/superpowers/specs/2026-07-26-wave5-structural-pins.md`, `2026-07-27-waveB-regexp-pins.md`, `2026-07-28-waveA-structural-tails.md`, `2026-07-28-stageB-multiplicity-pins.md`; current 550/678 recorded in `docs/known-limitations.md`). Nothing below was bought by relaxing that.

## 2. Where a call goes: the boundary decomposition (TASK-57)

TASK-57 (`backlog/tasks/task-57 - Input-marshalling-perf-lever-close-the-spec_dict-vs-python_dict-gap-on-the-ingest-side.md`) set out to close a 1.1–1.4× gap against the handcrafted Python twin and closed instead as a measurement result, shipping no code. The decomposition it produced (titanic scenario: 10 input columns, 31 output columns, n=64, release build, p50; also summarised in `docs/proposals/2026-07-28-columnar-path.md`):

| component | measured cost | how it was isolated |
|---|---|---|
| whole boundary floor (incl. 10-col ingest) | ~262 ns/row | a trivial 1-column query |
| output emission | ~37 ns per output column (~1.15 µs at 31 cols) | 10-col star passthrough at 594 ns/row |
| compute (the compiled program) | ~1.7 µs/row | full SQL at 3,127 ns/row, minus the above |
| handcrafted Python twin — *everything* | ~2.2 µs/row | 2,188 ns/row measured |

Two findings matter. First, the input side is essentially free already: model-instance rows versus plain-dict rows measured 3,127 vs 3,041 ns/row — ingest mode is irrelevant. Second, the plausible optimisation was falsified before it could ship. The candidate lever — for rows that are exact instances of the registered model, fetch `__dict__` once and do k dict lookups instead of k `getattr` calls — was built, benched, and came out **neutral** (titanic 1.50/1.52× before, 1.52/1.47× after; store_sales 1.44/1.37× — noise). pydantic v2 `getattr` is already dict-speed. The code was deleted. This is the bench-first discipline working as intended: a reasonable-sounding idea, an afternoon of implementation, and a measurement that killed it before it became permanent complexity.

What remains between us and the twin is not ingest but per-Python-object work on the way out: every output value boxed into a fresh PyObject, every row into a dict or model, while the twin pointer-copies passthrough fields. As `docs/proposals/2026-07-28-columnar-path.md` puts it, no row-at-a-time API can remove that floor — it *is* the API.

## 3. Us vs DuckDB per call: the fixed-cost story

The obvious question — why not just call DuckDB? — has a measured answer (`docs/proposals/2026-07-28-columnar-path.md`, 2026-07-28, titanic, p50 per call). DuckDB was given the serving-realistic path (a pre-built Arrow table each call: register + execute + fetch Arrow) and also its absolute floor (re-execute against an already-registered table). We ran the ordinary row path (`spec_dict`) — an upper bound for any columnar path of ours.

| n rows/call | duckdb | duckdb floor | us (row path) | ratio |
|---|---|---|---|---|
| 1 | 6.58 ms | 5.52 ms | 3.3 µs | 2055× faster |
| 8 | 6.28 ms | 5.86 ms | 24 µs | 265× |
| 64 | 6.75 ms | 5.94 ms | 206 µs | 33× |
| 1,024 | 7.75 ms | 6.93 ms | 3.42 ms | 2.3× |
| 16,384 | 20.4 ms | 18.7 ms | 60.5 ms | 0.3× (DuckDB wins) |
| 131,072 | 105 ms | 98 ms | 543 ms | 0.2× (DuckDB wins) |

The structure of the table is the point. DuckDB pays roughly 5.5–7 ms of per-query cost on titanic *every call* regardless of batch size (up to ~12 ms on the wider scenarios in the scaling sweep, section 6) — parse, bind, plan, and the surrounding machinery. We pay that class of cost exactly once, at `prepare`. So the serving regime (1–1k rows/call) is ours by one to three orders of magnitude, the crossover on this row path is around 2–3k rows/call, and past ~16k rows DuckDB's vectorised, parallel execution wins 3–5×. Neither side of the crossover is embarrassing to the other; they are different products.

## 4. The Arrow boundary: `infer_arrow` (TASK-60)

Section 2 established that the residual overhead is per-value PyObject traffic. The proposal's response (`docs/proposals/2026-07-28-columnar-path.md`) was to stop converting: the engine is already columnar inside (`ColData` lanes in, `OutCol` vectors out), so expose that directly. TASK-60 shipped `fn.infer_arrow(pa.Table) -> pa.Table` as an additive lane — the row APIs are untouched.

The mechanism, per the TASK-60 final summary (`backlog/tasks/task-60 - Pyarrow-input-output-infer_arrowpa.Table-pa.Table-—-the-columnar-boundary.md`; merged as PR #47): ingest walks raw pyarrow buffers via the Python buffer API — address + size, validity and bool bitmaps, utf8/large_utf8 offsets, non-zero slice offsets honoured — straight into `ColData`, with no `arrow-rs` dependency; output builds one `pa.Array.from_buffers` per output column from Rust-built buffers. One allocation per *column*, zero per-value PyObjects in either direction. Parity is by construction and by test: `infer_rows == infer_arrow` on every serving scenario, NULLs round-tripping, LEFT-join null-extensions under `shape='many'` included.

Measured results from that summary: at n ≥ 1,024 the Arrow lane beats the row-object path on every scenario (house_prices 5.31 ms → 2.80 ms per call) and beats the handcrafted twin on most of them — house_prices about 1.8× faster than the twin. The honest caveat: at n = 64 the fixed pyarrow API cost per call (~150 µs) exceeds the saving, and the row path remains the right choice there. `infer_arrow` is a large-batch lane, not a replacement.

## 5. The columnar core: built, measured, closed unmerged (TASK-61, PR #48)

The final roadmap item asked the natural next question: with the boundary columnar, would a columnar *execution core* — mask-vectorised, column-at-a-time kernels over the same IR (`packages/confit/src/specializer/ir/mod.rs`) — beat the row core on compute? The core was built properly (`exec/columnar.rs` on `origin/task-61-columnar-core`, +1,942 lines: all 37 instruction kernels, exact trap row-order parity, a 500-seed interpreter differential, all gates green) and then measured.

The verdict (TASK-61 final summary and PR #48 body): the v1 core computes at **row-core parity**, because its kernels call the same scalar helpers once per row — vectorised control flow around scalar work. Against the row path across the sweep: house_prices ~1.35× better, titanic and fraud_txn approximately ties, store_sales behind, plus a per-call lane-allocation cost that actively hurts tiny batches. Defaulting it on would have regressed the product regime, so it was wired opt-in (`SPECIALIZER_COLUMNAR=1`) — and then PR #48 was closed unmerged. The closing comment, verbatim:

> "Closed by decision, not by defect: the measured v1 core sits at row-core compute parity, and columnar execution above the serving regime is better served by DuckDB — we're strong before ~5k rows/call and that's the battle we're choosing."

This is the right engineering call, and worth spelling out. Making the columnar core actually fast would mean true vectorised kernels — SIMD string ops, branch-free lanes, lane reuse — which is competing with DuckDB at the thing DuckDB is best in the world at, in the regime (large analytic batches) that is explicitly not the product. Meanwhile the sub-5k regime is already won by orders of magnitude, and if the mid-band (1k–16k) ever matters commercially, the identified cheaper lever is parallelism over rows (rayon over the existing, already-correct row core) rather than a second execution engine — parallelism composes with the row core for free, where SIMD would have required maintaining and re-verifying a parallel kernel set. The work is preserved, not discarded: branches `task-61-columnar-core` and `task-61-columnar-exec` hold the complete, gates-green implementation should the calculus change.

## 6. The scaling data

The sweep behind that decision is `benchmarks/scaling_results.json` on the preserved branch (reproduce with `git fetch origin` then `git show origin/task-61-columnar-core:benchmarks/scaling_results.json`; produced by `scripts/bench_scaling.py`, same branch). Four engines — `columnar` (= `infer_arrow`, using the v1 columnar core where it compiled), `row` (= `infer_rows`, dict output), `python` (the handcrafted per-row twin), `duckdb` (per-call register + execute + fetch Arrow, statics pre-materialised) — across four scenarios and n = 1 → 262,144. Titanic, converted to ns/row (derived here from the JSON's per-call p50s):

| n | infer_arrow | row path | python twin | duckdb |
|---|---|---|---|---|
| 1 | 189,200 | 4,300 | 2,200 | 6,871,550 |
| 64 | 6,017 | 3,277 | 2,273 | 103,206 |
| 1,024 | 3,089 | 3,434 | 2,330 | 7,168 |
| 4,096 | 3,537 | 3,750 | 2,611 | 2,436 |
| 16,384 | 3,854 | 3,817 | 2,841 | 1,184 |
| 262,144 | 3,847 | 4,085 | 3,241 | 700 |

And house_prices, the scenario where the columnar lane does best:

| n | infer_arrow | row path | python twin | duckdb |
|---|---|---|---|---|
| 1 | 364,800 | 5,200 | 4,500 | 10,449,550 |
| 64 | 9,698 | 4,535 | 4,366 | 165,239 |
| 1,024 | 3,706 | 5,305 | 5,004 | 12,019 |
| 4,096 | 3,560 | 5,599 | 5,640 | 3,804 |
| 16,384 | 4,379 | 5,792 | 5,592 | 1,815 |
| 262,144 | 4,720 | 5,987 | 6,505 | 1,163 |

Readings, all visible in the raw JSON. Our per-row cost is essentially flat from n=1,024 upward (~3–5 µs/row depending on scenario) — the specializer does not get faster per row with scale, because it was never built to. DuckDB's per-row cost falls three orders of magnitude as its fixed cost amortises and its parallelism engages, crossing us at ~4k rows/call on titanic, fraud_txn and store_sales, and between 4k and 16k on house_prices (per call at 4,096: house 14.58 ms us vs 15.58 ms DuckDB — still ours; at 16,384: 71.8 ms vs 29.7 ms — DuckDB's). At the top end DuckDB wins by ~5× (titanic n=262,144: 700 vs 3,847 ns/row). The Arrow lane's fixed cost is stark at n=1 (189–445 µs across scenarios against a 4–10 µs row-path call) and fully amortised by ~1k, where it beats the twin on house_prices and titanic and trails it modestly on the string-heavy scenarios. One run-to-run honesty note: house_prices n=1,024 measured 2.80 ms per call in the TASK-60 bench and 3.79 ms in this sweep — different runs on different days with the columnar core active in the latter; treat gaps of that size between editions as within run variance, not as findings.

## 7. Methodology

All figures are p50 wall-clock per call on release builds. `benchmarks/bench_serving.py` samples under a 3-second-per-cell budget with a 30-sample minimum (so the ms-scale DuckDB rows cannot starve the sample count); `scripts/bench_scaling.py` uses a 0.5–2 s budget, 5-iteration minimum, 400 maximum. The parity gate runs before any timing.

Two incidents shaped the harness and deserve recording. First, the stale-wheel incident (documented in the `bench_serving.py` module docstring): a bench run against a stale installed wheel inflates *only* the engine rows and once manufactured a phantom 7× regression, caught by bisection on 2026-07-26 — the docstring now instructs a wheel rebuild before every run. Second, the debug-build guard: the native test suite's `maturin develop` can shadow the release wheel with a debug build, so the bench now detects and refuses debug builds automatically before timing anything. Both are the same lesson as the deleted `__dict__` fast path, applied to the harness itself: the measurement apparatus is as capable of lying as the hypothesis, and gets the same treatment.

**Provenance summary.** Boundary decomposition and the falsified ingest lever: `backlog/tasks/task-57 - ....md` (implementation notes) and `docs/proposals/2026-07-28-columnar-path.md`. DuckDB-per-call comparison: the measured table in the same proposal. Arrow boundary mechanism and results: TASK-60 final summary (`git show origin/task-61-columnar-core:"backlog/tasks/task-60 - ....md"`). Columnar-core verdict and closure: TASK-61 final summary and PR #48 (closed, comment quoted above). Scaling sweep: `origin/task-61-columnar-core:benchmarks/scaling_results.json`. Corpus and correctness contract: `packages/confit/tests/test_corpus_replay.py`, `packages/confit/tests/test_known_limitations.py`, `docs/known-limitations.md`. Oracle: DuckDB 1.5.5 throughout.
