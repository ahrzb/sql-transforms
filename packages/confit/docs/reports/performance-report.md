# Performance report: the serving regime, measured

**Scope.** Serving-path performance measurements of Confit (`DuckDBInferFn`): where a call's time goes, how the engine compares per call with DuckDB, and what the Arrow boundary costs and saves. Every number is a p50 from a release build on this project's benchmark harness, taken after a three-way parity gate (Confit == DuckDB == handcrafted Python) confirmed the outputs agree — a scenario that disagrees aborts the bench before any timing. Headline: for calls of 1–1,000 rows Confit is one to three orders of magnitude faster per call than invoking DuckDB, the crossover sits at a few thousand rows per call, and above that DuckDB wins by 3–5×.

## 1. The serving problem

The product is per-call latency at small batches: one request of one row, a few dozen, occasionally a thousand. Confit runs `prepare(sql, static_tables) -> f` once at fit time, then `f(batch)` per request. Large-batch throughput is DuckDB's regime.

These numbers hold under the correctness contract — *serve bit-exact with DuckDB 1.5.5 or refuse loudly at build time* (`packages/confit/docs/known-limitations.md`), enforced by the mined-corpus replay (`packages/confit/tests/test_corpus_replay.py`) with zero FAILs required.

## 2. Where a call goes: the boundary decomposition

Titanic scenario: 10 input columns, 31 output columns, n=64.

| component | measured cost | how it was isolated |
|---|---|---|
| whole boundary floor (incl. 10-col ingest) | ~262 ns/row | a trivial 1-column query |
| output emission | ~37 ns per output column (~1.15 µs at 31 cols) | 10-col star passthrough at 594 ns/row |
| compute (the compiled program) | ~1.7 µs/row | full SQL at 3,127 ns/row, minus the above |
| handcrafted Python twin — *everything* | ~2.2 µs/row | 2,188 ns/row measured |

The input side is essentially free: model-instance rows versus plain-dict rows measure 3,127 vs 3,041 ns/row, and pydantic v2 `getattr` ingest is already dict-speed (a `__dict__`-once ingest measured neutral: titanic 1.50/1.52× vs 1.52/1.47×, store_sales 1.44/1.37×). The gap to the twin is per-Python-object work on the way out: every output value boxed into a fresh PyObject and every row into a dict, where the twin pointer-copies passthrough fields. No row-at-a-time API removes that floor.

## 3. Confit vs DuckDB per call: the fixed-cost story

Titanic, p50 per call. DuckDB gets the serving-realistic path (a pre-built Arrow table each call: register + execute + fetch Arrow) and its floor (re-execute against an already-registered table). Confit runs the ordinary row path (`spec_dict`).

| n rows/call | duckdb | duckdb floor | Confit (row path) | ratio |
|---|---|---|---|---|
| 1 | 6.58 ms | 5.52 ms | 3.3 µs | 2055× faster |
| 8 | 6.28 ms | 5.86 ms | 24 µs | 265× |
| 64 | 6.75 ms | 5.94 ms | 206 µs | 33× |
| 1,024 | 7.75 ms | 6.93 ms | 3.42 ms | 2.3× |
| 16,384 | 20.4 ms | 18.7 ms | 60.5 ms | 0.3× (DuckDB wins) |
| 131,072 | 105 ms | 98 ms | 543 ms | 0.2× (DuckDB wins) |

DuckDB pays roughly 5.5–7 ms of per-query cost on titanic every call regardless of batch size (up to ~12 ms on the wider scenarios, §5) — parse, bind, plan, and the surrounding machinery. Confit pays that class of cost once, at `prepare`. The serving regime (1–1k rows/call) is Confit's by one to three orders of magnitude, the crossover on the row path is around 2–3k rows/call, and past ~16k rows DuckDB's vectorised, parallel execution wins 3–5×.

## 4. The Arrow boundary: `infer_arrow`

`fn.infer_arrow(pa.Table) -> pa.Table` exposes the engine's columnar interior (`ColData` lanes in, `OutCol` vectors out) directly, alongside the row APIs. Ingest walks raw pyarrow buffers via the Python buffer API — address + size, validity and bool bitmaps, utf8/large_utf8 offsets, non-zero slice offsets honoured — straight into `ColData`, with no `arrow-rs` dependency; output builds one `pa.Array.from_buffers` per output column from Rust-built buffers. One allocation per *column*, zero per-value PyObjects in either direction. `infer_rows == infer_arrow` on every serving scenario, NULLs and LEFT-join null-extensions under `shape='many'` included.

At n ≥ 1,024 the Arrow path beats the row-object path on every scenario (house_prices 5.31 ms → 2.80 ms per call) and beats the handcrafted twin on most — house_prices about 1.8× faster than the twin. At n = 64 the fixed pyarrow API cost per call (~150 µs) exceeds the saving, and the row path is faster there.

## 5. Scaling data

Sweep over n = 1 → 262,144, per-call p50 converted to ns/row: `row` (= `infer_rows`, dict output), `python` (the handcrafted per-row twin), `duckdb` (per-call register + execute + fetch Arrow, statics pre-materialised). Source: `benchmarks/scaling_results.json` from `scripts/bench_scaling.py`, both on the `task-61-columnar-core` branch.

Titanic:

| n | row path | python twin | duckdb |
|---|---|---|---|
| 1 | 4,300 | 2,200 | 6,871,550 |
| 64 | 3,277 | 2,273 | 103,206 |
| 1,024 | 3,434 | 2,330 | 7,168 |
| 4,096 | 3,750 | 2,611 | 2,436 |
| 16,384 | 3,817 | 2,841 | 1,184 |
| 262,144 | 4,085 | 3,241 | 700 |

House_prices:

| n | row path | python twin | duckdb |
|---|---|---|---|
| 1 | 5,200 | 4,500 | 10,449,550 |
| 64 | 4,535 | 4,366 | 165,239 |
| 1,024 | 5,305 | 5,004 | 12,019 |
| 4,096 | 5,599 | 5,640 | 3,804 |
| 16,384 | 5,792 | 5,592 | 1,815 |
| 262,144 | 5,987 | 6,505 | 1,163 |

Confit's per-row cost is flat from n=1,024 upward (~3–6 µs/row depending on scenario). DuckDB's per-row cost falls three orders of magnitude as its fixed cost amortises and its parallelism engages, crossing Confit at ~4k rows/call on titanic, fraud_txn and store_sales, and between 4k and 16k on house_prices. At the top end DuckDB wins by ~5× (titanic n=262,144: 700 vs 4,085 ns/row).

## 6. The engine against the Python twins, `a6fa318` to master

The serving table of 2026-08-04 (`a6fa318`) called the engine 1.5–1.7× faster than "a handwritten Python twin". That twin was the typed-model `python` row, since retired; the surviving `python_dict` row returns plain dicts. Measured on one Linux x86_64 machine, release builds of both commits, `bench_serving`, p50 per call at n=64 (2026-09-26):

| scenario | `a6fa318` `spec` | master `spec` (2 runs) | `a6fa318`: `spec` / `python` | `a6fa318`: `spec` / `python_dict` | master: `spec` / `python_dict` |
|---|---|---|---|---|---|
| titanic | 322–329 µs | 263 / 280 µs | 0.74 | 1.91 | 1.65–1.75 |
| house_prices | 429–443 µs | 382 / 387 µs | 0.57 | 1.39 | 1.23–1.25 |
| fraud_txn | 611–634 µs | 556 / 569 µs | 0.66 | 1.52 | 1.38–1.47 |
| store_sales | 618–634 µs | 541 / 556 µs | 0.81 | 1.83 | 1.60–1.64 |

The engine got 9–19% faster between the two commits. Against the typed-model twin it was 1.24–1.75× faster at `a6fa318`; against the plain-dict twin it was already 1.39–1.91× slower there and is 1.23–1.75× slower on master. The direction changed because the baseline changed — the typed-model twin costs 2.3–2.6× the dict twin — not because the engine regressed. (`a6fa318`'s harness measured `python_dict` but did not print it; its `--json` output carries the row.)

## 7. Methodology

All figures are p50 wall-clock per call on release builds. `benchmarks/bench_serving.py` samples under a 3-second-per-cell budget with a 30-sample minimum (so the ms-scale DuckDB rows cannot starve the sample count); `scripts/bench_scaling.py` uses a 0.5–2 s budget, 5-iteration minimum, 400 maximum. The parity gate runs before any timing.

Two harness guards, both in `bench_serving.py`: its module docstring requires a wheel rebuild before every run, because a stale installed wheel inflates only the engine rows; and `refuse_debug_build()` refuses to time a debug build, because the native test suite's `maturin develop` can shadow the release wheel with one.

**Provenance.** Corpus and correctness contract: `packages/confit/tests/test_corpus_replay.py`, `packages/confit/tests/test_known_limitations.py`, `packages/confit/docs/known-limitations.md`. Oracle: DuckDB 1.5.5 throughout.
