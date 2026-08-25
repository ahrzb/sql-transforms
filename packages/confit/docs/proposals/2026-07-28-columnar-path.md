# Proposal: own the columnar path

Status: **for discussion** (user + engine). Blocked work: none — this is
the candidate *next* wave after stage B. Decision owner: AmirHossein.

## Why (measured, not vibes)

TASK-57 closed with a decomposition of where serving time actually goes
(titanic scenario, 10 input cols, 31 output cols, release build, p50):

| component | cost |
|---|---|
| whole boundary floor (trivial 1-col query, incl. 10-col ingest) | ~262 ns/row |
| output emission | ~37 ns per output column → ~1.15 µs at 31 cols |
| compute (the compiled program) | ~1.7 µs/row |
| handcrafted python twin — EVERYTHING | ~2.2 µs/row |

Two conclusions. First, the input side is already near-free — a `__dict__`
fast path benched **neutral** and was deleted. Second, the remaining
overhead vs the twin is *per-Python-object work*: every output value is
boxed into a fresh PyObject, every output row into a dict/model, while the
twin pointer-copies passthrough fields. No row-at-a-time API can remove
that floor — it is the API, not the implementation.

Meanwhile the engine is **already columnar inside**: `ColData` lanes in,
`OutCol` vectors out. The row boundary is a conversion layer we bolt onto
both ends of a columnar core. The proposal is to stop converting.

## The proposal (option A, recommended)

```python
fn = DuckDBInferFn(sql, row_tables=..., static_tables=..., shape="map")

out: pa.Table = fn.infer_arrow(batch: pa.Table)   # arrow in, arrow out
```

- Input: one `pa.Table` (or `RecordBatch`) whose columns map 1:1 onto the
  engine's lanes (int64/float64/utf8/bool + validity = exactly our four
  types + null lanes). Ingest = buffer walks, zero per-value PyObjects.
- Output: arrow arrays built directly from `OutCol` vectors — one
  allocation per *column*, not per value.
- The row APIs (`infer`, `infer_rows`, pydantic models) stay untouched.
  This is an additive fast lane, not a migration.

### Why it composes with everything we just built

- **`shape="map"` is the natural contract here**: `out[i] ↔ in[i]` means
  the output table aligns positionally with the input table — the exact
  guarantee vectorized feature pipelines want. `shape="filter"`/`"many"`
  still work (output row count differs; optionally expose a
  `__row_index__` output column so callers can re-align — decision below).
- **The multi-language thesis**: the WASM spike proved one Rust build runs
  near-native in Go (wazero) and Java (chicory) with batching amortizing
  the boundary. Arrow IPC as the wire format + this columnar entry point
  is that story's missing half — the same engine serves Python, Go, and
  Java with no per-language marshalling code.

### Expected win (to be verified by bench, not promised)

Eliminates the ~1.4 µs/row of boxing/emit at titanic's width and the
per-row call machinery; the remaining cost is compute. That should put
`spec` decisively **ahead** of the handcrafted python twin at every batch
size — the twin cannot go columnar without becoming numpy-vectorized
code, which is exactly the authoring burden this library exists to remove.

## Costs and risks

- **Dependency**: arrow ingestion in Rust. Two routes: `arrow-rs` (heavy
  crate, full ecosystem) vs the Arrow **C Data Interface** via pyo3
  (lean — a few hundred lines for our four types + validity). Lean route
  recommended; we only need 4 primitive layouts + utf8 offsets.
- **Strings**: input utf8 can be read zero-copy from arrow buffers;
  output strings copy once from the arena into an arrow buffer (already
  cheaper than one PyUnicode per value).
- **Chunked tables**: accept only single-chunk input at v0 (`combine_chunks()`
  is the caller's one-liner) — named rejection otherwise.
- **API surface pre-1.0**: v0-no-compat makes this cheap to adjust later.
- **Bench honesty**: the comparison baseline for `infer_arrow` must be a
  *vectorized numpy/pandas twin*, not the per-row python twin — new bench
  rows, same three-way parity gate.

## Open questions for the discussion

1. **Entry point shape**: `infer_arrow(pa.Table) -> pa.Table`, or also
   accept/return a dict of numpy arrays? (Arrow-only keeps one code path;
   numpy is a `pa.table(...)` call away for callers.)
2. **Non-map shapes**: for `"filter"`/`"many"`, is an output-only
   `__row_index__` column the right re-alignment tool, or do we return
   `(table, indices)`?
3. **Zero-copy ambition**: v1 copies input buffers into `ColData` (simple,
   already a big win) vs true zero-copy lanes reading arrow buffers in
   place (bigger surgery in exec). Recommendation: copy first, measure,
   then decide if zero-copy is worth it.
4. **Does the columnar path also want `output_model`-style typed schemas**
   (arrow schema declaration up front), or is the derived schema enough?
5. **Multi-language sequencing**: is Python-only columnar the first ship,
   with the WASM/arrow-IPC endpoint as its own later proposal?

## Measured: us vs DuckDB-on-pyarrow (2026-07-28, titanic, p50 per call)

DuckDB gets a PRE-BUILT arrow table each call (register+execute+fetch
arrow = serving-realistic; "floor" = re-execute on an already-registered
table, its absolute best). We run today's ROW path (spec_dict) — an upper
bound for any columnar path of ours.

| n/call | duckdb | duckdb floor | us (row path, today) | us vs duckdb |
|---|---|---|---|---|
| 1 | 6.58 ms | 5.52 ms | 3.3 µs | **2055× faster** |
| 8 | 6.28 ms | 5.86 ms | 24 µs | **265×** |
| 64 | 6.75 ms | 5.94 ms | 206 µs | **33×** |
| 1024 | 7.75 ms | 6.93 ms | 3.42 ms | **2.3×** |
| 16384 | 20.4 ms | 18.7 ms | 60.5 ms | 0.3× (duckdb wins) |
| 131072 | 105 ms | 98 ms | 543 ms | 0.2× (duckdb wins) |

Reading: DuckDB pays ~5.5–7 ms of per-QUERY cost every call regardless of
batch size — we pay it once at build. The serving regime (1–1k rows/call)
is ours by 1–3 orders of magnitude ALREADY, on the row path. Today's
crossover is ~2–3k rows/call. The columnar path moves that crossover, not
the small-batch story: boundary-only (compute stays ~1.7 µs/row) puts
16k-row calls at ~29 ms vs DuckDB's 20 ms (near-tie); a vectorized
columnar CORE (2–3× on compute) would put the crossover at ~100k+
rows/call — competitive everywhere except true analytic scans, where
DuckDB's parallelism should keep the crown (and that's fine: that's not
serving).
