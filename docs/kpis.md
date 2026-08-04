# KPIs

Companion: `docs/properties.md` — the semantic laws the system holds; KPIs
measure, properties state what must remain true.

The measurable state of this project, in one place. Two kinds, optimized in
opposite directions:

- **Control KPIs** are invariants. Their target is a fixed point; they are
  never "improved", only *defended* — adversarial work (fuzz, differentials,
  refusal pins) exists to hunt counterexamples. A control that can only be
  held at 99% is not a control, it is an unacknowledged trade-off.
- **Drive KPIs** are the ones actually optimized: coverage up, latency down.
  A drive gain only counts if it lands inside every control — the pins make
  the difference impossible to smuggle past.

**Loosening a control is a design decision, never a fix.** The only
legitimate way a control moves: explicitly, in a spec/draft, with the new
bound named (precedent: matvec-tier parity got a *declared* per-family ulp
tolerance in DRAFT-23 — through review, not through a failing test).

Status date: 2026-08-03, master at "SQLProjection.infer/infer_batch through
Confit (DRAFT-22 closed)".

---

## Control KPIs (5)

### C1. Training-set round-trip

`fit(train)` + serving, applied to the training set, is **bit-exact** equal
to running the original SQL with `__THIS__` = train (both at
`SET threads = 1` — DuckDB's parallel window aggregation is not
bit-deterministic for floats).

- Enforced: `packages/sql-transform/sql_transform/_projection_test.py`
  (`gate()` across the admitted surface, plus the seeded differential fuzz —
  `MARGINALIZE_FUZZ_N`, seed 20260729; 1,500–2,000-case runs at each
  widening loop).
- Transformer columns are exempt from the *DuckDB* oracle (DuckDB cannot run
  them) and gate against C4 instead.

### C2. Engine parity

Confit serves **bit-for-bit identical to DuckDB** — with the same declared
UDFs registered via `create_function`, when there are any — **or refuses at
build with a named error**. No third behavior.

- Enforced: `packages/confit/tests/test_duckdb_*.py` (the wave suites),
  `test_params_joins.py`, `test_udfs.py` (`udf_check` — the parameterized
  form of the contract), `docs/known-limitations.md` (each refusal has an
  executable twin).
- Internal sub-invariant: cranelift ≡ interpreter, byte-for-byte —
  `packages/confit/src/specializer/exec/tests.rs` (500-seed random-IR
  differential; shared helper code is the structural argument).

### C3. Binding parity

`infer` / `infer_batch` (Confit row path) equals `transform` (DuckDB batch
path) **value-for-value** on the same fitted artifact — one artifact
(serving_sql + params tables + UDF objects), two bindings, no divergence.

- Enforced: `packages/sql-transform/sql_transform/_serving_test.py`
  (`serve_gate()` across aggregates, transformers width-1/width-k, author
  UDFs, chains, unseen-group NULLs; dict rows ≡ model rows).

### C4. Transformer parity

Transformer columns equal an **independent clone-per-group sklearn
reference** (fit and apply re-derived from scratch in the test, not via the
library code under test).

- Enforced: `packages/sql-transform/sql_transform/_transformers_test.py`
  (`_reference()`).
- Extension (DRAFT-23, when native families land): a native entry equals its
  `PythonTransform` fallback twin — bit-exact for scaler/tree tiers, within
  the *declared* per-family ulp bound for matvec tiers. The gate is
  swap-the-entry: same SQL, same statics, different udfs-list entry.

### C5. No third mode

Every query either **serves** (under C1–C4) or **refuses at construction
with an error naming the construct**. Silent wrongness — a query that
builds and quietly computes something else — is the one unrecoverable
state; the corpus's FAILED bucket is pinned empty.

- Enforced: `packages/sql-transform/sql_transform/_corpus_test.py` (three
  outcomes: MARGINALIZED / REFUSED / FAILED-must-be-empty), the refusal
  tables in every test module, `docs/known-limitations.md`.

## Drive KPIs (2)

### D1. SQL coverage ladder — current: 11/22 mined; 39/17/5 curated

How much of projection-SQL the marginalizer admits, pinned in
`packages/sql-transform/sql_transform/_corpus_test.py::test_progression_totals`
("the metric, in one place — edit these pins when a loop widens support"):

- Mined scoreboard (queries lifted verbatim from DuckDB's own window test
  suite, with provenance): **11 marginalized + 11 refused of 22, 0 failed**.
- Curated corpus: **39 marginalized / 17 refused / 5 schema-mode families**.

Progress = moving queries from REFUSED to MARGINALIZED (never to FAILED)
and growing the corpus. Known headroom, roughly in order of value: step
semantics for order-keyed windows off the training support (DRAFT-21),
static-table joins + frozen composition, IN-subqueries as fitted sets,
star bundles into transformers, typed takes (string features).

### D2. Serving latency vs baselines — measured 2026-08-04

Row-at-a-time serving cost on the wide-table scenarios in `benchmarks/`.
Two harnesses: `bench_serving.py` (pure-SQL path) and
`bench_transforms.py` (the transformer/UDF path).

**Pure SQL, p50 ns per call** (`uv run python -m benchmarks.bench_serving`):

| scenario | spec (n=1) | handcrafted python | duckdb per call |
|---|---|---|---|
| titanic | 4,000 | 6,000 | 6,224,800 |
| house_prices | 5,700 | 10,200 | 10,997,600 |
| fraud_txn | 6,100 | 10,200 | 11,685,100 |
| store_sales | 6,300 | 9,600 | 11,289,300 |

Healthy: ~1.5-1.7x faster than a handwritten Python microservice twin, and
three orders of magnitude off DuckDB-per-row (which is not a serving
engine). The interpreter backend runs ~1.3x the cranelift one; the
pre-marshaller boundary ~2.7x.

**Transformer path, p50 ns per row** (`bench_transforms.py`, same query
shape in every row so deltas are the transformer's cost alone):

| variant | n=1 | delta vs sql_only |
|---|---|---|
| sql_only (marginalized aggregates) | 1,500 | — |
| + one author `PythonUDF` | 2,100 | **+600** |
| + one fitted `StandardScaler` | 86,900 | **+85,400** |
| + two field accesses on one PCA(2) | 138,600 | +137,100 |
| + a bare width-2 PCA item | 136,700 | +135,200 |

**The finding that sets priorities:** the extern/UDF machinery is cheap
(+600ns for a plain UDF), and our own marshalling is 400ns. **93% of a
fitted transformer's per-row cost is sklearn's own `transform()`** —
measured 60,900ns for `StandardScaler.transform` on one row, versus
1,500ns for the identical arithmetic in numpy and 1,000ns in pure Python.
sklearn's per-call validation, not the boundary, is the bottleneck: one
transformer costs ~57x the entire rest of the query.

Levers, re-ordered by the measurement:

1. **Native UDF families (DRAFT-23)** — replaces the 60µs sklearn call
   with ~1µs of arithmetic. This is a ~100x lever on transformer queries
   and by far the dominant one.
2. **Single-evaluation field access (DRAFT-24 loop 4)** — k accessed
   fields currently re-run the transform k times (visible above:
   2 fields = 137µs vs 1 field = 85µs). Removes ~35% on multi-field
   queries; secondary to (1) but real.
3. Vectorized `apply_batch` for `infer_arrow`; marshaller work as measured.

---

## For a future session picking this up

1. Read this file, then `backlog/drafts/draft-22*` (the serving
   architecture, with addendum) and `draft-23*` (native families) — the
   controls' shapes are explained there.
2. To widen coverage (D1): add the query to the corpus first, watch it
   refuse, then implement until it marginalizes — and extend the C1 gate to
   the new family in the same loop. Update the pins deliberately.
3. To make serving faster (D2): measure first (`benchmarks/`), then swap
   entries behind the extern slots — C2/C3/C4 are the safety net; if an
   optimization needs a control loosened, that is a draft + review, not a
   code change.
4. Never trade a control for a drive gain. If a bar seems in the way, the
   move is a written, named tolerance — or a refusal.
