# Differential fuzzer for confit — Design

**Goal:** a seeded, stdlib-only fuzzer that generates random SQL + random Arrow
data over the full claimed confit surface, runs each case through DuckDB (the
oracle) and both confit backends, and reports any case where the contract —
*match DuckDB bit-for-bit or refuse by name, no third mode* — is broken.

Approved by AmirHossein 2026-08-11 ("Perfect", after the full-surface revision).

## Where it lives

```
packages/confit/fuzz/
  __init__.py
  gen.py       # seeded Case generator: tiny expression/query AST -> SQL text
  oracle.py    # one Case in, one Verdict out
  runner.py    # campaign: worker subprocesses, timeouts, findings.jsonl, stats
  shrink.py    # greedy AST minimizer; emits a ready-to-paste pin snippet
packages/confit/tests/test_fuzz_smoke.py   # fixed-seed CI smoke (~200 cases)
```

Campaigns run as `uv run --directory packages/confit python -m fuzz.runner
--seed 0 --n 20000 --workers 8`. No new dependencies: `random.Random(seed)`
gives reproducibility, the repro for any finding is its seed.

## Generator (`gen.py`)

A `Case` is: a row schema (bool/int64/double/string × nullable), 0–8 rows,
0–2 static tables, 0–2 declared UDFs (deterministic pure Python; scalar /
struct / fixed-list returns), occasionally a fitted sklearn tree ensemble, an
API choice (`shape=`, `output=`, `output_model=`), and a query built as a
small AST (dataclass nodes), rendered to SQL only at the edge. The AST — not
the SQL string — is what the shrinker edits.

Grammar productions and their intent:

- **Expression spine** — arithmetic, comparisons, AND/OR/NOT, CASE, CAST,
  BETWEEN / IN / IS [NOT] NULL / LIKE, COALESCE/NULLIF. The bulk of the weight.
- **Builtin catalogue** — signatures for the common ~30, plus a low-weight
  wildcard production over the full `BUILTIN_NAMES` list (92) with random
  args: wrong shapes refuse, right ones get differential coverage.
- **Query shapes** — projection+WHERE spine; `WITH` chains and nested SELECT
  in FROM; case-varied CTE/identifier names (the TASK-76 class); `*`,
  `* EXCLUDE`, `* REPLACE`, `COLUMNS('re')`; chained INNER/LEFT equi-joins to
  statics with ON residuals containing CASE/AND/OR (the TASK-73/74/75 class);
  GROUP BY + aggregates over static tables; struct_pack, wide-UDF projections,
  `f(x).a` lane reads.
- **Refused constructs on purpose** — QUALIFY, FETCH FIRST, TOP, OVER, FILTER,
  IGNORE NULLS, DISTINCT, ORDER BY/LIMIT. A loud refusal is a pass; building
  one is the TASK-69 silent-drop class and surfaces as a divergence.
- **Identifier/literal fuzz** — quoted unicode/space/case-colliding names,
  `__param_`-adjacent names (the P8 reserved-prefix finding), out-of-range
  integer literals, long float literals, escaped strings.
- **Boundary-biased data** — NULL-heavy (~30%), −0.0, NaN, ±inf, ±2^53±1,
  int64 min/max, quantised floats (the sklearn float32 lesson: continuous
  random data passes parity by luck), empty/unicode/quote-bearing strings.

## Oracle (`oracle.py`)

Verdicts, in decreasing order of interest:

```
DIVERGE_VALUE   outputs differ (bit-for-bit, schema included)
DIVERGE_BUILD   DuckDB errors on the SQL but confit builds — or the reverse
DIVERGE_TRAP    one side traps at infer time, the other returns rows
PANIC / TIMEOUT worker died or hung (the TASK-73 unbounded-recursion class)
AGREE           duckdb == confit(cranelift) == confit(interpreter)
AGREE_TRAP      both sides error at run time
REFUSED         confit refused by name (contract-permitted; counted per class)
```

Checks per case beyond the three-way engine comparison:

- `infer` (pydantic rows) vs `infer_arrow` (Arrow table) must agree with each
  other and with DuckDB — the `large_string` / int-width class lives at this
  boundary.
- Hostile Arrow input: the same table sliced (non-zero offset), multi-chunk,
  and empty — the TASK-67 unaligned-buffer class.
- Batch/single-row invariance: `fn([r0..rn])[i] == fn([ri])` — catches
  cross-row state leaks without needing DuckDB.
- Rebuild determinism: constructing the fn twice gives identical output.
- Tree cases add sklearn `predict` as a second ground truth.
- `fn.backend` is recorded; a cranelift request that silently fell back to the
  interpreter is counted (the fallback discards its compile error by design).

Known open divergences (today: TASK-79, int32 vs int64 on integer-literal
schemas) are matched by pattern, tagged with their ticket, and counted
separately — they must not drown new findings.

DuckDB setup mirrors `duck_check`: statics and the row table are materialized
as NATIVE tables (registered-arrow scans have different NaN pushdown
semantics), and value comparison is a repr-keyed multiset (row order is not
part of the contract across joins; repr keeps 1 / 1.0 / '1' apart and makes
NaN self-equal).

## Runner (`runner.py`)

Persistent worker subprocesses read seeds on stdin and write verdict JSON on
stdout; the parent enforces a per-case timeout, kills and restarts a dead or
hung worker, and blames the in-flight seed (PANIC/TIMEOUT finding). Output:
`findings.jsonl` (verdict, seed, sql, dedup key) plus a stats table — cases,
refusals by class, fallback count, and a construct-coverage histogram over
AGREE cases so a grammar hole is visible instead of silent.

Campaigns run against the **debug** build by default (`debug_assert!`
invariants are compiled out of release; the TASK-75 fix passed release and
panicked debug), with a release spot-check for the cranelift path.

## Shrinker (`shrink.py`)

Greedy passes over the Case AST, re-running the oracle after each cut, keeping
any edit that preserves the verdict: drop select items → drop WHERE/joins/CTEs
→ drop statics/UDFs → shrink rows toward one → replace expression nodes with a
child or a literal. Emits the minimal SQL + data + verdict and a ready-to-paste
`known_divergences/` pin snippet (then a single `test_known_divergences.py`).

## CI smoke (`test_fuzz_smoke.py`)

Fixed seed, ~200 cases, asserts the machinery rather than zero findings (the
fuzzer's job is to find live bugs, so "no findings" cannot be the invariant):
verdicts are produced, ≥1 AGREE and ≥1 REFUSED occur, same seed → identical
verdicts, and a planted `OVER ()` case yields DIVERGE_BUILD-or-REFUSED (today
it diverges — confit builds what DuckDB refuses; after the fix it must refuse).

## Findings process

Per the standing engine-bug rule: divergences are reported in chat, deduped
and personally reproduced; xfail-strict pins and tickets only on explicit go;
no inline fixes ride along with the fuzzer PR.

## Validation gate

The fuzzer must **rediscover the known live bug** (`abs(k) OVER ()` builds on
confit, DuckDB refuses) from pure random generation before any campaign
result is trusted. A fuzzer that cannot find a bug we know exists is not a
fuzzer.

## Out of scope

The `sql_transform` fit layer (the request is confit), perf measurement,
fixing anything the fuzzer finds, hypothesis/attrs or any new dependency.
