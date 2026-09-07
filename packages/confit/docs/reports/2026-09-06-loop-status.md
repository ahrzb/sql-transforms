# Loop status, 2026-09-07: what moved, what is crossed, what changed

**What this is.** A dated status reading of the standing loop (`make confit behavior match
goal.md`), refreshed at the close of iteration 8. Drives first with their lift, then any
control that is crossed, then every behavior change as executed code (each block was run on
2026-09-07 against master's engine at `8796bb2` and the tie branch's at `e6a3cd8`), then the
mechanism, the cost and the decisions. The narrative of iterations 1-8 is in
`2026-09-02-loop-report-1.md`. Slugs resolve as there.

---

## 1. Controls: one crossed, one restored {#controls}

> **CROSSED on master: a frozen fold that is not a function of the query.** exclusion:
> whole-relation-shapes says a static-tables-only query may be frozen only when what it
> selects is a function of the query text and the statics. Master's engine freezes and serves
> eighteen families of shapes that are not: a tie order, a file on the build machine's disk,
> a clock, the session time zone, the arrival order of a floating-point sum. Every block in
> the code section below marked *master* is that control being crossed today. The tie branch
> (`refuse-static-tie-order`, gated, unmerged) closes all eighteen; its own sixth review then
> found **four more** it does not, so the control stays crossed on the branch too until they
> close.

**Restored on master (2026-09-06): kpi: engine-parity.** At the loop's start one of 2000
campaign seeds served a value the oracle did not (seed 1804, `nan` for `-nan`). PR #202 closed
it; master reads `DIVERGE_VALUE` **0** of 2000. Not a headline: a control at its bound is the
expected state.

## 2. Drives, with lift {#drives}

| drive | master (`8796bb2`) | tie branch (`e6a3cd8`) | lift | why |
|---|---|---|---|---|
| campaign acceptance, seeds 0-1999 (constructor returns) | 1056 / 2000 = 52.8% | 1049 / 2000 = 52.45% | **-7 seeds (-0.35 pp)** | 44 seeds are now planted twins of which 24 must refuse by design; the like-for-like loss is 4 seeds (`avg` x3, `sum` over DOUBLE), a control bought with drive |
| mined corpus matches (of 678) | 547 | 540 | **-7** | 5 were never real matches (a harness variable, below); `geomean` and `test_all_types()` are the rule's price |
| dialect ladder L2 (of 678) | 288 | 288 | 0 | untouched |
| serving latency | unchanged | unchanged | 0 | the constant path replays frozen rows; no bench run this iteration |
| build time of a static-only fold | 1x | ~4-5x | **cost** | the parse and tie probes are DuckDB round trips at build |

The loop has so far **spent drive to restore a control**, which is what goal: parity-first
and the KPI law (never trade a control for a drive) tell it to do. No drive has been lifted
by it yet.

## 3. What changed, as code {#changed}

Each block: the call, what master's engine answers, what the tie branch answers. `ROW` is a
one-column `k BIGINT` row schema; `S` is `{"g": ["x","y","z"], "v": [1, 1, 2]}`; `S2` is
`{"st": [{"f1":1,"f2":0}, {"f1":2,"f2":0}], "a": [5, 5]}`.

**Merged (PR #202): a NaN's sign reaches the text.**

```python
fn = DuckDBInferFn("SELECT CAST(nextafter(d, 1.0e0) AS VARCHAR) AS o FROM __THIS__",
                   row_tables={"__THIS__": pa.schema([pa.field("d", pa.float64())])}, static_tables={})
fn.infer_rows([{"d": -float("nan")}, {"d": 1.0}])
# before (5819c3a):  [{'o': 'nan'},  {'o': '1.0'}]     DuckDB: '-nan'
# master now:        [{'o': '-nan'}, {'o': '1.0'}]     DuckDB: '-nan'
```

**On the tie branch: a tied ORDER BY refuses instead of freezing one order.**

```python
DuckDBInferFn("SELECT g AS o, min(v) AS t FROM s GROUP BY g ORDER BY t",
              row_tables={"__THIS__": ROW}, static_tables={"s": S})
# master: SERVES constant  [{'o': 'x', 't': 1}, {'o': 'y', 't': 1}, {'o': 'z', 't': 2}]
#         (x and y tie at 1; five settings a build machine picks gave five sequences)
# branch: ValueError: unsupported: tie-producing ORDER BY on a static-tables-only query
#         -- which of the tied rows comes first depends on scan order, not the query
```

**An order-dependent aggregate refuses.**

```python
DuckDBInferFn("SELECT avg(v) AS o FROM s", row_tables={"__THIS__": ROW}, static_tables={"s": S})
# master: SERVES constant  [{'o': 1.3333333333333333}]   (six answers over 200k rows, measured)
# branch: ValueError: unsupported: order-sensitive aggregate avg on a static-tables-only
#         query -- its answer follows scan order, and an ORDER BY inside the aggregate is not
#         read as a fix
```

**A file on the build machine's disk refuses.** (`demo_e2e.csv` holds two rows here.)

```python
DuckDBInferFn("SELECT * FROM 'demo_e2e.csv'", row_tables={"__THIS__": ROW}, static_tables={"s": S})
# master: SERVES constant  [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}]   (whatever the file held at build)
# branch: ValueError: unsupported: the table demo_e2e.csv on a static-tables-only query -- it
#         is not one of the query's static tables, so its rows are read off the file system
#         or the catalogue when the query runs, not fixed by the query
```

**A macro whose body is an order-dependent aggregate refuses under its own name.**

```python
DuckDBInferFn("SELECT json_group_array(v) AS o FROM s", row_tables={"__THIS__": ROW}, static_tables={"s": S})
# master: SERVES constant  [{'o': '[1,1,2]'}]   (a scan-order sequence; two answers across settings)
# branch: ValueError: unsupported: order-sensitive aggregate json_group_array on a
#         static-tables-only query -- its answer follows scan order, ...
```

**A clock read under a CONSISTENT flag refuses.**

```python
DuckDBInferFn("SELECT age(TIMESTAMP '2020-01-01')::VARCHAR AS o FROM s",
              row_tables={"__THIS__": ROW}, static_tables={"s": S})
# master: SERVES constant  [{'o': '6 years 8 months 6 days'}, ...]   (today's date, frozen)
# branch: ValueError: unsupported: the non-deterministic function age() on a
#         static-tables-only query -- its value is drawn when the query runs, not fixed by the query
```

**SUMMARIZE refuses.**

```python
DuckDBInferFn("SUMMARIZE s", row_tables={"__THIS__": ROW}, static_tables={"s": S})
# master: SERVES constant  [{'column_name': 'g', ..., 'approx_unique': 3, ...},
#                           {'column_name': 'v', ..., 'avg': '1.3333333333333333', 'std': '0.577...', ...}]
#         (seven settings, seven answers on a large static)
# branch: ValueError: unsupported: a SUMMARIZE, DESCRIBE or SHOW statement on a
#         static-tables-only query -- what it computes is chosen by DuckDB and read off the catalogue
```

**A value with a time zone refuses.**

```python
DuckDBInferFn("SELECT (TIMESTAMPTZ '2020-01-01 00:00:00+00')::VARCHAR AS o FROM s",
              row_tables={"__THIS__": ROW}, static_tables={"s": S})
# master: SERVES constant  [{'o': '2020-01-01 01:00:00+01'}, ...]   (this machine's zone, frozen)
# branch: ValueError: unsupported: a value WITH TIME ZONE (a cast to TIMESTAMP WITH TIME ZONE)
#         on a static-tables-only query -- its rendering reads the build machine's time zone
```

**An alias displaced by an expanding entry refuses instead of measuring the wrong column.**

```python
DuckDBInferFn("SELECT *, a AS k, unnest(st) FROM s ORDER BY k",
              row_tables={"__THIS__": ROW}, static_tables={"s": S2})
# master: SERVES constant  [{'st': {...}, 'a': 5, 'k': 5, 'f1': 1, 'f2': 0}, {..., 'k': 5, 'f1': 2, ...}]
#         (k ties at 5, 5)
# branch: ValueError: unsupported: a sort key whose output position this reading cannot place
#         on a static-tables-only query -- a tie among its rows could not be ruled out, ...
```

**Still crossed on the branch: what the sixth review measured** (its outputs, confirmed end
to end with a refusing control on identical data).

```python
# 1. a zoned type made from a STRING at bind time -- no cast, no column, no maker to sight
DuckDBInferFn("SELECT strptime('2020-01-01 00:00:00+05', '%Y-%m-%d %H:%M:%S%z')::VARCHAR AS d FROM s", ...)
# branch: SERVES constant  '2019-12-31 20:00:00+01'   (UTC '...19:00:00+00', Asia/Tokyo '2020-01-01 04:00:00+09')

# 2. a macro whose body calls another macro -- expansion stops one level short
DuckDBInferFn("SELECT geometric_mean(abs(d)) AS o FROM s", ...)      # -> geomean -> exp(avg(ln(x)))
# branch: SERVES constant  529251629.29637396           (four answers across settings)

# 3. a CTE declared inside a subquery whitelists its bare name for the outer FROM
DuckDBInferFn("SELECT count(*) AS o FROM 'e2e.csv' WHERE 1 IN (WITH \"e2e.csv\" AS (SELECT 1 AS a) SELECT a FROM \"e2e.csv\")", ...)
# branch: SERVES constant  o = 1 in one directory, o = 5 in another

# 4. a static named like a catalogue view's last segment whitelists the qualified read
DuckDBInferFn("SELECT count(*) AS o FROM information_schema.tables", ..., static_tables={"s": S, "tables": S})
# branch: SERVES constant  o = 4                          (a count of this build's catalogue)
```

Two more from that review are not fail-opens: the `age` arity read never reaches a macro body
(latent; no builtin macro calls `age`), and one zoned column in **any** static refuses every
query on that build, `SELECT 1` included, which is wider than its disclosure.

## 4. How the branch reads a statement {#mechanism}

Every refusal above is read off DuckDB's own parse (`json_serialize_sql`, walked with
`json_tree`) and DuckDB's own catalogue (`duckdb_functions()`, `duckdb_columns()`), asked of
the connection the statement already ran on. Three of iteration 8's five rules are allow-lists
or metadata reads rather than name lists: a base table must be a static or a CTE; a macro body
is parsed and fed into the same name reads as a call; a zoned value is sighted in a column's
declared type, a cast node, or a maker's catalogue return type. The four shapes still open are
defects in those readings (scope, qualification, depth, a type chosen at bind time from a
string), not new families.

**The corpus move, explained.** Five of the seven statements that left the match count
(`SELECT COUNT(*) FROM t`) had matched only because DuckDB's Python client resolves a bare
table name against the variables of the calling frame, and the replay left a pyarrow table
named `t` there. The FROM allow-list ended that accident; a harness fact as much as an engine
fact, now pinned at `MATCH_FLOOR` with its reason.

## 5. The gate on the rebased tip {#gate}

`e6a3cd8` is the gated `36ae02e` rebased onto master `8796bb2` (eleven commits, no
conflicts). Root suite **3480** passed / 1 skipped / 9 xfailed / 2 errors (absent `pyspark`):
every one of master's 3338 collected ids present with its outcome, plus exactly **154**
branch ids. `cargo test --release --lib` 269 / 5, the five pre-existing names. Corpus 540.
Campaign seeds 0-1999: `AGREE` 1008 / `REFUSED` 951 / `AGREE_TRAP` 20 / `UNSHIPPED` 14 /
`DIVERGE_OPT` 7 / `DIVERGE_VALUE` **0**; the one seed that moved against the pre-rebase gate
is 1804, agreeing now because the tree carries master's NaN fix. Mutation: five rules reverted
one at a time, 2 / 16 / 4 / 37 / 16 tests red each time. 48 of 48 hand probes, 29 must-serve
and 19 must-refuse.

## 6. Method, spend, decisions {#rest}

**Method.** The orchestrator's own read of every diff found the two items no gate did (the
`nextafter` sign bug; the displaced alias). A pin measured on Windows was not a pin on Linux
(the both-NaN `nextafter` case; the kernel now calls the platform's own C `nextafter`). A
gate on a tree thirteen commits behind master is a gate on something else; the branch is
rebased before it is presented. Two sentences that judged DuckDB's classification were
restated as the measurements they rest on.

**Spend.** Iterations 1-7 ~9.0M agent tokens; iteration 8 ~1.3M. The stop rule is roughly 70%
of the owner's weekly credit, owner-signalled, not yet given.

**Decisions that are the owner's.**
1. The fork: enumerate a seventh round, or pin the build-time fold's configuration through the
   oracle (ask: engine-fold-reading, ask: threads-and-value-order). The RFC with its
   three-question framework was put in the chat at the close of iteration 8.
2. ask: acceptance-target now (recommended: ratchet without a target); exclusion-ratification,
   kpi-set-change and next-query-classes wait on the fork.
3. The static-only acceptance price above: the policy says take it; it is measured, so it can
   be priced.
4. The weekly percentage.

**Next, in the goal's order.** The fork's answer; iteration 9 under it (close the four,
scope the over-refusal, re-gate, re-read); the report PR; then the refusal registry,
gap: undocumented-boolean-comparison, the red Rust unit gate CI cannot see, the bench baseline.
