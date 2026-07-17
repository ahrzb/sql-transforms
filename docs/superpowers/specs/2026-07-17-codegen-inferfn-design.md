# Codegen engine (a codegen `InferFn`) — Scope

> **STATUS: scope + front-end fork decided; plan written.** Front-end decided
> (fork B, Python/sqlglot — see below). Plan:
> [2026-07-17-codegen-inferfn.md](../plans/2026-07-17-codegen-inferfn.md).
>
> Still **not** settled, and NOT assumed by the plan: the two-engine framing and
> the doc reconciliation (codegen is still filed under BACKLOG "Considered —
> likely won't do"; the 2026-07-16 benchmark decision revived it as the *default*
> serving engine but ROADMAP/BACKLOG/VISION are not yet re-cut), and roadmap
> placement. Those are AmirHossein's/the PM's calls. The plan deliberately stops
> at "second engine proven equivalent" and does **not** touch engine-selection
> defaults.

**Goal.** A second serving engine that is **functionally equivalent to the Rust
`InferFn` interpreter**: same input (the post-fit rewritten `__STATE__`/`__THIS__`
SQL + row/static schemas), same validated Pydantic output model, same values on
every input, gated by the same differential harness — but executing as **generated,
cached Python** instead of the Rust interpreter, so the pyo3 boundary and dynamic
`Value`-enum dispatch leave the per-row hot path.

**Why this framing (PM's north star).** `InferFn` already *is* a precise,
harness-tested spec of the target semantics. "Make a codegen `InferFn`" gives a
complete existing target to hit rather than a subset to negotiate, and the existing
differential suite (~188 tests, `tests/differential.py`) is the correctness gate for
free — no new correctness burden.

**Why it exists at all (the benchmark).** Measured 2026-07-16: hand-written
codegen-proxy Python ~2 µs/row vs Rust `infer` ~26 µs/row (~8–13×) on a realistic
7-feature transform. The transform *compute* is trivial; the cost is the boundary
(per-column FFI getattr, enum dispatch, output construction). Generated straight-line
Python over native `int`/`float`/`str` skips all of it. (Caveat from the same
session's capstone: a Python-object-in / Pydantic-out contract still carries a
~1.2 µs/row boundary floor neither engine gets under — the columnar/numpy path that
removes *that* is explicitly **out of scope here**, see Deferred.)

---

## Where it plugs in

`SQLTransform.fit` already produces `self._rewritten_sql` and hands it to
`InferFn(sql, row_tables={"__THIS__": model}, static_tables=state)`
(`sql_transform/__init__.py:86`). The codegen engine is a **drop-in alternative at
exactly that seam** — same constructor inputs, same `.infer({"__THIS__": rows})`
call, same `list[output_model]` out. Engine selection (which one `infer`/
`infer_batch` dispatches to) is a `SQLTransform` concern; the default-vs-opt-in
policy from the 2026-07-16 decision is a **framing question for AmirHossein**, not
settled here. This scope covers only *building an engine equivalent to `InferFn`*.

---

## Front-end: DECIDED — re-analyze in Python (fork B)

Codegen needs the analyzed plan (relational tree + expr trees + lookup specs +
effective schemas + output model). `InferFn` derives all of that in Rust:
`parse (sqlparser) → build_plan → optimize (LookupJoin extraction) →
validate_columns (type inference, column checks, unnest expansion)`.

**Decided (AmirHossein, 2026-07-17): re-implement the front-end in Python on
sqlglot (already a dep).** The alternative — exposing the Rust-analyzed plan to
Python — was the smaller surface, but it makes the codegen engine depend on the
Rust crate, and **the Rust path may be axed entirely**. An engine that dies with
the thing it might replace is not a replacement. So codegen owns its whole
pipeline: parse → plan → optimize → validate → type-infer → lower → run.

Consequence, stated plainly: the SQL front-end is now **duplicated** while both
engines live. The differential harness is what keeps the two from drifting apart
on *analysis* as well as execution — every case runs through both.

---

## How the plan compiles to cached Python (assuming fork A)

At construction: pull the analyzed IR, walk it, emit one Python function body as a
string, `compile()` + `exec()` it once, cache the function on the engine. `infer`
just calls the cached function per row-batch. No `exec` at serve time.

**Relational nodes → nested Python** (each consumes an iterable of `row` dicts,
`row` = `{effective_table_name: {col: value}}`, mirroring the Rust `Row`):

| `RelNode` | Lowering |
|---|---|
| `TableScan{table}` | iterate input rows, wrap `{table: row}` |
| `SubqueryAlias{alias}` | rename the single inner table key → `alias` |
| `Filter{predicate}` | `if _truthy(<pred>): yield row` (keep only `Bool(true)`; NULL/false drop) |
| `CrossJoin` | nested loop, merge dicts (cartesian) |
| `Join{on}` (inner) | nested loop; keep when every ON pair compares equal and neither side NULL (NULL never matches) |
| `LookupJoin{table,keys,outer}` | build key tuple, dict-index the static table; hit → bind row; miss+outer → all-NULL value row; miss+inner → `ValueError` |
| `Unnest{list_expr,output_col}` | evaluate list; emit one row per element bound under the synthetic unnest key; NULL/empty → 0 rows *(deferred — see Deferred)* |

Projection: emit `out[alias] = <expr>` per projection item, build the output row,
then hand the collected rows to the (existing) Pydantic `output_model` for
validation — same output object `InferFn` returns.

**Expr nodes → Python expressions**, calling a small shared **semantics runtime**
(`_codegen_runtime.py`) for anything where naive Python diverges from the Rust/
DataFusion-parity semantics. Column refs lower to dict lookups with the same
qualified/unqualified + ambiguity rules the analyzer already resolved.

### The semantics runtime (the load-bearing part)

Straight-line Python is only correct where Python's semantics already match. These
are the **divergence landmines** the runtime must own so generated code stays
bit-identical to `InferFn`/DataFusion (each is exercised by the existing harness):

- **Integer division truncates toward zero**, not floors: `-7 / 2 == -3` (Python
  `//` gives `-4`). Runtime `_idiv` = `int(a / b)` guarded, or trunc-based.
- **Integer modulo takes the sign of the dividend** (C/Rust `%`), not the divisor
  (Python `%`): `-7 % 2 == -1`, not `1`. Runtime `_imod`.
- **Int div/mod by zero → `ValueError("division by zero")`** (matches Rust
  `InterpError::Eval`); float div-by-zero follows IEEE (`inf`/`nan`), not an error.
- **`ROUND` is round-half-away-from-zero** (Rust `f64::round`), not Python's
  banker's rounding: `round(0.5) → 1.0`, `round(2.5) → 3.0` (Python `round` gives
  `0`/`2`). Runtime `_round`.
- **Float→string formatting** (`CAST(x AS VARCHAR)`, `CONCAT`) follows Rust
  `f64::to_string` — e.g. `1.0 → "1"` (Python `str(1.0) → "1.0"`). Runtime
  `_display` mirroring `expr::display_value` (incl. struct/list rendering when
  those land).
- **Three-valued (Kleene) logic** for `AND`/`OR`/`NOT` with the exact NULL tables
  from `expr::logic`/`as_tribool`; non-bool operand → error.
- **NULL propagation** per operation family: arithmetic & comparison → NULL if
  either operand NULL; string builtins (`upper`/`lower`/`trim`/`substr`) → NULL on
  NULL arg; `concat` skips NULLs; `coalesce` first-non-NULL; `nullif` equal→NULL.
- **Comparison**: scalar ordering; structural eq for struct/list (deferred);
  NaN compare → error.
- **Casts** (`str`/`int`/`float`/`bool`) with the exact rules in `eval_cast`
  (float→int truncates; str→int/float parse-or-`ValueError`; bool→int/float; etc.).
- **`substr`** 1-indexed, character-based (not byte), with the clamping in
  `expr::substr`.

The runtime is a *tiny* hand-written module (one source of truth for these), imported
by every generated function — not re-emitted per query. Error **types** match
`InferFn` where the harness asserts them (`ValueError` on div-by-zero, bad cast);
error-message parity is a non-goal (per BACKLOG), only values/error-hierarchy matter.

---

## Parity surface: committed vs deferred

**Committed (MVP — `InferFn`'s scalar/relational surface):**
- Relational: `SELECT` projections, `WHERE`, `INNER JOIN ... ON`, `CROSS JOIN`,
  static `LookupJoin` (incl. LEFT-lookup NULL rows), `SubqueryAlias`.
- Expr: `Column` (qualified/unqualified + ambiguity), `Literal`, `BinaryOp`
  (arithmetic/comparison/logic), `Not`, `Function` (the shipped builtins:
  `upper`/`lower`/`trim`/`substr`/`substring`/`concat`/`abs`/`round`/`coalesce`/
  `nullif`), `Cast`.
- Full value semantics + NULL rules above; typed Pydantic output model.

**Deferred to fast-follows (non-blocking; codegen errors cleanly on these nodes
until implemented):**
- **Container exprs + `UNNEST`**: `Struct`, `List`, `FieldAccess`, `RelNode::Unnest`
  (and struct/list structural equality). `InferFn` covers these today, so *full*
  `InferFn` equivalence is reached in **two steps**: MVP (scalar) → containers. The
  `Unnest` row in the table above is designed here for coherence but not built in
  the MVP.
- **Vectorized/columnar codegen path** (batch-shaped generated code).
- **numpy-matrix output mode** (the boundary-floor-removing path; separate track).
- **`CASE WHEN`** (not in `InferFn` today either — genuine new surface, own item).

---

## Differential-harness wiring (the correctness gate)

`tests/differential.py` already runs every case through DataFusion (oracle) and
Rust `InferFn` and asserts agreement (`check(query, tables)`). Add codegen as a
**third engine**:

- New `_run_codegen(query, tables)` mirroring `_run_infer` (same typed row models
  via `synthesize_this_model`, same static tables, `.model_dump()` out).
- `check()` asserts **all three agree** (`datafusion == infer == codegen`), so the
  entire existing corpus regression-tests codegen with **no new test authoring**.
- `check_both_raise` generalizes to also require codegen rejects what both others
  reject (on the committed surface).
- Deferred-surface cases (struct/list/unnest) are skipped for codegen until step 2,
  via a capability guard — and the skip is **explicit/logged**, never silent, so a
  gap can't masquerade as coverage.

This reuse is the whole point of the PM's "codegen == `InferFn`" framing: the spec
is already written as executable tests.

---

## Open questions (framing calls — NOT resolved by the plan)

1. ~~Front-end fork A vs B~~ — **decided: B** (Python/sqlglot), see above.
2. **Engine-selection surface**: default codegen + opt-in Rust (per 2026-07-16
   decision) — how is it exposed on `SQLTransform`? (a ctor arg / method kwarg /
   global?) The plan builds the engine and proves it equivalent; it does **not**
   wire selection or change any default. That's a follow-up gated on (3)/(4).
3. **Roadmap placement**: foundational parallel track vs sequenced vs folded into
   M1 — TBD pending the two-engine framing decision; don't assume.
4. **Doc reconciliation** (PM-owned): move codegen off BACKLOG "won't do", add the
   engine epic, reflect the two-engine decision in ROADMAP/VISION. Held until the
   framing is ratified.

## DECIDED: match DataFusion — the Rust divergences are bugs

**Decision (AmirHossein, 2026-07-17): codegen matches the DataFusion oracle.**
Where the Rust `InferFn` disagrees with DataFusion, **Rust is wrong** — the harness
gates on the oracle, `transform` *is* DataFusion, and an engine meant to outlive the
Rust one should not inherit its quirks.

Consequence, accepted knowingly: on these cases codegen ≠ `InferFn`, so the honest
framing is **"a DataFusion-parity engine built by codegen"**, not "a codegen
`InferFn`". `InferFn` remains the spec for everything else — which is all of the
covered surface.

**Standing process for a Rust-engine bug found this way** (AmirHossein, 2026-07-17):
1. Add a differential test, **`xfail` on the rust backend only** (`strict=True`, so
   it flips to a failure the moment the bug is fixed), with the divergence described
   in the reason.
2. **Tell the PM to open a BACKLOG ticket** for the fix. Do not fix it inline — it
   is a separate concern from building codegen.

**Measured 2026-07-17** (probe against both engines, cases no test covers):

| Case | DataFusion (oracle) | Rust `InferFn` |
|---|---|---|
| `CAST(1.0 AS VARCHAR)` | `'1.0'` | `'1'` |
| `ROUND(3)` (int arg) | `3.0` (float) | `3` (int) |
| `NULLIF(1, 1.0)` | `NULL` | `1` (type-strict eq) |
| `-a`, `-1` (unary minus) | works | **rejected** (`Unsupported expression`) |
| `a \|\| b` | `'aa'` | **rejected** (`Unsupported operator`) |

The differential harness only pins the surface it *covers*; these are gaps, and on
them the two shipped engines return different values. **All five are bugs in the
Rust engine**, and they matter beyond codegen: the README promises `transform` and
`infer` return identical values, so each one is a live product defect where a user
gets a different answer at serving time than at batch time. Every one is `xfail`-ed
on the rust backend and ticketed with the PM per the process above.

Codegen's behaviour on each, therefore: render `1.0` as `'1.0'`, return `3.0` from
`ROUND(3)`, null `NULLIF(1, 1.0)`, and **support** unary minus and `||`.

## Appendix: verified findings (2026-07-17)

A full draft was written and deleted; a second, narrower spike then **validated**
these against the real toolchain. Everything below is measured, not reasoned. Items
marked ✅ confirm an earlier guess; ❌ mark a guess that was **wrong** and would have
shipped a defect.

### Toolchain facts (sqlglot 30.12)

- ❌ **`Select.args["from"]` does not exist — the key is `"from_"`** (renamed in
  sqlglot 30). Reading `"from"` returns `None`, which would raise "FROM clause is
  required" on *every query*. `_batch.py` never hit this because it uses the
  high-level `tree.select()`/`order_by()` API.
- ❌ **`s.a.b` parses as `exp.Column` with a `db` arg**, not `exp.Dot`. Code that
  only guards `exp.Dot` silently misreads it as `Column(table='a', name='b')` and
  drops `s` — a wrong answer, not an error. Reject any `Column` carrying
  `db`/`catalog`.
- ❌ **Variadic function args have per-function shapes**; one generic helper is
  wrong: `Concat` → `this=None`, args in `expressions`; `Coalesce` → first arg in
  `this`, rest in `expressions`; `Nullif` → `this` + `expression`. An
  `e.this not in args` de-dup guard silently drops an argument for `COALESCE(a, a)`
  (equal sub-expressions compare equal), yielding arity 1.
- ✅ `Substring` → `args["start"]` / `args["length"]`; `Cast` → `.to.sql()` gives
  `VARCHAR`/`BIGINT`/`DOUBLE`/`BOOLEAN` (prefix-matching works).
- ✅ Join `kind`/`side` come back as `None` (not `""`), so `(x or "").upper()` is
  required. Comma-join → `kind=None, side=None, on=None`; `CROSS JOIN` →
  `kind='CROSS'`; `INNER JOIN` → `kind='INNER'`; `LEFT JOIN` → `side='LEFT'`.
  Unaliased `Table.alias` is `''`, not `None`.
- ✅ Literals: `is_string` distinguishes; `1e3` has no `.` so exponent-sniffing is
  needed; `Boolean.this` is a real Python `bool`; `NULL` → `exp.Null`.

### Test-harness mechanics — the one that matters most

- ❌ **`metafunc.fixturenames.append(...)` + `parametrize(indirect=True)` produces
  the parametrized test IDs but NEVER RUNS THE FIXTURE.** Measured directly: the ID
  said `codegen` while the engine in use was still `rust`. In the deleted spike this
  meant all 160 differential runs silently executed the *rust* engine twice —
  186 passed, zero skips, codegen never executed once. A fully fake green.
- ❌ **"Check that the collected count doubled" is NOT a valid guard** — the count
  *did* double while codegen never ran. Counting proves parametrization, not
  execution.
- ✅ **`@pytest.fixture(autouse=True)` + `pytest_generate_tests` indirect
  parametrize works** — `autouse` is what forces instantiation. Non-harness modules
  (no `request.param`) fall through untouched and stay unparametrized.
- ✅ The only trustworthy guard is a test that asserts the **active engine** equals
  its own param, from inside the test body.

### Rust semantics (confirmed against the source AND at runtime)

- **`bool` is an `int` subclass in Python.** `Value::from_pyobject` checks `PyBool`
  before `PyInt`, and `as_f64` *errors* on a bool — so `True + 1` must raise, not
  return `2`. Every type test must be `type(v) is int` (identity), never
  `isinstance`. Same trap in comparison (`True == 1` is `True` in Python).
- **`Value`'s `Eq`/`Hash` are variant-tagged**: `Int(1) != Float(1.0)`. Python's
  `1 == 1.0` and `hash(1) == hash(1.0)`, so **JOIN-ON equality and lookup-join keys
  must be type-tagged** (`(tag, value)`) or they match rows Rust wouldn't.
- **Float division by zero**: Rust yields IEEE `inf`/`nan`; Python raises
  `ZeroDivisionError`. Only the *integer* path errors (`ValueError`).
- **Float `%`**: Rust `%` is C-style (`math.fmod`), not Python's `%`; `x % 0.0` is
  `NaN`, not an error.
- `RelNode::Filter` keeps a row only on `Value::Bool(true)` — so the emitted guard
  is `v is True`, not Python truthiness (`1`/`"a"` must not pass).
- `Expr::Function` args are evaluated eagerly before dispatch, and `logic()` calls
  `as_tribool` on **both** operands before matching — so `AND`/`OR` do **not**
  short-circuit, and a non-bool operand errors even when the other operand would
  determine the result. Emitting Python `and`/`or` would be wrong twice over.
- `math.floor`/`math.ceil` return `int` in Python; `ROUND` on a float must stay a
  float (`Value::Float`), so the half-away-from-zero helper must re-wrap.
- ✅ **Unary minus is rejected by the Rust engine** — measured: both `SELECT -1` and
  `SELECT -a` raise `Unsupported expression` (sqlparser's `convert_expr` handles
  only `Not`). DataFusion evaluates both. sqlglot parses both as `exp.Neg`, over a
  `Literal` and a `Column` respectively — so "fold literals only" would still leave
  `-a` unsupported. Which behaviour codegen adopts falls out of the oracle-vs-Rust
  decision above. **It is not committed surface** and must not be asserted as such.
- ✅ **`||` (`exp.DPipe`) is rejected by Rust** (`Unsupported operator: ||`) and
  evaluated by DataFusion (`'aa'`). Same decision applies.
- ✅ sqlglot gives dedicated nodes for `Upper`/`Lower`/`Trim`/`Substring`/`Concat`/
  `Abs`/`Round`/`Coalesce`/`Nullif`, plus `exp.Anonymous` for the rest — the
  builtin dispatch must cover both shapes.
- ✅ The harness's `_run_infer` is already engine-shaped: `Engine(query,
  row_tables=..., static_tables=...)` then `.infer(rows)` → `.model_dump()`. A
  second backend needs only that same duck-type, so **no test call sites change**.
- ✅ Test inventory: `tests/` collects **106** — 80 across the six harness modules,
  26 in `test_interpreter.py`. Full repo: 188. A correctly-wired second backend
  should take `tests/` to 186 — *but see the fake-green finding above; that number
  is necessary, not sufficient.*

## Non-goals (this engine)
- Error-*message* parity across engines (values + error hierarchy only; per BACKLOG).
- Removing the Python-object-in / Pydantic-out boundary floor (that's the columnar/
  numpy track — different engine mode, different scope).
- Any new SQL surface beyond what `InferFn` already accepts.
