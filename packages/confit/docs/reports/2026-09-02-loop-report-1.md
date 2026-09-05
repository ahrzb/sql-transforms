# The loop's first report: iterations 1-4 (2026-09-02, amended 2026-09-06)

**What this is.** A dated report of one loop's work against `packages/confit/docs/goal.md`,
read through the yardsticks the baseline reading measured
(`packages/confit/docs/reports/2026-09-02-goal-baseline.md`, reading **N=1**). It is not a
new reading: no census, floor or bench was re-taken here for its own sake. What it records is
**which distances from the target moved, by how much, and on whose measurement** — every
number below is quoted from a **gate record**, the independent leg that rebuilt master and
the branch itself, never from an implementer's own claim, and each is named to its gate.
Where only a review measured something, the item says "review record" and the claim stands as
a review's, not a gate's.

**What the amendment changed.** The 2026-09-02 text was written while two of its three items
were still moving: the modulo branch's own gate had not yet been read against the NaN
branch's, and the tie rule's rebuild had returned no record to the loop's journal. Iteration
4 closed both, and iteration 3's missing record has since been read. Every number added below
is quoted from a gate or review record taken after that text was written; wherever a figure
replaces one the first text carried, the old figure is named beside it so the two are not
silently conflated.

**Slugs.** `gap:` and `finding:` citations resolve in the baseline reading; `goal:`,
`kpi:`, `exclusion:` and `ask:` in `goal.md`; `claim:` and `divergence:` without a local
definition in `packages/confit/docs/oracle/`. Sections here carry kebab-case anchors and are
cited by slug, never by number.

---

## 1. What the loop is {#the-loop}

The owner's mandate, as it stood over these four iterations: **make confit match `goal.md`**
— close the distance the baseline reading measured, in the order the goal document's own
priorities give. Its working rules:

- **No tickets.** Work is dispatched against a `gap:` or `finding:` slug, not a ticket
  number, and nothing in this loop filed one.
- **No implementation check-ins.** The owner is not consulted mid-branch. What reaches him is
  a gated branch and a dated report.
- **Gated PRs and dated reports are the only outputs.** A branch is not done when its author
  says so; it is done when an independent gate has rebuilt both legs and reproduced or
  refuted every clause of its promise.
- **A merge needs the owner's own GitHub approval.** Nothing in this loop merged itself. One
  branch merged in this window, on his click.

The shape each iteration took: one implementer per item, then an independent **review** and
an independent **gate**, each in its own worktree, each building master and the branch from
source rather than trusting a shipped artifact. That structure is the reason this report can
name a gate for every number: the gate legs are what produced them.

**One measurement about the loop itself, worth keeping.** Every master leg the loop ran —
**eleven of them across four iterations** — reproduced the baseline's campaign census
exactly: `AGREE` 1013 / `REFUSED` 944 / `AGREE_TRAP` 21 / `UNSHIPPED` 14 / `DIVERGE_OPT` 7 /
`DIVERGE_VALUE` 1 over seeds 0-1999 at `--workers 8 --timeout 20`. Master advanced twice in
the window (`2ba96e5` -> `2c7c05c` -> `f81e17c`, docs plus one added assertion) and the census
did not move; it has stood at `f81e17c` since, across the four further master legs iterations
3 and 4 ran, each of which also re-checked the sanity seed and found 1804 the
`DIVERGE_VALUE`. claim: campaign-verdicts-today is reproducible on this machine, and a single
seed flipping is therefore signal, not scheduling noise.

---

## 2. What closed {#closed}

Three of the baseline reading's items were worked. Each is stated as its distance from the
target, then what closed it.

### 2.1 gap: corpus-match-slip

*Target:* kpi: coverage-ladder is a **drive** — the mined-corpus match count grows and never
silently shrinks.
*Was:* **547** replayed bit-exact where **550** was quoted at six unhedged sites; the count
was printed and never asserted, so three statements had gone missing with nothing noticing.
*Now:* the three are **named and classified**, and the count is **floored**.

**The diagnosis, reproduced by the gate rather than relayed.** Exactly three statements
flipped, all the same SQL from DuckDB's own `equality_join_limits` test at corpus indices
**250 / 251 / 252**, whose key columns are declared `UTINYINT`, `USMALLINT` and `UINTEGER`.
The gate rebuilt the engine at four historical commits and replayed all 678 cases at each:
**550** match / 128 clean-unsupported / 0 FAIL before the arrow surface, **547 / 131 / 0** at
the arrow-schema commit and at both later static-side commits, with the per-case diff over
all 678 showing **exactly those three** changing outcome and never to FAIL. The gate also
refuted its own first suspicion — that the branch had misattributed the cause to the row path
— by probing the flip commit and reading the refusal text there.

**The classification matters more than the count.** DuckDB answers those three at their
declared unsigned widths; the engine has no unsigned lane. The pre-flip "match" stood because
the replay compared `repr`'d **values** and never the **output schema** — so **550 was partly
a wrong answer**, and refusing is goal: two-outcome-contract working, not a regression. The
loss is a correctness gain that the ladder had no way to say out loud.

*What closed it:* `MATCH_FLOOR = 547` in `test_corpus_replay.py`, asserted inside the
existing three-outcome test (a new assertion, not a new node — junit node ids read **3287**
on both legs), plus two live docs corrected from 550 to 547.
*Gate:* suites **identical** on both legs (3281 passed, 1 skipped, 3 xfailed, 2 errors — the
two errors are the pre-existing absent-`pyspark` collection pair); corpus **547 / 131 / 0**
on both; campaign **zero flips**, the two 2000-record maps equal as whole objects; `cargo
test` 266 passed / 5 failed on both, the same five names. Mutation checked in **both**
directions: the floor at 548 goes red, an injected one-match drop goes red with the assertion
and stays **silent** without it — which is the ratchet's whole claim, demonstrated.
*Merged:* yes, as PR #200; master is `f81e17c`. It is the only merge in this window.
*Left open by the gate:* the doc cleanup is **4 of 6 sites short**. Two report files still
read 550 in the present tense and carry no date, and `known-limitations.md` still says "550
of 678 statements as of stage B" — the document the branch's own corrected README points at.
That is the precise failure the oracle spec's correction predicted for a partial remedy.

### 2.2 finding: seed-1804

*Target:* kpi: engine-parity is a **control**, fixed at 100% on the accepted surface.
*Was:* one live `DIVERGE_VALUE` in 2000 — a NaN **sign** reaching a string.
*Now:* **`DIVERGE_VALUE` 0 of 2000**, on three separate branch legs, with the root cause and
three of its neighbours named.

**Root cause.** `impl Display for DuckF64` (`specializer/exec/kernels.rs`) opened with an
unconditional `if self.0.is_nan() { return f.write_str("nan") }`, discarding the sign bit.
One IR instruction backs every double-to-text path on both backends, so the explicit `CAST`,
the implicit casts under `||` and `concat`, and struct fields all dropped the sign together.
The rule was read from the pinned DuckDB v1.5.5 source, not inferred from a probe: the
`DOUBLE -> VARCHAR` cast is `duckdb_fmt::format("{}", v)`, and the bundled writer takes the
sign from `std::signbit` **before** the finiteness branch. Both gates that checked this
citation confirmed it in the checkout.

**The neighbours it exposed** — this is the part worth carrying forward, because each was
invisible while every NaN rendered alike:

1. **Unary minus was not an IEEE sign flip.** The engine lowered `-x` on a DOUBLE as
   `-0.0 - x`, which returns a NaN operand with its own sign; DuckDB's is a plain negate. The
   first gate caught this as a caveat and named it exactly: one shape **agreed on master by
   accident** and diverged on the fixed branch. The second iteration replaced the subtraction
   with a total unary `Fneg` carried end to end (frontend, lowering, both backends, the
   constant fold, and the IR opcode, parser and generator).
2. **A classifier regression the whole suite was blind to.** The new node broke
   `scan_residual`, so a JOIN ON residual containing a negated DOUBLE refused where master
   compiled it. **A review found it, not a gate and not the suite** (3289 tests green over
   the defect). The third iteration closed it with one arm added to an existing or-pattern,
   red-first over four residual shapes.
3. **The double-modulo sign**, which the same rendering change made visible, went to its own
   branch rather than being folded in here. Iteration 4 established that branch was
   **redundant** and dropped it; see below.

**The `||` collapse the `Fneg` node broke, and its closure.** A review of the gated
`57a05ff` measured one HIGH the gate had not: the new total `Fneg` dropped the
fold-then-null-operand short circuit that `-0.0 - x` had carried for free, so
`- <DOUBLE NULL>` stopped binding to `NullOf` and the `||` SQLNULL collapse broke behind it.
Iteration 4 closed it red-first, and the red was wider than the review's report — **10 shapes
x 2 backends through the oracle, comparing the output SCHEMA as well as the rows, 12 failed
and 8 passed**, which is the reviewer's four divergences plus the two constructs master
served that had become bind refusals. The fix is arith's own rule spelled for the unary form:
fold the operand, and return `null_of(F64)` on a folded `NullOf` before reaching
`math1_node`. Pinned at `tests/known_divergences/test_literal_typing.py`.

**The modulo branch was redundant.** `fix-fmod-sign` was opened on the reading that `%` on
DOUBLE diverged. It did not. Iteration 4 established that **the kernel already matched DuckDB
bit-for-bit on both backends**, and the whole of its apparent divergence was the same
`DuckF64` NaN-sign short circuit this item fixes — one line, already carried here. The branch
is **dropped**: none of its production diff survives, because its `kernels.rs` change was
already on `fix-nan-sign-varchar` and its interpreter-test edit was not taken.

*What was salvaged from it:* the parity grid only — `MOD_SIGN_GRID` and
`test_double_mod_sign_grid_value_and_text`, moved into
`tests/test_duckdb_wave3_mathtail.py` and adjusted to project `x` and `y` so a mismatch names
its row (the comparison is a multiset). One test was deliberately **not** taken:
`test_mod_by_zero_nan_sign_is_not_a_property_of_the_row` asserts that the *oracle* is
self-inconsistent, which is a UCRT-dependent claim; its substance survives as the grid's
header comment naming why the invalid-operation domain (divisor zero, infinite dividend) is
excluded. The grid is mutation-proved where it now lives: drop `DuckF64`'s NaN sign and it
fails, naming the `fmod`-by-infinity rows; unmutated it passes on both backends.

*What closed it:* `fix-nan-sign-varchar`, four iterations deep — the formatter, then the
`Fneg` lowering plus eight review findings, then eight more (the residual classifier, a
bit-level two-backend `fneg` test, `-nan` surviving the IR text round trip, the cast-trap
text pinned as DuckDB's message truncated, and the stale docs), then iteration 4's eight as
`c361549` — the `||` collapse above plus seven comment, pin and coverage findings — and the
salvage as `550f949`. One review finding was **rejected with measurement** rather than
absorbed: the pin the review read as platform-dependent uses a defined constant's bit
pattern, which the repo rule exempts, and the new Rust pin builds both NaNs from explicit bit
patterns anyway.
*Gate (branch tip `550f949`, master `f81e17c`):* **PASS.** Suite 3281 -> **3320** passed, 1
skipped, 3 xfailed, the same two absent-`pyspark` collection errors, with **39 new node ids
all passing, 0 shared-id outcome changes, 0 disappeared**; the separately built **debug**
engine gives the identical set; `cargo test -p confit --lib` 266 -> **269** passed with the
failing **set** identical, the same five names; campaign `DIVERGE_VALUE` **1 -> 0** with
**exactly one seed flipped** of 2000 — seed 1804, its
`struct_pack(f0 := CAST(pow(-0.25e0, 0.1e0) AS VARCHAR))` where master wrote `nan` and DuckDB
writes `-nan` — every other 1999 seeds identical in (kind, klass), compared at full-record
granularity rather than at the summary level; corpus **547 / 131 / 0** and dialect L2
**288/678** on both legs; the public API diff is empty. Five mutations, each caught and each
restored by re-edit: the `DuckF64` sign -> 12 red, `fneg` back to `-0.0 - x` on both backends
-> 6, the fold plus the `NullOf` early return -> 20, unteaching `scan_residual` about `Fneg`
-> 4, the old residual message -> 1. The **+23** over the branch's own previous 3297 is 20
null-typing cases (10 shapes x 2 backends), the 2 salvaged mod-grid rows and 1 residual
refusal.
*Two caveats the gate names, neither an API or format change:* the out-of-range
DOUBLE-to-int cast trap **text** now spells the value DuckDB's way (`NaN` -> `nan` / `-nan`),
and JOIN residual refusals gained a second wording; both keep their `ValueError
unsupported:` / `Conversion Error` classes, and `known-limitations.md` records the remaining
` INT64` gap.
*Left open by a review of the gated tip:* **seven findings, none HIGH.** Three medium —
`Lit`'s `PartialEq` still compares opposite-signed NaNs equal, so the print/parse round trip
this branch strengthened cannot itself see a dropped sign; the IR generator gained `Fneg` but
no negative-NaN constant, so the fuzz round trip has zero coverage of the sign added; and the
new "classifier does not recognise" refusal selects on `known || !(left && right)`, which
still misdiagnoses a two-sided residual whose column references sit inside the unrecognised
node — and four low. Every one is a naming, coverage or comment defect over a production diff
the gate found clean; none is a wrong answer.

### 2.3 finding: static-only-tie-order

*Target:* exclusion: whole-relation-shapes — inside the static-tables-only carve-out, what a
whole-relation construct selects is frozen **only when it is a function of the query**.
*Was:* nothing refused a tie-producing `ORDER BY`; two builds of the same function could
freeze different orders. A silent-wrongness class.
*Now:* a rule that refuses it, rebuilt on the principle rather than the shape and generalized
well past both — gated **PASS**, and blocked on one node type the rule still reads wrong.

**The rule generalized.** The first iteration read the query with `sqlparser` and asked about
`ORDER BY` ties. Its gate returned **FAIL** on two independently reproduced defects: one pair
of parentheses around the whole query turned the check off (the probe read only the outermost
node, while its own sibling row-limit rule recursed), and a `DISTINCT` shape that master
served now refused while the branch's doc edit claimed it still served. A review added the
deeper one: any query `sqlparser` cannot parse **silently skipped the check entirely** — and
DuckDB-only dialect is exactly what the carve-out exists to serve.

The second iteration did not patch that; it **dropped all three commits** and rebuilt the
rule on the principle rather than the shape: *the fold is a pure function of the query and the
statics.* The reading is now **DuckDB's own parse**, via `json_serialize_sql` on the same
connection that folds the query, with `sqlparser` deleted from this path; where even DuckDB
will not serialize a statement, a tokenizer fallback **refuses** rather than falling through.

*What now refuses, and why:* row limits (`LIMIT` / `OFFSET` / `FETCH` / `SELECT TOP`) before
the fold, keeping master's messages byte-identical; then, after the fold, `USING SAMPLE` /
`TABLESAMPLE`, `DISTINCT ON`, `QUALIFY`, and the row-position window functions
(`row_number`/`ntile`/`lead`/`lag`/`first_value`/`last_value`/`nth_value`) — each of which
**picks** a row out of a group; then the tie probe on a top-level `ORDER BY`, run only when
the frozen result has more than one row. Plain `DISTINCT` still serves (it collapses a set,
it does not pick), and so does the rank family (a function of the key, deterministic under
ties). An `ORDER BY` below the top serves on a stated basis rather than an assumption: row
order on the constant path is not part of the contract, and a 60k-row measurement under five
DuckDB settings a build machine picks for itself gave **one** answer for the row set and
**five** for the sequence. One ceiling is stated rather than hidden: a window aggregate with
its own `ORDER BY` is order-dependent under ties and is not caught.

*What the rebuild added on top of that*, and what the first text could not see because the
branch returned no record: two **catalogue** rules, read off the same connection that folds
the query. A function whose `duckdb_functions().stability` is `VOLATILE` or
`CONSISTENT_WITHIN_QUERY` refuses (`random`, the clock family); and an aggregate refuses **by
name** unless it is one of the eleven DuckDB's own source flags `NOT_ORDER_DEPENDENT`, which
is **65 of the 88 distinct aggregate names**, `sum` and `avg` among them. The coarseness is
deliberate and fail-closed — DuckDB's flag is the authority and it defaults to
`ORDER_DEPENDENT` — and the upgrade path (read the bound overload off the result type) is
written down rather than guessed at. It is also the whole of the branch's acceptance loss;
the campaign delta below prices it.

*The generator's planted coverage, and the over-refusal detector.* The campaign could not
reach this shape at all — over seeds 0-39999 the gate found 28 static-only `ORDER BY` cases
and **none of them can tie**. So the generator plants both twins on an auxiliary stream
consulted before the main one: **1% tie, 1% unique**, 44 of 2000 seeds claimed, the other
1956 byte-identical. Grading covers both directions — a unique twin refused under the tie
class is a `DIVERGE_BUILD` **tie-over-refusal**, a tie twin that agrees is a
**tie-under-refusal**. That matters beyond this rule: as the first review established, the
campaign is **structurally blind to over-refusal** — `REFUSED` is terminal and is never
compared against the oracle — so this is the first check in the fuzzer that can see one.

*Gate (branch `21e3fdc`, iteration 3's rebuild):* **PASS WITH FINDINGS.** Suite 3281 ->
**3358** passed on release and on the debug build, 1 skipped, 3 xfailed, the same two
`pyspark` errors; `cargo test` 266 / 5, master's exact set; corpus **547 / 131 / 0** on both
legs; **84 flips**, every one attributed; mutation caught (`if false &&` on the tie-refusal
return -> 20 red), and a second mutation on the generator's own determinism caught too —
tying a unique twin while keeping its determined tag raises `DIVERGE_BUILD` on seeds
25 / 86 / 193. 22 adversarial probes beyond the suite all land right. The gate named two
over-refusals, both fail-closed, both disclosed in `known-limitations.md`:
`first(v ORDER BY k)` with a unique `k` refuses whole (the branch never reads an aggregate's
own `ORDER BY`), and `sum` refuses by name even over integers.

*The review that followed found eleven, three of them HIGH* — and all three were the same
shape: a **fail-open** path wherever the reading gave up. A `LIMIT`/`OFFSET` whose argument
is not a bare constant (`LIMIT 1+1`, `CAST(2 AS BIGINT)`, `(SELECT 2)`) read as a no-op limit
and **served** a scan-order row subset, a regression against master, whose deleted
`sqlparser` walk refused any limit clause. A string holding more than one statement returned
early and skipped **every** value rule, so `"SELECT 1 AS o; SELECT random() AS o FROM s"`
served three frozen draws. And a statement DuckDB will not serialize but will run —
`PIVOT s ON g USING first(v)` — took the same early return and served, which is exactly the
case the docs claimed "refuses rather than falling silent".

*What iteration 4's fix round closed:* **nine of the eleven; two rejected on the facts.** The
three HIGHs close by splitting the two things one predicate had conflated — JSON `null` (a
genuinely absent limit side, still a no-op) from a node that is present but unreadable (now a
real limit) — and by refusing on the **statement count** and on the serialization failure
alone, with no ordering word required. Four mediums close with them: built-in **macros** that
read the clock (`stability` is `NULL` for all 131 scalar-macro rows, so nine clock macros
served a frozen value), the catalogue being read before the query binds, `rowid` projecting a
physical scan position as an ordinary value, and the class the review named as `stability`
answering "constant within one query" rather than "a function of the query text". That last
one was **worse than reported**, and the fix round found the extra case in the pinned source
rather than by probing: `current_localtimestamp` is registered in
`extension/icu/icu-timezone.cpp` with no `SetStability`, so it inherits `CONSISTENT` — and
its value **moved between two connections 50 ms apart**, measured. The bare words
`localtime` / `localtimestamp` were already refused; the call spelling of the same function
was not.

*The two rejections, both measured rather than argued.* Adding `sum_no_overflow` to the
order-free list would be dead code: it is in the catalogue but it does not bind
(`sum_no_overflow is for internal use only!`), and a name no query can call cannot be
over-refused. And widening the over-refusal detector past the planted twin was rejected as
**unsound**, not deferred: the tag means "every value here is fixed by the query and the
statics", and the generator cannot assert that of a case it has just built without
reimplementing the rule under test — tagging a generated case that carries `first`, `avg` or
a real `LIMIT` would manufacture false findings, which is worse than the silence complained
of.

*Gate (branch `a08147e`, iteration 4):* **PASS**, with two over-refusals and a build-time
cost, all three intended and disclosed. Suite 3281 -> **3366** passed on release and on the
debug build, 1 skipped, 3 xfailed, the same two `pyspark` errors, with **85 new node ids, 0
disappeared, 0 shared-id outcome changes**; `cargo test` 266 / 5, master's exact set; corpus
**547 / 131 / 0**, `MATCH_FLOOR` met exactly; the public API diff is empty; two mutations
caught (22 red and 5 red), each restored by re-edit with a clean tree afterwards; nine hand
probes and five controls all land right, and all five of `goal.md`'s code claims reproduce
verbatim.

*Campaign delta, and the honest reading of it.* `AGREE` 1013 -> **1003**, `REFUSED` 944 ->
**955**, `AGREE_TRAP` 21 -> **20**, `UNSHIPPED` 14, `DIVERGE_OPT` 7, `DIVERGE_VALUE` 1, with
**0 `DIVERGE_BUILD`** and seed 1804 still the `DIVERGE_VALUE` on both legs. **84 flips, 33 of
them kind flips, every one attributed** — and the fix round adds **zero** of them: the 84 are
identical to the pre-fix branch's 84, and the SQL text is byte-identical on all 1956
unclaimed seeds. They split three ways. (A) **35** are seeds the planted static-order stream
claims — all 24 tie twins `REFUSED` under the tie class, all 20 unique twins `AGREE`, the
nine that did not flip already agreeing. (B) **41** are `REFUSED` -> `REFUSED` message-class
changes on identical SQL, every one verified **dynamic**: master mislabelled a row-path limit
error as a static-tables-only refusal, and the branch returns the row path's own wording. A
strict improvement. (C) **8** are `AGREE` -> `REFUSED` on identical SQL, all static-only
aggregates — `sum` at seeds 130 / 545 / 975 / 1484 / 1967 and `avg` at 113 / 1036 / 1314 —
which is the by-name coarsening above arriving as acceptance loss. The arithmetic reconciles
exactly: -13 +11 -8 = -10 on `AGREE`, +14 -11 +8 = +11 on `REFUSED`, -1 on `AGREE_TRAP`.
These figures replace the first text's for this branch (`4eee7f3`: 35 flips, `AGREE` 1011,
`REFUSED` 947); the count grew because the rule did, not because the campaign moved. What the
fix round newly refuses, the generator never produces, so the campaign is silent about it by
construction and each shape is pinned by a unit test instead.

*The cost, measured rather than estimated.* One static-only fold goes **15.6 -> 64.5 ms**
(`max(v)`) and **13.4 -> 72.5 ms** (`ORDER BY k`), roughly 4-5x, over 50 runs each, from the
extra `json_serialize_sql` / `duckdb_functions` / tie-probe round trips. The row path is
unchanged (0.20 -> 0.17 ms) and serving is untouched. That is the whole full-suite delta
(258s -> 365s), and it is identical on the debug build — which is what proves it is DuckDB
round trips and not our codegen.

*Status:* **blocked**, on one HIGH a review measured after the gate passed. `ORDER BY #N` —
DuckDB's positional **output** reference — is not one of the three node types the tie probe
reads, so it falls into the hidden-key arm and is re-emitted into the probe's `SELECT` list,
where `#N` binds to the **N-th input column** instead. The probe then measures the wrong key:
`SELECT g AS o, 1 AS c FROM s ORDER BY #2` serves a frozen order over a key that ties all
three rows, while the same query spelled `ORDER BY 2` refuses. That is the silent-wrongness
class this branch exists to close, still open inside it, and it is why the branch is not a
merge candidate as it stands. Four lesser findings ride with it (for-the-next-reading below).

---

## 3. Gate state, branch by branch {#gate-state}

| branch | tip | gate verdict | campaign delta vs master (2000 seeds) | corpus | state |
|---|---|---|---|---|---|
| `fix-corpus-slip` | `a7c5798` | PASS WITH FINDINGS | 0 flips; census identical | 547 / 131 / 0 both legs | **merged** (PR #200) |
| `fix-nan-sign-varchar` | `550f949` | **PASS** | 1 flip: seed 1804 `DIVERGE_VALUE` -> `AGREE`; `DIVERGE_VALUE` 1 -> 0 | 547 / 131 / 0 both legs | **awaiting the owner's approval click** |
| `fix-fmod-sign` | `ec71979` | PASS, then read as redundant | its one flip was the same seed 1804, already carried by the branch above | 547 / 131 / 0 both legs | **dropped**; grid salvaged |
| `refuse-static-tie-order` | `a08147e` | PASS (two disclosed over-refusals, 4-5x static-only build cost) | 84 flips, all attributed; `AGREE` 1013 -> 1003, `REFUSED` 944 -> 955, `AGREE_TRAP` 21 -> 20 | 547 / 131 / 0 both legs | **blocked** on one HIGH |

Suite counts, each from the gate that produced it: master **3281** passed / 1 skipped / 3
xfailed / 2 errors on every leg; `fix-nan-sign-varchar` **3320**; `refuse-static-tie-order`
**3366**; `fix-fmod-sign` stood at **3284** when it was gated, and is moot now. Every branch
was also run on a separately built **debug** engine with identical results and no
`debug_assert` firing. Dialect L2 reads **288/678** with 0 FAIL wherever it was taken. The
public API diff is empty on all four, and `cargo test`'s failing **set** is master's five on
all four.

**The contention resolved by measurement, not by a choice.** `fix-fmod-sign` and
`fix-nan-sign-varchar` both changed the same `DuckF64` NaN arm and both flipped the same
single seed, and the first text left the pick to the owner. Iteration 4 removed the pick:
`fix-fmod-sign`'s production change **was** that one line, already carried on the other
branch, so there was nothing to choose between. Its 29-row sign grid over `%` / `mod` /
`fmod` now lives on `fix-nan-sign-varchar` as tests, and the branch is dropped. One candidate
remains for kpi: engine-parity, and it is a click away rather than a decision away.

**Both live branches carry review findings their gates did not raise, and the two are not the
same weight.** On `fix-nan-sign-varchar` the seven are naming, coverage and comment defects
over a production diff the gate found clean — an equality that cannot see a NaN sign, a
generator that never emits one, a refusal selector that misdiagnoses a shape it already
refuses correctly. None of them answers a query wrongly. On `refuse-static-tie-order` one of
the five **does**: `ORDER BY #N` freezes a tied order and serves it. A control violation
inside the branch chartered to close that control is a different class of open item from a
stale comment, and the table above states the two differently on purpose.

---

## 4. Measured facts for the next full reading {#for-the-next-reading}

Facts this loop produced that belong in reading **N=2**, not in this report's conclusions.

**Acceptance changed, and not only by fixing things.** `refuse-static-tie-order` moves
`AGREE` 1013 -> **1003** and `REFUSED` 944 -> **955** on the same seed range — but the
comparison is **not like-for-like**, for two reasons that pull differently. 44 of the 2000
seeds are now planted twins rather than grammar draws, so roughly **2%** of every campaign is
two fixed queries; one displaced seed carried an `AGREE_TRAP` that nothing else covers (the
whole of the 21 -> 20 move), and seed numbers cited in older repros silently change meaning.
Separately, **8** of the flips are a deliberate over-refusal — `sum` and `avg` on the
static-only path — so part of the acceptance loss is a rule the loop chose, not a population
artefact and not a defect. Any next census over this generator is measuring a slightly
different population under a slightly stricter rule; the baseline's validity caveat under
acceptance-reading now has two reasons to bite rather than one.

**The over-refusal detector is class-agnostic; the generator is not, and that is now an
argued position rather than an oversight.** A test pins the detector as reading every refusal
class, but the determined tag is set only on the planted unique twin, so **no generated**
static-only case is ever graded for over-refusal — and the branch's new 65-name aggregate
refusal surface therefore has no campaign-level regression detector at all. The fix round
**rejected** widening it: the tag asserts "every value here is fixed by the query and the
statics", and a generator cannot assert that of a case it just built without reimplementing
the rule under test, which would manufacture false findings. So the silence stands, with a
reason. Every new message still carries the documented `unsupported:` prefix, so gap:
undocumented-refusal-prefixes does not grow — but the naming half of kpi: no-third-mode gains
its untested claims regardless.

**The corpus count did not move again.** 547 / 131 / 0 on every leg of every branch this loop
gated, iterations 3 and 4 included — and the tie branch is the interesting case, because it
adds a whole new refusal surface and still meets the floor **exactly**, neither tripping it
nor growing past it. `MATCH_FLOOR` holds at the current count with **zero headroom**, so the
next correct new refusal trips it on the day it lands, by design. Six gates have now named
that explicitly.

**The Rust unit gate is red on master and CI cannot see it.** `cargo test` is 266 passed / 5
failed on every master leg the loop ran, the same five names each time
(`pin_ftoi_rounding_and_traps`, `pin_ssubstr_window_arithmetic`,
`pin_stoi_trims_whitespace_like_duckdb_cast`, `table_and_custom_partition_the_catalogue`,
`substr_window_arithmetic_via_sql`), and CI runs only `pytest`. **Seven** separate gate
records now say so independently. A regression inside `exec::tests` would pass a green-bar
check today. This is an **enforcement fault**, the shape the baseline reading calls a finding
rather than a gap, and no item in this loop owned it — iterations 3 and 4 included, which
added four more records of it and no owner.

**Closed since the first text, each by a gate leg rather than by an author's claim:** the
`- <DOUBLE NULL>` collapse; the trailing-`;` and trailing-comment false refusals; the three
fail-open paths in the tie rule (expression limits, multi-statement strings, unserializable
statements); the clock functions DuckDB's own flag calls `CONSISTENT`; `rowid`; and the tie
rule's residual classes the first text listed — `ORDER BY COLUMNS(...)`, `ORDER BY *`, a
frozen `random()` sort key and the selection-by-position aggregates all refuse now, each
spot-checked on the built branch by a gate.

**Still open, each measured, none acted on:**

- **`ORDER BY #N` on `refuse-static-tie-order`** — the one open item that answers a query
  wrongly rather than refusing or misnaming. Positional **output** references are not among
  the node types the tie probe reads, so the probe re-emits `#N` into its own `SELECT` list
  where it binds to the N-th **input** column and measures the wrong key. `ORDER BY 2`
  refuses; `ORDER BY #2` over the same tied key serves.
- **Seven review findings on `fix-nan-sign-varchar`**, none HIGH: `Lit`'s `PartialEq`
  compares opposite-signed NaNs equal, so the print/parse round trip cannot see a dropped
  sign; the IR generator emits no negative-NaN constant, so the fuzz round trip never
  exercises the sign this branch added; the "classifier does not recognise" refusal selects
  on `known || !(left && right)` and still misdiagnoses a two-sided residual whose columns
  sit inside the unrecognised node; plus four low (a `snapshot_bits` doc that forbids what a
  test in the same crate correctly pins, a `known-limitations.md` cast-message claim that
  understates the divergence, a bind-time `fold` that reproduces arith's over-fold, and half
  the mod grid's excluded domain — the infinite-dividend half — covered by nothing).
- **The by-name aggregate coarsening**, disclosed rather than fixed: **65 of 88** aggregate
  names refuse on the static-only path, including `count_if`, `regr_count`,
  `approx_count_distinct`, `entropy` and the compensated accumulators `fsum` / `kahan_sum` /
  `favg` that exist to **be** order-stable. `first(v ORDER BY k)` with a unique `k` refuses
  whole for the same reason. The upgrade path (read the bound overload off the result type)
  is written down; the trade is the owner's to price.
- **A table function that reads the machine** (`SELECT ... FROM duckdb_settings()`) is the
  same class as the clock functions the fix round closed and is **not** covered. Named in
  `known-limitations.md`, not fixed: covering it means a reading of all 127 table-function
  rows, which is a different piece of work.
- **Four `known-limitations.md` line citations in `goal.md` are wrong** — rebased by +73 when
  the real shift is +161, so each now points at unrelated text. `goal.md`'s whole verified-by
  mechanism is line citations, which makes this a small edit against a load-bearing claim.
- **Two lesser tie-branch findings:** a refusal that names `SELECT TOP` when the query merely
  has a static column named `top`, and a `readable` guard that can never be false (one field,
  one initializer per arm, one dead `&&`).
- `x % y`'s NaN sign is **unmatchable in principle**, not merely unfixed: two gates
  independently reproduced DuckDB answering **43 identical rows two ways in one query** —
  the vectorized lanes give one bit pattern, the scalar tail another — stable across 20
  repeats. The salvaged grid correctly excludes the domain instead of pinning it, and
  iteration 4 confirmed the exclusion rather than narrowing it.
- A **single-side** negated DOUBLE in a JOIN ON residual is still a **provable
  over-refusal**: a sign flip is total, but `may_trap`'s catch-all still counts it as
  trapping. One line closes it; iteration 4 deliberately left it out again, and split the
  refusal's message instead, so acceptance did not widen inside a review-closure branch.
- The out-of-range cast trap's text gap is **wider than the first text said**: a review
  measured DuckDB's message as `... destination type INT64 when casting from source column f`
  — six extra words, one of them a column name — and the branch's own gate asserts only a
  `startswith`, so nothing in it can catch the overstatement.
- The unsigned-column refusal class — the reason three mined statements now refuse — is named
  **nowhere** in `known-limitations.md`. Same bookkeeping shape as gap:
  undocumented-boolean-comparison, and it belongs with that entry.
- The four stale `550` sites and the `known-limitations.md` line the corrected README points
  at (gap: corpus-match-slip's remaining half).

---

## 5. Spend {#spend}

| iteration | agent tokens |
|---|---|
| 1 (corpus slip, seed 1804, tie order — three implementers, three gates, two reviews) | ~1.45M |
| 2 (seed 1804 rework, tie order rebuild — two implementers, two gates, two reviews) | ~1.33M |
| 3 (seed 1804 closure, the modulo branch, the tie rule's rebuild and its gate) | **~1.76M** |
| 4 (seed 1804 review closure and salvage, the tie fix round — two implementers, two gates, two reviews) | see next report |

Iteration 3's figure is now recorded: **~1.76M**, the largest of the three, which is what a
gate that rebuilds both legs from source and a review that probes a built branch cost when
three items run at once. **Cumulative across iterations 1-3: ~4.5M agent tokens.** Iteration
4's figure is not recorded on the same basis in this run's journal, so it is left for the
next report rather than restated on a basis that would not compare. What is visible of it:
two branches, two independent gates, two reviews and one fix round, on the same three-role
shape.

**The standing stop rule is unchanged: roughly 70% of the owner's weekly credit, and it is
owner-signalled** — the loop does not infer it from its own accounting, and has not been
signalled to stop.

---

## 6. Next, in the goal's order {#next}

`goal.md` orders controls before drives, so the queue does too. Nothing here is chosen; it is
what the loop's own measurements rank.

1. **The parity control needs a click, not a decision.** kpi: engine-parity has
   `DIVERGE_VALUE` at **0** on exactly one branch now — `fix-nan-sign-varchar` at `550f949`,
   gate **PASS**, the `- <DOUBLE NULL>` regression closed, `fix-fmod-sign`'s grid carried as
   tests and that branch dropped. What the first text posed as a choice between two branches
   is settled by measurement: the loser had no production change of its own. The only
   question left is whether to merge as it stands or hold for the seven review findings, and
   none of those answers a query wrongly.
2. **finding: static-only-tie-order is one node type away from closed.** The branch is gated
   **PASS** at `a08147e` with three fail-open paths shut, but `ORDER BY #N` still freezes a
   tied order and serves it — the same silent-wrongness class the branch exists to close,
   now a bug with a named location rather than a rule question. A control violation is never
   a gap to live with, so this outranks every `gap:` entry below it.
   The rule question that remains is separate and is the owner's: the by-name aggregate
   refusal spends **65 of 88** aggregate names and **8 of 2000** campaign seeds to buy the
   control, and a static-only fold costs **4-5x** more build time for the same reason. The
   policy (never trade a control for a drive gain) says take it; the size is now measured, so
   the trade can be priced rather than assumed.
3. **The enforcement faults nobody owns.** The red Rust unit gate above, and finding:
   c1-depth, untouched by this loop and still routed to ask: kpi-set-change.
4. **gap: bench-baseline-flip's cheapest cause is still untested.** One re-run after
   `--reinstall-package` rules out the stale-wheel signature (d). No iteration in this loop
   touched it, and kpi: bench-refresh-cadence should not be adopted before it is settled.
5. **Then the gap ledger, in whatever order ask: next-query-classes gets answered.** That
   question is the owner's and remains open; the loop has added no evidence that reorders its
   candidates, only evidence that gap: corpus-match-slip's ratchet half is now real and its
   bookkeeping half is not.

**Reading N=2 replaces none of this.** This report is what moved between readings; the next
full reading is what the numbers are.
