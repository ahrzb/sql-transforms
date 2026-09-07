# The loop's first report: iterations 1-8 (2026-09-02, amended 2026-09-06 and 2026-09-07)

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

**What the second amendment changed.** Iterations 5, 6 and 7 (2026-09-05/06) ran the same
three-role shape on the two branches still live. The parity branch reached a ship gate and is
now PR #202. The tie branch went through two further fix-and-review rounds, is gated **PASS**
again, and was found fail-open again — by a fourth review, over a surface nobody has finished
enumerating. That repetition is itself a measurement and has its own section below
(enumeration-not-terminated); it is stated there as a **fork put to the owner**, not as a
decision this loop took.

**What the third amendment changed.** Iteration 8 (2026-09-07) closed the parity item and ran
a fifth round on the tie branch. The parity branch **merged** on the owner's approval, so its
row here is a record rather than a candidate; between its ship gate and that merge a design
pass and the orchestrator's own read of the whole diff found a **real parity bug** no gate had,
so that closure is written up with what found it. The tie branch closed the seventh fail-open
that same reading habit had found, then closed the round-4 review's six — three of the five
closures are **allow-lists and metadata reads** rather than longer name lists — and was gated
**PASS** again. A sixth review then found four more shapes that serve a value which is not a
function of the query. The enumeration section below says what that does to the sequence.

**Slugs.** `gap:` and `finding:` citations resolve in the baseline reading; `goal:`,
`kpi:`, `exclusion:` and `ask:` in `goal.md`; `claim:` and `divergence:` without a local
definition in `packages/confit/docs/oracle/`. Sections here carry kebab-case anchors and are
cited by slug, never by number.

---

## 1. What the loop is {#the-loop}

The owner's mandate, as it stood over these seven iterations: **make confit match `goal.md`**
— close the distance the baseline reading measured, in the order the goal document's own
priorities give. Its working rules:

- **No tickets.** Work is dispatched against a `gap:` or `finding:` slug, not a ticket
  number, and nothing in this loop filed one.
- **No implementation check-ins.** The owner is not consulted mid-branch. What reaches him is
  a gated branch and a dated report.
- **Gated PRs and dated reports are the only outputs.** A branch is not done when its author
  says so; it is done when an independent gate has rebuilt both legs and reproduced or
  refuted every clause of its promise.
- **A merge needs the owner's own GitHub approval.** Nothing in this loop merged itself. **Two**
  branches merged in this window, each on his click.

The shape each iteration took: one implementer per item, then an independent **review** and
an independent **gate**, each in its own worktree, each building master and the branch from
source rather than trusting a shipped artifact. That structure is the reason this report can
name a gate for every number: the gate legs are what produced them.

**A fourth leg was added at the owner's correction, and it paid immediately.** From 2026-09-06
the orchestrator reads every diff itself rather than treating a PASS gate as the review. That
reading is what found the `nextafter` NaN-sign parity bug on a branch two gates had passed, the
tie branch's seventh fail-open, and four doctrine slips in `goal.md` — none of which a gate or
a probing review had. Gates are evidence; they are not the review. The rule that follows from
it is also now standing practice: findings are closed **before** a PR is presented, so a branch
goes back to draft rather than forward with a list.

**One measurement about the loop itself, worth keeping.** Every master leg the loop ran —
**eleven of them across iterations 1-4, and every leg since** — reproduced the baseline's
campaign census exactly: `AGREE` 1013 / `REFUSED` 944 / `AGREE_TRAP` 21 / `UNSHIPPED` 14 /
`DIVERGE_OPT` 7 / `DIVERGE_VALUE` 1 over seeds 0-1999 at `--workers 8 --timeout 20`. Master
advanced twice in the window (`2ba96e5` -> `2c7c05c` -> `f81e17c`, docs plus one added
assertion) and the census did not move; it has stood at `f81e17c` since, across the four
further master legs iterations 3 and 4 ran, each of which also re-checked the sanity seed and
found 1804 the `DIVERGE_VALUE`. Iteration 7's gate reproduced the same six counts at the same
tip four days later, seed 1804 included. claim: campaign-verdicts-today is reproducible on
this machine, and a single seed flipping is therefore signal, not scheduling noise.

**Iteration 8 is where that census finally moves, and it moves for the stated reason.** Master
advanced a third time, to `8796bb2`, and the counts went to `AGREE` **1014** / `DIVERGE_VALUE`
**0** — one seed, 1804, the one the parity branch was chartered to fix. Every other count is
unchanged. So the invariant this loop has been leaning on holds in the strongest form
available: eleven-plus master legs held the census fixed while master moved for docs, and the
one master change that was supposed to move it moved exactly the one seed it named.

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

*Shipped (iteration 5).* All three mediums are the subject of `5819c3a` — `Lit` equality and
the IR round trip can both see a NaN's sign now, the generator emits both signs and `-0.0`,
and the join-residual refusal names the node it actually failed to recognise — and the
**ship gate** read that tip and returned **PASS**. The branch became **PR #202** and stood open
for the owner's approval click, the loop's only candidate for kpi: engine-parity. Its
ship-gate numbers over the same seeds 0-1999 at `--workers 8 --timeout 20`: master `AGREE`
1013 / `REFUSED` 944 / `AGREE_TRAP` 21 / `UNSHIPPED` 14 / `DIVERGE_OPT` 7 /
`DIVERGE_VALUE` 1, against branch `DIVERGE_VALUE` **0** and `AGREE` **1014** — **one flip**,
seed 1804 — with the suite outcome-identical on every shared node id on release **and** debug,
`cargo
test`'s failing set unchanged, the corpus **547 of 678** at its floor, and the public API
surface unchanged.

*One process note on it, the same shape the tie branch's gate raised below.* PR #202's head is
`81e8fa2`, **one commit past the gated `5819c3a`**: a docs-and-comments commit that
reclassifies the out-of-range cast trap's **text** as a **diagnostics pin rather than a
divergence** — both engines erroring at run time is `AGREE_TRAP` and the two messages are
never compared, so that pin was never recording a divergence — and drops the
`known-limitations.md` row that listed it as a limitation. It touches no production code and
it closes one of the four lows above by reclassifying it, but no gate record in this loop's
journal named that tip when it was written. Iteration 8 closed that gap the only way it can be
closed: a gate read the **final** tip, after the design pass and the review fixes, and it is
the histogram quoted below.

**Merged (iteration 8, 2026-09-06).** PR #202 is on master, rebased, at tip `8796bb2`, on the
owner's approval click. kpi: engine-parity now reads `DIVERGE_VALUE` **0** on master rather
than on a branch. The final-tip gate over seeds 0-1999 reads `AGREE` **1014** / `REFUSED`
**944** / `AGREE_TRAP` **21** / `UNSHIPPED` **14** / `DIVERGE_OPT` **7** / `DIVERGE_VALUE`
**0** — a histogram **identical** to the ship gate's, so nothing the merge window added moved
a verdict — with the root suite at **3326** passed / 1 skipped / 9 xfailed / 2 errors (the
same absent-`pyspark` collection pair) and `cargo test` on release **and** debug showing the
five pre-existing failures and no others.

**What the merge window found, and it was not the gate that found it.** Between the ship gate
at `5819c3a` and the merge the branch took a **design pass** — one home per duplicated rule:
`fold_operand`, `out_of_range_trap`, the NaN-sign argument stated once at `Lit`, one
`inf`/`nan` token path — and then the orchestrator's own read of the whole diff. That read
returned four items a **PASS** gate had not: a doc over-claim (`fold_operand` said *every*
strict numeric operator, while comparisons deliberately elide no NULL), a stale `cmp` comment,
one production `f64::NAN` literal, and a **real parity bug**. `duck_nextafter` answered every
NaN input with a fresh **positive** NaN, where DuckDB hands a NaN operand back **with its own
sign**: `nextafter(-nan, 1.0)::VARCHAR` printed `nan` against DuckDB's `-nan`. The campaign
could not have caught it — it never feeds a NaN into `nextafter`, and that blind spot is now
named rather than assumed.

**The fix took two attempts, and the second one is the lesson.** The first pinned *which*
operand is returned when **both** are NaN, measured on Windows; Linux CI measured the **other**
operand, because that choice belongs to the platform's C runtime and its compiler. The kernel
now calls the platform's own C `nextafter` — the function DuckDB's `std::nextafter` calls — so
it matches by construction wherever it is built: lone-NaN rows are pinned, both-NaN rows are
**compared against the machine's own oracle and never pinned**. Three facts carry forward. A
sign a libm or a compiler picked is compared, never pinned. A pin measured on one platform is
not a pin, and **CI is the cross-platform leg**. And a gate that rebuilds both legs is
evidence, not a review: reading the diff is what found this one.

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
one was **worse than reported**, and the fix round found the extra case by reading rather than
by probing: `current_localtimestamp`'s catalogue row says `CONSISTENT`, and its value
**moved between two connections 50 ms apart**, measured. The bare words
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

*Status at the end of iteration 4:* **blocked**, on one HIGH a review measured after the
gate passed. `ORDER BY #N` —
DuckDB's positional **output** reference — is not one of the three node types the tie probe
reads, so it falls into the hidden-key arm and is re-emitted into the probe's `SELECT` list,
where `#N` binds to the **N-th input column** instead. The probe then measures the wrong key:
`SELECT g AS o, 1 AS c FROM s ORDER BY #2` serves a frozen order over a key that ties all
three rows, while the same query spelled `ORDER BY 2` refuses. That is the silent-wrongness
class this branch exists to close, still open inside it, and it is why the branch is not a
merge candidate as it stands. Four lesser findings ride with it (for-the-next-reading below).

**Round 3 (iterations 5-6, `9f16246` then `e85aa09`): the positional key, the last alias, and
a typed sum.** `ORDER BY #N` closed first — a positional **output** reference is read as a
position in the output now, so the probe measures the key the query names rather than the
N-th input column. That round's own ship gate then found a second one no review had: the
branch resolved an `ORDER BY` alias to the **first** match where DuckDB's binder takes the
**last**, so a repeated output name measured the wrong column. And the round-3 review added
two more of the by-then-familiar shape — `POSITIONAL JOIN`, which pairs row *i* with row *i*
and is therefore a scan-order construct, and the question left unasked for every other join
reference type. Its gate returned **PASS** with a battery of roughly **110 probes** clean.

*The typed-sum rule, and what it bought back.* Iteration 4 refused `sum` **by name**, which
cost eight campaign seeds and was the whole of the branch's acceptance loss. Round 3 replaced
the name with **DuckDB's own overload resolution**: `DESCRIBE` the folded statement, read the
result type, and let an **exact** accumulator — integer, hugeint, decimal — serve while the
floating overload keeps refusing. That is the upgrade path iteration 3 wrote down rather than
guessed at, taken. The effect reconciles exactly against the flips iteration 4 named: the four
exact-`sum` seeds it lost (130 / 545 / 975 / 1484) come back, and the fifth, 1967, does not,
because that seed's generated schema types `c0` as `DOUBLE`.

**Round 4 (iteration 7, `2d24744`): four more fail-opens, closed by naming what serves.** The
round-3 review found four, each the same shape — a reading that gave up and served. `ASOF
JOIN` draws **one** of the tied inequality matches: measured at **15 distinct answers** for a
single scalar sum, 3000 x 150000 rows with 50 ties each, seven DuckDB settings x three
connections. `min` / `max` under a **collation** pick a representative among values the
collation calls equal. The `OrderBinder` fallback re-emitted an unresolved key into the
probe's own `SELECT` list, where it bound over the **input** columns and measured the wrong
thing. And a **table function that reads the machine** — the item the first amendment listed
as open and not covered.

*Both fixes are allow-lists, and that is the point.* Join reference types are now decided for
all **six** of DuckDB's `JoinRefType` values: `REGULAR` / `NATURAL` / `CROSS` serve, because
they pair by **values** under any `join_type`; `POSITIONAL` and `ASOF` refuse by name;
`DEPENDENT` is unspellable (`LATERAL` serializes as one of the first three); and anything
DuckDB adds later refuses, because the list enumerates what **serves**. The table-function
rule inverts its polarity the same way — `range` / `generate_series` / `unnest` / `repeat` /
`repeat_row` serve and **every other table function refuses by name**, which covers all **39**
nullary catalogue entries (`duckdb_settings`, `pragma_version`, `duckdb_functions`,
`test_all_types`, ...) and `read_csv` / `read_parquet` / `glob` / `query` / `query_table` for
free. The name is read from `$..function.function_name`, so a scalar `repeat()` or `range()`
sharing a spelling is untouched.

*Gate (branch `2d24744`, master `f81e17c`):* **PASS**, with one process finding worth keeping
as a loop fact: **the branch tip was one commit ahead of the tip the dispatch named**, and the
gate verified the real tip. It also built the named `e85aa09` to check that commit's own
claim — the two are identical seed-for-seed on the campaign, so the claim holds, but
`e85aa09`'s corpus is 547 and `2d24744`'s is 546, so reviewing the named tip would have missed
four rules and a floor move. Suite **3281 -> 3415** passed on release and on the separately
built debug engine, 1 skipped, 3 xfailed, the same two absent-`pyspark` errors, with **134 new
node ids and not one id removed, renamed or reparametrised** — `comm` over the two collected
lists is empty, so every shared id has the same outcome. `cargo test` 266 / 5, master's exact
five names. The public API diff is empty and the live surface is unchanged
(`confit.__all__ == ['BUILD_PROFILE', 'DuckDBInferFn']`); five build legs, each rebuilt from
source with the installed `.pyd` verified by path, mtime and size. Mutation caught: putting
`ASOF` back in the serve list turns exactly **two** named tests red and the gate's own battery
**60/60 -> 58/60**, restored by re-edit. **92 hand probes**, written against the promise rather
than against the branch's tests — every serving case additionally required to be on backend
`constant` **and** to match DuckDB's own rows as an unordered multiset — all land right, and
the second battery of 32 is weighted deliberately towards **over**-refusal, the expensive
failure mode here. All five of `goal.md`'s code claims reproduce verbatim, printed outputs
included.

*Campaign delta, and the one number that moved the good way.* `AGREE` 1013 -> **1007**,
`REFUSED` 944 -> **951**, `AGREE_TRAP` 21 -> **20**, `UNSHIPPED` 14, `DIVERGE_OPT` 7,
`DIVERGE_VALUE` 1, `DIVERGE_BUILD` 0 — against iteration 4's `AGREE` **1003** / `REFUSED`
**955**, so the typed-sum rule bought back four seeds of acceptance without spending any part
of the control. **89 flips, every one attributed.** 44 are the planted twins, whose SQL itself
changed (24 tie twins all `REFUSED`, 20 unique twins all `AGREE` — the rule firing in both
directions inside the campaign). 41 are `REFUSED` -> `REFUSED` message-class renames on
identical SQL, every one verified **dynamic** and not one claiming the static-only path:
master's mislabelled row-limit refusal becomes the row path's own wording, which is exactly
the promise that dynamic queries keep the row path's errors. And **4** are `AGREE` ->
`REFUSED`: `avg` at 113 / 1036 / 1314, and `sum` at 1967 over that `DOUBLE`-typed column.
Zero flips the other way outside the planted set; the `DIVERGE_OPT` seed set is identical on
both legs, the single `DIVERGE_VALUE` is 1804 on both, and the `AGREE_TRAP` 21 -> 20 move is
itself a planted-twin seed (1435), so no trap was lost. **Round 4 adds none of the 89**: its
own delta against round 3 is **zero**, seed-by-seed on SQL, kind and refusal class, with zero
occurrences of the four new messages — the generator emits no `ASOF`, no `POSITIONAL`, no
`COLLATE` and no table function, so each new rule is pinned by a unit test instead.

*One correction the gate makes to the branch's own claim.* "Existing seeds byte-identical" is
**1906 byte-identical + 5 whitespace-fixed**, not 1911. Master's row-limit refusal format
string literally contains a long run of spaces (`on a`, twenty-six spaces, then
`static-tables-only query`); the branch's helper emits clean text. Same kind, same class,
whitespace-normalised details identical on all five seeds — a real fix, but one the branch's
attribution did not name.

*The corpus floor moved, once, and it was earned.* **547 -> 546**, and the gate reproduced the
justification rather than relaying it. Exactly one statement of 678 changed outcome — index
626, `select round(100::INTEGER, int) from test_all_types();` from DuckDB's own
`test_round_integers.test` — and it is precisely the statement `MATCH_FLOOR`'s own comment
names: `test_all_types()` carries a `TIMESTAMPTZ` column that renders in the **build
machine's** session time zone, and `max(timestamp_tz)::VARCHAR` gives four different values
under `TimeZone` unset / UTC / Asia/Tehran / America/New_York. Zero other statements changed
outcome and zero changed detail, 677 of 678 identical. This is the ratchet working as
designed: a correct new refusal trips the floor on the day it lands, and the floor moves only
with a reproduced reason beside it.

*Status:* gated **PASS** at `2d24744`, and **not a merge candidate** — for the fourth round
running, because the review of the gated tip found **six** more, four of them HIGH, and all
four are the same fail-open shape the three previous rounds closed elsewhere. `FROM '<path>'`,
DuckDB's implicit file scan, serializes as a `BASE_TABLE` whose `table_name` **is** the file
path, so the table-function rule never sees a function name and a CSV, a parquet file or a
**glob** freezes the build machine's file system into the constant — while `read_csv()` on the
same file refuses with the message that names exactly that class. A **macro** whose body calls
an order-dependent aggregate serves, because the aggregate scan matches
`function_type = 'aggregate'` and a macro is `'macro'`: `json_group_array`,
`json_group_object`, `weighted_avg` and `geomean` leak at 2, 2, 4 and 7 distinct answers
across settings, each wrapping an aggregate that refuses when spelled directly. One-argument
`age(TIMESTAMP)` answers with the transaction clock (measured: it moves across transactions and
time zones while `age(a, b)` does not) and its catalogue row says `CONSISTENT`, so it passes
every stability rule, and two builds a day apart freeze two different constants. And
`SUMMARIZE` serializes as an opaque `SHOW_REF` node that names none of the `avg` / `stddev` /
`approx_quantile` aggregates it actually runs, so no value rule reaches them: seven settings,
**seven** distinct answers. A medium and a low ride with them — **any `TIMESTAMPTZ` rendered
or decomposed** freezes the build machine's session time zone with no function name involved
anywhere, the same effect `known-limitations.md` already names as disqualifying for
`test_all_types()`, now reaching ordinary queries; and `SHOW TABLES` / `DESCRIBE` leak the
harness's own internal registration name (`__arrow_s`) into a user-visible constant.

*That is a review record, and it earned the weight.* Its method was to rebuild the branch's
five reading functions as a Python replica, validate the replica against **29** known verdicts
from the branch's own test file (29/29 agree), sweep candidate shapes, measure every serving
case across seven DuckDB settings x 2-3 reps on 200k-500k-row statics, and then re-confirm
each finding **end to end against the built branch** with a refusing control on identical
data. The same sweep verified a long list **clean** by that method: `LATERAL` and correlated
subqueries, `unnest` ordering, list / array / struct constructors, `PIVOT` and `UNPIVOT`,
recursive CTEs including the `USING KEY` form, sampling inside CTEs and derived tables,
`EXPLAIN` / `PRAGMA` / `CALL`, the window-only function escape hatch, the full 88-name
aggregate catalogue, deep nesting and 300-call select lists, and values that compare **equal**
but stay **distinguishable** (`0.0` vs `-0.0`, `'a'` vs `'A' COLLATE NOCASE`, 200k of each) —
each measured across all seven settings, each giving a single answer.

**The seventh fail-open, and what found it (`04f113a`).** Before round 5 opened, the
orchestrator's read of the whole diff — not a probe, not a gate — found one more, and it is
the only one in this branch's history found by **reading the rule** rather than by asking
DuckDB questions. An alias behind a star is placed by counting from the wrong end when a
top-level `unnest(struct)` also expands: `SELECT *, a AS k, unnest(st) FROM s ORDER BY k`
served a **tied** `k`. Closed test-first, with the suite at **3419** passed and the campaign
**byte-identical**. Three doctrine slips in `goal.md` went with it — a today-state sentence
and a mechanism paragraph, neither of which belongs in a target document, and the serving
example that sat under a `REFUSES:` heading, now labelled `SERVES:` — and so did the
`MATCH_FLOOR` comment, which argued a skew for a statement whose answer has none: the loss is
the table-function allow-list's price, and the comment now says so. The owner paused the loop
when PR #202 merged and restarted it for this iteration.

**Round 5 (iteration 8, `36ae02e`): the six the round-4 review found, closed by reading
DuckDB's own metadata rather than by listing more names.** Five rules, and the shape of three
of them is the point — this round stopped extending name lists and started reading the
catalogue and the parse.

- **The `FROM` allow-list.** Every `BASE_TABLE` must name a **static or a CTE**; everything
  else refuses by name. That closes the implicit file scan (a `BASE_TABLE` whose name **is**
  the path), the catalogue views reached as tables (`duckdb_tables`,
  `information_schema.tables`), and the harness's own `__arrow_s` registration name, with one
  rule instead of three lists.
- **`SHOW_REF` refuses by node.** `SUMMARIZE`, `DESCRIBE` and `SHOW` serialize as a node that
  names none of the aggregates it runs, so the node itself is the refusal.
- **A macro is read as its call.** The aggregate, stability, clock and maker readings are fed
  from the **parsed definitions** of the catalogue macros a statement names, so
  `json_group_array`, `json_group_object`, `weighted_avg` and `geomean` refuse through a macro
  exactly as they do when spelled directly.
- **One-argument `age()` refuses by arity**, read out of the statement's own parse.
- **Zoned types refuse by DuckDB's own metadata**, in three sightings: a static column's
  declared type, a `cast_type` node in the parse, and a maker function's `return_type` from
  the catalogue.

*One reading, not two.* The macro rule's parse-based stability read reproduces the old
regex-based one name for name — `ago`, `current_catalog`, `current_database`, `current_query`,
`current_schema`, `current_schemas`, `pg_conf_load_time`, `pg_postmaster_start_time`,
`pg_sleep` — so the regex is **deleted** and the branch carries a single reading of that
question. Four table-macro bodies do not parse as their definition text
(`duckdb_logs_parsed`, `duckdb_profiling_settings`, `histogram`, `histogram_values`);
`histogram` is caught by the aggregate catalogue under its own name and the other three by the
table-function rule, so that gap costs nothing. One implementation ceiling is recorded rather
than hidden: pyo3's 12-tuple extraction limit put the new names into the existing one-column
name list rather than into columns of their own.

*Gate (branch `36ae02e`, master `8796bb2`):* **PASS**, with a caveat the gate raised first and
this report keeps first: **the branch is not rebased**. Its merge-base is `f81e17c` and master
has advanced **13 commits** past it, so every number below measures the branch **as pushed**,
not the merged result. Suite: master **3326** passed / 1 skipped / 9 xfailed / 2 errors over
3338 ids, branch **3435** / 1 / 3 / 2 over 3441 ids, with **3286 shared ids and zero outcome
changes**, 155 branch-only ids all passing, and 52 master-only ids every one of which sits in a
file only master touched since the merge-base. The branch deletes no `def test_` and no
`#[test]`. A separately built **debug** engine gives the identical 3441 ids and the identical
outcomes. `cargo test --release --lib` reads master 269 / 5 and branch 266 / 5 with the **same
five names** on both sides — the three-test gap is master's own new Rust tests, not a branch
deletion — `cargo check --all-targets` is byte-identical to the baseline at 2 warnings, `ruff`
is clean, and the public API diff is empty.

*Campaign, and the one flip that is master's rather than the branch's.* Seeds 0-1999 at
`--workers 8 --timeout 60`: `AGREE` **1007** / `REFUSED` **951** / `AGREE_TRAP` **20** /
`UNSHIPPED` **14** / `DIVERGE_OPT` **7** / `DIVERGE_VALUE` **1**, against master's 7 findings
in 3 classes and the branch's 8 in 4. **Against `04f113a` the delta is zero** — the round adds
no flip, and the findings are byte-identical after sorting, 8 on each side. Against master the
per-seed delta is **81**: 44 planted-twin seeds whose SQL itself changed (35 of them changing
verdict), 41 identical-SQL `REFUSED` -> `REFUSED` message renames, 4 identical-SQL `AGREE` ->
`REFUSED` on the order-sensitive aggregate rule (seeds 113 / 1036 / 1314 / 1967), and **seed
1804** `AGREE` -> `DIVERGE_VALUE`. That last one is **not the branch's**: it reproduces at the
merge-base with the identical `nan` versus `-nan` detail, so it is the NaN-sign work master has
merged and this branch does not yet carry. It is the first time the two lines have met, and it
is the gate's own argument for rebasing before merging.

*Corpus: **546 -> 540**, with a reproduced reason per statement.* Against master's 547 the gate
attributes **seven** moved, all `match` -> `unsupported`, zero FAIL: one is round 4's
`test_all_types()` statement, and six are new. Five are the same `SELECT COUNT(*) FROM t` from
`test_issue_1812.test` over the **driving** table, and the reason is measured rather than
argued — the replay's own caller frame held a pyarrow table named `t` and the unqualified name
resolved against it, so the build was handed **zero** statics and still produced a constant.
With one more Python frame between, the same statement refuses; the gate reproduced both
directions (`backend='constant'` rows `[6]` equal to the mined answer with that local present,
refused without it). The sixth is `geomean`, whose catalogue body is `exp(avg(ln(x)))` and
whose `avg` is order-dependent — the macro rule reaching a statement the corpus mined. **Zero**
statements moved because of the zoned rule.

*Mutation, on both legs.* The author reverted each of the five rules by re-edit, rebuilt,
re-ran and restored by re-edit: the `FROM` allow-list gives 2 red, the `SHOW_REF` marker 1, the
aggregate read taken back to the statement's own names 1, the `age` arity arm 1, and the zoned
marker 4. The gate ran its own five against the standing rules and got 2, 16, 4, 37 and 16 red
— every rule fenced, each restored with a zero-byte `git diff` afterwards. **48 hand probes**
beyond the suite, 29 must-serve (required to be on backend `constant` **and** to equal DuckDB's
own rows as an unordered multiset) and 19 must-refuse-by-name: **48/48**.

*Two over-refusals disclosed, both stated as prices.* `TIME WITH TIME ZONE` is sighted with the
zoned class although it renders **without** the session zone (measured) — because
`DATE + TIMETZ` produces a `TIMESTAMPTZ`, and separating the two would require the reading to
type every expression rather than to read declared types. And a static column carrying the
zoned marker is sighted wherever it sits **among the query's own statics**, not only where the
statement selects it. Both are in `known-limitations.md` and in the test section's own header;
the collation that takes `min` / `max` off the order-free list is carried forward unchanged.

*The review that followed (round 6) found **five** shapes and one over-refusal, and its method
is why the count carries weight.* It rebuilt the branch's shape reading, its exact-sum reading,
its ordering-word reading and its refusal ladder as a **Python replica**, mined every SQL-shaped
literal out of the branch's own test file (162 candidates, 41 not runnable as a single
statement), and ran the remaining **121 against the built branch across five static-table
shapes**: **121/121 agreement, zero disagreements**. Then it probed for what the replica and the
branch **both** miss. It re-ran the gate itself first: 3435 passed / 1 skipped / 3 xfailed / 2
errors, the rule's own file **150 passed**, corpus replay green at `MATCH_FLOOR` 540.

1. **HIGH — a `TIMESTAMPTZ` built from a string argument escapes all three zoned sightings.**
   `strptime` / `try_strptime` with `%z`, and `json_transform` / `from_json` and their
   `_strict` forms with a `"TIMESTAMP WITH TIME ZONE"` structure string, are typed at **bind**
   time from an argument, so the catalogue's `return_type` says `TIMESTAMP` or `ANY`, the parse
   carries no `cast_type`, and no static column is involved. Measured: one such select froze
   `'2019-12-31 20:00:00+01'` on a Europe/Berlin machine while raw DuckDB over seven settings x
   two reps gave **three** answers across zones; `date_part('hour', ...)` freezes 20 against
   19 / 14 / 4. The control is exact — the same instant written as a `TIMESTAMPTZ` literal cast,
   same statics, same connection, **refuses**.
2. **HIGH — a macro whose body calls another macro is classified by nothing.** The expansion is
   exactly **one level** deep, and the inner name is itself a macro, so the aggregate, stability
   and maker reads all see a macro with a `NULL` stability. `geometric_mean` froze one value
   where raw DuckDB gave **4** distinct answers over 300k rows, and `wavg` froze one where raw
   gave **5**; the controls one level shallower — `geomean`, `weighted_avg` — refuse by name on
   identical data.
3. **HIGH — the `FROM` allow-list is flat and unscoped.** CTE names are gathered by recursive
   descent over the whole parse, so a CTE declared inside **any** subquery whitelists that bare
   name for an outer `FROM` that cannot see it. `FROM 'e2e.csv'` with an inner CTE quoted to the
   same name serves and freezes the **file system**: the identical query text froze `o=1` in one
   working directory and `o=5` in another. Without the CTE, the same statement refuses in both.
4. **MEDIUM — the allow-list compares only the last path segment.** A static named `tables`
   whitelists `information_schema.tables`, and one named `duckdb_tables` whitelists
   `system.main.duckdb_tables`; the qualifier the query wrote is dropped before the membership
   test, so the frozen counts (4 and 2) move with how many statics the caller happened to
   register. Removing the colliding static is the only difference between serving and refusing.
5. **LOW — the `age` arity reading is the one name reading that stops at the statement**, and
   its arm reports the **inner** name where the four beside it report the outer one. Nothing
   escapes today: all 131 catalogue macro definitions were enumerated and **zero** call `age`
   or any zoned maker, and every direct spelling — bare, schema-qualified, named-argument, and
   inside a lambda — refuses.
6. **MEDIUM, and the opposite of an escape — one zoned column in any static refuses every query
   on that build.** The static-column sighting asks the catalogue about all of the caller's
   statics with no reference to the statement and sits above the other arms, so `SELECT 1 AS o`
   refuses when an unrelated static carries a `TIMESTAMPTZ`. Drop that static and the same three
   statements serve. The disclosure above reads as a projection-level cost; measured, it is a
   per-build switch, and this report states it that way.

*The review's clean list, by the same method.* The whole over-refusal battery serves: naive
`TIMESTAMP` / `DATE` arithmetic, `strptime` **without** `%z`, `epoch_ms`, `INTERVAL`,
two-argument `age`, `list_sum` / `list_avg` / `array_to_string`, schema- and catalog-qualified
statics, quoted and aliased statics, CTEs used only in nested subqueries and in set-operation
branches, `WITH RECURSIVE` self-reference, a CTE named like a file, a static named like a
catalogue view, and a struct field literally spelled `with time zone`. A systematic sweep of
**all 1343 `CONSISTENT` catalogue scalars** across seven environments (three time zones, a
non-Gregorian calendar, one and eight threads, a default collation) x two working directories
found **24** that answer more than one way, and the fold refuses **22** — the two exceptions are
the bind-time zoned pair above. The calendar setting reaches ICU only through `TIMESTAMPTZ`, which refuses, so
naive temporal types are unaffected by it. `DESCRIBE` nested in a subquery still refuses;
`sqlite_master`, `pg_catalog.pg_class`, `__arrow_s`, `query_table()`, `duckdb_settings` and the
file and glob scans all refuse by name; and the exact-sum rule behaves as it did at `04f113a`.

*Its design findings, kept because they are about maintainability rather than answers.* Six,
one of them clean. The six offenders the shape reading returns travel as a **positional array**
whose position-to-meaning binding lives in three places no compiler checks, where named columns
and a named struct would make a seventh sighting a compile error rather than a swapped message.
The arity arm's odd column is unmarked, so a reader cannot tell a decision from a typo. The
macro paragraph in `known-limitations.md` over-claims completeness — it is the stability, clock,
aggregate and maker reads only, stopping at one level, which is exactly the
macro-inside-a-macro hole — and the
eight maker names beside it are a hand-copied snapshot of a catalogue query that no test pins.
The `age` paragraph argues about a third party's own classification where a **measured** fact is
available and stronger, which is both the house rule and the better sentence. And four comments
carry two different counts of "three" and "four" readings for two different groupings, where
naming the three zoned sightings once and referring to them by that name would never go stale.
The clean one is the comment policy: **zero** ticket or PR references and **zero** dates across
the whole diff, and the `MATCH_FLOOR` comment names its six statements as standing facts rather
than as a changelog.

*Status:* gated **PASS** at `36ae02e`, and **not a merge candidate**, for two separate reasons
this time. Four of round 6's shapes serve a value that is not a function of the query — three
HIGH and one MEDIUM — which is the same control violation the branch exists to close. And the
branch is 13 commits behind master, so its numbers describe a tree that exists nowhere else:
rebase onto `8796bb2` and re-gate is a precondition, not a tidy-up.

---

## 3. Gate state, branch by branch {#gate-state}

| branch | tip | gate verdict | campaign delta vs master (2000 seeds) | corpus | state |
|---|---|---|---|---|---|
| `fix-corpus-slip` | `a7c5798` | PASS WITH FINDINGS | 0 flips; census identical | 547 / 131 / 0 both legs | **merged** (PR #200) |
| `fix-nan-sign-varchar` | `5819c3a` gated; merged, master `8796bb2` | **PASS** (ship gate, then a final-tip gate at the merged tree) | 1 flip: seed 1804 `DIVERGE_VALUE` -> `AGREE`; final tip `AGREE` **1014** / `REFUSED` 944 / `AGREE_TRAP` 21 / `UNSHIPPED` 14 / `DIVERGE_OPT` 7 / `DIVERGE_VALUE` **0**, histogram identical to the ship gate's | 547 / 131 / 0 both legs | **merged** (PR #202, 2026-09-06, on the owner's click) |
| `fix-fmod-sign` | `ec71979` | PASS, then read as redundant | its one flip was the same seed 1804, already carried by the branch above | 547 / 131 / 0 both legs | **dropped**; grid salvaged |
| `refuse-static-tie-order` | `36ae02e` | **PASS** (two disclosed over-refusals, 4-5x static-only build cost; **not rebased** — 13 commits behind master) | 81-seed delta vs master `8796bb2`, all attributed, one of them master's own; **zero** vs the branch's previous tip `04f113a`; `AGREE` **1007** / `REFUSED` **951** / `AGREE_TRAP` **20** | 540 / 138 / 0; floor moved twice, each statement's reason reproduced | **not a merge candidate**: round-6 review found four shapes that answer wrongly, three HIGH; and a rebase is a precondition |

Suite counts, each from the gate that produced it: master **3281** passed / 1 skipped / 3
xfailed / 2 errors on every leg up to `f81e17c`, and **3326** / 1 / **9** / 2 at `8796bb2`
once the parity branch merged; `fix-nan-sign-varchar` **3320** at `550f949` and
outcome-identical on every shared node id at its ship gate; `refuse-static-tie-order` **3366**
at `a08147e`, **3415** at `2d24744`, **3419** at `04f113a` and **3435** at `36ae02e`;
`fix-fmod-sign` stood at **3284** when it was gated, and is moot now. Every branch was also run
on a separately built **debug** engine with identical results and no `debug_assert` firing.
Dialect L2 reads **288/678** with 0 FAIL wherever it was taken. The public API diff is empty on
all four, and `cargo test`'s failing **set** is master's five on all four. The corpus reads
547 / 131 / 0 on every leg of every branch except `refuse-static-tie-order`, which reads
**546 / 132 / 0** at `2d24744` and **540 / 138 / 0** at `36ae02e` — seven statements against
master, each moved for a reason the gate reproduced rather than relayed.

**The contention resolved by measurement, not by a choice.** `fix-fmod-sign` and
`fix-nan-sign-varchar` both changed the same `DuckF64` NaN arm and both flipped the same
single seed, and the first text left the pick to the owner. Iteration 4 removed the pick:
`fix-fmod-sign`'s production change **was** that one line, already carried on the other
branch, so there was nothing to choose between. Its 29-row sign grid over `%` / `mod` /
`fmod` now lives on `fix-nan-sign-varchar` as tests, and the branch is dropped. One candidate
remained for kpi: engine-parity, and iteration 8 spent the click: it is master's now.

**One live branch is left, and its open items are still a different class from the parity
branch's were.** On `fix-nan-sign-varchar` the review findings were naming, coverage and
comment defects over a production diff the gate found clean — an equality that could not see a
NaN sign, a generator that never emitted one, a refusal selector that misdiagnosed a shape it
already refused correctly. None answered a query wrongly. The one thing on that branch that
did was found by neither a gate nor a review but by **reading the diff**, and it was fixed
before the merge (the NaN-sign `nextafter` bug above). On `refuse-static-tie-order`, four of
round 6's items **do** answer a query wrongly: a `TIMESTAMPTZ` built from a string argument, a
macro one level too deep, a CTE name escaping its scope, and a catalogue view reached past its
qualifier each freeze something that is not a function of the query. A control violation inside
the branch chartered to close that control is a different class of open item from a stale
comment, and the table above states the two differently on purpose. What was new at iteration 7
is that the **repetition**, rather than any one finding, is the measurement; iteration 8 adds a
fifth round to it without ending it.

---

## 4. The enumeration has not terminated {#enumeration-not-terminated}

This is a measurement about the tie branch's **method**, not a finding against it, and it is
the reason a design question already put to the owner now has evidence under it.

**What five review rounds did.** Each round closed every fail-open the last one found, and each
next round found more — from the same surface, by the same method: an independent reader
probing DuckDB for shapes whose answer is not a function of the query text and the statics.

```
ties -> ORDER BY #N -> the LAST alias -> POSITIONAL JOIN -> ASOF JOIN ->
collated min/max -> the OrderBinder fallback -> machine-state table functions ->
the implicit file scan -> a macro over an order-dependent aggregate ->
one-argument age() -> SUMMARIZE -> any rendered TIMESTAMPTZ ->
an alias behind a star that unnest(struct) displaces ->
a TIMESTAMPTZ typed at bind time from a string argument ->
a macro whose body calls another macro ->
a CTE name escaping its subquery -> a catalogue view reached past its qualifier
```

**Round 6 adds to the sequence; it is not the round that came back empty.** Of its five
shapes, three are HIGH and one is MEDIUM, and each of those four **serves** a value that is not
a function of the query — a frozen session time zone, a frozen order-dependent aggregate, a
frozen file system, a frozen catalogue count. Only the fifth is different in kind: the
one-argument `age` reading is statement-only, but all 131 catalogue macro definitions were
enumerated and none calls `age`, so nothing escapes through it **today** — a latent hole, not a
wrong answer. The sixth item is an over-refusal, which is the fail-closed direction. So the
count of rounds that came back with nothing wrongly served is still **zero**.

Every entry is closed or open on its own merits, and every fix is right. What the **sequence**
measures is the shape of the work: five rounds, no round empty, no round's findings predicted
by the one before it. The rule is chasing a surface — DuckDB's whole function, join,
table-function, macro and session-setting catalogue — that neither the branch nor five rounds of
independent review have been able to enumerate, and nothing this loop measured says the next
round is empty.

**Why that reads as structural rather than as a run of bad luck.** Three things now. The
findings get **narrower in kind** each round — a clause a parser can see, then a binder rule,
then a catalogue flag, then a serialization node, then a type the catalogue does not carry — so
the question has moved from "did we cover the shapes" to "can this reading see the shape at
all". Rounds 4 and 5 both found cases where **no name appears anywhere in the parse**:
`SUMMARIZE`'s `SHOW_REF`, `FROM '<path>'`'s `BASE_TABLE`, and a `TIMESTAMPTZ` rendered by a
plain cast. And round 6 goes one step past that: `strptime('...','%z')` carries a name the
reading **does** find, in a catalogue that reports its return type as naive, because the zoned
type is chosen at bind time from a **string argument**. A rule that decides by reading names and
declared types cannot be completed against shapes whose type is a value.

**One thing round 5 does change, and it is a point for the enumerating side.** Three of its
five closures are **allow-lists and metadata reads** rather than longer name lists: every
`BASE_TABLE` must name a static or a CTE, macros are read through their own parsed definitions,
and zoned types are read off DuckDB's declared types. Each of those covers a class rather than a
list, and the corpus and campaign price them exactly. Round 6's two allow-list findings are then
**defects in that reading itself** — scope and qualification — not new shapes to enumerate,
which is a smaller kind of open item than the shapes rounds 1-4 kept producing. The
two HIGHs that are not of that kind (bind-time zoned types, macros one level too deep) are the
ones that keep the sequence going.

**The fork, stated as a fork.** The alternative already on the table is to stop deciding
*which shapes are pure* and instead **pin the build-time fold's configuration through the
oracle**, so that the answer is deterministic by construction: one reading, one settings set,
one thread count, fixed at the door the constant is folded behind — and a query whose answer
moves under that fixed configuration becomes a bug rather than a shape to enumerate. That is
the oracle spec's ask: engine-fold-reading (does the engine's build-time fold move to the
oracle's reading, given it folds optimizer-ON while the oracle is optimizer-OFF) and
ask: threads-and-value-order (does `threads` join the oracle constant, and what disposition
covers order *inside* a value). Both are **stated, not ruled**, and both are the owner's.

What this loop is claiming, and what it is not. It is **not** claiming the fork is decided,
that enumeration is the wrong approach, or that the tie branch should be abandoned — the
branch closes a real silent-wrongness class, every rule in it is measured, and its gate is
PASS. It **is** recording that five consecutive rounds of enumeration have not terminated,
that what they find trends away from what a name-reading rule can see, and that this is the
first evidence the loop has produced bearing on those two asks. The fork's own claim is that
pinning the configuration changes what the rule must enumerate from "every impure shape DuckDB
offers" to "every shape that moves under a configuration we control"; it would not make this
branch redundant, because a frozen tie order still is not a function of the query. Pricing
that claim is the decision, and it is not this report's.

---

## 5. Measured facts for the next full reading {#for-the-next-reading}

Facts this loop produced that belong in reading **N=2**, not in this report's conclusions.

**Acceptance changed, and not only by fixing things.** `refuse-static-tie-order` moves
`AGREE` 1013 -> **1007** and `REFUSED` 944 -> **951** on the same seed range (the first
amendment read 1003 / 955 at `a08147e`; round 3's typed-sum rule recovered four seeds) — but
the comparison is **not like-for-like**, for two reasons that pull differently. 44 of the 2000
seeds are now planted twins rather than grammar draws, so roughly **2%** of every campaign is
two fixed queries; one displaced seed (1435) carried an `AGREE_TRAP` that nothing else covers,
which is the whole of the 21 -> 20 move, and seed numbers cited in older repros silently
change meaning. Separately, **4** of the flips are a deliberate over-refusal — `avg` at
113 / 1036 / 1314 and `sum` over a `DOUBLE` at 1967 — so part of the acceptance loss is still
a rule the loop chose, not a population artefact and not a defect. Any next census over this
generator is measuring a slightly different population under a slightly stricter rule; the
baseline's validity caveat under acceptance-reading now has two reasons to bite rather than
one. A third arrives with iteration 8: **master itself moved**. At `8796bb2` master reads
`AGREE` **1014** and `DIVERGE_VALUE` **0**, so the tie branch's 1007 is a delta against a
baseline the parity merge changed, and the 81-seed comparison the gate ran includes one flip
(seed 1804) that belongs to master rather than to the branch. Whichever leg reading N=2 takes,
it should name the master tip beside the number.

**The over-refusal detector is class-agnostic; the generator is not, and that is now an
argued position rather than an oversight.** A test pins the detector as reading every refusal
class, but the determined tag is set only on the planted unique twin, so **no generated**
static-only case is ever graded for over-refusal — and the branch's new 65-name aggregate
refusal surface therefore has no campaign-level regression detector at all. The fix round
**rejected** widening it: the tag asserts "every value here is fixed by the query and the
statics", and a generator cannot assert that of a case it just built without reimplementing
the rule under test, which would manufacture false findings. So the silence stands, with a
reason — and it is wider now than when that reason was written, because rounds 3 and 4 added
refusals by name for three join reference types, every table function outside a five-name
allow-list, collations and macros, none of which the generator emits and none of which the
campaign can therefore grade in either direction. Every new message still carries the
documented `unsupported:` prefix, so gap: undocumented-refusal-prefixes does not grow — but
the naming half of kpi: no-third-mode gains its untested claims regardless.

**The corpus count moved, and it was the first amendment's own prediction that moved it.**
That text said `MATCH_FLOOR` held at 547 with **zero headroom**, so the next correct new
refusal would trip it on the day it landed. Round 4 landed one and it did: **547 -> 546**,
exactly one statement of 678, `select round(100::INTEGER, int) from test_all_types();` —
`test_all_types()` carries a `TIMESTAMPTZ` column that renders in the **build machine's**
session time zone, four time zones giving four answers, reproduced by the gate rather than
relayed. Every other leg of every branch this loop gated still reads 547 / 131 / 0. The
mechanism is now demonstrated rather than argued: the ladder cannot shrink in silence, and a
correct new refusal costs one documented line each time it lands.

**And it moved again, six times at once, which is the first test of that mechanism at scale.**
Round 5 takes the floor **546 -> 540**. Five of the six are the same `SELECT COUNT(*) FROM t`
whose constant the build produced with **zero** statics in hand, because the replay's own caller
frame carried a pyarrow table of that name; the sixth is `geomean`, whose catalogue body is
`exp(avg(ln(x)))`. Every one has a reproduced reason beside it and none is a FAIL. Two things
follow for reading N=2. The ratchet's cost is **not** one line per refusal in general — a rule
that covers a class costs however many mined statements that class holds — and five of these six
say as much about the **replay harness** as about the engine, since the same statement refuses
when one more Python frame stands between the replay and the build. kpi: coverage-ladder is
measured through that harness, so the harness's own name resolution is part of what the number
means.

**The Rust unit gate is red on master and CI cannot see it.** `cargo test` is 266 passed / 5
failed on every master leg the loop ran, the same five names each time
(`pin_ftoi_rounding_and_traps`, `pin_ssubstr_window_arithmetic`,
`pin_stoi_trims_whitespace_like_duckdb_cast`, `table_and_custom_partition_the_catalogue`,
`substr_window_arithmetic_via_sql`), and CI runs only `pytest`. **Seven** separate gate
records said so by iteration 4, and every gate leg since has added another — iteration 8's
reads the same five names on both legs, at master `8796bb2` (269 passed / 5 failed) and on the
tie branch (266 / 5). A regression inside `exec::tests` would pass a green-bar check today.
This is an **enforcement fault**, the shape the baseline reading calls a finding rather than a
gap, and no item in this loop owned it — eight iterations in, which is now itself the
measurement.

**Closed since the first text, each by a gate leg rather than by an author's claim:** the
`- <DOUBLE NULL>` collapse; the trailing-`;` and trailing-comment false refusals; the three
fail-open paths in the tie rule (expression limits, multi-statement strings, unserializable
statements); the clock functions DuckDB's own flag calls `CONSISTENT`; `rowid`; and the tie
rule's residual classes the first text listed — `ORDER BY COLUMNS(...)`, `ORDER BY *`, a
frozen `random()` sort key and the selection-by-position aggregates all refuse now, each
spot-checked on the built branch by a gate.

**Closed in iterations 5-7, again each by a gate leg rather than an author's claim:**
`ORDER BY #N`; the first-versus-last alias binding in `ORDER BY`; `POSITIONAL JOIN` and
`ASOF JOIN`, with all six `JoinRefType` values decided; `min` / `max` under a collation; the
`OrderBinder` fallback that measured an input column; **machine-state table functions** — the
item the first amendment listed as open and not covered, closed by inverting the rule's
polarity; the by-name `sum` coarsening, for every exact accumulator; the `SELECT TOP` refusal
that fired on a static column merely named `top`; and, on the parity branch, the sign-blind
`Lit` equality, the generator that emitted no negative NaN, and the misdiagnosing residual
selector. One item is **reclassified** rather than fixed: the out-of-range cast trap's text
sits outside the comparison contract — both engines erroring at run time is `AGREE_TRAP` and
the messages are never compared — so that pin was never recording a divergence, and the
`known-limitations.md` row went with it.

**Closed in iteration 8, each by a gate leg or by a reading of the diff rather than an author's
claim:** on the parity branch, the `nextafter` NaN-sign bug and the doc, comment and literal
items the design pass and the diff read found — and the branch itself, **merged**. On the tie
branch, all seven the round-4 review and the diff read had left open: DuckDB's implicit file
scan and the catalogue views and `__arrow_s` with it (one `FROM` allow-list); `SUMMARIZE` /
`DESCRIBE` / `SHOW` (one node); macros over order-dependent aggregates (macro bodies parsed and
fed into the same readings); one-argument `age()` (by arity); every `TIMESTAMPTZ` a static
column, a cast node or a catalogue return type can name; and the alias behind a star that a
top-level `unnest(struct)` displaced. One duplicate reading is **deleted** rather than fixed:
the regex-based macro stability read reproduced the parse-based one name for name, so the branch
carries one reading of that question instead of two.

**Still open, each measured, none acted on:**

- **Four fail-open shapes on `refuse-static-tie-order` at `36ae02e`**, each measured by the
  round-6 review of the gated tip against a refusing control on identical data, each answering
  a query wrongly rather than refusing. A **`TIMESTAMPTZ` typed at bind time from a string
  argument** — `strptime` / `try_strptime` with `%z`, `json_transform` / `from_json` and their
  `_strict` forms with a zoned structure string — escapes all three zoned sightings, because
  the catalogue's return type is naive, the parse carries no cast node and no static column is
  involved: one frozen value against **three** raw answers across zones. A **macro whose body
  calls another macro** is classified by nothing, since expansion stops at one level:
  `geometric_mean` froze one value against **4** raw answers and `wavg` one against **5**,
  while `geomean` and `weighted_avg` one level shallower refuse. A **CTE declared in any
  subquery** whitelists its bare name for an outer `FROM` that cannot see it, so `FROM
  'e2e.csv'` froze `o=1` in one working directory and `o=5` in another. And the allow-list
  compares only the **last path segment**, so a static named `tables` or `duckdb_tables`
  whitelists the qualified catalogue view and freezes a count of what the build's own database
  holds.
- **Two lesser round-6 items with them.** The `age` arity reading is the one name reading that
  stops at the statement and never reaches a macro body, and its arm reports the inner name
  where the four beside it report the outer one — a latent hole rather than a live one, since
  all 131 catalogue macro definitions were enumerated and none calls `age` or a zoned maker.
  And its **design** findings: a positional array whose position-to-meaning binding lives in
  three unchecked places, an over-claiming macro paragraph in `known-limitations.md` beside an
  unpinned hand-copied maker list, an `age` paragraph that argues where a measured fact is
  available, and four comments carrying two different counts for two different groupings.
- **One over-refusal that is wider than its disclosure says.** A single zoned column in **any**
  static refuses **every** query on that build, `SELECT 1 AS o` included, because the
  static-column sighting reads the caller's statics with no reference to the statement.
  `known-limitations.md` describes it as a projection-level cost; measured, it is a per-build
  switch.
- **Three low review findings left on `fix-nan-sign-varchar`**, now on master, none answering a
  query wrongly: a `snapshot_bits` doc that forbids what a test in the same crate correctly
  pins, a bind-time `fold` that reproduces arith's over-fold, and the infinite-dividend half of
  the mod grid's excluded domain, covered by nothing. The three mediums closed in `5819c3a`;
  the fourth low closed by reclassification in `81e8fa2`.
- **The tie branch is 13 commits behind master and must be rebased before it can be gated for
  merge.** Its gate proved the two lines have already met: seed 1804 flips `AGREE` ->
  `DIVERGE_VALUE` against `8796bb2` and reproduces at the merge-base, so that flip is the
  NaN-sign work the branch does not yet carry rather than anything the branch did.
- **The by-name aggregate coarsening, narrowed but not closed.** Round 3 gave `sum` its typed
  rule through DuckDB's own overload resolution, so every **exact** accumulator serves. The
  by-name list was **65 of 88**, and `sum`'s exact overloads are all that came off it:
  `count_if`, `regr_count`, `approx_count_distinct`, `entropy` and the compensated
  accumulators `fsum` / `kahan_sum` / `favg` that exist to **be** order-stable all still
  refuse by name, and `first(v ORDER BY k)` with a unique `k` still refuses whole. The same
  overload reading is the upgrade path for the rest; the trade is the owner's to price.
- **Three allow-lists are deliberately conservative, and every one of them is an over-refusal
  the campaign cannot see**, by the blindness above. Five table-function names serve and
  everything else refuses, so a genuinely pure table function DuckDB adds later refuses until
  someone lists it; every `BASE_TABLE` must name a static or a CTE, which is what costs the
  corpus its `test_all_types()` statement; and `TIME WITH TIME ZONE` is sighted with the zoned
  class although it renders **without** the session zone, because `DATE + TIMETZ` produces a
  `TIMESTAMPTZ` and separating them would require typing every expression rather than reading
  declared types. All three are the fail-closed direction and all three are disclosed.
- **Four `known-limitations.md` line citations in `goal.md` are wrong** — rebased by +73 when
  the real shift is +161, so each now points at unrelated text. `goal.md`'s whole verified-by
  mechanism is line citations, which makes this a small edit against a load-bearing claim. The
  presentation defect that stood beside them is closed: the serving counter-example under a
  `REFUSES:` heading now reads `SERVES:`, along with two other doctrine slips in the target
  document.
- **One lesser tie-branch finding left:** a `readable` guard that can never be false (one
  field, one initializer per arm, one dead `&&`).
- `x % y`'s NaN sign is **unmatchable in principle**, not merely unfixed: two gates
  independently reproduced DuckDB answering **43 identical rows two ways in one query** —
  the vectorized lanes give one bit pattern, the scalar tail another — stable across 20
  repeats. The salvaged grid correctly excludes the domain instead of pinning it, and
  iteration 4 confirmed the exclusion rather than narrowing it.
- A **single-side** negated DOUBLE in a JOIN ON residual is still a **provable
  over-refusal**: a sign flip is total, but `may_trap`'s catch-all still counts it as
  trapping. One line closes it; iteration 4 deliberately left it out again, and split the
  refusal's message instead, so acceptance did not widen inside a review-closure branch.
- The unsigned-column refusal class — the reason three mined statements now refuse — is named
  **nowhere** in `known-limitations.md`. Same bookkeeping shape as gap:
  undocumented-boolean-comparison, and it belongs with that entry.
- The four stale `550` sites and the `known-limitations.md` line the corrected README points
  at (gap: corpus-match-slip's remaining half).

---

## 6. Spend {#spend}

| iteration | agent tokens |
|---|---|
| 1 (corpus slip, seed 1804, tie order — three implementers, three gates, two reviews) | ~1.45M |
| 2 (seed 1804 rework, tie order rebuild — two implementers, two gates, two reviews) | ~1.33M |
| 3 (seed 1804 closure, the modulo branch, the tie rule's rebuild and its gate) | **~1.76M** |
| 4 (seed 1804 review closure and salvage, the tie fix round — two implementers, two gates, two reviews) | not recorded on a comparable basis |
| 5 (the parity branch's ship gate and PR, the tie branch's positional-key round) | ~1.0M |
| 6 (the tie branch's typed sum, last alias and positional join — one implementer, one gate, one review) | ~0.75M |
| 7 (the tie branch's round 4, its gate, its review, and the second amendment) | not closed when it was written; the 2026-09-06 status reading puts iterations 1-7 together at ~9.0M+ |
| 8 (the parity branch's merge with its design pass and diff read, the tie branch's seventh fail-open and round 5, its gate, its review, and this amendment) | this run, not closed |

Iteration 3's figure is now recorded: **~1.76M**, the largest of the three, which is what a
gate that rebuilds both legs from source and a review that probes a built branch cost when
three items run at once. **Cumulative across iterations 1-3: ~4.5M agent tokens.** Iteration
4's figure is not recorded on the same basis in this run's journal, so it is not restated on
a basis that would not compare. What is visible of it: two branches, two independent gates,
two reviews and one fix round, on the same three-role shape.

Iterations 5 and 6 are recorded: **~1.0M** and **~0.75M**, both smaller than any of the first
four, because each ran a single item through the three roles rather than three items at once.
Iterations 7 and 8 are not: 7 was open when it was written up, and 8 is this run. The nearest
figure on a stated basis is the 2026-09-06 status reading's **~9.0M+ across iterations 1-7**,
plus **~0.4M** in subagents for that session's own orchestrator-driven review, fixes and gates,
which is not a workflow and does not compare with the rows above. **Cumulative across the loop:
~9.4M+ agent tokens**, where the `+` covers iterations 4, 7 and 8 and the orchestrator's own
context throughout.

**A cost that is not in the table, and iteration 8 is the reason to name it.** The diff read
that found the `nextafter` parity bug and the seventh tie-branch fail-open is the
orchestrator's own context rather than a subagent's, so it is invisible to every figure above
while being the leg that found the two items no gate did. Any future accounting of this loop's
shape should say so rather than compare gate costs alone.

**One operational cost, recorded because it is not free.** The loop's worktree-per-role shape
put `C:` at 100% on 2026-09-06, and **35 finished workflow worktrees** were removed to clear
it. Every gate leg keeps a full release build, and a debug build for as long as it runs —
which is the same independence every number in this report depends on, so this is the shape's
own cost rather than an accident. It wants a sweep between iterations rather than after a full
disk.

**The standing stop rule is unchanged: roughly 70% of the owner's weekly credit, and it is
owner-signalled** — the loop does not infer it from its own accounting. It has not been
signalled to stop, through iteration 8 included; it was **paused** once, when PR #202 merged,
and restarted for this iteration.

---

## 7. Next, in the goal's order {#next}

`goal.md` orders controls before drives, so the queue does too. Nothing here is chosen; it is
what the loop's own measurements rank.

1. **The parity control is closed; what is left of it is three lows.** kpi: engine-parity reads
   `DIVERGE_VALUE` **0** on **master** at `8796bb2`, not on a branch: PR #202 merged on the
   owner's click and a final-tip gate reproduced the ship gate's histogram exactly. The three
   low review findings above ride on master now and none answers a query wrongly. Two facts
   from the merge window belong in the next reading rather than in a queue item: the campaign
   never feeds a NaN into `nextafter`, so that class of parity bug is invisible to it, and a
   sign a platform's C runtime picked is compared, never pinned.
2. **The tie branch needs a rebase before it needs anything else.** It is 13 commits behind
   `8796bb2`, and its own gate showed the two lines have met — seed 1804 flips against master
   for a reason that is master's. Rebase and re-gate is a precondition for reading any of its
   numbers as a merge candidate's; closing round 6's four shapes is the work after that.
3. **kpi: named-refusal-share, and the refusal registry it would need.** This loop roughly
   doubled the engine's refusal vocabulary and nothing lists it. `refuse-static-tie-order`
   alone refuses by name across the aggregate catalogue, three of six join reference types,
   every table function outside a five-name allow-list, collations, macros and the clock
   family — while the campaign stays **structurally blind to over-refusal**,
   `known-limitations.md` is already drifting (four wrong line citations in `goal.md`, four
   stale `550` sites, one refusal class named nowhere), and the corpus floor now moves
   whenever a refusal lands. One registry — every refusal class with its message, its reason
   and the test that pins it — is what makes that vocabulary auditable, and it is exactly what
   kpi: named-refusal-share would measure. Adopting the KPI routes through
   ask: kpi-set-change; building the registry does not, and the naming half of
   kpi: no-third-mode is the loop's largest untested claim without it. Round 5 widens the case:
   the vocabulary now includes a `FROM` allow-list, a node-level refusal, macro expansion and
   three zoned sightings, and the round-6 review had to **rebuild the reading as a replica**
   to audit it at all.
4. **gap: undocumented-boolean-comparison, and the unsigned class that belongs with it.** Same
   bookkeeping shape, both small, both against load-bearing claims: boolean comparison is
   undocumented, and the **unsigned-column** refusal — the reason three mined statements
   stopped matching, and the loop's clearest worked example of a refusal that is a correctness
   *gain* — is named nowhere in `known-limitations.md`. The cheapest items on this list, and
   the first two rows any registry would want.
5. **The fork: the oracle spec's ask: engine-fold-reading and ask: threads-and-value-order.**
   Five rounds of enumeration have not terminated (enumeration-not-terminated above); rounds 4
   and 5 found shapes that carry **no name for a rule to read**, and round 6 found one whose
   zoned type is chosen at bind time from a **string argument**, so neither a name nor a
   declared type reaches it. That is measured evidence bearing on two questions that are stated
   and not ruled, and the tie branch's disposition hangs on the answer: enumerate a sixth round,
   or pin the build-time fold's configuration through the oracle so the answer is deterministic
   by construction. This report takes no position. It states the fork and prices what the
   enumerating side has cost so far — four iterations, five review rounds, **4-5x** static-only
   build time, seven mined statements off the corpus floor, and a branch that is gated PASS and
   still not mergeable.
6. **The enforcement faults nobody owns.** The red Rust unit gate above — eight iterations of
   gate records and no owner — and finding: c1-depth, untouched by this loop and still routed
   to ask: kpi-set-change.
7. **gap: bench-baseline-flip's cheapest cause is still untested.** One re-run after
   `--reinstall-package` rules out the stale-wheel signature (d). No iteration in this loop
   touched it, and kpi: bench-refresh-cadence should not be adopted before it is settled.
8. **Then the gap ledger, in whatever order ask: next-query-classes gets answered.** That
   question is the owner's and remains open; the loop has added no evidence that reorders its
   candidates, only evidence that gap: corpus-match-slip's ratchet half is real, that its
   bookkeeping half is not, and that the ratchet now has **seven** earned moves on the record —
   one statement at round 4 and six at round 5, each with its reason reproduced by a gate.

**Reading N=2 replaces none of this.** This report is what moved between readings; the next
full reading is what the numbers are.
