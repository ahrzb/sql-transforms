# The loop's first report: iterations 1-3 (2026-09-02)

**What this is.** A dated report of one loop's work against `packages/confit/docs/goal.md`,
read through the yardsticks the baseline reading measured
(`packages/confit/docs/reports/2026-09-02-goal-baseline.md`, reading **N=1**). It is not a
new reading: no census, floor or bench was re-taken here for its own sake. What it records is
**which distances from the target moved, by how much, and on whose measurement** — every
number below is quoted from a **gate record**, the independent leg that rebuilt master and
the branch itself, never from an implementer's own claim, and each is named to its gate.
Where only a review measured something, the item says "review record" and the claim stands as
a review's, not a gate's.

**Slugs.** `gap:` and `finding:` citations resolve in the baseline reading; `goal:`,
`kpi:`, `exclusion:` and `ask:` in `goal.md`; `claim:` and `divergence:` without a local
definition in `packages/confit/docs/oracle/`. Sections here carry kebab-case anchors and are
cited by slug, never by number.

---

## 1. What the loop is {#the-loop}

The owner's mandate, as it stood over these three iterations: **make confit match `goal.md`**
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
**seven of them across three iterations** — reproduced the baseline's campaign census
exactly: `AGREE` 1013 / `REFUSED` 944 / `AGREE_TRAP` 21 / `UNSHIPPED` 14 / `DIVERGE_OPT` 7 /
`DIVERGE_VALUE` 1 over seeds 0-1999 at `--workers 8 --timeout 20`. Master advanced twice in
the window (`2ba96e5` -> `2c7c05c` -> `f81e17c`, docs plus one added assertion) and the census
did not move. claim: campaign-verdicts-today is reproducible on this machine, and a single
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
   branch rather than being folded in here.

*What closed it:* `fix-nan-sign-varchar`, three iterations deep — the formatter, then the
`Fneg` lowering plus eight review findings, then eight more (the residual classifier, a
bit-level two-backend `fneg` test, `-nan` surviving the IR text round trip, the cast-trap
text pinned as DuckDB's message truncated, and the stale docs). One review finding was
**rejected with measurement** rather than absorbed: the pin the review read as
platform-dependent uses a defined constant's bit pattern, which the repo rule exempts, and
the new Rust pin builds both NaNs from explicit bit patterns anyway.
*Gate (branch tip `57a05ff`, master `f81e17c`):* **PASS, no defects found.** Suite 3281 ->
**3297** passed (junit 3287 -> 3303, **16 new ids all passed, 0 shared-id outcome changes, 0
disappeared**); the debug build gives the identical set from a genuinely different engine
(23.2 MB against 17.6 MB); `cargo test` 266 -> **268** passed with the failing **set**
identical; campaign `DIVERGE_VALUE` **1 -> 0** with **exactly one seed flipped**, compared at
full-record granularity, not at the summary level; corpus **547 / 131 / 0** and dialect L2
**288/678** on both legs; four independent mutations each caught and each restored by
re-edit; the public API diff is empty.

### 2.3 finding: static-only-tie-order

*Target:* exclusion: whole-relation-shapes — inside the static-tables-only carve-out, what a
whole-relation construct selects is frozen **only when it is a function of the query**.
*Was:* nothing refused a tie-producing `ORDER BY`; two builds of the same function could
freeze different orders. A silent-wrongness class.
*Now:* a rule that refuses it, generalized well past the shape it started as — and gated to
one named defect.

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

*The generator's planted coverage, and the over-refusal detector.* The campaign could not
reach this shape at all — over seeds 0-39999 the gate found 28 static-only `ORDER BY` cases
and **none of them can tie**. So the generator plants both twins on an auxiliary stream
consulted before the main one: **1% tie, 1% unique**, 44 of 2000 seeds claimed, the other
1956 byte-identical. Grading covers both directions — a unique twin refused under the tie
class is a `DIVERGE_BUILD` **tie-over-refusal**, a tie twin that agrees is a
**tie-under-refusal**. That matters beyond this rule: as the first review established, the
campaign is **structurally blind to over-refusal** — `REFUSED` is terminal and is never
compared against the oracle — so this is the first check in the fuzzer that can see one.

*Gate (branch `4eee7f3`):* **PASS WITH ONE DEFECT.** Suite 3281 -> **3326** passed (45 new
ids, all passed, **0 shared-id outcome changes**); debug identical; `cargo test` 266 / 5, the
pre-existing set; campaign `AGREE` 1013 -> **1011**, `REFUSED` 944 -> **947**, `AGREE_TRAP`
21 -> **20**, with **35 flips, every one a seed the planted stream claims and zero on any
unclaimed seed**; all 24 tie twins `REFUSED` under the tie class, all 20 unique twins
`AGREE`, **zero** grader findings in either direction; mutation caught (18 tests red). The
gate's own 33 hand probes found **no silent bypass**, including seven DuckDB-only forms
outside `sqlparser`'s grammar.
*The defect:* a trailing `;` or a trailing `--` comment makes the probe's wrapper a parse
error, which the code converts into a refusal — so `... ORDER BY t;` over a **unique** key
refuses where master served, and the message names a cause that is not the real one. Fails
closed, so nothing wrong is frozen; the fix is already in the branch (the hidden-key path
round-trips through DuckDB and drops the trailing token for free).
*Status:* the third iteration's work on this item **returned no record to the loop's
journal**. `origin/refuse-static-tie-order` has since advanced one commit past the gated tip,
**ungated in this loop**. Its last gated state is the one above.

---

## 3. Gate state, branch by branch {#gate-state}

| branch | tip | gate verdict | campaign delta vs master (2000 seeds) | corpus | state |
|---|---|---|---|---|---|
| `fix-corpus-slip` | `a7c5798` | PASS WITH FINDINGS | 0 flips; census identical | 547 / 131 / 0 both legs | **merged** (PR #200) |
| `fix-nan-sign-varchar` | `57a05ff` | **PASS**, no defects | 1 flip: seed 1804 `DIVERGE_VALUE` -> `AGREE`; `DIVERGE_VALUE` 1 -> 0 | 547 / 131 / 0 both legs | awaiting the owner's approval |
| `fix-fmod-sign` | `ec71979` | PASS, with a scope correction | 1 flip: the same seed 1804; `DIVERGE_VALUE` 1 -> 0 | 547 / 131 / 0 both legs | awaiting the owner's approval |
| `refuse-static-tie-order` | `4eee7f3` gated; tip now `21e3fdc` | PASS WITH ONE DEFECT (at `4eee7f3`) | 35 flips, all planted seeds; `AGREE` 1013 -> 1011, `REFUSED` 944 -> 947, `AGREE_TRAP` 21 -> 20 | 547 / 131 / 0 both legs | not ready; tip ungated |

Suite counts, each from the gate that produced it: master **3281** passed / 1 skipped / 3
xfailed / 2 errors on every leg; `fix-nan-sign-varchar` **3297**; `fix-fmod-sign` **3284**;
`refuse-static-tie-order` **3326**. Every branch was also run on a separately built **debug**
engine with identical results and no `debug_assert` firing. Dialect L2 reads **288/678** with
0 FAIL wherever it was taken. The public API diff is empty on all four.

**Two branches contend for one line.** `fix-fmod-sign` and `fix-nan-sign-varchar` both change
the same `DuckF64` NaN arm, and both flip the same single seed. `fix-fmod-sign` is the
narrower one: its own gate records that the promise it was given ("`%` on DOUBLE matches") is
**wider than the change** — the modulo values already matched bit-for-bit on both backends,
and only the text was wrong. What it adds beyond the shared line is a 29-row sign grid over
`%` / `mod` / `fmod` on a value leg and a text leg, on both backends. `fix-nan-sign-varchar`
carries the same fix plus the `Fneg` lowering, the residual classifier and the IR text form.
They are not both mergeable as they stand; the second subsumes the first's production change.

**A review of the gated `fix-nan-sign-varchar` commit measured one HIGH the gate did not.**
The `Fneg` lowering drops the fold-then-null-operand short circuit that `-0.0 - x` had, so
`- <DOUBLE NULL>` no longer binds to `NullOf` and the `||` SQLNULL collapse breaks: **five of
eight probed shapes diverge from DuckDB on the branch where all eight matched on master, and
two constructs master served are now bind refusals**. The full suite is blind to all six
(3297 passed over it). The review states the one-line restoration and reports it keeps the
branch's own tests green. That is a review record, not a gate's — but it is measured, and it
is the reason this branch is not simply ready.

---

## 4. Measured facts for the next full reading {#for-the-next-reading}

Facts this loop produced that belong in reading **N=2**, not in this report's conclusions.

**Acceptance changed, and not only by fixing things.** `refuse-static-tie-order` moves
`AGREE` 1013 -> 1011 and `REFUSED` 944 -> 947 on the same seed range — but the comparison is
**not like-for-like**: 44 of the 2000 seeds are now planted twins rather than grammar draws,
so roughly **2% of every campaign** is two fixed queries. One displaced seed carried an
`AGREE_TRAP` that nothing else covers (which is the whole of the 21 -> 20 move), and seed
numbers cited in older repros silently change meaning. Any next census over this generator is
measuring a slightly different population; the baseline's validity caveat under
acceptance-reading now has a second reason to bite.

**Five new refusal messages exist and only one is graded.** The tie rule's over-refusal
detector recognizes the tie class alone, so an over-refusal arriving under any of its other
four messages is filed as a plain `REFUSED` and never reported (review record, reproduced).
Every new message carries the documented `unsupported:` prefix, so gap:
undocumented-refusal-prefixes does not grow — but the naming half of kpi: no-third-mode gains
five more untested claims.

**The corpus count did not move again.** 547 / 131 / 0 on every leg of every branch this loop
gated. `MATCH_FLOOR` now holds at exactly the current count, **with zero headroom** — so the
next correct new refusal trips it on the day it lands, by design. Two gates named that
explicitly.

**The Rust unit gate is red on master and CI cannot see it.** `cargo test` is 266 passed / 5
failed on every master leg the loop ran, the same five names each time, and CI runs only
`pytest`. Three separate gate records say so independently. A regression inside
`exec::tests` would pass a green-bar check today. This is an **enforcement fault**, the shape
the baseline reading calls a finding rather than a gap, and no item in this loop owned it.

**Still open, each measured, none acted on:**

- The `- <DOUBLE NULL>` collapse on `fix-nan-sign-varchar` (the-gate-state section above).
- `x % y`'s NaN sign is **unmatchable in principle**, not merely unfixed: two gates
  independently reproduced DuckDB answering **43 identical rows two ways in one query** —
  the vectorized lanes give one bit pattern, the scalar tail another — stable across 20
  repeats. No engine value can be right there, and the fmod branch's grid correctly excludes
  the domain instead of pinning it.
- A **single-side** negated DOUBLE in a JOIN ON residual is now a **provable over-refusal**:
  a sign flip is total, but `may_trap`'s catch-all still counts it as trapping. One line
  closes it; it was deliberately left out because it accepts SQL master refused, and widening
  acceptance unmeasured inside a review-closure branch is the wrong place for it.
- The tie rule's residual classes, all measured by review against the branch build:
  `ORDER BY COLUMNS(...)` read as `ORDER BY ALL`; `ORDER BY *` never reaching the star arm;
  a nondeterministic sort key (`random()`) frozen — **eight builds gave four different
  sequences**; selection-by-position **aggregates** (`first` / `any_value` / `arg_max` /
  `string_agg` / `list`) absent from the refusal set and measured to move under five DuckDB
  settings; and the inner-`ORDER BY` carve-out unsound when an order-sensitive consumer sits
  above it.
- The out-of-range cast trap stops one word short of DuckDB's text (`... destination type`
  against `... destination type INT64`). Naming the type means plumbing the SQL destination
  width to the trap site; the branch pins ours as DuckDB's message **truncated** and says so
  in `known-limitations.md` rather than overclaiming.
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
| 3 (seed 1804 closure, modulo sign — two implementers, two gates, two reviews) | see next report |

Iteration 3's figure is not recorded on the same basis in this run's journal, so it is left
for the next report rather than restated on a basis that would not compare. What is visible:
iteration 3 ran the same three-role shape at the same fan-out as iteration 1.

**The standing stop rule is unchanged: roughly 70% of the owner's weekly credit, and it is
owner-signalled** — the loop does not infer it from its own accounting.

---

## 6. Next, in the goal's order {#next}

`goal.md` orders controls before drives, so the queue does too. Nothing here is chosen; it is
what the loop's own measurements rank.

1. **Finish the parity control in flight.** kpi: engine-parity has `DIVERGE_VALUE` at 0 on
   two branches that cannot both land. The decision the owner owns: take
   `fix-nan-sign-varchar` (which subsumes the other's production change) with the
   `- <DOUBLE NULL>` regression closed first, and take `fix-fmod-sign`'s grid as tests only —
   or reverse it and lose the `Fneg` work. This is the one item where a merge is blocked on a
   choice rather than on more measurement.
2. **finding: static-only-tie-order is still a silent-wrongness class.** The gated branch
   fails closed on one named over-refusal and leaves five measured shapes serving a frozen
   arbitrary answer. A control violation is never a gap to live with, so this outranks every
   `gap:` entry below it — but the branch's tip is ungated and its residuals are a rule
   question (which order-dependent constructs must refuse), not a bug list.
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
