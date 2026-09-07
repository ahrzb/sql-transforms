# Loop status, 2026-09-06 (refreshed 2026-09-07): where the goal loop stands after eight iterations

**What this is.** A dated status reading of the standing loop (`make confit behavior match
goal.md`), written on the owner's ask for "the current situation of the loop" and refreshed at
the close of iteration 8. It states what has shipped, what is open, what the loop has learned
about its own method, what it has cost, and which decisions are the owner's. The narrative of
iterations 1-8 lives in `2026-09-02-loop-report-1.md`; this file is the snapshot, not the
story. Slugs resolve as in that report: `goal:` / `kpi:` / `exclusion:` / `ask:` in `goal.md`,
`finding:` and `gap:` in the baseline reading, `claim:` in the oracle spec.

**Where the tree stands.** Master is `8796bb2`. The one live branch is
`refuse-static-tie-order` at `e6a3cd8`: the gated `36ae02e` rebased onto that master (eleven
commits, no conflicts), with its clock rules restated as measurements, and re-gated there.

---

## 1. Shipped {#shipped}

| item | state | what it closed |
|---|---|---|
| PR #200 `fix-corpus-slip` | **merged** (master `f81e17c`) | the corpus floor (`MATCH_FLOOR`), and the finding that the 550 -> 547 slip was a correctness gain (three unsigned join columns had matched only because the replay ignored output type) |
| PR #202 `fix-nan-sign-varchar` | **merged** 2026-09-06 on the owner's click; master `8796bb2`, rebased | finding: seed-1804 (`-nan` rendered `nan`); unary minus on DOUBLE is a real negation (`fneg`); the IR carries a NaN's sign; `nextafter` hands a NaN operand back as DuckDB does |

**kpi: engine-parity, on master rather than on a branch.** Seeds 0-1999 at the final-tip gate:
`AGREE` **1014** / `REFUSED` **944** / `AGREE_TRAP` **21** / `UNSHIPPED` **14** /
`DIVERGE_OPT` **7** / `DIVERGE_VALUE` **0** — a histogram identical to the ship gate's, so
nothing the merge window added moved a verdict. Root suite **3326** passed / 1 skipped / 9
xfailed / 2 errors (absent `pyspark`); `cargo test` on release and debug shows the five
pre-existing failures and no others. The seven `DIVERGE_OPT` seeds are the optimizer-bracket
set the goal excludes (exclusion: optimizer-on-answers).

**The control reads clean on the accepted surface, with its blind spots named rather than
assumed.** The campaign never feeds a NaN into `nextafter`; its comparison contract spells
every NaN `nan` (claim: repr-equality); and it cannot see over-refusal outside the two planted
twins.

**What the merge window cost and taught.** Between the ship gate and the merge the branch took
a design pass (one home per duplicated rule) and the orchestrator's own read of the whole diff.
That read found a **real parity bug** two gates had passed over: `duck_nextafter` answered
every NaN input with a fresh positive NaN where DuckDB returns a NaN operand with its own sign.
Its first fix pinned the both-NaN case as measured on Windows; Linux CI measured the other
operand, because that choice is the platform C runtime's. The kernel now calls the platform's
own C `nextafter`, so lone-NaN rows are pinned and both-NaN rows are compared against the
machine's own oracle and never pinned. Three standing lessons: a sign a libm or a compiler
chose is compared, never pinned; a pin measured on one platform is not a pin, and CI is the
cross-platform leg; a gate is evidence, not the review.

---

## 2. Open: the static-only carve-out {#static-only}

**Target.** exclusion: whole-relation-shapes: a static-tables-only query is folded once at
build and frozen, and what a whole-relation construct selects is frozen only when it is a
function of the query text and the statics.

**Branch `refuse-static-tie-order`, tip `e6a3cd8`, gated PASS on master.** Five fix-and-review rounds have
closed, each gated PASS, in order: tie-producing `ORDER BY` (measured by DuckDB over the frozen
result); every selection by position (`LIMIT`/`OFFSET`/`FETCH`/`SAMPLE`/`DISTINCT ON`/
`QUALIFY`/row-position window functions); non-deterministic functions by DuckDB's own stability
flag, plus macros read through their definitions and four run-state names the flag misses;
order-dependent aggregates by DuckDB's own flag, with `sum` read per overload through DuckDB's
binder; row-counted window frames; `ORDER BY #N`; the last-alias and OrderBinder-fallback name
bindings; `POSITIONAL` and `ASOF` joins (all six `JoinRefType` values decided, three serve);
collated `min`/`max`; table functions by a five-name allow-list; `rowid`; unreadable and
multi-statement strings refused whole.

**Iteration 8's closures, all seven confirmed by the gate at `36ae02e`.** Three of the five
rules are allow-lists or metadata reads rather than longer name lists.

| # | shape | severity | how it escaped the reading | closure |
|---|---|---|---|---|
| 1 | `FROM 'file.csv'` (implicit file scan; also parquet, relative paths, globs) | high | a `BASE_TABLE` whose name is the path; no function name for the table-function rule to see | **closed in iteration 8**: every `BASE_TABLE` must name a static or a CTE (the `FROM` allow-list) |
| 2 | macros over order-dependent aggregates (`json_group_array`, `json_group_object`, `weighted_avg`, `geomean`) | high | the macro scan read bodies for stability only | **closed in iteration 8**: the aggregate, stability, clock and maker readings are fed from the parsed macro definitions |
| 3 | one-argument `age()` | high | reads the transaction clock under a `CONSISTENT` flag | **closed in iteration 8**: by arity, out of the statement's own parse |
| 4 | `SUMMARIZE` (and `DESCRIBE`/`SHOW`) | high | an opaque `SHOW_REF` node names none of the aggregates it runs | **closed in iteration 8**: any `SHOW_REF` refuses by node |
| 5 | any `TIMESTAMPTZ` rendered or decomposed | medium | no name anywhere; the build machine's `TimeZone` is read | **closed in iteration 8**: three sightings off DuckDB's own metadata — a static column's declared type, a `cast_type` node in the parse, a maker's catalogue return type |
| 6 | `SHOW TABLES` leaks the harness's `__arrow_s` | low | same `SHOW_REF` node | **closed in iteration 8** by the `SHOW_REF` rule and by the `FROM` allow-list |
| 7 | `SELECT *, a AS k, unnest(st) FROM s ORDER BY k` serves a tied `k` | high | a top-level `unnest(struct)` expands to columns, so an alias after a star is placed by counting from the wrong end | **closed in iteration 8** (`04f113a`), test-first |

The displaced alias was the first fail-open found by reading the branch rather than by probing
it, and it is the one the campaign, four gates and four reviews all missed.

**The new open set: what round six found at the gated tip.** Four shapes serve a value that is
not a function of the query; one is latent; one is an over-refusal wider than its disclosure.

| # | shape | severity | how it escapes the reading | closure planned |
|---|---|---|---|---|
| 1 | a `TIMESTAMPTZ` typed at bind time from a string argument: `strptime`/`try_strptime` with `%z`, `json_transform`/`from_json` and their `_strict` forms with a zoned structure string | high | the catalogue's return type is naive or `ANY`, the parse carries no cast node, and no static column is involved, so all three zoned sightings miss it | read the folded statement's own result types rather than the declarations around it |
| 2 | a builtin macro whose body calls another builtin macro (`geometric_mean` -> `geomean`, `wavg` -> `weighted_avg`, `json_group_structure` -> `json_group_array`) | high | expansion stops at one level, and the inner name is itself a macro | expand to a fixed point rather than one level |
| 3 | a CTE declared in any subquery whitelists its bare name for an outer `FROM` | high | CTE names are gathered by recursive descent over the whole parse, so lexical scope is not modelled | gather CTE names per scope, not per statement |
| 4 | a static whose name equals a catalogue view's last path segment whitelists the qualified read | medium | the membership test drops the schema and catalog qualifiers the query wrote | compare the qualified name |
| 5 | the `age` arity reading never reaches a macro body, and its arm reports the inner name | low | `tree` is fed from the statement's parse while the name reads are fed from macro bodies too | feed `tree` from the macro parses and report the outer name |
| 6 | **over-refusal**: one zoned column in any static refuses every query on that build, `SELECT 1 AS o` included | medium | the static-column sighting reads the caller's statics with no reference to the statement, above the other arms | sight the column where the statement reaches it, or restate the disclosure as a per-build switch |

Nothing escapes through the `age` arity gap today: all 131 catalogue macro definitions were
enumerated and none calls `age` or a zoned maker. The review also returned six design findings (a positional
array whose position-to-meaning binding lives in three unchecked places; an unmarked odd column
in the arity arm; an over-claiming macro paragraph beside an unpinned maker list; a paragraph
that argues where a measured fact is available; four comments carrying two different counts;
and one clean: zero ticket or PR references and zero dates across the whole diff).

**How round six was measured, which is why its count carries weight.** The reviewer rebuilt the
branch's shape, exact-sum, ordering-word and refusal readings as a Python replica, mined 162
SQL-shaped literals from the branch's own test file, and ran the 121 runnable ones against the
built branch across five static-table shapes: **121/121 agreement**. Every finding is confirmed
end to end with a refusing control on identical data. A sweep of all **1343 `CONSISTENT`
catalogue scalars** across seven environments x two working directories found 24 that answer
more than one way; the fold refuses **22**, the two exceptions being the bind-time zoned pair.

**The gate's numbers at `36ae02e`.** Suite: branch **3435** passed / 1 skipped / 3 xfailed / 2
errors over 3441 ids against master's 3326 / 1 / 9 / 2 over 3338, with **3286 shared ids and
zero outcome changes**, 155 branch-only ids all passing, and no test deleted or renamed; a
separately built debug engine gives the identical ids and outcomes. `cargo test --release
--lib`: master 269 / 5, branch 266 / 5, the same five pre-existing names on both sides. Public
API diff empty. Campaign seeds 0-1999: `AGREE` **1007** / `REFUSED` **951** / `AGREE_TRAP`
**20** / `UNSHIPPED` 14 / `DIVERGE_OPT` 7 / `DIVERGE_VALUE` **1** — **zero flips against the
branch's own previous tip**, and of the 81-seed delta against master the single
`DIVERGE_VALUE` is **master's**, the NaN-sign work this branch does not yet carry. Corpus
**540 / 138 / 0** against master's 547 / 131 / 0, seven statements moved with a reproduced
reason each and zero FAIL. Mutation: five rules reverted one at a time, 2 / 16 / 4 / 37 / 16
red, each restored by re-edit. **48/48** hand probes, 29 must-serve on backend `constant` and
19 must-refuse-by-name.

**The re-gate on the rebased tip `e6a3cd8`, taken by the orchestrator.** Root suite **3480**
passed / 1 skipped / 9 xfailed / 2 errors (absent `pyspark`): master's 3338 collected ids are
all present with their outcomes, and the branch adds exactly **154** (150 in the static-only
test file, 4 in the fuzz smoke file), so the count is master's 3326 plus those. `cargo test
--release --lib` 269 / 5, the five pre-existing names. Corpus **540** holds. Campaign seeds
0-1999: `AGREE` **1008** / `REFUSED` **951** / `AGREE_TRAP` **20** / `UNSHIPPED` 14 /
`DIVERGE_OPT` 7 / `DIVERGE_VALUE` **0** — the seven findings are master's own optimizer-bracket
seeds, and the single seed that moved against the pre-rebase gate is 1804, now `AGREE` because
the rebased tree carries master's NaN-sign fix.

**Why it is still not a merge candidate.** Four of round six's shapes answer a query wrongly,
which is the control the branch exists to close; the rebase precondition the gate named is
met.

**The enumeration has not terminated.** That is the report's own section, and round six adds to
the sequence rather than ending it: four of its five shapes serve wrongly, so the count of
rounds that came back with nothing wrongly served is still zero. The structural alternative,
pinning the build-time fold's configuration through the oracle so the answer is deterministic by
construction, is stated in the report as a fork against ask: engine-fold-reading and
ask: threads-and-value-order. It is the owner's call, and the loop proceeds under the goal as
written until it is made.

**The price already paid, disclosed.** 64 of DuckDB's 88 aggregate names refuse on the
static-only path by DuckDB's flag (including the compensated sums that exist to be
order-stable); a collation anywhere takes `min`/`max` off the served list; every table function
outside five names refuses, and now every `BASE_TABLE` that is not a static or a CTE;
`TIME WITH TIME ZONE` is sighted with the zoned class although it renders without the session
zone; the fold costs 4-5x more build time; the corpus floor has moved **547 -> 546 -> 540**.
Five of those six new statements are the same `SELECT COUNT(*) FROM t`, whose constant the build
produced with **zero** statics in hand because the replay's own caller frame carried a pyarrow
table of that name — a fact about the replay harness as much as about the engine. The campaign
cannot see over-refusal in this class, so all of it is pinned by unit tests instead.

---

## 3. What the loop learned about itself {#method}

- **The orchestrator reads every diff.** The owner's correction of 2026-09-06 ("you must review
  the model outputs"). It has now paid twice: the `nextafter` parity bug on a branch two gates
  had passed, and fail-open 7 on the tie branch, plus four doctrine slips in `goal.md` and one
  over-claiming doc comment. Gates are evidence; they are not the review.
- **Findings are fixed before a PR is presented.** PR #202 went back to draft for the design
  pass and the review fixes before it merged; the tie branch stays in draft until round six's
  set closes and the diff is re-read.
- **A pin measured on one platform is not a pin.** The both-NaN `nextafter` case above.
- **Rebase before gating for merge.** The tie branch's gate measured a tree 13 commits behind
  master and had to attribute one campaign flip to master to stay honest. A gate that is not on
  the merge result is a gate on something else.
- **Allow-lists and metadata reads beat name lists.** Three of iteration 8's five closures read
  DuckDB's own catalogue and parse or enumerate what *serves*; each covers a class, and the two
  round-six findings against them are defects in the reading (scope, qualification) rather than
  new shapes.
- **Reviewers apply the design lens by name.** The `fix-nan-sign-varchar` design pass gave each
  duplicated rule one home; the tie branch's standing structural item is a positional array
  where a named struct belongs.
- **Third-party defect descriptions stay out of the tree** until the owner has seen them; the
  branches state measured facts and our consequence only.

---

## 4. Spend and operations {#spend}

| span | agent tokens (approx.) |
|---|---|
| iterations 1-7 (workflows) | ~9.0M+ |
| iteration 8: the merge window's design pass and diff read, the tie branch's round 5, its gate, its review | ~0.4M in subagents on 2026-09-06, plus iteration 8's own workflow legs, not closed as this is written |

The orchestrator's own context is in none of these figures, and it is the leg that found the
two items no gate did. The stop rule stands at roughly 70% of the owner's weekly credit,
owner-signalled; the loop cannot read that meter and has not been given a percentage. It has
not been signalled to stop; it was paused once, when PR #202 merged, and restarted for
iteration 8. `C:` reached 100% once during iteration 6; 35 finished workflow worktrees were
removed and a sweep between iterations is now part of the routine.

---

## 5. Decisions that are the owner's {#decisions}

1. **The fork**: enumerate a sixth round, or pin the build-time fold's configuration through
   the oracle (ask: engine-fold-reading, ask: threads-and-value-order). Evidence: five rounds,
   none empty, four shapes still open, and the newest one carries a type chosen at bind time
   from a string argument, which neither a name nor a declared type can reach.
2. **The four goal asks** still open on master: acceptance-target, next-query-classes,
   exclusion-ratification, kpi-set-change.
3. **The static-only acceptance price**: 64/88 aggregate names, collations, table functions, the
   `FROM` allow-list, `TIMETZ`, 4-5x build time, seven mined statements off the corpus floor.
   The policy (never trade a control for a drive) says take it; the size is measured, so it can
   be priced rather than assumed.
4. **The weekly percentage**, whenever the stop rule should bite.

---

## 6. Next, in the goal's order {#next}

1. The owner's answer to the fork (the RFC put to him at the close of iteration 8): it
   decides whether iteration 9 enumerates round six's shapes or pins the fold.
2. Iteration 9 on the tie branch, under whichever answer: close round six's four serving shapes by the closures in the
   table, decide the over-refusal (fix or restate), re-gate, re-read the whole diff.
3. Open the report PR (`loop-report-1`: the iterations 1-8 narrative plus this status file).
4. Then, unchanged from the report's queue: the refusal registry behind kpi:
   named-refusal-share, gap: undocumented-boolean-comparison with the unsigned class, the
   enforcement faults (the red Rust unit gate CI cannot see, eight iterations of records and no
   owner), and the bench baseline.
