# Loop status, 2026-09-06: where the goal loop stands after eight iterations

**What this is.** A dated status reading of the standing loop (`make confit behavior match
goal.md`), written on the owner's ask for "the current situation of the loop". It states
what has shipped, what is open, what the loop has learned about its own method, what it has
cost, and which decisions are the owner's. The narrative of iterations 1-7 lives in
`2026-09-02-loop-report-1.md` (amended today to 838 lines); this file is the snapshot, not
the story. Slugs resolve as in that report: `goal:` / `kpi:` / `exclusion:` / `ask:` in
`goal.md`, `finding:` and `gap:` in the baseline reading, `claim:` in the oracle spec.

---

## 1. Shipped, and one click away {#shipped}

| item | state | what it closed |
|---|---|---|
| PR #200 `fix-corpus-slip` | **merged** (master `f81e17c`) | the corpus floor (`MATCH_FLOOR`), and the finding that the 550 -> 547 slip was a correctness gain (three unsigned join columns had matched only because the replay ignored output type) |
| PR #202 `fix-nan-sign-varchar` | **approved by the owner**; merge blocked on one CI failure, fix in flight | finding: seed-1804 (`-nan` rendered `nan`); unary minus on DOUBLE is a real negation (`fneg`); the IR carries a NaN's sign; `nextafter` hands a NaN operand back as DuckDB does |

**The CI failure on PR #202, and what it taught.** The last commit pinned which operand
`nextafter` returns when *both* are NaN, measured on Windows. Linux CI measured the other
operand. That choice is the platform compiler's, so the fix routes the kernel through the C
runtime's own `nextafter` (the function DuckDB's wheel calls), which matches by construction
on every platform, and leaves the both-NaN column compared against the machine's own oracle
and never pinned. The lesson joins the branch's doctrine: a sign a libm or a compiler chose
is compared, never pinned, and a local Windows gate is not a Linux gate.

**kpi: engine-parity after the merge.** Seeds 0-1999: `DIVERGE_VALUE` 0 (master: 1), `AGREE`
1014 / `REFUSED` 944 / `AGREE_TRAP` 21 / `UNSHIPPED` 14 / `DIVERGE_OPT` 7. The seven
`DIVERGE_OPT` seeds are the optimizer-bracket set the goal excludes (exclusion:
optimizer-on-answers). The control reads clean on the accepted surface, with the campaign's
blind spots named rather than assumed: it never feeds a NaN into `nextafter`, its comparison
contract spells every NaN `nan` (claim: repr-equality), and it cannot see over-refusal outside
the two planted twins.

---

## 2. Open: the static-only carve-out {#static-only}

**Target.** exclusion: whole-relation-shapes: a static-tables-only query is folded once at
build and frozen, and what a whole-relation construct selects is frozen only when it is a
function of the query text and the statics.

**Branch `refuse-static-tie-order`, tip `ebfbdb2`.** Four fix-and-review rounds have closed,
each gated PASS, in order: tie-producing `ORDER BY` (measured by DuckDB over the frozen
result); every selection by position (`LIMIT`/`OFFSET`/`FETCH`/`SAMPLE`/`DISTINCT ON`/
`QUALIFY`/row-position window functions); non-deterministic functions by DuckDB's own
stability flag, plus macros read through their definitions and four run-state names the
flag misses; order-dependent aggregates by DuckDB's own `SetOrderDependent`, with `sum`
read per overload through DuckDB's binder; row-counted window frames; `ORDER BY #N`; the
last-alias and OrderBinder-fallback name bindings; `POSITIONAL` and `ASOF` joins (all six
`JoinRefType` values decided, three serve); collated `min`/`max`; table functions by a
five-name allow-list; `rowid`; unreadable and multi-statement strings refused whole.

**Why it is not a merge candidate.** The review of the gated tip found six more fail-opens,
and the orchestrator's own read of the whole diff found a seventh:

| # | shape | severity | how it escapes the reading | closure planned |
|---|---|---|---|---|
| 1 | `FROM 'file.csv'` (implicit file scan; also parquet, relative paths, globs) | high | a `BASE_TABLE` whose name is the path; no function name for the table-function rule to see | every `BASE_TABLE` must name a static or a CTE (FROM allow-list) |
| 2 | macros over order-dependent aggregates (`json_group_array`, `json_group_object`, `weighted_avg`, `geomean`) | high | the macro scan reads bodies for stability only | classify a macro body exactly as a call: parse the definition, reuse the same name sets |
| 3 | one-argument `age()` | high | reads the transaction clock under a `CONSISTENT` flag | by arity, through `json_tree` (present in 1.5.5) |
| 4 | `SUMMARIZE` (and `DESCRIBE`/`SHOW`) | high | an opaque `SHOW_REF` node names none of the aggregates it runs | any `SHOW_REF` refuses by name |
| 5 | any `TIMESTAMPTZ` rendered or decomposed | medium | no name anywhere; the build machine's `TimeZone` is read | refuse zoned types by DuckDB's own metadata: static column types, cast and literal types in the parse, and maker functions from the catalogue |
| 6 | `SHOW TABLES` leaks the harness's `__arrow_s` | low | same `SHOW_REF` node | closed with 4 |
| 7 | `SELECT *, a AS k, unnest(st) FROM s ORDER BY k` serves a tied `k` | high | a top-level `unnest(struct)` expands to columns but is `FUNCTION`, not `STAR`, so an alias after a star is placed by counting from the wrong end | `SelectList.expands` (TDD, in progress, measured red) |

Number 7 is the first fail-open found by reading the branch rather than by probing it, and
it is the one the campaign, four gates and four reviews all missed.

**The enumeration has not terminated.** That sentence is the report's own section 4, and
nothing in iteration 8 changes it: each round closes everything the last found and the next
round finds more, and the newest shapes carry no name for a rule to read. The structural
alternative, pinning the build-time fold's configuration through the oracle so the answer is
deterministic by construction, is stated in the report as a fork against the oracle spec's
ask: engine-fold-reading and ask: threads-and-value-order. It is the owner's call, and the
loop proceeds under the goal as written until it is made. What iteration 8 does change is the
*shape* of the closures: three of the five are allow-lists or metadata reads (what serves,
DuckDB's own catalogue and parse), not more names.

**The price already paid, disclosed.** 64 of DuckDB's 88 aggregate names refuse on the
static-only path by DuckDB's flag (including `bit_and`, `histogram`, `count_if` and the
compensated sums that exist to be order-stable); a collation anywhere takes `min`/`max` off
the served list; every table function outside five names refuses; the fold costs 4-5x more
build time; the corpus floor moved 547 -> 546 for one statement whose own answer is fixed (the
allow-list's price, not a skew). The campaign cannot see over-refusal in this class, so these
are pinned by unit tests instead.

---

## 3. What the loop learned about itself {#method}

- **The orchestrator reads every diff.** The owner's correction of 2026-09-06 ("you must
  review the model outputs"). Applied since to both live branches: it produced findings 7
  above and the `nextafter` parity bug, four doctrine slips in `goal.md` (a today-state
  sentence and a mechanism paragraph in the target document, a serving example under a
  `REFUSES` label, a corpus comment arguing a skew for a statement that has none), and one
  over-claiming doc comment. Gates are evidence; they are not the review.
- **Findings are fixed before a PR is presented.** PR #202 went back to draft for the design
  pass and the review fixes; the tie branch stays in draft until all seven close and the diff
  is re-read.
- **A pin measured on one platform is not a pin.** The both-NaN `nextafter` case above.
- **Reviewers apply the design lens by name.** The `fix-nan-sign-varchar` design pass gave
  each duplicated rule one home (`fold_operand`, `out_of_range_trap`, the NaN-sign argument at
  `Lit`, one `inf`/`nan` token path); the structural item still open is that node classifiers
  are hand-enumerated across five files (a node should own its own properties).
- **Third-party defect descriptions stay out of the tree** until the owner has seen them; the
  branch states measured facts and our consequence only.

---

## 4. Spend and operations {#spend}

| span | agent tokens (approx.) |
|---|---|
| iterations 1-7 (workflows) | ~9.0M+ |
| this session's own review, fixes and gates (orchestrator-driven, not a workflow) | ~0.4M in subagents, plus the orchestrator's own context |

The stop rule stands at roughly 70% of the owner's weekly credit, owner-signalled; the loop
cannot read that meter and has not been given a percentage. `C:` reached 100% once during
iteration 6; 35 finished workflow worktrees were removed and a sweep between iterations is
now part of the routine.

---

## 5. Decisions that are the owner's {#decisions}

1. **The fork**: enumerate a sixth round, or pin the build-time fold's configuration through
   the oracle (ask: engine-fold-reading, ask: threads-and-value-order). Evidence: four
   rounds, seven still open, the last three shapes nameless.
2. **The four goal asks** still open on master: acceptance-target, next-query-classes,
   exclusion-ratification, kpi-set-change.
3. **The static-only acceptance price**: 64/88 aggregate names, collations, table functions,
   4-5x build time. The policy (never trade a control for a drive) says take it; the size is
   measured, so it can be priced rather than assumed.
4. **The weekly percentage**, whenever the stop rule should bite.

---

## 6. Next, in the goal's order {#next}

1. Land PR #202 once CI is green (the C-runtime `nextafter` commit).
2. Iteration 8 on the tie branch: close the seven fail-opens by the closures in the table,
   re-gate, re-read the whole diff, amend the report, open the PR for the click.
3. Open the report PR (`loop-report-1`: the iterations 1-7 narrative plus this status file).
4. Then, unchanged from the report's queue: the refusal registry behind kpi:
   named-refusal-share, gap: undocumented-boolean-comparison with the unsigned class, the
   enforcement faults (the red Rust unit gate CI cannot see), and the bench baseline.
