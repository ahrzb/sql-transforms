---
id: TASK-129
title: >-
  Define order-sensitivity per query, and make the oracle comparison honor it
status: Done
assignee: []
created_date: '2026-08-19 00:00'
labels:
  - m-8
  - parity
  - fuzzer
dependencies: []
type: bug
ordinal: 114000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rewritten 2026-08-19 per AmirHossein: the row-order problem is an issue with
HOW THE ORACLE COMPARISON IS DEFINED, not (only) with any one code path.
Determine per query whether output order is defined, and compare accordingly.

What the comparison does TODAY (`fuzz/oracle.py:390`):

```python
def _key(rows):
    return sorted(sorted((k, repr(v)) for k, v in r.items()) for r in rows)
```

Every leg goes through `_key` -- the DuckDB legs, cranelift-vs-interpreter,
infer_rows-vs-infer_arrow. The oracle is MULTISET everywhere. That is:

* accidentally right for unordered constant results (the 12-orders-in-12-
  connections measurement from TASK-128's AC #3 can never fire it), and
* too WEAK on the row path: a map-shaped query's contract is out[i] <-> in[i],
  and `_key` accepts our rows in ANY permutation. An order bug on the serving
  path is invisible to the fuzzer today. Seed 1784 was caught only because a
  LIMIT changes which rows EXIST (a multiset difference), never their order.

"Bit-for-bit identical to DuckDB" is undefinable for sequences SQL leaves
unordered -- DuckDB itself has no repeatable answer there (measured: 12
distinct row orders over 12 fresh connections, unordered 200-group GROUP BY,
int and varchar keys both). The honest contract: bit-for-bit on VALUES;
sequence exactly where a sequence is DEFINED; validity rather than
byte-equality under ORDER BY ties.

Order-relevance is mechanically determinable in our surface -- three modes:

1. ROW PATH (map / filter / many). Order is defined by the SERVING contract,
   not by SQL: output rows follow input rows. DuckDB's own order is not a
   function of the query even here, so the check is NOT "match DuckDB's
   sequence":
   - vs DuckDB: multiset, as today;
   - NEW self-leg, no DuckDB involved: `infer_rows(rows[::-1])` must equal
     `infer_rows(rows)[::-1]` (for shape='many': the per-input-row BLOCKS
     reverse; within a block stays multiset, per the documented join-order
     accident).

2. CONSTANT PATH WITH ORDER BY. Ties make DuckDB's sequence one of several
   valid answers (measured on TASK-128: a GROUP-BY-fed tie flipped in 20
   runs), so byte-equality is the wrong test even here. Check instead:
   multiset equality AND our sequence SATISFIES the ORDER BY keys. Restrict
   to ORDER BY over output columns (what the fuzz grammar emits); anything
   fancier falls back to multiset with a logged note, never silently.

3. CONSTANT PATH WITHOUT ORDER BY. SQL defines no order. Multiset, as today
   -- and the contract doc SAYS so, in words (this is the old ticket's
   option (c), now grounded in the oracle definition instead of ad hoc).

Row limits are NOT reopened by any of this: TASK-128's refusal guards the
multiset itself (which rows exist), which no comparison mode can absorb.

SEPARATE but adjacent, kept out of scope here: build-vs-build repeatability.
Even with order declared unspecified, two builds of the same fn disagreeing
on frozen row order makes downstream golden-file tests flake. Our arrow
materialization measured stable over 6 builds (bulk CREATE-AS scan) -- luck,
not contract. If that ever matters, sort-at-freeze is the artifact-level fix
and is orthogonal to the oracle definition; decide it on its own evidence.

History: the original TASK-129 ("an unordered multi-row constant result
freezes an accidental row order") offered sort-at-freeze / require-ORDER-BY /
declare-unspecified as options. This rewrite supersedes it -- the measurements
(12/12 raw DuckDB, 1/6 our path, fuzzer statics are 0-6 rows so the class was
unreachable) carry over as the ground.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 the oracle states each case's comparison MODE explicitly (an enum
      derived from the case: row-path / constant-ordered / constant-unordered),
      instead of one implicit `_key` for everything
- [x] #2 the row-path reversal leg exists and FAILS if output order stops
      following input order -- prove it red by scrambling output order in a
      scratch build before trusting it green; shape='many' reverses blocks,
      multiset within a block
- [x] #3 the constant-ordered mode checks multiset equality plus our-side
      sortedness on the ORDER BY keys; a tie must NOT fail when the multiset
      matches and sortedness holds
- [x] #4 an ORDER BY the checker cannot evaluate (expression over non-output
      columns) falls back to multiset WITH a logged note -- never silently
- [x] #5 the fuzz grammar generates a static with enough groups (>= 50) for
      hash order to actually vary, so mode 3 is exercised for real, and a
      campaign stays clean after
- [x] #6 known-limitations.md states the order contract in one place: values
      bit-for-bit; sequence where the serving contract or a total ORDER BY
      defines one; otherwise unspecified
- [x] #7 nothing in TASK-128 reopens: every row-limit refusal test still
      passes unchanged
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Done 2026-08-19. `compare_mode` (oracle.py) states the three modes; every
case is tagged `cmp=<mode>` so coverage shows which mode ran.

Mode 1 (row path): the existing batch-vs-single leg now compares SEQUENCES
(`_seq`, order-preserving) instead of `_key`, splitting its verdict into
`batch-vs-single` (values) vs `batch-vs-single-order` (order only); a new
reversal leg feeds the batch backwards and requires the per-row blocks to
reverse. Both self-legs, no DuckDB involved. AC #2's red proof is a
PERMANENT capability test (`test_fuzz_order_legs.py`): a real engine wrapped
in a batch-reversing scrambler must be caught, and a mutation check
(re-routing the legs through `_key`, then restoring by re-edit) showed the
test fail and recover.

Mode 2: the static-only branch of `against()` adds our-side sortedness on
the ORDER BY column (DuckDB defaults: ASC, NULLS LAST, NaN above numbers)
on top of the multiset; an unevaluatable ORDER BY appends an
`order-by-unevaluated` tag -- logged, never silent.

Mode 3: unchanged multiset, now stated in known-limitations.md alongside
the oracle definition (AC #6).

AC #5: 8% of generated statics are wide (50-200 rows, ints spread mod 97 so
every width fits and ~97 groups exist). The FIRST attempt at this leaked
its loop variables over gen()'s own `rows`, shipping static rows as the row
table -- caught by the campaign (212 bogus SKIPs), not by my own generation
diff, whose module-cache shadowing made old and new gen look identical.
Fixed with hygienic names; the campaign after: 24 raw / 8 classes, all
known residue (TASK-121 ambiguous, documented DIVERGE_OPT, decimal-literal
family, TASK-123 timeout, TASK-113's cast class -- noted: that one is a
trap-where-DuckDB-serves, already ticketed, now fuzzer-reachable). Zero
findings from the new order legs: the engine's row path is order-faithful.

Note: the 8% draw shifts rng for static-bearing seeds, so residue seed IDs
moved; classes are what stayed comparable. The committed findings.jsonl
baseline is a historical artifact and was not regenerated here.

AC #7: all TASK-128 refusal tests pass unchanged in the full suite.
<!-- SECTION:NOTES:END -->
