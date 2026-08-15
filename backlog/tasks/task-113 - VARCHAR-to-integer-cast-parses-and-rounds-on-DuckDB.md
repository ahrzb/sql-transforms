---
id: TASK-113
title: >-
  CAST(VARCHAR AS integer) parses and rounds on DuckDB; we refuse, and the
  refusal text is wrong about DuckDB
status: To Do
assignee: []
created_date: '2026-08-15 13:40'
labels:
  - m-8
dependencies: []
type: bug
ordinal: 98000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The last unticketed family from the 2026-08-13 fuzz triage (§3, seeds 12626
and 13560). DuckDB casts a decimal-looking string to an integer by parsing
and rounding; we refuse. Measured on master `bace06a` against DuckDB 1.5.5:

```
CAST('1.5'  AS BIGINT)   duck=2     ours=refuse
CAST('2.5'  AS BIGINT)   duck=3     ours=refuse
CAST('-1.5' AS BIGINT)   duck=-2    ours=refuse
CAST('1e2'  AS BIGINT)   duck=100   ours=refuse
CAST(' 7 '  AS BIGINT)   duck=7     ours=7        <- already works
```

Two separate defects, one root:

**1. The refusal states a falsehood about the oracle.** The bind-time
message reads

> constant cast fails on every row: CAST('1.5' AS BIGINT) -- DuckDB errors
> at plan time; TRY_CAST is the NULL-yielding spelling

DuckDB does not error at plan time on this input; it answers `2`. And
`TRY_CAST('1.5' AS BIGINT)` is also `2`, not NULL. Both clauses are wrong
for exactly the inputs that reach them. A refusal that misdescribes the
oracle is worse than no message: it sends the author to a spelling that
does not do what the text claims.

**2. The rounding mode is not the one we already implement.** VARCHAR ->
integer rounds **half away from zero**; DOUBLE -> integer rounds **half to
even**. They differ:

```
CAST('2.5'  AS BIGINT) = 3      CAST(2.5e0 AS BIGINT) = 2
CAST('3.5'  AS BIGINT) = 4      CAST(3.5e0 AS BIGINT) = 4
CAST('-2.5' AS BIGINT) = -3
```

TASK-70 is the DOUBLE half-to-even member. This is the sibling, and reusing
that path would be wrong.

**Parsing is exact decimal, not a double round-trip.** Two inputs prove it:

```
CAST('9223372036854775807.4'  AS BIGINT) = 9223372036854775807   (no overflow)
CAST('1.4999999999999999999'  AS BIGINT) = 1
```

Through `f64` the first is 9223372036854775808.0 and overflows, and the
second rounds to 1.5 before the integer round. So the implementation reads
the digits, rounds on the fractional part, and only then range-checks.

Accepted spellings measured: leading/trailing whitespace, a leading `+`,
a bare `.5` (-> 1), scientific `1e2`. Non-numeric still errors on DuckDB
(`'abc'`), and `TRY_CAST` yields NULL there, so the TRY_CAST channel is
correct for genuinely unparseable input and only wrong for parseable ones.

Pre-existing; reproduces on pre-migration master `217799d`. Not caused by
the arrow schema API.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 the false refusal text is gone first, as its own commit — whatever
      is decided about the cast, no message may assert DuckDB errors at plan
      time on an input where it serves a value
- [ ] #2 VARCHAR -> integer parses the decimal digits exactly and rounds
      half away from zero, matching DuckDB on '1.5' '2.5' '-2.5' '.5' '+1.5'
      and the whitespace and scientific-notation spellings
- [ ] #3 the exactness cases are pinned: '9223372036854775807.4' serves
      INT64_MAX and does not overflow, '1.4999999999999999999' serves 1 —
      i.e. no f64 appears anywhere on this path
- [ ] #4 every narrow target width (TINYINT/SMALLINT/INTEGER) rounds first
      and range-checks after, refusing by name on genuine overflow
- [ ] #5 unparseable input keeps today's behaviour: CAST errors, TRY_CAST
      serves NULL
- [ ] #6 the DOUBLE -> integer path (TASK-70, half to even) is untouched and
      a test asserts the two modes stay different
- [ ] #7 fuzz seeds 12626 and 13560 replay clean
<!-- AC:END -->
