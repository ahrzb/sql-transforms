# Inherited DuckDB quirks

## Default disposition

**claim: reproduce-not-fix.** Reproduce the configured DuckDB reference unless a
recorded decision explicitly refuses the construct. This is a compatibility contract,
not a correction layer for DuckDB or the SQL standard. A DuckDB bug can therefore be
the required confit behavior.

*Evidence:* `packages/confit/docs/reports/pins-first-methodology.md:39` defines pins as
engine-equals-reference contracts.

## Ruled quirks

**claim: enumerated-quirks.** These oddities have specific dispositions. This is not the
complete unsupported-syntax inventory; that remains in
`packages/confit/docs/known-limitations.md:190-207`.

| quirk | reference behavior | disposition | evidence |
|---|---|---|---|
| `^` | power, not bit-xor | refuse: `sqlparser` assigns different precedence, so mapping it would change the tree; use `pow()` | `known-limitations.md:197` |
| `~` | full match, not search | reproduce | `reports/pins-first-methodology.md:39` |
| `SIMILAR TO` | no wildcard translation | reproduce | `reports/pins-first-methodology.md:39` |
| `SIMILAR TO ... ESCAPE` | unimplemented in DuckDB | refuse | `known-limitations.md:201` |
| `reverse()` | ASCII uses byte reversal, including splitting CRLF; non-ASCII uses graphemes | reproduce both paths | `specs/pins-waveA/reverse-graphemes.json`; `specs/2026-07-28-waveA-structural-tails.md` §4 |
| paren-less `* REPLACE e AS c` | consumes one item; a following comma starts another select item and may duplicate the name | reproduce | `specs/pins-waveA/columns-replace.json` |
| quoted struct `EXCLUDE` names | still case-insensitive | reproduce | `specs/pins-waveA/struct-star.json` |
| `* EXCLUDE (t.key)` after `USING` | unmerges the coalesced column and restores the right copy at its original position | refuse; measured but not modeled | `known-limitations.md:202` |
| `BETWEEN` / `IN` mixing non-numeric strings and numbers | conversion happens at execution, so empty input succeeds | conservatively refuse | `known-limitations.md:203` |
| `repeat(NULL, n)` with bare `NULL` | selects the BLOB overload | refuse; `CAST(NULL AS VARCHAR)` is supported and types equally | `known-limitations.md:207` |
| regex `\B` | can crash DuckDB on non-ASCII | reject-list | `known-limitations.md:199` |
| non-final regex `$` | row evaluation may literal-optimize to PREFIX while constant evaluation performs a normal match | refuse by name under claim: evaluation-path-disagreement | `specs/pins-waveB/fuzzer-20260728.json` |

*Evidence:* each row's cited measurement or limitation. The disposition, rather than the
surprising behavior alone, is the contract.

## New oddities

**claim: unlisted-oddity.** Report and measure a newly observed oddity; neither silently
“fix” it nor silently inherit it. Until its behavior and scope are established,
“DuckDB looks wrong” and “confit diverges” are the same observation. Adding a new
reproduction or refusal is a disposition decision.

*Evidence:* `packages/confit/docs/known-limitations.md:284-285` requires behavior absent
from both documentation and tests to be reported.
