## 2. Inherited quirks

**ORC-10.** Where DuckDB's behavior is a quirk, the quirk is reproduced, not fixed.
Because the contract is bit-for-bit DuckDB rather than the SQL standard, DuckDB's
oddities *are* our normative behavior, including ones DuckDB would call bugs.
*Verified-by:* `packages/confit/docs/reports/pins-first-methodology.md:39` ("pins are
engine==oracle contracts").

**ORC-11.** The enumerated quirks — *the oddities whose disposition needed a decision*,
not every descoped construct. This list exists so a future reader who meets an inherited
oddity has something to check it against before "fixing" it. It is short today and cheap
to write; reconstructing it later is not. The fuller descope list, including `#`,
`NOT GLOB` and the twelve fuzzer-found regex reject classes, is
`known-limitations.md:190-207`; a construct there but not here was descoped without an
inherited-oddity ruling to record.

| quirk | DuckDB's behavior | what we do | evidence |
|---|---|---|---|
| `^` | is `pow`, not bit-xor | descoped: sqlparser's precedence differs, so mapping it computes a wrong tree silently. Use `pow()`. | `known-limitations.md:197` |
| `~` | full-match, not search | reproduced | `pins-first-methodology.md:39` |
| `SIMILAR TO` | no wildcard translation | reproduced | `pins-first-methodology.md:39` |
| `SIMILAR TO ... ESCAPE` | not implemented in DuckDB itself | refused | `known-limitations.md:201` |
| `reverse()` | byte-reverses all-ASCII input, splitting CRLF (`'a\r\nb'` -> `'b\n\ra'`), violating UAX-29; only non-ASCII takes the grapheme path | both paths reproduced | `pins-waveA/reverse-graphemes.json`; `specs/2026-07-28-waveA-structural-tails.md` section 4 |
| paren-less `* REPLACE e AS c` | consumes exactly one item; a following comma starts a new select item, yielding a duplicate name | reproduced | `pins-waveA/columns-replace.json` |
| double-quoted identifiers in struct `EXCLUDE` | still case-insensitive: `a.* EXCLUDE("J")` removes field `j` | reproduced | `pins-waveA/struct-star.json` |
| `* EXCLUDE (t.key)` on a `USING` join | UNMERGES the coalesced column, which reappears at the right table's position | descoped (measured, not modeled) | `known-limitations.md:202` |
| `BETWEEN`/`IN` mixing non-numeric strings with numbers | converts at EXECUTION time, so an empty input succeeds | conservatively refused | `known-limitations.md:203` |
| `repeat(NULL, n)` on a bare NULL | picks the **BLOB** overload | refused; `CAST(NULL AS VARCHAR)` types identically on both | `known-limitations.md:207` |
| `\B` in a regex | crashes DuckDB at runtime on non-ASCII | reject-listed | `known-limitations.md:199` |
| `$` anchor in non-final position | the row path literal-optimizes `$`+literal into a PREFIX match while DuckDB's own constant fold matches normally — the oracle disagrees with itself | rejected by name (section 3.6) | `pins-waveB/fuzzer-20260728.json` |

*Verified-by:* each row cites its own evidence; the reproduce-don't-fix rule is ORC-10.

**ORC-12.** Meeting an inherited oddity that is not in the table above is a report,
not a fix. Adding a row is a decision and goes through the owner, because "this looks
wrong" and "this is a divergence" are the same observation until somebody measures.
*Verified-by:* the owner's standing governance rule — the oracle spec states what is
considered correct, and every contradiction goes through the owner — which is what makes
adding a row a decision rather than an edit; `known-limitations.md:284-285` ("If a
message you hit isn't in this document or the tests, that's a bug in our bookkeeping —
file it") is the existing half that makes it a report.

---
