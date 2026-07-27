# Wave B — regexp family: pins (TASK-53)

Measured 2026-07-27 against DuckDB 1.5.5 by a 6-agent fleet (5 DuckDB areas
+ one DuckDB-RE2 vs rust-regex DIFFERENTIAL battery); every claim is backed
by an executed query/program recorded verbatim in `pins-waveB/*.json`.
Baseline at master ba8bf98: 484 match / 192 unsupported / 2 known-divergent.

## The engine decision (pins-waveB/re2-vs-rust-regex.json)

**rust `regex` 1.13.1 is the engine, behind a bind-time translation layer;
raw pass-through is wrong for any pattern with a Perl class.** All 98
differential battery entries are byte-identical or identically-rejected
with the following applied:

- Config: `RegexBuilder::octal(true)`; **keep default Unicode mode** —
  `unicode(false)` was measured to BREAK `(?i)` folding parity (KELVIN,
  sharp-s agree by default; the "obvious" global switch is the wrong fix).
- Pattern rewrite (Perl classes are the whole Unicode gap — RE2 ASCII,
  rust Unicode), each verified byte-for-byte incl. inside `[...]` classes:
  `\d`→`(?-u:\d)`, `\D`→`[^0-9]`, `\w`→`[0-9A-Za-z_]`,
  `\W`→`[^0-9A-Za-z_]`, `\s`→`[\t\n\f\r ]`, `\S`→`[^\t\n\f\r ]`,
  `\b`→`(?-u:\b)`. NOTE: `(?-u:\s)` is the WRONG `\s` rewrite (rust ASCII
  `\s` includes VT); `(?-u:\W)` does not compile — negated classes must be
  spelled as Unicode-mode `[^...]`.
- Bind-time REJECT list (each measured; these differ or are unservable):
  `\B` (DuckDB itself dies at runtime on non-ASCII — RE2's ASCII \B matches
  inside multibyte chars), `(?<name>...)` angle groups (DuckDB rejects),
  duplicate group names (rust rejects, DuckDB accepts), repetition bounds
  > 1000 (`a{1001}` DuckDB error), **stacked quantifiers `a*+` (DuckDB
  errors; rust silently reinterprets as `(a*)+` — a wrong-answer risk, not
  an error-shape mismatch)**, `\uXXXX` escapes, `\Q...\E` (rust rejects;
  optionally pre-expandable via regex::escape).
- Replacement templates translate, not the pattern: DuckDB is `\N` with
  literal `$`; rust is `$N` with literal `\N` — map `\N`→`${N}`, `$`→`$$`.
- Agrees with zero config: leftmost-first alternation, greedy/lazy/(?U)/
  (?s), empty-match iteration incl. replace_all adjacency skip, `\A`/`\z`,
  `$` without trailing-newline magic, mid-pattern inline flags, `\p{...}`,
  `[]a]` (legal, same meaning), backrefs/lookaround/`\Z`/`a{2,1}` rejected
  by both.

## Operators and match functions (pins-waveB/matches-operators.json)

- `regexp_matches(s, p)` = **unanchored SEARCH**;
  `regexp_full_match(s, p)` = whole-string.
- **`~` is FULL match, not search** (binds to regexp_full_match — the
  binder error names it; diverges from PostgreSQL). `!~` = NOT(~) with
  standard NULL propagation.
- **SIMILAR TO does NO wildcard translation**: the pattern goes RAW to RE2
  with full-match anchoring (`'hello' SIMILAR TO 'h%o'` is FALSE; `h.llo`
  is TRUE). It is exactly regexp_full_match. `SIMILAR TO ... ESCAPE` is
  DuckDB "Not implemented Error: Custom escape in SIMILAR TO".
- Empty pattern: regexp_matches(s,'') true for EVERY non-NULL string;
  full-match/`~` with '' true only for ''.
- NULL string/pattern → NULL; **NULL options arg → error** ("Regex options
  field must not be NULL"), not NULL. All results BOOLEAN.
- Constant patterns compile at BIND time (errors fire under WHERE 1=0 /
  EXPLAIN / PREPARE); column patterns compile at EXECUTION (WHERE false
  suppresses) — column patterns stay unsupported in v0 (prepare-time
  compilation is the engine model).

## regexp_extract (pins-waveB/extract.json)

Default/group-0 = whole match; groups 1..n; **no match → `''`, NEVER
NULL** — same for non-participating groups and in-range indexes above the
pattern's group count. Unanchored leftmost-first search. **Group index is a
flat 0..9 range check unrelated to the pattern** ("Group index must be
between 0 and 9!" even when 10 groups exist; also for negatives; fires even
on non-matching subjects). NULL subject/pattern → NULL but **NULL group →
`''`**. Options as 4th arg only (no (V,V,VARCHAR) overload — 3rd-arg
options is a binder ambiguity). Group index and options must be constants.
Name-list STRUCT form: non-scalar, classify clean. Result VARCHAR.

## regexp_replace (pins-waveB/replace.json)

First-match-only by default; `'g'` global. Rewrite grammar: `\0` whole
match, `\1`-`\9`, `\\` literal backslash, `\10` = `\1` + literal '0';
`$1`/`$0`/`&` are PLAIN LITERALS. **Invalid rewrites never error, and
global vs non-global diverge**: non-global → input unchanged for ANY
invalid rewrite; global → out-of-range backref is still a full no-op
(MaxSubmatch pre-check) but a bad escape CONSUMES each match and emits only
the prefix before the bad escape (`('hello','h','\x','g')` = `'ello'`).
Non-participating in-range group → empty. Empty-match global insert at
every codepoint boundary incl. both ends (`'abc'` → `'XaXbXcX'`), no empty
match immediately after a nonempty one, never inside UTF-8 bytes. ANY NULL
arg (incl. options — asymmetric with the other functions!) → NULL.

## Options + compile errors (pins-waveB/options-errors.json)

Alphabet: `c i l m n p s` everywhere, `g` ONLY regexp_replace ("Option 'g'
(global replace) is only valid for regexp_replace" elsewhere); unknown →
"Unrecognized Regex option <ch>". Parsed left-to-right, whitespace
skipped, LAST conflicting letter wins ('ci' insensitive, 'ic' sensitive).
**`m`/`n`/`p` are functional no-ops** (they do NOT enable per-line ^/$;
multiline exists only via inline `(?m)`); only `s` (dotall) changes
matching; `l` = literal. Inline flags override the options arg. Verbatim
compile-error texts recorded (missing ]/), bad repetition, `**`, trailing
backslash, `\1` backref, lookahead) — all "Invalid Input Error: " + RE2
text; ours mirror the shape with rust-regex's message where texts differ
(class-correct; corpus success-cases never see them).

## Star forms (pins-waveB/star-similar-columns.json)

Deepens the wave-5 star pins: positive `* SIMILAR TO` / `COLUMNS('re')` =
unanchored RE2 search over declared-case names; `* NOT SIMILAR TO` =
NOT-full-match — **two independent predicates, never derived from each
other** ('a.*' selects {abc,abd,Weird Name} while NOT 'a.*' selects
{xyz,Weird Name}: "Weird Name" is in BOTH). NOT-form cannot produce the
zero-match error at all. Pattern must be a bare string literal. COLUMNS
expansion is always table-declaration order (even for alternation and
list forms); an alias stamps EVERY expansion (duplicates OK — feeds the
wave-5 dedup); un-aliased COLUMNS exprs keep bare column names. Grammar:
EXCLUDE-then-filter only. COLUMNS('re') in scope: SELECT-list expansion
(incl. inside an expression, zipped per-column); COLUMNS in WHERE
(AND-conjunction) and multi-set expressions stay unsupported.

## Implementation stages (each lands with tests + corpus replay green)

1. Regex infrastructure: `regex = "1"` dependency; translation module
   (pattern rewrite + reject list + options parser + replacement-template
   translation); Program-level compiled-regex table (patterns are
   prepare-time constants); IR ops + print/parse/verify + both backends.
2. Functions + operators: regexp_matches / regexp_full_match /
   regexp_extract / regexp_replace, `~` / `!~`, SIMILAR TO on values.
3. Star forms: * SIMILAR TO / NOT SIMILAR TO name filters (marker codes in
   rewrite.rs), COLUMNS('re') interception + expansion.
4. Census + bench parity + close-out.
