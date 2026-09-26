# Regexp pins: engine translation, functions, operators, options, star forms

DuckDB 1.5.5 (RE2). Every claim is backed by an executed query/program
recorded verbatim in `pins-waveB/*.json`. Tests:
`tests/test_duckdb_waveB_regexp.py`, `tests/test_duckdb_regexp_fuzz.py`.

## Engine: rust `regex` behind a translation layer (`pins-waveB/re2-vs-rust-regex.json`)

confit runs rust `regex` (`regex = "1"`) behind a bind-time translation
layer (`src/specializer/retrans.rs`). Raw pass-through is wrong for any
pattern with a Perl class. With the following applied, every entry of the
98-entry differential battery is byte-identical or identically rejected:

- Config: `RegexBuilder::octal(true)`; default Unicode mode kept —
  `unicode(false)` BREAKS `(?i)` folding parity (KELVIN and sharp-s agree
  by default).
- Pattern rewrite (Perl classes are the whole Unicode gap — RE2 is ASCII,
  rust Unicode), each verified byte for byte incl. inside `[...]` classes:
  `\d`→`(?-u:\d)`, `\D`→`[^0-9]`, `\w`→`[0-9A-Za-z_]`,
  `\W`→`[^0-9A-Za-z_]`, `\s`→`[\t\n\f\r ]`, `\S`→`[^\t\n\f\r ]`,
  `\b`→`(?-u:\b)`. `(?-u:\s)` is the WRONG `\s` rewrite (rust ASCII `\s`
  includes VT); `(?-u:\W)` does not compile — negated classes are spelled
  as Unicode-mode `[^...]`.
- Replacement templates translate: DuckDB is `\N` with literal `$`; rust is
  `$N` with literal `\N` — `\N`→`${N}`, `$`→`$$`.
- Agree with zero config: leftmost-first alternation, greedy/lazy/(?U)/
  (?s), empty-match iteration incl. replace_all adjacency skip, `\A`/`\z`,
  `$` without trailing-newline magic, mid-pattern inline flags, `\p{...}`,
  `[]a]`; backrefs/lookaround/`\Z`/`a{2,1}` rejected by both.

confit refuses at bind (each measured to differ or be unservable):

- `\B` (RE2's ASCII `\B` matches inside multibyte chars; DuckDB itself dies
  at runtime on non-ASCII);
- `(?<name>...)` angle groups (DuckDB rejects); duplicate group names
  (rust rejects, DuckDB accepts);
- repetition bounds > 1000 (`a{1001}` is a DuckDB error), and NESTED
  counted-repetition bound products over 1000 (RE2 "invalid repetition
  size"; an unbounded inner count counts as over the cap);
- stacked quantifiers: one lazy `?` is the only legal follower; `a*+`,
  `{2}*`, `?*`, `*{2}`, `a???` are DuckDB errors that rust silently
  reinterprets;
- `\uXXXX` escapes, `\Q...\E`;
- `\1`-`\9` outside a class (RE2 backref reject vs rust octal escape);
  inside a class both read octal and it serves;
- `{1, 3}` with whitespace (literal to RE2, a repetition to rust);
- in character classes: `--`/`&&`/`~~` (rust set operations), non-POSIX
  nested `[`, Perl-class range endpoints, ranges starting at a
  class-leading `]`. POSIX `[:...:]` elements are consumed atomically;
- anchor-only patterns (2+ text anchors with only flag/`(?:)` noise):
  DuckDB is self-inconsistent — the row path literal-optimizes `'$\z'` to
  string equality (FALSE for 'hello') while its constant fold says TRUE;
- capturing `(x){0}` (rust erases the group, shifting group numbering).

## Differential fuzzer (`pins-waveB/fuzzer-task54.json`, `pins-waveB/fuzzer-20260728.json`)

`tests/test_duckdb_regexp_fuzz.py` runs in the normal gate. Contract per
case: identical rows, or engine build-time reject, or both error; a DuckDB
error where the engine serves, or a mismatch, fails with seed + case index
+ SQL in the message. `REGEXP_FUZZ_SEED` / `REGEXP_FUZZ_N` override the
fixed defaults for deep runs. The JSON files record each divergence class
the fuzzer found with witnesses and measured outputs.

## Operators and match functions (`pins-waveB/matches-operators.json`)

- `regexp_matches(s, p)` = **unanchored SEARCH**; `regexp_full_match(s,
  p)` = whole string.
- **`~` is FULL match, not search** (binds to regexp_full_match; diverges
  from PostgreSQL). `!~` = NOT(~) with standard NULL propagation.
- **SIMILAR TO does NO wildcard translation**: the pattern goes RAW to RE2
  with full-match anchoring (`'hello' SIMILAR TO 'h%o'` is FALSE; `h.llo`
  is TRUE) — exactly regexp_full_match. `SIMILAR TO ... ESCAPE` is "Not
  implemented Error: Custom escape in SIMILAR TO"; confit refuses it.
- Empty pattern: regexp_matches(s, '') is true for EVERY non-NULL string;
  full-match/`~` with '' is true only for ''.
- NULL string/pattern → NULL; **NULL options arg → error** ("Regex options
  field must not be NULL"), not NULL. All results BOOLEAN.
- Constant patterns compile at BIND time (errors fire under WHERE 1=0 /
  EXPLAIN / PREPARE); column patterns compile at EXECUTION (WHERE false
  suppresses). confit compiles patterns at prepare and refuses non-constant
  patterns.

## regexp_extract (`pins-waveB/extract.json`)

Default/group 0 = whole match; groups 1..n; **no match → `''`, never
NULL** — same for non-participating groups and in-range indexes above the
pattern's group count. Unanchored leftmost-first search. **The group index
is a flat 0..9 range check unrelated to the pattern** ("Group index must be
between 0 and 9!" even when 10 groups exist; also for negatives; fires even
on non-matching subjects). NULL subject/pattern → NULL but **NULL group →
`''`** for a non-NULL subject. Options only as the 4th arg (no
(V,V,VARCHAR) overload). Group index and options must be constants. The
name-list STRUCT form is non-scalar; confit refuses it. Result VARCHAR.

## regexp_replace (`pins-waveB/replace.json`)

First match only by default; `'g'` global. Rewrite grammar: `\0` whole
match, `\1`-`\9`, `\\` literal backslash, `\10` = `\1` + literal '0';
`$1`/`$0`/`&` are PLAIN LITERALS. **Invalid rewrites never error, and
global vs non-global diverge**: non-global → input unchanged for ANY
invalid rewrite; global → an out-of-range backref is still a full no-op
(RE2's MaxSubmatch pre-check, which outranks a bad escape even when the bad
escape comes first), but a bad escape CONSUMES each match and emits only
the prefix before the bad escape (`('hello','h','\x','g')` = `'ello'`).
A non-participating in-range group → empty. Empty-match global inserts at
every codepoint boundary incl. both ends (`'abc'` → `'XaXbXcX'`), no empty
match immediately after a nonempty one, never inside UTF-8 bytes. ANY NULL
arg (incl. options — asymmetric with the other functions) → NULL.

NULL constant patterns: `2026-07-28-waveA-structural-tails.md`.

## Options + compile errors (`pins-waveB/options-errors.json`)

Alphabet: `c i l m n p s` everywhere, `g` ONLY in regexp_replace ("Option
'g' (global replace) is only valid for regexp_replace" elsewhere); unknown
→ "Unrecognized Regex option <ch>". Parsed left to right, whitespace
skipped, the LAST conflicting letter wins ('ci' insensitive, 'ic'
sensitive). **`m`/`n`/`p` are functional no-ops** (they do NOT enable
per-line ^/$; multiline exists only via inline `(?m)`); only `s` (dotall)
changes matching; `l` = literal. Inline flags override the options arg.
Compile errors are "Invalid Input Error: " + RE2 text (verbatim texts in
the JSON); confit mirrors the shape with rust-regex's message where the
texts differ.

## Star forms (`pins-waveB/star-similar-columns.json`)

Positive `* SIMILAR TO` and `COLUMNS('re')` = unanchored RE2 search over
declared-case names; `* NOT SIMILAR TO` = NOT-full-match — **two
independent predicates, never derived from each other** ('a.*' selects
{abc, abd, Weird Name} while NOT 'a.*' selects {xyz, Weird Name}). The NOT
form cannot produce the zero-match error. The pattern must be a bare
string literal. COLUMNS expansion is always table-declaration order (even
for alternation and list forms); an alias stamps EVERY expansion
(duplicates then go through the duplicate-name rename); un-aliased COLUMNS
expressions keep bare column names. Grammar: EXCLUDE then filter only.
DuckDB also expands COLUMNS inside an expression (zipped per column).

confit serves `* [NOT] SIMILAR TO` and `COLUMNS('re')` / `COLUMNS(*)` as a
bare SELECT item; it refuses COLUMNS inside an expression, in WHERE, and
with a non-constant or list argument.
