# Join-form pins: comma, cross, star, USING, ON residuals

DuckDB 1.5.5, native tables. Full pin tables in `pins-wave4/*.json`.
Nothing inferred from documentation. Output multiplicity (duplicate keys,
keyless N-row joins) is in `2026-07-28-stageB-multiplicity-pins.md`.

## Comma joins / cross / star (`pins-wave4/comma_cross_star.json`)

- `FROM t, u WHERE t.k = u.k` is bit-identical to `INNER JOIN ON` (rows AND
  star column order); WHERE conjuncts split freely into equi keys +
  residuals; a three-way comma == nested INNER. Row ORDER of cross products
  is an artifact (flips with the projection shape), so comparison is by
  multiset.
- Cross join to a 1-row static: output rowcount == driving rowcount; a
  0-row static annihilates (INNER semantics). LEFT JOIN ON TRUE to an empty
  table keeps rows. confit serves a keyless comma table as an empty-key
  probe; the duplicate-key check enforces a single entry at compile time.
- Star expands in FROM order, declared column order, duplicate names KEPT
  verbatim (`[id, lv, id, rv]`). confit applies DuckDB's duplicate-name
  rename at its output boundary (`2026-07-26-wave5-structural-pins.md`).
- `FROM a, a a2`: bare `a` binds the unaliased occurrence (aliasing
  normally shadows the base name entirely). Schema-qualified same-name
  statics (`s1.t1, s2.t1`) bind per schema; confit refuses a comma table
  that is not a provided static.

## USING (`pins-wave4/using_desugar.json`)

- `USING (a)` == `ON t1.a = t2.a` row-wise for INNER and LEFT; NULL keys
  never match.
- Star: the merged column appears ONCE, at the LEFT table's declared
  position (not hoisted to the front, unlike PostgreSQL); USING-list order
  is irrelevant; output spelling = the LEFT table's declared
  capitalization; matching is case-insensitive EVEN QUOTED.
- Merged value = COALESCE(left, right) ≡ the LEFT value under INNER/LEFT.
  `t1.a` and `t2.a` both stay addressable; `t2.a` is NULL on a LEFT miss.
- Chained `USING (a)` binds the MERGED column; `USING (a)` after a prior ON
  join that left duplicate visible `a`s is a binder AMBIGUITY error.
- `USING (a, a)` silently dedupes; a USING column missing on one side has
  side-specific error texts (left checked first); the same static joined
  twice under different aliases is legal.

## Residual ON predicates (`pins-wave4/equi_on_residuals.json`)

- INNER: an ON residual ≡ WHERE (rows and columns).
- LEFT: a key-matching row whose residual fails SURVIVES with an all-NULL
  right side (incl. the right key column) — ON filters matches, never
  rows. `AND false` ⇒ every driving row with a NULL side; `AND true` is a
  no-op; `ON NULL = 2` binds (INNER empty / LEFT all-NULL).
- Match rule: `match = key_hit AND residual`, with a NULL residual counted
  as non-match.
- Error eagerness: DuckDB pushes SINGLE-side residuals into that side's
  scan, so a trapping residual fires on rows/build entries whose key never
  matches. BOTH-sides residuals are lazy per candidate pair. confit
  evaluates residuals per key-matched pair, so it serves both-sides
  residuals that may trap and refuses single-side residual conjuncts that
  are not trap-free (columns, literals, comparisons, IS NULL, logic).
- All-key statics (no value columns) serve; `r.id` of the probed key
  column is `CASE match THEN probe-key ELSE NULL` (`SKind::JoinHit`).
