# Wave-4 join-form pins — DuckDB 1.5.5, measured 2026-07-26

The implementation contract for TASK-50 stage A. Three probe agents,
native tables, full pin tables in `pins-wave4/*.json`. Nothing inferred
from documentation.

## Comma joins / cross / star (pins-wave4/comma_cross_star.json)

- `FROM t, u WHERE t.k = u.k` is bit-identical to `INNER JOIN ON` (rows
  AND star column order); WHERE conjuncts split freely into equi keys +
  residuals; three-way comma == nested INNER; the comma→probe rewrite is
  safe. Row ORDER of cross products is an artifact (flips with the
  projection shape — measured) — multiset comparison is mandatory.
- Cross join to a 1-row static: output rowcount == driving rowcount; a
  0-row static annihilates (INNER semantics) — so an empty-keys probe
  with a compile-time exactly-one-entry check is exact for 1 row and a
  clean reject beyond. LEFT JOIN ON TRUE to an empty table keeps rows.
- Star expands FROM-order, declared column order, duplicate names KEPT
  verbatim into the Arrow schema (`[id, lv, id, rv]`) — our typed output
  model cannot hold duplicate field names, so duplicate-output-name star
  shapes stay cleanly unsupported (documented model constraint).
- `FROM a, a a2`: bare `a` binds the unaliased occurrence (aliasing
  normally shadows the base name entirely). Schema-qualified same-name
  statics (`s1.t1, s2.t1`) bind per-schema — out of v0's bare-name
  static catalog, stays unsupported.

## USING (pins-wave4/using_desugar.json)

- `USING (a)` == `ON t1.a = t2.a` row-wise for INNER and LEFT; NULL keys
  never match.
- Star: the merged column appears ONCE, at the LEFT table's declared
  position (NOT hoisted to the front — refutes the PostgreSQL habit);
  USING-list order is irrelevant; output spelling = the LEFT table's
  declared capitalization; matching is case-insensitive EVEN QUOTED.
- Merged value = COALESCE(left, right) ≡ the LEFT value under
  INNER/LEFT-only — bit-exact simplification. `t1.a` and `t2.a` both
  stay addressable; `t2.a` is NULL on a LEFT miss (not coalesced).
- Chained `USING (a)` binds the MERGED column (≡ left under our kinds);
  `USING (a)` after a prior ON join that left duplicate visible `a`s is
  a binder AMBIGUITY error — never silently pick a side.
- `USING (a, a)` silently dedupes; USING a col missing on one side has
  side-specific error texts (left checked first); same static joined
  twice under different aliases is legal.

## Residual ON predicates (pins-wave4/equi_on_residuals.json)

- INNER: ON residual ≡ WHERE (rows and columns) — the rewrite target.
- LEFT: a key-matching row whose residual fails SURVIVES with an
  all-NULL right side (incl. the right key column) — ON filters matches,
  never rows. `AND false` ⇒ every driving row with NULL side; `AND
  true` no-op; `ON NULL = 2` binds fine (INNER empty / LEFT all-NULL).
- Match rule for the probe plan: `match = key_hit AND residual` with
  3VL residual collapse (NULL ⇒ non-match).
- ERROR EAGERNESS (the trap): DuckDB pushes SINGLE-side residuals into
  that side's scan — a trapping residual fires on rows/build-entries
  whose key never matches. BOTH-sides residuals are lazy per candidate
  pair (both kinds) — identical to our hit-guarded evaluation. v0 rule:
  single-side residual conjuncts must be conservatively TRAP-FREE
  (columns, literals, comparisons, IS NULL, logic) or the join rejects
  by name; both-sides residuals may trap (evaluated per key-matched
  pair, matching DuckDB's laziness bit-exactly).

## Stage-A scope consequences

- All-key statics: lift the no-value-columns rejection (empty probe dst
  list); join hit becomes first-class in the plan (SKind::JoinHit) for
  key-column reconstruction: `r.id` ≡ CASE match THEN dyn-key ELSE NULL.
- Expected non-flips recorded honestly: `SELECT *` shapes that produce
  duplicate output names (both-keys star, self comma joins) and
  schema-qualified statics stay cleanly unsupported.
- Stage B (output multiplicity — pure non-equi ON, range comma joins,
  N-row cross) is design-gated: written proposal before any build.
