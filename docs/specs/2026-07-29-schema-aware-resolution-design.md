# Schema-aware resolution

**Date:** 2026-07-29
**Status:** loop 4 of SQLProjection marginalization; extends loops 1–3.

## What changes

`marginalize(sql, columns=None)` optionally takes the `__THIS__` column
names; `SQLProjection(sql, this_model=SomeModel)` passes its pydantic field
names (in definition order — the model is the declared, authoritative
schema). `fit(table)` then canonicalizes with `table.select(columns)`: model
order wins, extra table columns drop, missing ones fail with a named error.
`this_model` moves from `fit` to the constructor (v0 break) — errors surface
at construction and the plan is final before any data exists.

Schema-free mode is unchanged: every loop-1–3 behavior and refusal stands.

## The mechanism: the base environment becomes explicit

Schema knowledge is one change, not many: the base level's substitution
environment stops being a star-fallback and becomes an explicit ordered list
of `(name, __cf_t.name)` entries — base columns are just substitution
expressions like any projected column. Everything else falls out:

- **Unknown columns refuse at construction** with a bind-style message.
- **Struct access composes through plain columns**: a substitution target
  that is itself a pure column reference extends (`s.f` → `__cf_t.s.f`) —
  including through a CTE that projected the column; only access through a
  *computed* expression stays refused.
- **Every star expands to named items**, so `* EXCLUDE/REPLACE/RENAME`
  compose at any level (previously final-over-base only), and star-through-
  CTE needs no special cases.
- **`COLUMNS('regex')` expands** against the known names — the pattern is
  matched *by DuckDB itself* (`regexp_matches` at marginalize time), never by
  Python's `re`, so the dialect of the regex is the oracle's. Lambda and
  EXCLUDE forms of COLUMNS stay refused by name. Select-item position only.
- **Lateral aliases resolve exactly by DuckDB's rule**: a source column wins
  over an earlier item's alias; otherwise the alias's rewritten expression is
  inlined; otherwise unknown-column. The schema-free ambiguity refusal
  remains only in schema-free mode. Inside window arguments/keys/filters a
  lateral alias stays refused (the window's fit-side text runs against the
  source relation, which has no such column) — a named error with the same
  disambiguation hint.

## Gate

The round-trip invariant is unchanged and the fuzz gains a schema mode: half
of all generated cases construct with a `this_model` built from the table, so
every resolution path runs explicit. Corpus: new curated families (COLUMNS,
star modifiers through CTEs, resolved lateral aliases, struct-through-CTE)
move the pinned scoreboard up.

## Out of scope

Step semantics for off-support order keys (DRAFT-21, next); t-strings
(deprioritized); joins/static tables; `COLUMNS` inside expressions.
