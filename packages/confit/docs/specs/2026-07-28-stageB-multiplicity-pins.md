# Join multiplicity under shape='many'

DuckDB 1.5.5. Raw pins: `pins-stageB/*.json` (57 pins). Tests:
`tests/test_duckdb_stageb_many.py`. Multiplicity constructs (duplicate
build keys, keyless N-row joins, self-joins) build ONLY under
`shape='many'`; under `filter`/`map` confit refuses them by name
("duplicate map key", "dynamic table to itself").

## The row-SET contract

1. **Inner equi-join**: one output row per (left row, matching right row)
   pair; the full per-key cross product when BOTH sides have duplicate keys
   (2×3 = 6 verified).
2. **LEFT**: per left row, `max(1, surviving matches)` rows — exactly one
   null-extended row when zero matches survive, INCLUDING when a residual
   ON predicate filtered every match away.
3. **NULL keys match nothing** — under `=`, `>`, and `<>` (NULL <> NULL is
   NULL, not TRUE). A NULL-keyed left row is unmatched: dropped by INNER,
   one null-extension by LEFT. (The 9-case `left_join_issue_1172` corpus
   family reduces to this: the probe is NULL, so every case is a single
   null-extended row.)
4. **Residual ON predicates** filter per match pair; WHERE composes
   orthogonally (applied after the join).
5. **Keyless joins** (comma, CROSS, `ON` inequality / one-sided /
   constant): exactly cross-product-then-filter. `ON NULL = 2` → zero
   matches (LEFT null-extends every left row). Empty right side: INNER → 0
   rows, LEFT → all null-extended.
6. **USING** merges the key into ONE leading output column carrying the
   LEFT value (on null-extended rows: the probe key, not NULL). With ON,
   both copies appear, renamed by the duplicate-name rule (`id, id_1`) —
   see `2026-07-26-wave5-structural-pins.md`.
7. **Self-join stars**: `SELECT *` = left columns then right columns in
   declaration order; unqualified EXCLUDE strips EVERY copy; qualified
   EXCLUDE strips one side's (the USING survivor's position depends on
   which copy went — pinned); excluding all of one side's star is legal;
   bare names (and rowid) are ambiguity errors.

## Order is not part of the contract

DuckDB's join output order is a hash-join artifact on three independent
axes: the optimizer picks the streamed side by cost (not FROM order, not
reliably size); dup-key matches emit in REVERSE build-insertion order (LIFO
chains) in per-2048-row lockstep passes; and at threads > 1 with ~500k+
rows the order differs run to run on the same connection. Even
single-threaded, one probe row's matches land ~2048 rows apart.

Parity for `shape='many'` is therefore **multiset** (both sides sorted
before comparing). confit's own order is deterministic: probe rows in input
order; all matches for one probe row contiguous, in build INSERTION order;
the LEFT null-extension in place. confit refuses ORDER BY.

## confit's implementation

- The single join lowers as a loop over the multimap row range for the
  probe key (`ProbeRange`; a NULL key forces an empty range). The residual
  gates match-ness; WHERE gates emission only; LEFT emits the
  null-extension row when no match survived (`lower_many_loop` in
  `src/specializer/lower.rs`).
- Under `shape='many'` confit refuses more than one join per query and
  IS NOT DISTINCT FROM join keys.
- The corpus replay (`tests/test_corpus_replay.py`) retries with
  `shape='many'` only on the named multiplicity refusals, so the
  default-shape refusals stay proven.
