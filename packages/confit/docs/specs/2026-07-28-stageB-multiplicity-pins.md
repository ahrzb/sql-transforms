# Stage B: join multiplicity under shape='many' — pins + design (TASK-59)

Measured against DuckDB 1.5.5, 2026-07-28. Raw pins: `pins-stageB/*.json`
(5 fleet agents, 57 pins). Gate: every construct here builds ONLY under
`shape='many'` (TASK-58); under `filter`/`map` the rejections stay exactly
as today.

## The row-SET contract (stable everywhere — this is what we serve)

1. **Inner equi-join**: one output row per (left row, matching right row)
   pair; full per-key cross product when BOTH sides have duplicate keys
   (2×3 = 6 verified).
2. **LEFT**: per left row, `max(1, surviving matches)` rows — exactly one
   null-extended row when zero matches survive, INCLUDING when a residual
   ON predicate filtered every match away.
3. **NULL keys match nothing** — under `=`, `>`, and crucially `<>`
   (NULL <> NULL is NULL, not TRUE). A NULL-keyed left row = unmatched:
   dropped by INNER, one null-extension by LEFT. (The entire 9-case
   left_join_issue_1172 corpus family reduces to this — probe is NULL, so
   every case is a single null-extended row; the dup keys are inert.)
4. **Residual ON predicates** filter per match-pair; WHERE composes
   orthogonally (left rows removed first).
5. **Keyless joins** (comma, CROSS, `ON` inequality / one-sided /
   constant): exactly cross-product-then-filter, verified bit-for-bit.
   `ON NULL = 2` → zero matches (LEFT: null-extends every left row).
   Empty right side: INNER → 0 rows, LEFT → all null-extended.
6. **USING** merges the key into ONE leading output column carrying the
   LEFT value (visible on null-extended rows: the probe key, not NULL).
   With ON, both copies appear; the model/dict boundary applies the
   wave-5 dedup rename (`id, id_1`) — same convention as DuckDB's own
   `.df()`.
7. **Self-join stars**: `SELECT *` = left cols then right cols in
   declaration order; unqualified EXCLUDE strips EVERY copy; qualified
   EXCLUDE strips one side's (USING survivor's position depends on which
   copy went — pinned); excluding all of one side's star is legal; bare
   names (and rowid) are ambiguity errors.

## ORDER: an accident, not a contract (the fleet's central finding)

DuckDB's join output order is a hash-join artifact on three independent
axes: the optimizer picks the streamed side by cost (not FROM order, not
reliably size), dup-key matches emit in REVERSE build-insertion order
(LIFO chains) in per-2048-row lockstep passes, and at threads>1 with
~500k+ rows the order differs RUN-TO-RUN on the same connection. Even
single-threaded, one probe row's matches land ~2048 rows apart — a
row-at-a-time engine cannot naturally reproduce it and must not try.

**Decision**: parity for shape='many' is **multiset** (sort both sides
before comparing — the corpus already does; duck_check gains a sorted
mode for 'many' tests). The engine defines its OWN deterministic order,
documented: probe rows in input order; all matches for one probe row
contiguous; matches in build INSERTION order; the LEFT null-extension
in place. This intentionally diverges from DuckDB's accidental order —
SQL without ORDER BY promises none, and we reject ORDER BY anyway.

## Engine design (validated against the emit-machinery study, TASK-59 plan)

- Emission becomes explicit: `Inst::EmitRow` appends the out-lane values;
  `CTerm::Emit` retires to end-of-row. Loops use the existing Brif/Jump
  blocks with back-edges (verify must accept them — first change).
- `StaticTy::Map` values become per-key row LISTS (flat arena +
  (offset, len) per key). New ops `ProbeStart`/`ProbeNext` iterate them;
  keyless joins are the degenerate one-bucket scan with the residual in
  the loop body; LEFT keeps an `any_emitted` flag and emits the
  null-extension when the loop closes dry.
- Self-joins build the batch-side map per call before the row loop.
- Cranelift: emission via an `h_emit_row` helper (helpers already write
  through Cx), loops as normal CLIF blocks; the row_fn return code stays
  for end/trap only. Fallback if heavy: interpreter-first for 'many'
  (documented, backend-identity tests exempt 'many').
- Corpus harness: replay retries with shape='many' ONLY on the named
  multiplicity rejections, so the default-shape rejections stay proven.
