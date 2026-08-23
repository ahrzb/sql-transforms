# Selection context: where boolean laziness lives, measured (TASK-124)

The engine decides AND/OR short-circuit per STATEMENT (`FB::in_filter`).
DuckDB decides it per CONTEXT, and the context recurses. That gap is the one
severity-1 contract break standing: we trap where both DuckDB readings serve,
one nesting level below a top-level `AND`, and we answer queries DuckDB
refuses under `NOT`/`IS NULL`. This spec fixes the MODEL; every rule below is
a measurement, not a guess.

## 1. The measured model

All measurements 2026-08-19, DuckDB 1.5.5, `PRAGMA disable_optimizer`,
`t(b BOOLEAN, s VARCHAR)`, `T = CAST(s AS DOUBLE) > 1`, the discriminating
row `(NULL, 'abc')` (plus `(true,'1.5')`, `(false,'abc')` where 3 rows shown).

**Where selection context ENTERS** — exactly two places:

```sql
WHERE <pred>                                   -- the predicate root
CASE WHEN <cond> ...                           -- every condition, ANYWHERE:
SELECT CASE WHEN (b AND T) THEN 1 ELSE 2 END   --   [2,1,2] serves, projection
```

**And nowhere else.** A boolean tree in a projection is value context even
when it looks like a filter:

```sql
SELECT (b AND T) FROM t              -- TRAP  (dfb3a99's fix, still right)
SELECT ((b AND T) OR TRUE) FROM t    -- TRAP  (ctx does NOT blanket projections)
```

**How it propagates once entered:**

| node | children's ctx | evaluation in selection ctx |
|---|---|---|
| `AND` | both selection | LEFT always evaluated (`WHERE T AND b` traps, even `b` false); RIGHT skipped when left is not TRUE (`WHERE b AND T` serves on b NULL/false) |
| `OR` | both selection | BOTH always evaluated — `WHERE coalesce(b,TRUE) OR T` TRAPS; laziness reaches only ANDs inside (`WHERE (b AND T) OR TRUE` serves) |
| `CASE` | condition selection; ARMS VALUE | `WHERE CASE WHEN ((b AND T) OR TRUE) THEN ...` serves; `WHERE CASE WHEN TRUE THEN (b AND T) ELSE TRUE END` TRAPS — a taken arm is eager |
| `NOT`, `IS [NOT] NULL`, comparisons, function args, `coalesce`/`nullif`/`IN` operands | value | all measured TRAP: `NOT (b AND T)`, `(b AND T) IS NULL`, `(b AND T) = TRUE`, `coalesce((b AND T), TRUE)`, `nullif((b AND T), FALSE)`, `(b AND T) IN (TRUE, NULL)` |

Nesting recurses arbitrarily: `((b AND T) OR TRUE) AND TRUE` and
`((b OR TRUE) AND (b AND T)) OR TRUE` both serve 3/3; `(b AND T) AND TRUE`
serves in a filter (left child keeps ctx) and traps in a projection.

Why AND may skip its right on a not-TRUE left without computing Kleene
exactly: the consumer of a selection-context boolean asks only "is it TRUE".
`NULL AND FALSE` is FALSE while our skip yields NULL — but FALSE and NULL are
the same answer to that question, and `OR` of two truths preserves it
(NULL-as-false OR x is TRUE exactly when Kleene's is). `NOT` would tell the
two apart, which is exactly where DuckDB reverts to value context.

## 2. The design

One new emission mode in `lower.rs`, alongside `emit` (value):

```rust
/// Selection context: the value is consumed only for TRUE-ness, so the
/// result is a bare i1 -- no null flag. Entered from the WHERE root and
/// from every CASE condition; exits to `emit` at every operator that can
/// tell NULL from FALSE.
fn emit_truth(&mut self, e: &SExpr, live: &mut Live) -> Result<Value, PrepareError> {
    match &e.kind {
        SKind::And { a, b } => {
            let at = self.emit_truth(a, live)?;
            // brif: not-TRUE left -> merge(false) WITHOUT touching b;
            // the existing trap_if/narrow_trap CFG plumbing is the model
            ...skip-right block split...
        }
        SKind::Or { a, b } => {
            let at = self.emit_truth(a, live)?;   // both sides ALWAYS run
            let bt = self.emit_truth(b, live)?;
            Ok(self.bin(BinOp::Or, at, bt))
        }
        SKind::Case { .. } => ...condition via emit_truth, arms via emit
                              (value), each arm's Lane reduced to truth at
                              the merge...,
        // everything else: value context, then reduce
        _ => {
            let l = self.emit(e, live)?;
            Ok(self.truth_of(l))   // flag && val for a bool lane
        }
    }
}
```

Consumers:

- the scalar WHERE: `let t = emit_truth(pred); brif !t -> drop row` —
  replaces the `flatten_and` spine, whose per-conjunct drop is the special
  case this generalizes (top-level AND chain short-circuits left-to-right).
- the many-join WHERE sites (the two `emit_many` filters): same call —
  this is what closes the F2 half of the ticket.
- every CASE **condition**, in `emit`'s own Case arm too: conditions lower
  via `emit_truth` regardless of where the CASE sits. Arms stay `emit`.

Deleted: `FB::in_filter`, and the `may_trap` gating on `kleene` — in value
context AND/OR are ALWAYS eager (both operands emitted, Kleene combine), in
selection context laziness lives only in `emit_truth`. The statement-level
flag has no reader left.

NOT of a bool lane in value context needs no change: operands were already
eager there.

Out of scope, unchanged: JOIN ON residuals (their trap-freeness guard is
`plan::may_trap`, a build-time refusal — different mechanism, measured pins
of its own), COALESCE argument laziness (a CASE desugar whose conditions are
`IS NOT NULL` — arms eager per this model, matching the measurements).

## 3. What flips, what gets added

The 7 existing xfail-strict pins flip (nested OR, CASE-condition projection
and filter, NOT, IS NULL, many-join JOIN/LEFT). New pins from this spec's
measurements, all currently WRONG in the engine in the mirror direction
(we short-circuit where DuckDB is eager — we ANSWER what it refuses):

```sql
WHERE coalesce(b, TRUE) OR T          -- oracle TRAP, we serve today
WHERE T AND b                         -- oracle TRAP (left always runs)
WHERE CASE WHEN TRUE THEN (b AND T) ELSE TRUE END   -- oracle TRAP (arm eager)
SELECT ((b AND T) OR TRUE)            -- oracle TRAP (no ctx in projections)
```

plus serving controls: `(b AND T) AND TRUE` filter serves, cond→OR→AND
serves in both filter and projection.

## 4. Verification

- red first: run the 7 pins + the new measurements as failing tests before
  the lowering change; each must fail for its own reason.
- the fuzz grammar (AC #6): emit nullable-bool conjuncts under `OR` and as
  CASE WHEN conditions with trapping siblings, so this class is reachable —
  today it structurally is not, which is how the 4000-seed gate over-claimed.
- a 4000-seed campaign after, expected residue = known classes only; the
  TASK-129 order legs are already in the harness and must stay clean.
- full suite in release AND debug (debug_asserts catch CFG mistakes the
  release build forgives — the 2026-08-08 sweep lesson).
