# Struct-valued transform calls (the subtraction loop)

Ruled by AmirHossein 2026-08-04: loops 1–4 introduced two owned surface
rules with no oracle reading; delete them now rather than waiting for
DRAFT-25 (the fit/transform-split epic), and refuse where the epic will
later provide the nested boundary. Sequenced FIRST: this loop →
composition (DRAFT-24 loop 5, with partition composition) → private
columns (TASK-64) → DRAFT-25.

## The rule

**Anything with declared output field names is struct-valued at every
width; only field reads are scalars.** The discriminator is
`return_names` being non-empty — true for every fitted transformer and
every `Named(...)`; false for author scalar UDFs (`PythonUDF`, width-1
scalar as today) and for unnamed width-k externs (direct
`DuckDBInferFn(udfs=...)` users keep the engine's `list | None`
boundary untouched).

```sql
-- stays (the one spelling for consuming a transform):
sc(struct_pack(a := age)) OVER (PARTITION BY g).a * 10 AS z     -- field read: scalar, composes

-- deleted behavior #1 — width-1 auto-unwrap. These now REFUSE:
sc(struct_pack(a := age)) OVER () * 10 AS z    -- a struct has no scalar reading (any width now)
sc(struct_pack(a := age)) OVER () AS z         -- a struct output column needs DRAFT-25's boundary

-- deleted behavior #2 — flat alias-prefixed expansion. This now REFUSES:
pca(struct_pack(a := age, f := fare)) OVER () AS e     -- was: e_pca0, e_pca1 columns
-- refusal names the fix: "address a field (pca(...).name); nested outputs arrive with DRAFT-25"
```

The bare-item refusal is **construction-time** (P7 proper, not the
carve-out): a bare transformer call as a select item is a struct by
construction — no learned width needed to refuse it. Field-name
*validation* stays fit-time (T is learned).

## What gets deleted (code)

- `_marginalize.py`: the width-1 collapse in `_finalize_fields` (field
  reads survive uniformly as `(call).field` in serving SQL);
  `expand_wide_items` and `_check_no_nested_wide` entirely — subsumed by
  the construction-time bare-call refusal (a call under a field read is
  the one permitted position).
- `_projection.py`: the `expand_wide_items` call and widths plumbing in
  `fit`.
- Engine (`frontend.rs`): the TASK-63 width-1 field-access refusal
  FLIPS — with registration now struct-at-every-width, `(sc(x)).a` binds
  lane 0 of a width-1 named extern (C2: DuckDB reads the single-field
  struct the same way), and a NAMED extern in any non-field-read position
  refuses (scalar expression AND bare item — generalizing the
  named-wide-bare-item refusal the review added). Unnamed externs:
  completely unchanged.
- `_udf.py` `_duck_signature`: named ⇒ `duckdb.struct_type(...)` at every
  width (drop the `len > 1` condition); `_scalar` returns the dict for
  named width-1 too.

## What stays untouched

Name-keyed identity and fit-time validation (P16a and the C5 refit
protection), `Named`, the `output_names` ladder, the one-call/lane-read
engine machinery and its counting guarantees (TASK-63), the case-collision
and alias-soundness refusals, the NULL story (NULL id → NULL struct →
NULL fields), the unnamed-extern engine boundary and its tests.

## Gates and moved pins

- C1–C5 unchanged in meaning. C4's transformer tests and C3's serve gate
  re-spell bare/arithmetic width-1 usages as field reads; loop-3's
  flat-expansion tests become refusal pins (construction-time, exact
  message pinned).
- Engine tests: width-1 field access flips from refusal pin to a
  bind+execute test; a named-extern-as-scalar refusal pin is added;
  `named_wide_extern_as_a_bare_item_refuses` stays (message may
  generalize); `test_wide_udf_bare_item_is_list_field` (unnamed) stays
  byte-identical.
- Bench: `tf_width1` already field-addressed (no change);
  `tf_bare2` is no longer legal SQL — dropped from the scenario table
  with a kpis.md note (its measurement burden is carried by `tf_fields2`).
  D2 re-run to confirm no regression from single-field struct
  registration on the width-1 path.
- Corpus/D1: no transformer entries — untouched.

## Laws

P16 (width rules) rewrites: "a call with declared output names is
struct-valued at every width; only field reads cross the serving output
boundary; a bare call as an output refuses until the nested boundary
(DRAFT-25); bundles destructure at construction — nothing struct-shaped
flows into computation." The loop-3 flat-boundary clause and its
NULL-distinction note are deleted (DRAFT-25 restores the distinction
properly via nested outputs). DRAFT-24's loop 1/3 sections get dated
supersession notes.

## Why now (recorded)

Every surviving behavior after this loop has either an oracle reading
("what would DuckDB compute with these registered") or a named refusal —
the hygiene bar set with DRAFT-25. Doing it before composition removes
the width-1 special case from member inlining entirely.
