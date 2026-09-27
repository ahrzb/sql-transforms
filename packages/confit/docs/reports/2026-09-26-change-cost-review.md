# Making change cheaper: codebase and testing review (2026-09-26)

Measured on master `d04f1d9` (DuckDB 1.5.5 through `confit.oracle.Oracle`,
optimizer off; Linux x86_64, 4 cores).

## 1. The answer

Both the codebase and the testing scheme need work, and four experiments say
where. The fuzzer is run too shallow: 50,000 seeds found **4 live defects**
that 2,000 seeds never reach. Over-refusals are mostly spelling
inconsistencies, which a cheap metamorphic suite catches without DuckDB (it
found one in 9 seconds). `frontend.rs`, the code that changes most, is too big
to change safely. The gate spends 62% of its time waiting on one known DuckDB
hang.

| change | effort | what it buys |
|---|---|---|
| fix the 4 live defects | ~1 day | parity: wrong answers or wrong traps today |
| gate: marker-timed hang pin + `pytest -n 4` | hours | gate 6.5 → ~1.3 min |
| metamorphic spelling suite | ~1 day | the over-refusal class in seconds, no oracle |
| nightly deep campaign (50k–100k seeds) | hours | the depth that found tonight's defects |
| one parity harness + `confit.probe` | ~half a day | uniform test strictness; no throwaway probe scripts |
| split `frontend.rs`, one relation resolver | ~1 day | safer subquery work, fewer merge conflicts |
| explicit Cranelift fallback | hours | a codegen failure can no longer hide |

Hypothesis-style property testing would add little: the campaign already is a
grammar-based property test with a shrinker. What is missing is its depth,
metamorphic properties, and a generator kept level with the engine.

## 2. Evidence: where the day's defects came from

The 2,000-seed campaign found no wrong answer all day. Every defect was found by
targeted probing or read off the campaign's refusal histogram:

| defect | kind | found by |
|---|---|---|
| CAST to a target with no lane | wrong answer | reading the code |
| EXCLUDE of the left USING qualifier | wrong answer | reading the code |
| bind-time folding beyond DuckDB's binder | wrong answer | open xfail pins |
| presence lane accepted a scalar | wrong input accepted | reading the code |
| `other.__THIS__.a` served | served an error case | while building |
| VARCHAR key vs a narrower probe (pre-merge) | wrong answer | while building |
| static struct leaf as a join key | over-refusal | probing an adjacent feature |
| dotted struct field names | over-refusal | PLANS |
| `s['f']`, `struct_extract`, `v.*` | over-refusal | PLANS |
| quoted table names | over-refusal | campaign refusal list |
| `CROSS JOIN` | over-refusal | campaign refusal list |
| schema-qualified join keys | over-refusal | probing |

Most over-refusals were one spelling serving while an equivalent spelling
refused. And every served form added that day needed a generator update, most of
which it did not get: the fuzzer only sees what its grammar generates.

## 3. The 6 interpreter fallbacks

All 6 in the 2,000-seed campaign (seeds 486, 533, 717, 1098, 1156, 1636) are
`shape='many'` programs: Cranelift rejects every multiplicity program
(`exec/cranelift.rs`, `has_multiplicity`) because it lacks the MultiMap and
BatchMap statics, `ProbeRange`, `ProbeRead` and the `EmitTo` terminator. So
every 1:N join serves on the interpreter, about 1.4× slower.

The fallback also discards every Cranelift error (`Err(_) =>` in
`duckdb/mod.rs`), so a real codegen failure would silently serve on the
interpreter. Fix: fall back only for the named reason, treat any other error as
a bug, and report fallbacks by reason in the campaign. Teach Cranelift the four
constructs when the subquery pipeline rewrites the lowering.

## 4. Testing

| experiment | cost | result |
|---|---|---|
| campaign at 50,000 seeds | ~12 min, 4 workers | 4 live defect classes (below); 16 `DIVERGE_TRAP`, 2 `DIVERGE_VALUE` |
| metamorphic spelling harness (prototype) | 9 s for 2,000 seeds × 7 rewrites | 1 over-refusal: a static struct-leaf join key spelled `c0['f0']`, `(c0).f0` or `struct_extract` refuses |
| kernel sweep: 108 expressions × 2,000 edge values | 11 min | only `cbrt`'s approved one-ulp bound |
| `pytest -n 4` | 74 s vs 390 s | parallel-safe as it stands |

The live defects, each minimized with `fuzz.shrink`:

1. `SELECT c2 - c1` with `c2` NULL and `c1` = −2⁶³ (seed 22366): confit traps,
   DuckDB returns NULL. The overflow check runs on a NULL row's payload.
2. `CASE WHEN FALSE THEN c1 ELSE c0 * c0 END`, `coalesce(c0 * c0, c1)` with
   `c0` TINYINT (seed 5008 and 12 more): confit returns 16384, DuckDB overflows.
   Unification with a wider arm casts the operands instead of the result.
3. `(udf0(NULL, CAST(34 AS DOUBLE))).f2` (seeds 40473, 49961): typed DOUBLE,
   DuckDB INTEGER.
4. `(CASE WHEN 0.75e0 THEN CAST('' AS DOUBLE) … END) / floor(TRY_CAST('%_' AS
   DOUBLE))` (seed 16617): DuckDB returns NULL without evaluating the CASE;
   confit traps.

Recommendations:

1. Run the campaign deep and on a schedule (nightly 50k–100k), findings shrunk
   and filed automatically.
2. A standing metamorphic spelling suite: every served feature adds its
   equivalent spellings to one table (quoting, case, qualification, struct
   access forms, CAST synonyms, `a, b` / `CROSS JOIN`, `x <> y` / `NOT (x = y)`,
   later `q` / `SELECT * FROM (q)`). Equivalent spellings both serve with equal
   answers or both refuse.
3. One parity harness: the tests carry about 25 hand-rolled parity helpers and
   51 direct `Oracle()` uses of differing strictness. One `assert_parity` built
   on the campaign's verdict code.
4. Operator sweeps with NULLs and mixed widths: defects 1 and 2 live in operand
   typing and NULL masking, one layer above the kernels.
5. Adding a feature's spellings to the generator is part of its definition of
   done.

## 5. Codebase

1. **Split `frontend.rs`** (9,045 lines, 158 functions, 26 commits this month,
   two of the day's merge conflicts) along its responsibilities: `from.rs`,
   `resolve.rs`, `joins.rs`, `star.rs`, `functions.rs`, `typing.rs`,
   `refusal.rs`. Mechanical, no behavior change; the subquery design reshapes
   FROM handling anyway.
2. **One relation resolver.** The JOIN and comma paths compute a relation's
   scope name, schema, alias rename and duplicate check separately (the schema
   expression appears 6 times); the schema-qualifier and quoted-name fixes each
   had to be made in several places.
3. **The staged pipeline** unifies the single-row and `many` lowering paths;
   teach Cranelift the multiplicity constructs then.
4. **Explicit Cranelift fallback** (section 3).
5. **A written unification rule** (DuckDB casts an arm's finished result; it
   never re-types the arm's operands), pinned per operator family.

## 6. Workflow friction

| friction | measured | fix |
|---|---|---|
| the gate | 6 min 31 s; **240 s** of it is `_duckdb_arrow_test.py::test_a_lazy_reader_round_trips_through_register` waiting out a known DuckDB hang (120 s timeout, two cases) | time the hang from a marker the child prints after `import duckdb` (~20 s), same protection |
| the gate is serial | 390 s; 74 s with `pytest -n 4` | `pytest-xdist` in the dev group and the gate script |
| probing | ~25 throwaway comparison scripts in one day | `python -m confit.probe`, sharing the parity harness |
| docs upkeep | a behavior change touched 3–6 docs; the corpus count lives in 4 places | one home per fact; a generated refusal inventory |
| appended tests | 2 merge conflicts | topic-sized test files |
| rebuild after a Rust edit | 24 s | fine |

## 7. Proposed order

1. Fix the 4 live defects, each pinned first.
2. Gate speed (hang pin timeout, `-n 4`).
3. Metamorphic suite (and the struct-leaf key spelling it found), parity harness
   and `confit.probe`, nightly deep campaign.
4. Split `frontend.rs`, one relation resolver, explicit Cranelift fallback.
5. Then the subquery design, with `q` ≡ `SELECT * FROM (q)` as the metamorphic
   suite's first new relation.

Not recommended: Hypothesis as a framework; Rust line coverage or mutation
testing for now; dropping the interpreter; rewriting the docs system.
