# TASK-38 — Fold an unquoted struct-column qualifier (`S.x`)

Design for TASK-38. Specs it builds on: TASK-28 (identifier folding), doc-1
(DataFusion function catalogue). DataFusion is the parity oracle (decision-1).

Everything below was measured against the code at `origin/master`, not inferred.
Where a measurement contradicts the ticket's framing, the measurement wins and the
disagreement is called out.

## The bug

```sql
SELECT S.x AS v FROM __THIS__     -- struct column `s`, field `x`
```

DataFusion folds the unquoted qualifier `S` → `s` like any identifier and returns
`7`. Native does not fold it and raises `Unknown column: S`. So TASK-28's
unquoted-folding rule silently does not apply in the struct-qualifier position.

This is TASK-28's AC#5 ceiling, which shipped with an explicit note that a
struct-column qualifier is not folded — "unreachable today … flagged for when
qualified tables become reachable". Struct field access made it reachable. The
flag did its job.

Severity: fails **loudly**. Not a silent value divergence. Workaround is
lowercase-or-quote once you know.

## Findings that changed the ticket

1. **The one-line fix breaks 96 tests.** The ticket framed this as folding the
   qualifier in `expr_build.rs`. Measured:

   ```
   expr_build.rs:44   Some(parts[0].value.clone())  ->  Some(fold_ident(&parts[0]))
   result:            96 failed, 477 passed
   ```

   Our own rewrite emits internally-qualified SQL —
   `SELECT __THIS__.age / __STATE__.avg_age … FROM __THIS__ LEFT JOIN __STATE__` —
   so folding at parse turns `__THIS__` into `__this__` and every windowed
   transform on native dies.

2. **`expr_build` cannot make the fold decision.** A qualifier is either a
   relation alias or a struct column, and that is only known in
   `plan.rs::validate_expr`, where `resolved.get(t)` decides. If it resolves, the
   qualifier names a relation and must **not** fold. If it misses, it is
   reinterpreted as a struct column and **must** fold. So the quoting has to
   survive from parse to that decision point — which is what this design carries.

3. **`unnest_display_name` also needs the fold**, and the ticket does not mention
   it. Measured on the oracle:

   ```
   SELECT unnest(t.s) FROM t   ->  cols ['t.s.x', 't.s.y']
   SELECT unnest(T.s) FROM t   ->  cols ['t.s.x', 't.s.y']     <- qualifier normalised
   ```

   Native renders `{t}.{name}` raw, so `unnest(T.s)` would emit `T.s.x` and
   diverge on output column *names*. Two consumers need the fold decision, not
   one — which is why a bare `table_quoted: bool` was rejected in favour of a type
   that any consumer can ask.

4. **The relation branch is inverted vs the oracle — and is out of scope.**
   Measured (AC#6, discharged):

   | query | oracle | native |
   |---|---|---|
   | `__this__.age` | `[{'v': 7}]` | `Unknown column: __this__` |
   | `"__this__".age` | `[{'v': 7}]` | `Unknown column: __this__` |
   | `"__THIS__".age` | `No field named "__THIS__"` | `[{'v': 7}]` |
   | `__THIS__.age` | `[{'v': 7}]` | `[{'v': 7}]` |

   Root cause: DataFusion **registers** the relation under a folded name
   (`from_arrow(name="__THIS__")` is stored as `__this__`), so quoted-exact misses
   and quoted-lower hits. Native compares raw and gets the opposite answer.

   Unreachable through the public API: `SQLTransform` raises a clean
   `ValueError: Column qualifier '__this__' does not reference…` at `fit()`, and
   the quoted form agrees on both engines. Only a direct `InferFn` call reaches
   it, and it fails loudly. Filed as **DRAFT-22** (Low). **Not** part of this
   ticket — different branch, and fixing it means matching DataFusion's
   register-time relation folding in `plan.rs`, a second core change.

## Design

### Carry the qualifier as a type, not a string

```rust
// src/expr.rs — replaces `table: Option<String>`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QualifiedName {
    pub value: String,   // as written: "S", "__THIS__", "t"
    pub quoted: bool,    // was it double-quoted in the SQL?
}

impl QualifiedName {
    /// DataFusion's rule, identical to `fold_ident`: unquoted folds to
    /// lowercase, quoted stays case-exact.
    pub fn folded(&self) -> String {
        if self.quoted { self.value.clone() } else { self.value.to_lowercase() }
    }
}

Column { table: Option<QualifiedName>, name: String }
```

`expr_build` stores the qualifier **unfolded** and records its quoting. It does
not decide; it preserves. Every consumer then asks for the form it needs.

### Three consumers, two different needs

```rust
// plan.rs:1062  relation branch — RAW. __THIS__ is registered raw on our side;
//               folding here is exactly what produced the 96 failures.
if let Some((real, is_row)) = resolved.get(t.value.as_str()) { … }

// plan.rs:1077  struct-column fallback — FOLD. Only runs once the relation
//               lookup above has MISSED, so `t` names data, not a relation.
base: Expr::Column { table: None, name: t.folded() }

// plan.rs:1017  unnest_display_name — FOLD. Matches the oracle's normalised
//               output column names (finding 3).
Ok(format!("{}.{name}", t.folded()))
```

The invariant: **fold only where the qualifier names data; never where it names a
relation.** That boundary is what keeps this ticket out of DRAFT-22's territory.

### Blast radius

Measured, narrower than the ticket's "12 sites" estimate:

```
15 Expr::Column sites
 ├─  2 construct WITH a qualifier   expr_build.rs:23,43   -> build QualifiedName
 ├─  4 read .table                  plan.rs:412,1015,1059; expr.rs:339 / types.rs:39
 └─  9 use `table: None` or `..`    mechanical, no logic change
```

`expr.rs:339` and `types.rs:39` pass the qualifier to `resolve_column` /
`resolve_column_type`, which look it up against relation schemas — the relation
branch, so they take `.value`, not `.folded()`.

## Testing

| test | asserts | state |
|---|---|---|
| `test_uppercase_qualifier_field_access` | `S.x` folds → 7, both engines | exists, xfail flipped (AC#4) |
| `test_quoted_qualifier_stays_case_exact` | `"S".x` does **not** fold → raises on both | **new** (AC#2) |
| `test_qualified_struct_field_access` | `t.s.x` still works | exists, must stay green (AC#3) |
| `test_unnest_uppercase_qualifier_output_names` | `unnest(T.s)` → cols `t.s.x` | **new** (finding 3) |

The AC#2 test is the load-bearing addition: without it the fix could over-fold —
folding the quoted form too — and still look correct, because nothing currently
pins that half.

Mutation checks: force `folded()` to always fold (the quoted test must fail) and
to never fold (the AC#1 test must fail). Plus the full suite, which is the real
guard on AC#3 — the 96-test signature is unmistakable.

Expected suite: 572 baseline + 2 new tests, one xfail flipped → 574 passed, 11
xfailed.

## AC#5 — the stale ceiling note

`expr_build.rs:38-42` currently reads:

> ponytail: a real CamelCase table or struct-column qualifier won't fold like
> DataFusion here; not reachable today (qualifiers are always library-internal).
> Widen to fold parts[0] too if user-named CamelCase relations ever appear.

Both claims become false when this lands: the struct-column case *is* reachable
and is now fixed, and "widen to fold parts[0]" is precisely the change that breaks
96 tests. It is replaced with what is true — the qualifier is carried unfolded by
design because only `plan.rs` can tell a relation from a struct column — plus a
pointer to DRAFT-22 for the relation half that remains divergent.

## Out of scope

- **DRAFT-22** — relation-qualifier folding inversion (finding 4).
- **DRAFT-21** — `array(...)` native dispatch, parked from TASK-37.
- Changing how relations are registered or resolved. This ticket touches only how
  a qualifier that turned out **not** to be a relation is interpreted.
