# Type unification

How confit gives several operands one type, as DuckDB 1.5.5 does (optimizer off).
Pinned per operator family in `packages/confit/tests/test_arm_widening.py`.

## The rule

**claim: unify-by-cast.** When operands of different numeric types meet, the common
type is chosen first, and then each operand's *finished* result is cast to it. An
operand's own operands are never re-typed. Each computed operand keeps its own width,
and with it its own overflow check.

So `c0 * c0` over TINYINT traps at `c0 = -128` whatever wider operand sits beside it.
This holds in `CASE WHEN c1 > 100 THEN c1 ELSE c0 * c0 END`, in
`coalesce(c0 * c0, c1)` and in `c0 * c0 = c1`. The whole expression is BIGINT, but the
multiplication is still TINYINT.

*Evidence:* the fuzz campaign's 14 seeds of this class (5008 and 13 more; report
`docs/reports/2026-09-26-change-cost-review.md` §4). Each family's pin traps on the
overflowing row and agrees on value and output type for in-range rows, on both backends.

## Choosing the common type

- **Integers** widen along TINYINT < SMALLINT < INTEGER < BIGINT. The wider side wins,
  with one exception. When a syntactic literal of a wider type meets a narrower
  non-literal and the literal's value fits the narrow type, the result is the narrow
  type. So `c8 + 127` is TINYINT, and `c8 + 128` is INTEGER.
  - The families below (CASE, COALESCE, GREATEST and LEAST) apply only the wider-side
    rule.
  - The code is `int_width_promote` in `src/specializer/frontend/typing.rs`.
- **An integer meeting a DOUBLE** is DOUBLE. The integer operand is converted with
  `promote_f64`, which is exact below 2⁵³.
- **A DECIMAL** unifies only with an identical DECIMAL. Anything else is refused by name
  (see [known limitations](../known-limitations.md)).
- **A bare NULL** takes the type of its context. When every arm is a bare NULL, the
  result is SQLNULL, which is INTEGER at the output.

## Where the rule applies

| family | where it binds | how a narrow operand is widened |
|---|---|---|
| CASE (searched and simple) | `Binder::case` (`frontend/expr.rs`) | `widen_int` |
| COALESCE, IFNULL | `frontend/functions.rs` | `widen_int` |
| GREATEST, LEAST | `frontend/functions.rs` | `widen_int` |
| NULLIF | `frontend/functions.rs` | none: it compares at the common type, and the result keeps the first argument's own type (`nullif(1, 1.0)` is INTEGER) |
| IN, BETWEEN | `Binder::unify_family` (`frontend/typing.rs`) | via comparisons |
| comparisons, IS [NOT] DISTINCT FROM | `Binder::cmp` (`frontend/expr.rs`) | the comparison runs at the common width; operands keep theirs |
| arithmetic | `numeric_promote` (`frontend/typing.rs`) | the operation runs at the common width; operands keep theirs |
| a UDF's BIGINT or DOUBLE parameter | `Binder::bind_udf_args` (`frontend/udf.rs`) | `widen_int` / `promote_f64` |

`widen_int` changes the type in place only for a literal or a typed NULL, which carry
no check. Any computed operand is wrapped in a width-only cast. Changing a computed
node's type in place would move its range check to the wider width, which is exactly
the defect this rule exists to prevent.

## Adding a family

A new construct that unifies operand types must use `widen_int` or `promote_f64`, never
assign `.ty` on a bound operand. Add the construct to `ARMS` in
`test_arm_widening.py`, so both of its pins run on both backends.
