# Decimal expressions: DuckDB's rules, read from source

Status: rules verified, implementation not started. Source: DuckDB `v1.5.5`
(commit `d8cdaa33`), the version the oracle runs. Every rule below is fixed-width
integer arithmetic: a `DECIMAL(p,s)` value is an integer `v` meaning `v / 10^s`,
with `1 <= p <= 38`. Nothing is arbitrary precision and nothing is floating
point, so confit can follow it exactly on its `Ty::Dec(p,s)` / `i128` lane.

Why: confit types decimal literals as f64 today (`typing.rs::literal`), which
gives silently wrong values wherever a decimal result becomes an integer or a
boolean:

- `CAST(-2.5 AS BIGINT)` is `-3` on DuckDB, `-2` on confit (nightly seed 1119492).
- `0.1 + 0.2 = 0.3` is `true` on DuckDB, `false` on confit.
- `a * 0.1 = 0.3` with `a = 3` is `true` on DuckDB, `false` on confit.

## 1. Literal typing

`src/parser/transform/expression/transform_constant.cpp`, `T_PGFloat`.

- Any `e`/`E` in the text makes the literal DOUBLE.
- A literal with a `.` becomes `DECIMAL(width, scale)`, where width is the digit
  count (a leading minus excluded, leading zeros included) and scale is the
  digit count after the point. Underscores are not digits. This applies when
  width is at most 38; otherwise the literal is DOUBLE.
  - `2.5` is `DECIMAL(2,1)`; `0.1` is `DECIMAL(2,1)`; `.5` is `DECIMAL(1,1)`;
    `1.` is `DECIMAL(1,0)`.
- Without `.` or `e`: BIGINT if it fits, then HUGEINT, then UHUGEINT, then
  DOUBLE. (Small integers arrive earlier as `T_PGInteger`, typed INTEGER.)

## 2. Integers meeting a decimal

`LogicalType::GetDecimalProperties`, `src/common/types.cpp:729`. BOOLEAN is
`DECIMAL(1,0)`, TINYINT `(3,0)`, SMALLINT `(5,0)`, INTEGER `(10,0)`, BIGINT
`(19,0)`. confit already has this as `typing.rs::int_dec_width`.

## 3. `+` and `-`

`BindDecimalArithmetic`, `src/function/scalar/operator/arithmetic.cpp:193`.

Over the inputs' `(p_i, s_i)`:

- `s = max s_i`
- `p = max(s + max(p_i - s_i), max p_i) + 1`
- If every `p_i <= 18` and `p > 18`: `p = 18`, overflow-checked.
- If `p > 38`: `p = 38`, overflow-checked.

Both inputs are rescaled to `s` (multiplied by a power of ten) before the
operation.

## 4. `*`

`BindDecimalMultiply`, `arithmetic.cpp:836`.

- `s = sum s_i`. If `s > 38`, it is a bind error: "Needed scale %d to
  accurately represent the multiplication result, but this is out of range of
  the DECIMAL type. …"
- `p = sum p_i`.
- If every `p_i <= 18`, `p > 18` and `s < 18`: `p = 18`, overflow-checked.
- If `p > 38`: `p = 38`, overflow-checked.

No rescaling happens: the product of the scaled integers is already at scale `s`.

## 5. `%` and `/`

- `%` uses `BindDecimalArithmetic<IS_MODULO=true>`: the same as §3 without the
  `+1` and without the 18-digit cap.
- `/` has no decimal overload; the result is DOUBLE (measured: `2.5 / 2` is
  DOUBLE).

## 6. Overflow checks

`src/function/scalar/operator/{add,subtract,multiply}.cpp` and
`src/include/duckdb/common/operator/{add,subtract,multiply}.hpp`.

These run only where §3 or §4 set the checked flag. The bound is the storage
type's limit, not the declared width:

| Storage | Bound |
|---|---|
| int16 | 9999 |
| int32 | 999,999,999 |
| int64 | 18 nines |
| int128 | `abs(v) < 10^38` |

The caps are exactly 18 and 38, so this equals the capped width. The error
texts:

- `Overflow in addition of DECIMAL(18) (%d + %d). You might want to add an explicit cast to a bigger decimal.`
  (the subtraction and multiplication texts follow the same pattern)
- `Overflow in addition of DECIMAL(38) (%s + %s);`
- `Overflow in multiplication of DECIMAL(38) (%s * %s). You might want to add an explicit cast to a decimal with a smaller scale.`

Unchecked widths cannot overflow by construction.

## 7. Common type (comparisons, CASE, COALESCE, IN)

`src/common/types.cpp`, `MaxLogicalType` / `DecimalSizeCheck`.

- DECIMAL with DECIMAL: `int = max(p_i - s_i)`, `s = max s_i`, `p = int + s`. If
  `p > 38`, then `p = 38` and `s = 38 - int` (the scale is truncated).
- DECIMAL `(p,s)` with an integer of decimal width `w` (§2): if `w > p - s`,
  the result is `DECIMAL(min(w + s, 38), s)`; otherwise it is `(p,s)`. confit's
  `promote_dec` covers the comparison side of this today.
- DECIMAL with DOUBLE: DOUBLE (`dec_to_float` exists).

## 8. Casts out of and within DECIMAL

- DECIMAL to integer (`src/common/operator/cast_operators.cpp:2570`): round
  half away from zero, `(v + sign(v) * 10^s / 2) / 10^s` with truncating
  division, then a range check against the target width, which errors when
  out of range. DOUBLE to integer is half to even instead; confit already
  matches that.
- DECIMAL to a smaller scale (`DecimalScaleDownOperator`,
  `src/function/cast/decimal_cast.cpp:100`): also half away from zero,
  computed as `((v / (f/2)) ± 1) / 2`, where `f` is the power of ten dropped.
- DECIMAL to a larger scale: multiply by a power of ten, range-checked against
  the target width.
- DECIMAL to DOUBLE: served today (`DecToFloat`, DuckDB's own algorithm).

## What confit has, and the plan

Already present: the `Ty::Dec(p,s)` IR type and `i128` lane, DECIMAL output
columns (`OutCol::Dec`, Arrow `decimal128`), integer-to-decimal promotion for
comparisons, decimal-to-DOUBLE, and static-table decimal columns.

Missing (listed in PLANS as "Decimal expressions"): §1, §3–§6, the full §7, and
§8 apart from DOUBLE. These must land together, because typing literals as
decimals before arithmetic is served would turn today's `c * 0.5` into a
refusal. Suggested order, one PR each:

1. Decimal-typed literals, `+ - * %` with §6's checks, unary minus, and the
   §8 casts.
2. The full §7 common type in CASE, COALESCE and IN, plus `round`, `abs` and
   `sign` over decimals (each needs its own source read).
3. The generator emits decimal arithmetic, and the `decimals` feature leaves
   `UNSHIPPED`.
