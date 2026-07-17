# Codegen Engine (a codegen `InferFn`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a second serving engine, functionally equivalent to the Rust `InferFn` interpreter, that compiles the post-fit rewritten SQL into a cached Python function — proven equivalent by running the existing differential suite against both engines.

**Architecture:** A standalone Python pipeline: sqlglot parse → plan IR → optimize (static joins become lookup joins) → validate/type-infer → emit Python source → `compile()`/`exec()` once → cached function runs per row. Nothing depends on the Rust crate (it may be axed). All value semantics route through a hand-written runtime module, because naive Python diverges from Rust/DataFusion in ~8 verified places.

**Tech Stack:** Python 3.13+, sqlglot 30.12 (already a dep), pydantic v2, pyarrow. No new dependencies.

## The parity target: DataFusion, not Rust

**Measured 2026-07-17: the Rust engine and DataFusion already disagree on cases no test covers.**

| Case | DataFusion — codegen matches this | Rust `InferFn` (bug) |
|---|---|---|
| `CAST(1.0 AS VARCHAR)` | `'1.0'` | `'1'` |
| `ROUND(3)` (int arg) | `3.0` (float) | `3` (int) |
| `NULLIF(1, 1.0)` | `NULL` | `1` |
| `-a`, `-1` | evaluates | **rejected** |
| `a \|\| b` | `'aa'` | **rejected** |

**Decided (AmirHossein, 2026-07-17): codegen matches the DataFusion oracle. Where Rust disagrees, Rust is wrong.** Codegen therefore renders `1.0` as `'1.0'`, returns `3.0` from `ROUND(3)`, and nulls `NULLIF(1, 1.0)`.

`InferFn` remains the spec for everything else — which is all of the covered surface. The honest framing is **"a DataFusion-parity engine built by codegen"**, not "a codegen `InferFn`".

**Two distinct kinds of Rust defect here, handled differently:**
- **Value divergences** (rows 1–3): both engines accept the query and return *different values*. Codegen matches the oracle and PASSES; rust is `xfail`-ed. → Task 11.
- **Missing surface** (rows 4–5, unary minus and `||`): Rust *rejects* what DataFusion evaluates. "Match DataFusion" does **not** mean "implement everything DataFusion has" — that is unbounded. The committed surface stays `InferFn`'s, so **codegen defers these too** and they are ticketed as Rust gaps, not built here. Adding them is a follow-up, not scope creep into this plan.

All five are **bugs in the Rust engine** regardless: the README promises `transform` and `infer` return identical values, so each is a live defect where a user gets a different answer at serving time than at batch time.

**Standing process when a Rust-engine bug turns up (AmirHossein, 2026-07-17):**
1. Add a differential test, `xfail` **on the rust backend only**, `strict=True`, describing the divergence in the reason. Codegen must PASS it.
2. Tell the PM to open a BACKLOG ticket. **Do not fix the Rust engine inline** — separate concern.

Task 11 does exactly this for all five.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-17-codegen-inferfn-design.md`. Its "Appendix: verified findings" is measured, not reasoned — trust it over intuition, and re-measure rather than assume.
- **Parity target: DataFusion.** When in doubt, *run both engines* (`tests/differential.py` exposes `_run_datafusion` and `_run_infer`) rather than reading either implementation and inferring. If they disagree, the oracle wins, and Rust gets an xfail + a PM ticket.
- **Never trust a green bar for a new backend.** This wiring already produced one fake pass: correct IDs, doubled count, everything green, codegen never executed. `tests/test_backend_wiring.py` (Task 8) is the guard; do not weaken or skip it.
- **Error-message parity is a NON-GOAL** (per `docs/BACKLOG.md`). Match error *values* and the raised *type* where the harness asserts it (`ValueError` on int div-by-zero and bad casts; `KeyError` on an inner lookup-join miss). Do not match message text.
- **Never use `isinstance` for value type tests** — `bool` is an `int` subclass in Python and Rust treats them as distinct variants. Always `type(v) is int`.
- **Committed surface:** SELECT projections, WHERE, INNER JOIN ... ON, CROSS JOIN, LEFT/inner static `LookupJoin`, table aliases, and the expression set (`Column`, `Literal`, `BinaryOp`, `Not`, `Cast`, and builtins `upper`/`lower`/`trim`/`substr`/`substring`/`concat`/`abs`/`round`/`coalesce`/`nullif`).
- **Deferred surface:** struct/list/`UNNEST`/`FieldAccess`, vectorized/columnar path, numpy output, `CASE WHEN`. These MUST raise `UnsupportedInCodegen` at build time — never silently produce a wrong answer.
- **Out of scope:** wiring engine selection into `SQLTransform` or changing any default. This plan ends at "second engine, proven equivalent". Selection/defaults are gated on framing calls that are not yet made (see spec, "Open questions").
- **File convention:** flat modules under `sql_transform/`, unit tests co-located as `<module>_test.py` (matches `_batch.py`/`_batch_test.py`).
- **Commands:** `uv run pytest <path> -v` for a file; `mise run test` for everything; `mise run fmt` before committing.

## File Structure

| File | Responsibility |
|---|---|
| `sql_transform/_codegen_runtime.py` (create) | Value semantics: NULL propagation, arithmetic, comparison, Kleene logic, casts, builtins, type tags, display. Pure functions, no SQL knowledge. This is where every Python/Rust divergence is contained. |
| `sql_transform/_codegen_runtime_test.py` (create) | Unit tests for the above — one per divergence. |
| `sql_transform/_codegen_plan.py` (create) | Front-end: type system (`FieldType`/`Base`), schema extraction (pydantic/arrow), IR dataclasses, sqlglot→`Plan`, `optimize`, `validate_columns`, `infer_type`, `compatible`. |
| `sql_transform/_codegen_plan_test.py` (create) | Unit tests for parse/optimize/validate/type-inference. |
| `sql_transform/_codegen.py` (create) | Emitter (`Plan`→Python source) + `CodegenFn` engine class + `UnsupportedInCodegen`. |
| `sql_transform/_codegen_test.py` (create) | Unit tests for the engine end-to-end. |
| `tests/differential.py` (modify) | Add codegen as a selectable backend alongside rust. |
| `tests/conftest.py` (create) | Autouse fixture parametrizing every harness test over both backends, plus `rust_bug` for xfail-on-rust. |
| `tests/test_backend_wiring.py` (create) | Proves the backend fixture actually switches the engine — the guard against a fake green. |
| `tests/test_codegen_coverage.py` (create) | Pins codegen's committed vs deferred surface. |
| `tests/test_diff_rust_bugs.py` (create) | The three Rust/DataFusion value divergences, xfail-on-rust (strict). |

---

### Task 1: Runtime — values, type tags, display, arithmetic

**Files:**
- Create: `sql_transform/_codegen_runtime.py`
- Test: `sql_transform/_codegen_runtime_test.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces: `type_name(v) -> str`, `display(v) -> str`, `tag(v) -> int`, `key(v) -> tuple`, `val_eq(a, b) -> bool`, `truthy(v) -> bool`, `add/sub/mul/div/mod(l, r) -> Any`. All take/return native Python values; `None` is SQL NULL.

- [ ] **Step 1: Write the failing tests**

Create `sql_transform/_codegen_runtime_test.py`:

```python
"""Unit tests for the codegen semantics runtime.

Each test here pins a place where naive Python diverges from the Rust engine
(src/expr.rs). See docs/superpowers/specs/2026-07-17-codegen-inferfn-design.md.
"""

import math

import pytest

from sql_transform import _codegen_runtime as rt


def test_int_division_truncates_toward_zero():
    # Python's -7 // 2 == -4 (floor); Rust's -7 / 2 == -3 (truncate).
    assert rt.div(-7, 2) == -3
    assert rt.div(7, 2) == 3
    assert rt.div(-7, -2) == 3


def test_int_mod_takes_sign_of_dividend():
    # Python's -7 % 2 == 1 (sign of divisor); Rust's == -1 (sign of dividend).
    assert rt.mod(-7, 2) == -1
    assert rt.mod(7, -2) == 1


def test_int_div_and_mod_by_zero_raise():
    with pytest.raises(ValueError, match="division by zero"):
        rt.div(1, 0)
    with pytest.raises(ValueError, match="division by zero"):
        rt.mod(1, 0)


def test_float_division_by_zero_is_ieee_not_an_error():
    # Rust yields inf/nan; Python's / would raise ZeroDivisionError.
    assert rt.div(1.0, 0.0) == math.inf
    assert rt.div(-1.0, 0.0) == -math.inf
    assert math.isnan(rt.div(0.0, 0.0))


def test_float_mod_is_c_style():
    assert rt.mod(-7.0, 2.0) == math.fmod(-7.0, 2.0)
    assert math.isnan(rt.mod(1.0, 0.0))


def test_arithmetic_propagates_null():
    assert rt.add(None, 1) is None
    assert rt.add(1, None) is None
    assert rt.mul(None, None) is None


def test_int_int_stays_int_mixed_promotes_to_float():
    assert rt.add(1, 2) == 3
    assert type(rt.add(1, 2)) is int
    assert type(rt.add(1, 2.0)) is float


def test_bool_is_not_a_number():
    # Python would give True + 1 == 2; Rust's as_f64 errors on a bool.
    with pytest.raises(ValueError, match="arithmetic"):
        rt.add(True, 1)


def test_string_is_not_a_number():
    with pytest.raises(ValueError, match="arithmetic"):
        rt.add("a", 1)


def test_tags_keep_int_float_bool_distinct():
    # Value's Eq/Hash are variant-tagged: Int(1) != Float(1.0), True is not 1.
    assert rt.tag(1) != rt.tag(1.0)
    assert rt.tag(True) != rt.tag(1)
    assert rt.key(1) != rt.key(1.0)
    assert not rt.val_eq(1, 1.0)
    assert not rt.val_eq(True, 1)
    assert rt.val_eq(1, 1)
    assert rt.val_eq(None, None)


def test_truthy_is_strict_bool_true():
    # RelNode::Filter keeps a row only on Value::Bool(true).
    assert rt.truthy(True)
    assert not rt.truthy(False)
    assert not rt.truthy(None)
    assert not rt.truthy(1)
    assert not rt.truthy("yes")


@pytest.mark.parametrize(
    "value, expected",
    [
        # Every expectation below was MEASURED against DataFusion, not reasoned.
        (1.0, "1.0"),  # Rust gives "1" here -- a Rust bug, xfail-ed in Task 11
        (1.5, "1.5"),
        (0.1, "0.1"),
        (1e300, "1e300"),  # Python str() gives "1e+300"; Rust gives 300 digits
        (math.nan, "NaN"),  # Python str() gives "nan"
        (math.inf, "inf"),
        (-math.inf, "-inf"),
        (None, ""),
        (True, "true"),
        (False, "false"),
        (42, "42"),
        ("hi", "hi"),
    ],
)
def test_display_matches_datafusion(value, expected):
    assert rt.display(value) == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/_codegen_runtime_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sql_transform._codegen_runtime'`

- [ ] **Step 3: Write the implementation**

Create `sql_transform/_codegen_runtime.py`:

```python
"""Value semantics for the codegen engine — the DataFusion-parity helpers that
generated code calls.

Mirrors src/expr.rs. Generated code never uses bare Python operators, because
Python diverges from the Rust engine in ways that are silent rather than loud:
integer division floors instead of truncating, `%` takes the divisor's sign,
`bool` is an `int` subclass, float division by zero raises instead of yielding
IEEE inf/nan, and `1 == 1.0`. Every such case is contained in this module.

NULL is Python None throughout. Type tests are `type(v) is X` on purpose --
`isinstance` would let `True` pass as an int, which Rust's Value never does.
"""

from __future__ import annotations

import math
from typing import Any


def type_name(v: Any) -> str:
    """Human-readable type name, matching expr::type_name for error messages."""
    if v is None:
        return "null"
    t = type(v)
    if t is bool:
        return "bool"
    if t is int:
        return "int"
    if t is float:
        return "float"
    if t is str:
        return "string"
    return "object"


def _fmt_float(f: float) -> str:
    """Render a float the way DataFusion does (NOT the way Rust does).

    Measured 2026-07-17 -- all three engines disagree here, and plain str() is
    wrong twice:
                      DataFusion    Rust        Python str()
        1.0           "1.0"         "1"         "1.0"   <- str ok
        0.1           "0.1"         "0.1"       "0.1"   <- str ok
        NaN           "NaN"         "NaN"       "nan"   <- str WRONG
        inf/-inf      "inf"/"-inf"  same        same    <- str ok
        1e300         "1e300"       "1000...0"  "1e+300" <- str WRONG

    Rust's "1" and its 300-digit 1e300 are both bugs (xfail-ed + ticketed).
    """
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    # Python writes exponents as "1e+300"; DataFusion omits the "+".
    return str(f).replace("e+", "e")


def display(v: Any) -> str:
    """String form used by CONCAT and CAST(.. AS VARCHAR).

    Matches DataFusion, NOT Rust's expr::display_value -- see _fmt_float.
    """
    if v is None:
        return ""
    t = type(v)
    if t is bool:
        return "true" if v else "false"
    if t is str:
        return v
    if t is int:
        return str(v)
    if t is float:
        return _fmt_float(v)
    return "<object>"


def tag(v: Any) -> int:
    """Variant tag mirroring Value's Eq/Hash: Int(1) != Float(1.0), True != 1."""
    if v is None:
        return 4
    t = type(v)
    if t is bool:
        return 3
    if t is int:
        return 0
    if t is float:
        return 1
    if t is str:
        return 2
    return 5


def key(v: Any) -> tuple:
    """Hashable, type-tagged key component for lookup-join indexes."""
    return (tag(v), v)


def val_eq(a: Any, b: Any) -> bool:
    """Value-level equality: type-strict, like Value's PartialEq."""
    return tag(a) == tag(b) and a == b


def truthy(v: Any) -> bool:
    """RelNode::Filter keeps a row only when the predicate is Value::Bool(true)."""
    return v is True


def as_f(v: Any) -> float:
    t = type(v)
    if t is int:
        return float(v)
    if t is float:
        return v
    raise ValueError(f"Cannot use a {type_name(v)} value in an arithmetic expression")


def _trunc_div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def _float_div(a: float, b: float) -> float:
    if b == 0.0:
        if a == 0.0 or math.isnan(a):
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


def add(l: Any, r: Any) -> Any:
    if l is None or r is None:
        return None
    if type(l) is int and type(r) is int:
        return l + r
    return as_f(l) + as_f(r)


def sub(l: Any, r: Any) -> Any:
    if l is None or r is None:
        return None
    if type(l) is int and type(r) is int:
        return l - r
    return as_f(l) - as_f(r)


def mul(l: Any, r: Any) -> Any:
    if l is None or r is None:
        return None
    if type(l) is int and type(r) is int:
        return l * r
    return as_f(l) * as_f(r)


def div(l: Any, r: Any) -> Any:
    if l is None or r is None:
        return None
    if type(l) is int and type(r) is int:
        if r == 0:
            raise ValueError("division by zero")
        return _trunc_div(l, r)
    return _float_div(as_f(l), as_f(r))


def mod(l: Any, r: Any) -> Any:
    if l is None or r is None:
        return None
    if type(l) is int and type(r) is int:
        if r == 0:
            raise ValueError("division by zero")
        return l - r * _trunc_div(l, r)
    a, b = as_f(l), as_f(r)
    if b == 0.0:
        return math.nan
    return math.fmod(a, b)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/_codegen_runtime_test.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
mise run fmt
git add sql_transform/_codegen_runtime.py sql_transform/_codegen_runtime_test.py
git commit -m "feat: codegen runtime — values, tags, display, arithmetic"
```

---

### Task 2: Runtime — comparison and three-valued logic

**Files:**
- Modify: `sql_transform/_codegen_runtime.py` (append)
- Test: `sql_transform/_codegen_runtime_test.py` (append)

**Interfaces:**
- Consumes: `as_f`, `type_name`, `tag` from Task 1.
- Produces: `eq/neq/lt/gt/lte/gte(l, r) -> bool | None`, `and_(l, r)`, `or_(l, r)`, `not_(v)`, `join_eq(a, b) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `sql_transform/_codegen_runtime_test.py`:

```python
def test_comparison_propagates_null():
    assert rt.eq(None, 1) is None
    assert rt.lt(1, None) is None


def test_comparison_across_int_and_float():
    assert rt.eq(1, 1.0) is True
    assert rt.lt(1, 1.5) is True
    assert rt.gte(2.0, 2) is True


def test_comparison_of_strings_and_bools():
    assert rt.lt("a", "b") is True
    assert rt.eq("a", "a") is True
    assert rt.lt(False, True) is True


def test_comparison_of_mismatched_types_errors():
    # Rust falls through to as_f64, which errors on a string.
    with pytest.raises(ValueError, match="arithmetic"):
        rt.eq("a", 1)


def test_comparison_of_nan_errors():
    with pytest.raises(ValueError, match="NaN"):
        rt.lt(math.nan, 1.0)


def test_kleene_and():
    assert rt.and_(True, True) is True
    assert rt.and_(True, False) is False
    assert rt.and_(False, None) is False  # false dominates
    assert rt.and_(True, None) is None
    assert rt.and_(None, None) is None


def test_kleene_or():
    assert rt.or_(False, False) is False
    assert rt.or_(True, None) is True  # true dominates
    assert rt.or_(False, None) is None
    assert rt.or_(None, None) is None


def test_not_is_tri_valued():
    assert rt.not_(True) is False
    assert rt.not_(False) is True
    assert rt.not_(None) is None


def test_logic_rejects_non_bool_even_when_other_side_decides():
    # Rust's logic() calls as_tribool on BOTH operands before matching, so this
    # errors rather than short-circuiting to False.
    with pytest.raises(ValueError, match="boolean"):
        rt.and_(False, 1)
    with pytest.raises(ValueError, match="boolean"):
        rt.or_(True, "x")


def test_join_eq_is_type_strict_and_null_never_matches():
    assert rt.join_eq(1, 1) is True
    assert rt.join_eq(1, 1.0) is False  # Value::Int(1) != Value::Float(1.0)
    assert rt.join_eq(None, None) is False  # NULL keys never match
    assert rt.join_eq(None, 1) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/_codegen_runtime_test.py -v -k "comparison or kleene or not_is or logic or join_eq"`
Expected: FAIL with `AttributeError: module 'sql_transform._codegen_runtime' has no attribute 'eq'`

- [ ] **Step 3: Write the implementation**

Append to `sql_transform/_codegen_runtime.py`:

```python
def _cmp(l: Any, r: Any) -> int:
    """Ordering mirroring expr::compare_values: same-type int/str/bool compare
    directly, everything else goes through as_f (which errors on non-numbers)."""
    tl, tr = type(l), type(r)
    if (tl is int and tr is int) or (tl is str and tr is str) or (tl is bool and tr is bool):
        return -1 if l < r else (0 if l == r else 1)
    a, b = as_f(l), as_f(r)
    if math.isnan(a) or math.isnan(b):
        raise ValueError("Cannot compare NaN")
    return -1 if a < b else (0 if a == b else 1)


def eq(l: Any, r: Any) -> Any:
    return None if l is None or r is None else _cmp(l, r) == 0


def neq(l: Any, r: Any) -> Any:
    return None if l is None or r is None else _cmp(l, r) != 0


def lt(l: Any, r: Any) -> Any:
    return None if l is None or r is None else _cmp(l, r) < 0


def gt(l: Any, r: Any) -> Any:
    return None if l is None or r is None else _cmp(l, r) > 0


def lte(l: Any, r: Any) -> Any:
    return None if l is None or r is None else _cmp(l, r) <= 0


def gte(l: Any, r: Any) -> Any:
    return None if l is None or r is None else _cmp(l, r) >= 0


def _tribool(v: Any) -> bool | None:
    if v is True:
        return True
    if v is False:
        return False
    if v is None:
        return None
    raise ValueError(f"Expected a boolean expression, got a {type_name(v)} value")


def and_(l: Any, r: Any) -> Any:
    # Both operands are converted before matching (expr::logic), so a non-bool
    # errors even when the other operand would decide the result. No short-circuit.
    lb, rb = _tribool(l), _tribool(r)
    if lb is False or rb is False:
        return False
    if lb is True and rb is True:
        return True
    return None


def or_(l: Any, r: Any) -> Any:
    lb, rb = _tribool(l), _tribool(r)
    if lb is True or rb is True:
        return True
    if lb is False and rb is False:
        return False
    return None


def not_(v: Any) -> Any:
    b = _tribool(v)
    return None if b is None else (not b)


def join_eq(a: Any, b: Any) -> bool:
    """JOIN ON equality: a NULL on either side never matches, and equality is
    type-strict (RelNode::Join compares Values, not Python numbers)."""
    if a is None or b is None:
        return False
    return val_eq(a, b)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/_codegen_runtime_test.py -v`
Expected: PASS (all tests, including Task 1's)

- [ ] **Step 5: Commit**

```bash
mise run fmt
git add sql_transform/_codegen_runtime.py sql_transform/_codegen_runtime_test.py
git commit -m "feat: codegen runtime — comparison and Kleene logic"
```

---

### Task 3: Runtime — casts and builtins

**Files:**
- Modify: `sql_transform/_codegen_runtime.py` (append)
- Test: `sql_transform/_codegen_runtime_test.py` (append)

**Interfaces:**
- Consumes: `display`, `type_name` from Task 1; `eq` from Task 2 (`nullif` coerces numerically, so it uses `eq`, NOT `val_eq`).
- Produces: `as_s`, `as_i`, `cast_str/cast_int/cast_float/cast_bool(v)`, `upper/lower/trim/substr/concat/abs_/round_/coalesce/nullif(*args)`, `miss(table, key) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `sql_transform/_codegen_runtime_test.py`:

```python
def test_casts_propagate_null():
    assert rt.cast_int(None) is None
    assert rt.cast_str(None) is None


def test_cast_int_truncates_floats():
    assert rt.cast_int(1.9) == 1
    assert rt.cast_int(-1.9) == -1


def test_cast_int_parses_strings_and_errors_on_junk():
    assert rt.cast_int(" 42 ") == 42
    with pytest.raises(ValueError, match="Cannot cast"):
        rt.cast_int("abc")


def test_cast_bool_from_string_is_case_insensitive_true():
    assert rt.cast_bool("TRUE") is True
    assert rt.cast_bool("yes") is False


def test_cast_float_and_bool_from_numbers():
    assert rt.cast_float(1) == 1.0
    assert rt.cast_float(True) == 1.0
    assert rt.cast_bool(0) is False
    assert rt.cast_bool(2) is True


def test_cast_str_uses_datafusion_float_formatting():
    # DataFusion renders 1.0 as "1.0" (via display); Rust gives "1" -- a Rust bug.
    assert rt.cast_str(1.0) == "1.0"


def test_round_is_half_away_from_zero_not_bankers():
    # Python's round(0.5) == 0 and round(2.5) == 2 (banker's); Rust rounds away.
    assert rt.round_(0.5) == 1.0
    assert rt.round_(2.5) == 3.0
    assert rt.round_(-2.5) == -3.0
    assert type(rt.round_(2.5)) is float  # must stay a float, not become an int


def test_round_of_an_int_returns_a_float():
    # Measured: DataFusion's ROUND(3) is 3.0. Rust returns int 3 -- a Rust bug.
    assert rt.round_(3) == 3.0
    assert type(rt.round_(3)) is float


def test_abs_preserves_its_argument_type():
    assert rt.abs_(-3) == 3
    assert type(rt.abs_(-3)) is int
    assert rt.abs_(-3.5) == 3.5


def test_abs_and_round_reject_non_numbers():
    with pytest.raises(ValueError, match="ABS"):
        rt.abs_("x")
    with pytest.raises(ValueError, match="ROUND"):
        rt.round_("x")


def test_string_builtins_propagate_null():
    assert rt.upper(None) is None
    assert rt.trim(None) is None
    assert rt.substr(None, 1) is None
    assert rt.substr("abc", None) is None


def test_string_builtins():
    assert rt.upper("aB") == "AB"
    assert rt.lower("aB") == "ab"
    assert rt.trim("  x  ") == "x"


def test_substr_is_one_indexed_and_clamps():
    assert rt.substr("hello", 2) == "ello"
    assert rt.substr("hello", 2, 3) == "ell"
    assert rt.substr("hello", 0) == "hello"  # start <= 0 clamps to the beginning
    assert rt.substr("hello", 2, 99) == "ello"  # length clamps to the end
    assert rt.substr("hello", 99) == ""


def test_concat_skips_nulls_and_never_returns_null():
    assert rt.concat("a", None, "b") == "ab"
    assert rt.concat(None) == ""
    assert rt.concat("n=", 1) == "n=1"


def test_coalesce_returns_first_non_null():
    assert rt.coalesce(None, None, 3) == 3
    assert rt.coalesce(None, None) is None


def test_nullif_compares_numerically_across_int_and_float():
    assert rt.nullif(1, 1) is None
    assert rt.nullif(1, 2) == 1
    # Measured: DataFusion nulls this. Rust returns 1 (variant-tagged eq) -- a bug.
    assert rt.nullif(1, 1.0) is None
    assert rt.nullif(None, None) is None
    assert rt.nullif(None, 1) is None  # a[0] is NULL, so the result is NULL
    assert rt.nullif("a", "b") == "a"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/_codegen_runtime_test.py -v -k "cast or round or abs or string_builtins or substr or concat or coalesce or nullif"`
Expected: FAIL with `AttributeError: module 'sql_transform._codegen_runtime' has no attribute 'cast_int'`

- [ ] **Step 3: Write the implementation**

Append to `sql_transform/_codegen_runtime.py`:

```python
def as_s(v: Any) -> str:
    if type(v) is str:
        return v
    raise ValueError(f"Expected a string argument, got {type_name(v)}")


def as_i(v: Any) -> int:
    if type(v) is int:
        return v
    raise ValueError(f"Expected an integer argument, got {type_name(v)}")


def cast_str(v: Any) -> Any:
    return None if v is None else display(v)


def cast_int(v: Any) -> Any:
    if v is None:
        return None
    t = type(v)
    if t is bool:
        return 1 if v else 0
    if t is int:
        return v
    if t is float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Cannot cast this value to INT")
        return int(math.trunc(v))
    if t is str:
        try:
            return int(v.strip())
        except ValueError:
            raise ValueError(f"Cannot cast '{v}' to INT") from None
    raise ValueError("Cannot cast this value to INT")


def cast_float(v: Any) -> Any:
    if v is None:
        return None
    t = type(v)
    if t is bool:
        return 1.0 if v else 0.0
    if t is int:
        return float(v)
    if t is float:
        return v
    if t is str:
        try:
            return float(v.strip())
        except ValueError:
            raise ValueError(f"Cannot cast '{v}' to FLOAT") from None
    raise ValueError("Cannot cast this value to FLOAT")


def cast_bool(v: Any) -> Any:
    if v is None:
        return None
    t = type(v)
    if t is bool:
        return v
    if t is int:
        return v != 0
    if t is float:
        return v != 0.0
    if t is str:
        return v.lower() == "true"
    raise ValueError("Cannot cast this value to BOOLEAN")


def upper(*a: Any) -> Any:
    return None if any(x is None for x in a) else as_s(a[0]).upper()


def lower(*a: Any) -> Any:
    return None if any(x is None for x in a) else as_s(a[0]).lower()


def trim(*a: Any) -> Any:
    return None if any(x is None for x in a) else as_s(a[0]).strip()


def substr(*a: Any) -> Any:
    if any(x is None for x in a):
        return None
    s = as_s(a[0])
    start = as_i(a[1])
    length = as_i(a[2]) if len(a) > 2 else None
    idx = min((start - 1) if start > 0 else 0, len(s))
    end = min(idx + max(length, 0), len(s)) if length is not None else len(s)
    return s[idx:end]


def concat(*a: Any) -> str:
    return "".join(display(x) for x in a if x is not None)


def abs_(*a: Any) -> Any:
    v = a[0]
    if v is None:
        return None
    t = type(v)
    if t is int or t is float:
        return abs(v)
    raise ValueError(f"ABS expects a number, got a {type_name(v)} value")


def round_(*a: Any) -> Any:
    """ROUND always returns a float, including for an int argument.

    Measured: DataFusion's ROUND(3) is 3.0; Rust returns 3 (int) -- a Rust bug
    (xfail-ed + ticketed, Task 11). Rounding is half-away-from-zero, which
    DataFusion and Rust agree on and Python's banker's round() does NOT.
    """
    v = a[0]
    if v is None:
        return None
    t = type(v)
    if t is int:
        return float(v)
    if t is float:
        if math.isnan(v) or math.isinf(v):
            return v
        # math.floor/ceil return ints, so re-wrap to keep this a float.
        return float(math.floor(v + 0.5)) if v >= 0 else float(math.ceil(v - 0.5))
    raise ValueError(f"ROUND expects a number, got a {type_name(v)} value")


def coalesce(*a: Any) -> Any:
    for v in a:
        if v is not None:
            return v
    return None


def nullif(*a: Any) -> Any:
    """NULLIF compares numerically across int/float, not by variant.

    Measured: DataFusion's NULLIF(1, 1.0) is NULL; Rust returns 1 because its
    Value equality is variant-tagged -- a Rust bug (xfail-ed + ticketed).
    Deliberately NOT val_eq: lookup-join keys stay type-strict, NULLIF coerces.
    `eq` returns None when either side is NULL, and NULLIF(NULL, NULL) must be
    NULL -- which falls out, since a[0] is then NULL anyway.
    """
    if len(a) != 2:
        raise ValueError("NULLIF expects 2 arguments")
    return None if eq(a[0], a[1]) is True else a[0]


def miss(table: str, k: tuple) -> str:
    """Message for an inner lookup-join miss (plan.rs InterpError::MissingKey)."""
    rendered = ", ".join(display(part[1]) for part in k)
    return f"No row in static table '{table}' matches key ({rendered})"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/_codegen_runtime_test.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
mise run fmt
git add sql_transform/_codegen_runtime.py sql_transform/_codegen_runtime_test.py
git commit -m "feat: codegen runtime — casts and builtins"
```

---

### Task 4: Type system and schema extraction

**Files:**
- Create: `sql_transform/_codegen_plan.py`
- Test: `sql_transform/_codegen_plan_test.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `INT`/`FLOAT`/`STR`/`BOOL`/`OTHER` base constants, `StructBase(fields)`, `ListBase(elem)`, `FieldType(base, nullable)`, `is_container(base) -> bool`, `schema_from_pydantic(model) -> dict[str, FieldType]`, `schema_from_arrow(table) -> dict[str, FieldType]`, `field_type_to_python(ft) -> Any`, `compatible(inferred, declared) -> bool`.

- [ ] **Step 1: Write the failing tests**

Create `sql_transform/_codegen_plan_test.py`:

```python
"""Unit tests for the codegen front-end (types, parse, optimize, validate)."""

import typing

import pyarrow as pa
import pytest
from pydantic import BaseModel

from sql_transform import _codegen_plan as cp


def test_schema_from_pydantic_reads_types_and_nullability():
    class Row(BaseModel):
        a: int
        b: float | None
        c: str
        d: bool

    schema = cp.schema_from_pydantic(Row)
    assert schema["a"] == cp.FieldType(cp.INT, False)
    assert schema["b"] == cp.FieldType(cp.FLOAT, True)
    assert schema["c"] == cp.FieldType(cp.STR, False)
    assert schema["d"] == cp.FieldType(cp.BOOL, False)


def test_schema_from_pydantic_optional_and_any():
    class Row(BaseModel):
        a: typing.Optional[int]
        b: typing.Any

    schema = cp.schema_from_pydantic(Row)
    assert schema["a"] == cp.FieldType(cp.INT, True)
    assert schema["b"].base == cp.OTHER


def test_schema_from_pydantic_nested_model_is_a_struct():
    class Inner(BaseModel):
        x: int

    class Row(BaseModel):
        s: Inner

    schema = cp.schema_from_pydantic(Row)
    assert schema["s"].base == cp.StructBase((("x", cp.FieldType(cp.INT, False)),))
    assert cp.is_container(schema["s"].base)


def test_schema_from_arrow_reads_types_and_nullability():
    table = pa.table(
        {"a": pa.array([1], type=pa.int64()), "b": pa.array([1.0], type=pa.float64())},
        schema=pa.schema(
            [pa.field("a", pa.int64(), nullable=False), pa.field("b", pa.float64())]
        ),
    )
    schema = cp.schema_from_arrow(table)
    assert schema["a"] == cp.FieldType(cp.INT, False)
    assert schema["b"] == cp.FieldType(cp.FLOAT, True)


def test_field_type_to_python_round_trips():
    assert cp.field_type_to_python(cp.FieldType(cp.INT, False)) is int
    assert cp.field_type_to_python(cp.FieldType(cp.INT, True)) == typing.Optional[int]
    assert cp.field_type_to_python(cp.FieldType(cp.OTHER, False)) is typing.Any


def test_compatible_allows_int_into_float_and_unknown_into_anything():
    assert cp.compatible(cp.INT, cp.FLOAT)
    assert cp.compatible(cp.OTHER, cp.STR)
    assert cp.compatible(cp.INT, cp.INT)
    assert not cp.compatible(cp.STR, cp.INT)
    assert not cp.compatible(cp.FLOAT, cp.INT)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/_codegen_plan_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sql_transform._codegen_plan'`

- [ ] **Step 3: Write the implementation**

Create `sql_transform/_codegen_plan.py`:

```python
"""Front-end for the codegen engine: type system, schema extraction, plan IR,
sqlglot parsing, optimization and validation.

Mirrors src/types.rs, src/schema.rs, src/expr_build.rs and src/plan.rs, but
standalone on sqlglot -- the codegen engine must not depend on the Rust crate.
"""

from __future__ import annotations

import types as pytypes
import typing
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, create_model

INT = "int"
FLOAT = "float"
STR = "str"
BOOL = "bool"
OTHER = "other"  # unresolvable: passthrough column, union, unsupported generic


@dataclass(frozen=True)
class StructBase:
    fields: tuple  # tuple[tuple[str, FieldType], ...]; order is significant


@dataclass(frozen=True)
class ListBase:
    elem: Any  # FieldType


@dataclass(frozen=True)
class FieldType:
    base: Any
    nullable: bool


def is_container(base: Any) -> bool:
    return isinstance(base, (StructBase, ListBase))


def schema_from_pydantic(model: type[BaseModel]) -> dict:
    return dict(_pydantic_fields_ordered(model))


def _pydantic_fields_ordered(model: type[BaseModel]) -> list:
    fields = getattr(model, "model_fields", None)
    if fields is None:
        raise ValueError("Not a Pydantic v2 model class")
    return [(name, _annotation_to_field_type(f.annotation)) for name, f in fields.items()]


def _annotation_to_field_type(annotation: Any) -> FieldType:
    origin = typing.get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return FieldType(StructBase(tuple(_pydantic_fields_ordered(annotation))), False)
        return FieldType(_python_type_to_base(annotation), False)
    if origin is typing.Union or origin is pytypes.UnionType:
        args = typing.get_args(annotation)
        nullable = any(a is type(None) for a in args)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            inner = _annotation_to_field_type(non_none[0])
            return FieldType(inner.base, nullable or inner.nullable)
        return FieldType(OTHER, nullable)
    if origin is list:
        args = typing.get_args(annotation)
        if len(args) == 1:
            return FieldType(ListBase(_annotation_to_field_type(args[0])), False)
    return FieldType(OTHER, False)


def _python_type_to_base(t: Any) -> Any:
    return {int: INT, float: FLOAT, str: STR, bool: BOOL}.get(t, OTHER)


def schema_from_arrow(table: Any) -> dict:
    return {f.name: _arrow_field_to_field_type(f) for f in table.schema}


def _arrow_field_to_field_type(field: Any) -> FieldType:
    return FieldType(_arrow_type_to_base(field.type), field.nullable)


def _arrow_type_to_base(t: Any) -> Any:
    if pa.types.is_struct(t):
        return StructBase(
            tuple(
                (t.field(i).name, _arrow_field_to_field_type(t.field(i)))
                for i in range(t.num_fields)
            )
        )
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return ListBase(_arrow_field_to_field_type(t.value_field))
    if pa.types.is_integer(t):
        return INT
    if pa.types.is_floating(t) or pa.types.is_decimal(t):
        return FLOAT
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return STR
    if pa.types.is_boolean(t):
        return BOOL
    return OTHER


def field_type_to_python(ft: FieldType) -> Any:
    base = ft.base
    if isinstance(base, StructBase):
        inner = {n: (field_type_to_python(f), ...) for n, f in base.fields}
        py: Any = create_model("StructModel", **inner)
    elif isinstance(base, ListBase):
        py = list[field_type_to_python(base.elem)]
    else:
        py = {INT: int, FLOAT: float, STR: str, BOOL: bool}.get(base, typing.Any)
    return py if not ft.nullable else typing.Optional[py]


def compatible(inferred: Any, declared: Any) -> bool:
    """Is `inferred` provably safe to store in a field declared as `declared`?
    Only rejects what can be proven wrong; Pydantic validates the rest at call
    time (mirrors types::compatible)."""
    if inferred == declared:
        return True
    if inferred == INT and declared == FLOAT:
        return True
    if inferred == OTHER:
        return True
    if isinstance(inferred, StructBase) and isinstance(declared, StructBase):
        d = dict(declared.fields)
        return len(inferred.fields) == len(declared.fields) and all(
            n in d and compatible(ft.base, d[n].base) for n, ft in inferred.fields
        )
    if isinstance(inferred, ListBase) and isinstance(declared, ListBase):
        return compatible(inferred.elem.base, declared.elem.base)
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/_codegen_plan_test.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
mise run fmt
git add sql_transform/_codegen_plan.py sql_transform/_codegen_plan_test.py
git commit -m "feat: codegen front-end — type system and schema extraction"
```

---

### Task 5: Plan IR and sqlglot parsing

**Files:**
- Modify: `sql_transform/_codegen_plan.py` (append)
- Test: `sql_transform/_codegen_plan_test.py` (append)

**Interfaces:**
- Consumes: Task 4's module.
- Produces: IR dataclasses `Column(table, name)`, `Literal(value)`, `BinaryOp(op, left, right)`, `Not(inner)`, `Func(name, args)`, `Cast(expr, target)`; rel nodes `TableScan(table)`, `Filter(input, predicate)`, `CrossJoin(left, right)`, `Join(left, right, on, outer)`, `SubqueryAlias(input, alias)`, `LookupJoin(input, table, keys, outer)`; `Plan(projection, input)`; `build_plan(sql) -> Plan`; `UnsupportedInCodegen`. `op` is one of `add/sub/mul/div/mod/eq/neq/lt/gt/lte/gte/and/or` — matching `_codegen_runtime` function names.

- [ ] **Step 1: Write the failing tests**

Append to `sql_transform/_codegen_plan_test.py`:

```python
def test_build_plan_simple_projection():
    plan = cp.build_plan("SELECT a AS x FROM t")
    assert plan.input == cp.TableScan("t")
    assert plan.projection == [("x", cp.Column(None, "a"))]


def test_build_plan_unaliased_column_keeps_its_name():
    plan = cp.build_plan("SELECT a FROM t")
    assert plan.projection == [("a", cp.Column(None, "a"))]


def test_build_plan_qualified_column():
    plan = cp.build_plan("SELECT t.a AS x FROM t")
    assert plan.projection == [("x", cp.Column("t", "a"))]


def test_build_plan_requires_an_alias_for_expressions():
    with pytest.raises(ValueError, match="alias"):
        cp.build_plan("SELECT a + 1 FROM t")


def test_build_plan_where_becomes_a_filter():
    plan = cp.build_plan("SELECT a AS x FROM t WHERE a > 1")
    assert isinstance(plan.input, cp.Filter)
    assert plan.input.predicate == cp.BinaryOp("gt", cp.Column(None, "a"), cp.Literal(1))


def test_build_plan_binary_ops_and_literals():
    plan = cp.build_plan("SELECT a + 1 AS x, b / 2.0 AS y, c AS z FROM t WHERE c = 'hi'")
    assert plan.projection[0][1] == cp.BinaryOp("add", cp.Column(None, "a"), cp.Literal(1))
    assert plan.projection[1][1] == cp.BinaryOp("div", cp.Column(None, "b"), cp.Literal(2.0))
    assert plan.input.predicate == cp.BinaryOp("eq", cp.Column(None, "c"), cp.Literal("hi"))


def test_build_plan_null_and_boolean_literals():
    plan = cp.build_plan("SELECT NULL AS x, TRUE AS y FROM t")
    assert plan.projection == [("x", cp.Literal(None)), ("y", cp.Literal(True))]


def test_build_plan_alias():
    plan = cp.build_plan("SELECT s.a AS x FROM t AS s")
    assert plan.input == cp.SubqueryAlias(cp.TableScan("t"), "s")


def test_build_plan_rejects_duplicate_relations():
    with pytest.raises(ValueError, match="more than once"):
        cp.build_plan("SELECT a AS x FROM t JOIN t ON t.a = t.a")


def test_build_plan_cross_join():
    plan = cp.build_plan("SELECT a AS x FROM t CROSS JOIN u")
    assert plan.input == cp.CrossJoin(cp.TableScan("t"), cp.TableScan("u"))


def test_build_plan_comma_join_is_a_cross_join():
    plan = cp.build_plan("SELECT a AS x FROM t, u")
    assert plan.input == cp.CrossJoin(cp.TableScan("t"), cp.TableScan("u"))


def test_build_plan_inner_join_extracts_equality_keys():
    plan = cp.build_plan("SELECT a AS x FROM t JOIN u ON t.k = u.k")
    assert plan.input == cp.Join(
        cp.TableScan("t"),
        cp.TableScan("u"),
        [(cp.Column("t", "k"), cp.Column("u", "k"))],
        False,
    )


def test_build_plan_left_join_is_outer():
    plan = cp.build_plan("SELECT a AS x FROM t LEFT JOIN u ON t.k = u.k")
    assert plan.input.outer is True


def test_build_plan_join_on_and_of_equalities():
    plan = cp.build_plan("SELECT a AS x FROM t JOIN u ON t.k = u.k AND t.j = u.j")
    assert len(plan.input.on) == 2


def test_build_plan_rejects_non_equality_join_on():
    with pytest.raises(ValueError, match="equalit"):
        cp.build_plan("SELECT a AS x FROM t JOIN u ON t.k > u.k")


def test_build_plan_functions_and_cast():
    plan = cp.build_plan(
        "SELECT UPPER(a) AS u, SUBSTR(a, 2, 3) AS s, COALESCE(a, b) AS c, "
        "CAST(a AS VARCHAR) AS v FROM t"
    )
    assert plan.projection[0][1] == cp.Func("upper", [cp.Column(None, "a")])
    assert plan.projection[1][1] == cp.Func(
        "substr", [cp.Column(None, "a"), cp.Literal(2), cp.Literal(3)]
    )
    assert plan.projection[2][1] == cp.Func(
        "coalesce", [cp.Column(None, "a"), cp.Column(None, "b")]
    )
    assert plan.projection[3][1] == cp.Cast(cp.Column(None, "a"), cp.STR)


def test_build_plan_cast_targets():
    plan = cp.build_plan(
        "SELECT CAST(a AS BIGINT) AS i, CAST(a AS DOUBLE) AS f, "
        "CAST(a AS BOOLEAN) AS b FROM t"
    )
    assert [e.target for _, e in plan.projection] == [cp.INT, cp.FLOAT, cp.BOOL]


def test_build_plan_not_and_logic():
    plan = cp.build_plan("SELECT a AS x FROM t WHERE NOT (a AND b)")
    assert plan.input.predicate == cp.Not(
        cp.BinaryOp("and", cp.Column(None, "a"), cp.Column(None, "b"))
    )


def test_build_plan_defers_containers():
    with pytest.raises(cp.UnsupportedInCodegen):
        cp.build_plan("SELECT unnest(a) AS x FROM t")
    with pytest.raises(cp.UnsupportedInCodegen):
        cp.build_plan("SELECT named_struct('k', a) AS x FROM t")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/_codegen_plan_test.py -v -k build_plan`
Expected: FAIL with `AttributeError: module 'sql_transform._codegen_plan' has no attribute 'build_plan'`

- [ ] **Step 3: Write the implementation**

Append to `sql_transform/_codegen_plan.py` (and add `import sqlglot` / `from sqlglot import expressions as exp` to the imports at the top):

```python
class UnsupportedInCodegen(NotImplementedError):
    """Surface the Rust InferFn covers that the codegen engine defers.

    A distinct type keeps the gap explicit at the harness boundary instead of
    letting a deferred feature silently pass as a rejection.
    """


@dataclass
class Column:
    table: str | None
    name: str


@dataclass
class Literal:
    value: Any  # native Python value; None == SQL NULL


@dataclass
class BinaryOp:
    op: str  # matches a function name in _codegen_runtime
    left: Any
    right: Any


@dataclass
class Not:
    inner: Any


@dataclass
class Func:
    name: str
    args: list


@dataclass
class Cast:
    expr: Any
    target: str


@dataclass
class TableScan:
    table: str


@dataclass
class Filter:
    input: Any
    predicate: Any


@dataclass
class CrossJoin:
    left: Any
    right: Any


@dataclass
class Join:
    left: Any
    right: Any
    on: list
    outer: bool


@dataclass
class SubqueryAlias:
    input: Any
    alias: str


@dataclass
class LookupJoin:
    input: Any
    table: str
    keys: list
    outer: bool


@dataclass
class Plan:
    projection: list
    input: Any


_BINOPS = {
    exp.Add: "add",
    exp.Sub: "sub",
    exp.Mul: "mul",
    exp.Div: "div",
    exp.Mod: "mod",
    exp.EQ: "eq",
    exp.NEQ: "neq",
    exp.LT: "lt",
    exp.GT: "gt",
    exp.LTE: "lte",
    exp.GTE: "gte",
    exp.And: "and",
    exp.Or: "or",
}

_SIMPLE_FUNCS = {exp.Upper: "upper", exp.Lower: "lower", exp.Abs: "abs", exp.Round: "round"}
_DEFERRED_FUNCS = ("named_struct", "struct", "unnest", "make_array")

# Measured: sqlglot spreads variadic args differently PER FUNCTION, so each needs
# its own extraction. Concat  -> this=None, args all in .expressions
#                    Coalesce -> first arg in .this, rest in .expressions
#                    Nullif   -> .this + .expression
# One generic helper gets this wrong; in particular a `this not in expressions`
# de-dup guard drops an argument for COALESCE(a, a), whose sub-expressions
# compare equal -- yielding arity 1 and a silently different answer.
_CONCAT_ARGS = lambda e: list(e.expressions)  # noqa: E731
_COALESCE_ARGS = lambda e: [e.this, *e.expressions]  # noqa: E731
_NULLIF_ARGS = lambda e: [e.this, e.expression]  # noqa: E731
_VARIADIC_FUNCS = {
    exp.Concat: ("concat", _CONCAT_ARGS),
    exp.Coalesce: ("coalesce", _COALESCE_ARGS),
    exp.Nullif: ("nullif", _NULLIF_ARGS),
}


def _convert_expr(e: exp.Expression) -> Any:
    if isinstance(e, (exp.Paren, exp.Alias)):
        return _convert_expr(e.this)
    if isinstance(e, exp.Dot):
        raise UnsupportedInCodegen("struct field access is not supported in codegen yet")
    if isinstance(e, exp.Column):
        # `s.a.b` parses as a 3-part Column carrying `db` (NOT exp.Dot). Reading
        # only .table/.name would silently misread it as Column('a', 'b') and drop
        # `s` -- a wrong answer rather than an error. It is struct field access,
        # which is deferred.
        if e.args.get("db") or e.args.get("catalog"):
            raise UnsupportedInCodegen(
                "struct field access is not supported in codegen yet"
            )
        return Column(table=e.table or None, name=e.name)
    if isinstance(e, exp.Null):
        return Literal(None)
    if isinstance(e, exp.Boolean):
        return Literal(bool(e.this))
    if isinstance(e, exp.Literal):
        return Literal(_convert_literal(e))
    if isinstance(e, exp.Neg):
        inner = _convert_expr(e.this)
        if isinstance(inner, Literal) and type(inner.value) in (int, float):
            return Literal(-inner.value)
        raise ValueError(f"Unsupported expression: {e.sql()}")
    if isinstance(e, exp.Not):
        return Not(_convert_expr(e.this))
    for cls, op in _BINOPS.items():
        if isinstance(e, cls):
            return BinaryOp(op, _convert_expr(e.this), _convert_expr(e.expression))
    if isinstance(e, exp.Cast):
        return Cast(_convert_expr(e.this), _cast_target(e.to.sql()))
    if isinstance(e, (exp.Struct, exp.Array)):
        raise UnsupportedInCodegen("struct/list construction is not supported in codegen yet")
    if isinstance(e, exp.Trim):
        if e.args.get("position") or e.expression:
            raise ValueError("Only plain TRIM(expr) is supported")
        return Func("trim", [_convert_expr(e.this)])
    if isinstance(e, exp.Substring):
        args = [_convert_expr(e.this)]
        start = e.args.get("start")
        args.append(_convert_expr(start) if start is not None else Literal(1))
        length = e.args.get("length")
        if length is not None:
            args.append(_convert_expr(length))
        return Func("substr", args)
    for cls, name in _SIMPLE_FUNCS.items():
        if isinstance(e, cls):
            return Func(name, [_convert_expr(e.this)])
    for cls, (name, extract) in _VARIADIC_FUNCS.items():
        if isinstance(e, cls):
            return Func(name, [_convert_expr(a) for a in extract(e)])
    if isinstance(e, exp.Anonymous):
        name = e.name.lower()
        if name in _DEFERRED_FUNCS:
            raise UnsupportedInCodegen(f"{name}() is not supported in codegen yet")
        return Func(name, [_convert_expr(a) for a in e.expressions])
    raise ValueError(f"Unsupported expression: {e.sql()}")


def _convert_literal(e: exp.Literal) -> Any:
    if e.is_string:
        return e.this
    text = e.this
    return float(text) if "." in text or "e" in text.lower() else int(text)


def _cast_target(name: str) -> str:
    name = name.upper()
    if name.startswith(("VARCHAR", "TEXT", "STRING", "CHAR")):
        return STR
    if name.startswith(("BIGINT", "INT")):
        return INT
    if name.startswith(("DOUBLE", "FLOAT", "REAL", "DECIMAL")):
        return FLOAT
    if name.startswith("BOOL"):
        return BOOL
    raise ValueError(f"Unsupported CAST target type: {name}")


def build_plan(sql: str) -> Plan:
    tree = sqlglot.parse_one(sql)
    if not isinstance(tree, exp.Select):
        raise ValueError("Only SELECT queries are supported")
    node = _build_from(tree)
    where = tree.args.get("where")
    if where is not None:
        node = Filter(node, _convert_expr(where.this))
    return Plan(_build_projection(tree.expressions), node)


def _build_from(tree: exp.Select) -> Any:
    # sqlglot 30 renamed this arg "from" -> "from_". Reading "from" silently
    # returns None, which would reject every query as missing a FROM clause.
    from_ = tree.args.get("from_")
    if from_ is None:
        raise ValueError("FROM clause is required")
    seen: set = set()
    node = _table_factor(from_.this, seen)
    for join in tree.args.get("joins") or []:
        node = _build_join(node, join, seen)
    return node


def _build_join(left: Any, join: exp.Join, seen: set) -> Any:
    right = _table_factor(join.this, seen)
    kind = (join.args.get("kind") or "").upper()
    side = (join.args.get("side") or "").upper()
    on = join.args.get("on")
    if kind == "CROSS" or (on is None and not side and not kind):
        return CrossJoin(left, right)
    if on is None:
        raise ValueError("JOIN requires an ON condition")
    if side not in ("", "LEFT"):
        raise ValueError(
            f"Unsupported JOIN type: {side} {kind} — only inner JOIN ... ON, "
            "LEFT JOIN ... ON and CROSS JOIN are supported"
        )
    return Join(left, right, _equality_keys(on), side == "LEFT")


def _table_factor(factor: exp.Expression, seen: set) -> Any:
    if not isinstance(factor, exp.Table):
        raise ValueError("Unsupported FROM clause")
    name = factor.name
    alias = factor.alias or None
    # Track the EFFECTIVE name: a collision would silently overwrite one side's
    # data when rows merge (plan.rs build_table_factor).
    effective = alias or name
    if effective in seen:
        raise ValueError(
            f"table '{effective}' is referenced more than once in FROM/JOIN — "
            "self-joins and alias collisions are not supported"
        )
    seen.add(effective)
    scan = TableScan(name)
    return SubqueryAlias(scan, alias) if alias else scan


def _equality_keys(e: exp.Expression) -> list:
    if isinstance(e, exp.Paren):
        return _equality_keys(e.this)
    if isinstance(e, exp.And):
        return _equality_keys(e.this) + _equality_keys(e.expression)
    if isinstance(e, exp.EQ):
        return [(_convert_expr(e.this), _convert_expr(e.expression))]
    raise ValueError(
        "JOIN ON condition must be an equality, or an AND of equalities, between columns"
    )


def _build_projection(items: list) -> list:
    out = []
    for item in items:
        if isinstance(item, exp.Alias):
            out.append((item.alias, _convert_expr(item.this)))
        elif isinstance(item, exp.Column):
            out.append((item.name, _convert_expr(item)))
        elif isinstance(item, exp.Star):
            raise ValueError("Unsupported SELECT item: *")
        else:
            raise ValueError("Expression in SELECT list needs an alias (AS name)")
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/_codegen_plan_test.py -v`
Expected: PASS. The arg shapes here were measured against sqlglot 30.12, not inferred. If a case fails after a sqlglot bump, print the tree with `print(sqlglot.parse_one(sql).__repr__())` and re-measure — arg placement is version-sensitive and silent (the `from` → `from_` rename is exactly this).

- [ ] **Step 5: Commit**

```bash
mise run fmt
git add sql_transform/_codegen_plan.py sql_transform/_codegen_plan_test.py
git commit -m "feat: codegen front-end — plan IR and sqlglot parsing"
```

---

### Task 6: Optimize, validate, and infer types

**Files:**
- Modify: `sql_transform/_codegen_plan.py` (append)
- Test: `sql_transform/_codegen_plan_test.py` (append)

**Interfaces:**
- Consumes: Tasks 4–5.
- Produces: `LookupSpec(static_table, key_columns)`, `optimize(plan, static_tables) -> (Plan, list[LookupSpec])`, `ColumnValidation(row_table_columns, effective_schemas)`, `validate_columns(plan, row_table_names, row_schemas, static_schemas) -> ColumnValidation`, `infer_type(expr, schemas) -> FieldType`, `scan_name(node) -> str | None`, `referenced_tables(node) -> list[str]`. `validate_columns` mutates unqualified `Column.table` to the resolved effective name.

- [ ] **Step 1: Write the failing tests**

Append to `sql_transform/_codegen_plan_test.py`:

```python
def _schemas():
    return (
        {"t": {"a": cp.FieldType(cp.INT, False), "k": cp.FieldType(cp.INT, False)}},
        {"s": {"k": cp.FieldType(cp.INT, False), "v": cp.FieldType(cp.FLOAT, False)}},
    )


def test_optimize_rewrites_a_static_join_into_a_lookup_join():
    plan = cp.build_plan("SELECT v AS x FROM t JOIN s ON t.k = s.k")
    plan, specs = cp.optimize(plan, {"s"})
    assert isinstance(plan.input, cp.LookupJoin)
    assert plan.input.table == "s"
    assert plan.input.keys == [cp.Column("t", "k")]
    assert specs[0].static_table == "s"
    assert specs[0].key_columns == ["k"]


def test_optimize_finds_the_static_side_regardless_of_on_order():
    plan = cp.build_plan("SELECT v AS x FROM t JOIN s ON s.k = t.k")
    plan, _ = cp.optimize(plan, {"s"})
    assert plan.input.keys == [cp.Column("t", "k")]


def test_optimize_rejects_a_static_to_static_join():
    plan = cp.build_plan("SELECT v AS x FROM s JOIN s2 ON s.k = s2.k")
    with pytest.raises(ValueError, match="two static tables"):
        cp.optimize(plan, {"s", "s2"})


def test_optimize_rejects_a_row_to_row_left_join():
    plan = cp.build_plan("SELECT a AS x FROM t LEFT JOIN u ON t.k = u.k")
    with pytest.raises(ValueError, match="only supported against a static"):
        cp.optimize(plan, set())


def test_validate_resolves_unqualified_columns_and_collects_used():
    row, static = _schemas()
    plan = cp.build_plan("SELECT a AS x FROM t")
    v = cp.validate_columns(plan, {"t"}, row, static)
    assert plan.projection[0][1] == cp.Column("t", "a")  # rewritten in place
    assert v.row_table_columns == {"t": ["a"]}


def test_validate_rejects_unknown_and_ambiguous_columns():
    row, static = _schemas()
    plan = cp.build_plan("SELECT nope AS x FROM t")
    with pytest.raises(ValueError, match="Unknown column"):
        cp.validate_columns(plan, {"t"}, row, static)

    row2 = {"t": {"a": cp.FieldType(cp.INT, False)}, "u": {"a": cp.FieldType(cp.INT, False)}}
    plan = cp.build_plan("SELECT a AS x FROM t CROSS JOIN u")
    with pytest.raises(ValueError, match="Ambiguous"):
        cp.validate_columns(plan, {"t", "u"}, row2, {})


def test_validate_widens_the_outer_side_of_a_left_lookup_join_to_nullable():
    row, static = _schemas()
    plan = cp.build_plan("SELECT v AS x FROM t LEFT JOIN s ON t.k = s.k")
    plan, _ = cp.optimize(plan, {"s"})
    v = cp.validate_columns(plan, {"t"}, row, static)
    assert v.effective_schemas["s"]["v"].nullable is True


def test_validate_resolves_through_an_alias():
    row, static = _schemas()
    plan = cp.build_plan("SELECT z.a AS x FROM t AS z")
    v = cp.validate_columns(plan, {"t"}, row, static)
    assert v.effective_schemas["z"]["a"] == cp.FieldType(cp.INT, False)
    assert v.row_table_columns == {"t": ["a"]}


def test_infer_type_arithmetic_and_nullability():
    schemas = {
        "t": {
            "i": cp.FieldType(cp.INT, False),
            "f": cp.FieldType(cp.FLOAT, False),
            "n": cp.FieldType(cp.INT, True),
        }
    }
    assert cp.infer_type(cp.BinaryOp("add", cp.Column("t", "i"), cp.Literal(1)), schemas) == (
        cp.FieldType(cp.INT, False)
    )
    assert cp.infer_type(cp.BinaryOp("add", cp.Column("t", "i"), cp.Column("t", "f")), schemas) == (
        cp.FieldType(cp.FLOAT, False)
    )
    assert cp.infer_type(cp.BinaryOp("add", cp.Column("t", "i"), cp.Column("t", "n")), schemas) == (
        cp.FieldType(cp.INT, True)
    )
    assert cp.infer_type(cp.BinaryOp("gt", cp.Column("t", "i"), cp.Literal(1)), schemas) == (
        cp.FieldType(cp.BOOL, False)
    )


def test_infer_type_functions_and_casts():
    schemas = {"t": {"s": cp.FieldType(cp.STR, False), "i": cp.FieldType(cp.INT, True)}}
    assert cp.infer_type(cp.Func("upper", [cp.Column("t", "s")]), schemas).base == cp.STR
    assert cp.infer_type(cp.Func("concat", [cp.Column("t", "i")]), schemas) == (
        cp.FieldType(cp.STR, False)  # concat is never null
    )
    assert cp.infer_type(cp.Func("abs", [cp.Column("t", "i")]), schemas) == (
        cp.FieldType(cp.INT, True)  # abs keeps its argument's base
    )
    assert cp.infer_type(cp.Func("coalesce", [cp.Column("t", "i")]), schemas).nullable is True
    assert cp.infer_type(cp.Cast(cp.Column("t", "i"), cp.STR), schemas) == (
        cp.FieldType(cp.STR, True)
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/_codegen_plan_test.py -v -k "optimize or validate or infer_type"`
Expected: FAIL with `AttributeError: module 'sql_transform._codegen_plan' has no attribute 'optimize'`

- [ ] **Step 3: Write the implementation**

Append to `sql_transform/_codegen_plan.py`:

```python
@dataclass
class LookupSpec:
    static_table: str
    key_columns: list


def optimize(plan: Plan, static_tables: set) -> tuple:
    """Rewrite every Join with exactly one static side into a LookupJoin
    (mirrors plan::optimize)."""
    specs: list = []
    return Plan(plan.projection, _optimize_rel(plan.input, static_tables, specs)), specs


def _optimize_rel(node: Any, static_tables: set, specs: list) -> Any:
    if isinstance(node, Join):
        left = _optimize_rel(node.left, static_tables, specs)
        right = _optimize_rel(node.right, static_tables, specs)
        left_name, right_name = scan_name(left), scan_name(right)
        left_static = left_name if left_name in static_tables else None
        right_static = right_name if right_name in static_tables else None
        if left_static and right_static:
            raise ValueError("Joining two static tables together is not supported")
        if right_static or left_static:
            table = right_static or left_static
            other = left if right_static else right
            keys, key_columns = _split_keys(node.on, table)
            specs.append(LookupSpec(table, key_columns))
            return LookupJoin(other, table, keys, node.outer)
        if node.outer:
            raise ValueError("LEFT JOIN is only supported against a static lookup table")
        return Join(left, right, node.on, node.outer)
    if isinstance(node, CrossJoin):
        return CrossJoin(
            _optimize_rel(node.left, static_tables, specs),
            _optimize_rel(node.right, static_tables, specs),
        )
    if isinstance(node, Filter):
        return Filter(_optimize_rel(node.input, static_tables, specs), node.predicate)
    if isinstance(node, SubqueryAlias):
        return SubqueryAlias(_optimize_rel(node.input, static_tables, specs), node.alias)
    return node


def scan_name(node: Any) -> str | None:
    """The real table name a scan (possibly aliased) reads from."""
    if isinstance(node, TableScan):
        return node.table
    if isinstance(node, SubqueryAlias):
        return scan_name(node.input)
    return None


def _split_keys(on: list, static_table: str) -> tuple:
    """Split each ON equality into (row-side expression, static key column).
    The static side is identified per-pair by qualifier, since `a = b` vs
    `b = a` is independent of which side is structurally left/right."""
    row_keys, static_cols = [], []
    for left, right in on:
        lq = left.table if isinstance(left, Column) else None
        rq = right.table if isinstance(right, Column) else None
        if lq == static_table:
            static_expr, row_expr = left, right
        elif rq == static_table:
            static_expr, row_expr = right, left
        else:
            raise ValueError(
                f"JOIN ON keys against static table '{static_table}' must reference "
                f"the static table's columns by name (e.g. {static_table}.col)"
            )
        if not isinstance(static_expr, Column):
            raise ValueError(
                f"JOIN ON keys against static table '{static_table}' must be plain columns"
            )
        static_cols.append(static_expr.name)
        row_keys.append(row_expr)
    return row_keys, static_cols


@dataclass
class ColumnValidation:
    row_table_columns: dict
    effective_schemas: dict


def validate_columns(
    plan: Plan, row_table_names: set, row_schemas: dict, static_schemas: dict
) -> ColumnValidation:
    """Validate every column reference against the resolved table schemas,
    rewrite unqualified refs to their effective table, and collect (per row
    table's REAL name) the columns the query actually reads."""
    resolved: dict = {}
    nullable_tables: set = set()
    _resolve_tables(plan.input, row_table_names, False, resolved, nullable_tables)

    effective_schemas: dict = {}
    for effective, (real, is_row) in resolved.items():
        schema = (row_schemas if is_row else static_schemas).get(real)
        if schema is None:
            continue
        if effective in nullable_tables:
            # An unmatched outer row makes every column on that side NULL, so
            # the synthesized output type must be nullable even when the source
            # declares otherwise.
            schema = {k: FieldType(v.base, True) for k, v in schema.items()}
        effective_schemas[effective] = schema

    used: dict = {}
    for _, e in plan.projection:
        _validate_expr(e, resolved, row_schemas, static_schemas, used)
    _validate_rel(plan.input, resolved, row_schemas, static_schemas, used)
    return ColumnValidation({k: sorted(v) for k, v in used.items()}, effective_schemas)


def _resolve_tables(
    node: Any, row_table_names: set, nullable: bool, out: dict, nullable_out: set
) -> None:
    if isinstance(node, TableScan):
        out[node.table] = (node.table, node.table in row_table_names)
        if nullable:
            nullable_out.add(node.table)
    elif isinstance(node, SubqueryAlias):
        real = scan_name(node.input)
        if real is not None:
            out[node.alias] = (real, real in row_table_names)
            if nullable:
                nullable_out.add(node.alias)
    elif isinstance(node, Filter):
        _resolve_tables(node.input, row_table_names, nullable, out, nullable_out)
    elif isinstance(node, CrossJoin):
        _resolve_tables(node.left, row_table_names, nullable, out, nullable_out)
        _resolve_tables(node.right, row_table_names, nullable, out, nullable_out)
    elif isinstance(node, Join):
        _resolve_tables(node.left, row_table_names, nullable, out, nullable_out)
        _resolve_tables(node.right, row_table_names, nullable or node.outer, out, nullable_out)
    elif isinstance(node, LookupJoin):
        _resolve_tables(node.input, row_table_names, nullable, out, nullable_out)
        out[node.table] = (node.table, False)
        if nullable or node.outer:
            nullable_out.add(node.table)


def _validate_rel(node: Any, resolved, row_schemas, static_schemas, used) -> None:
    if isinstance(node, Filter):
        _validate_expr(node.predicate, resolved, row_schemas, static_schemas, used)
        _validate_rel(node.input, resolved, row_schemas, static_schemas, used)
    elif isinstance(node, CrossJoin):
        _validate_rel(node.left, resolved, row_schemas, static_schemas, used)
        _validate_rel(node.right, resolved, row_schemas, static_schemas, used)
    elif isinstance(node, Join):
        for left, right in node.on:
            _validate_expr(left, resolved, row_schemas, static_schemas, used)
            _validate_expr(right, resolved, row_schemas, static_schemas, used)
        _validate_rel(node.left, resolved, row_schemas, static_schemas, used)
        _validate_rel(node.right, resolved, row_schemas, static_schemas, used)
    elif isinstance(node, SubqueryAlias):
        _validate_rel(node.input, resolved, row_schemas, static_schemas, used)
    elif isinstance(node, LookupJoin):
        for k in node.keys:
            _validate_expr(k, resolved, row_schemas, static_schemas, used)
        _validate_rel(node.input, resolved, row_schemas, static_schemas, used)


def _validate_expr(e: Any, resolved, row_schemas, static_schemas, used) -> None:
    if isinstance(e, Column):
        if e.table is not None:
            entry = resolved.get(e.table)
            if entry is None:
                # Rust reinterprets this as struct field access; containers are
                # deferred here, so it is simply an unknown reference.
                raise ValueError(f"Unknown table: {e.table}")
            real, is_row = entry
            schema = (row_schemas if is_row else static_schemas).get(real)
            if schema is None:
                raise ValueError(f"Unknown table: {real}")
            if e.name not in schema:
                raise ValueError(f"Unknown column: {real}.{e.name}")
            if is_row:
                used.setdefault(real, set()).add(e.name)
            return
        matches = [
            (effective, real, is_row)
            for effective, (real, is_row) in resolved.items()
            if e.name in ((row_schemas if is_row else static_schemas).get(real) or {})
        ]
        if not matches:
            raise ValueError(f"Unknown column: {e.name}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous column reference: {e.name}")
        effective, real, is_row = matches[0]
        e.table = effective  # codegen emits a direct subscript off this
        if is_row:
            used.setdefault(real, set()).add(e.name)
    elif isinstance(e, BinaryOp):
        _validate_expr(e.left, resolved, row_schemas, static_schemas, used)
        _validate_expr(e.right, resolved, row_schemas, static_schemas, used)
    elif isinstance(e, Not):
        _validate_expr(e.inner, resolved, row_schemas, static_schemas, used)
    elif isinstance(e, Cast):
        _validate_expr(e.expr, resolved, row_schemas, static_schemas, used)
    elif isinstance(e, Func):
        for a in e.args:
            _validate_expr(a, resolved, row_schemas, static_schemas, used)


_STR_FUNCS = frozenset({"upper", "lower", "trim", "substr", "substring"})


def infer_type(e: Any, schemas: dict) -> FieldType:
    """Statically infer a projection's FieldType, mirroring types::infer_type.
    Sound but not tight on nullability: nullable means "cannot prove non-NULL"."""
    if isinstance(e, Column):
        return _resolve_column_type(e.table, e.name, schemas)
    if isinstance(e, Literal):
        return _literal_type(e.value)
    if isinstance(e, BinaryOp):
        left, right = infer_type(e.left, schemas), infer_type(e.right, schemas)
        nullable = left.nullable or right.nullable
        if e.op in ("add", "sub", "mul", "div", "mod"):
            return FieldType(INT if left.base == INT and right.base == INT else FLOAT, nullable)
        return FieldType(BOOL, nullable)
    if isinstance(e, Not):
        return FieldType(BOOL, infer_type(e.inner, schemas).nullable)
    if isinstance(e, Cast):
        return FieldType(e.target, infer_type(e.expr, schemas).nullable)
    if isinstance(e, Func):
        return _function_type(e.name, [infer_type(a, schemas) for a in e.args])
    raise UnsupportedInCodegen(f"cannot infer the type of {type(e).__name__}")


def _resolve_column_type(table: str | None, name: str, schemas: dict) -> FieldType:
    if table is not None:
        schema = schemas.get(table)
        if schema is None or name not in schema:
            raise ValueError(f"Unknown column: {table}.{name}")
        return schema[name]
    found = None
    for schema in schemas.values():
        if name in schema:
            if found is not None:
                raise ValueError(f"Ambiguous column reference: {name}")
            found = schema[name]
    if found is None:
        raise ValueError(f"Unknown column: {name}")
    return found


def _literal_type(v: Any) -> FieldType:
    if v is None:
        return FieldType(OTHER, True)
    t = type(v)
    if t is bool:
        return FieldType(BOOL, False)
    if t is int:
        return FieldType(INT, False)
    if t is float:
        return FieldType(FLOAT, False)
    if t is str:
        return FieldType(STR, False)
    return FieldType(OTHER, True)


def _function_type(name: str, args: list) -> FieldType:
    any_nullable = any(a.nullable for a in args)
    if name in _STR_FUNCS:
        return FieldType(STR, any_nullable)
    if name == "abs":
        return FieldType(args[0].base if args else OTHER, any_nullable)
    if name == "round":
        # DataFusion ROUND always yields a float, even for an int argument
        # (measured: ROUND(3) -> 3.0). Rust types it as the arg base -- a bug.
        return FieldType(FLOAT, any_nullable)
    if name == "concat":
        return FieldType(STR, False)
    if name in ("coalesce", "nullif"):
        return FieldType(args[0].base if args else OTHER, True)
    return FieldType(OTHER, True)


def referenced_tables(node: Any) -> list:
    if isinstance(node, TableScan):
        return [node.table]
    if isinstance(node, (SubqueryAlias, Filter, LookupJoin)):
        return referenced_tables(node.input)
    if isinstance(node, (CrossJoin, Join)):
        return referenced_tables(node.left) + referenced_tables(node.right)
    return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/_codegen_plan_test.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
mise run fmt
git add sql_transform/_codegen_plan.py sql_transform/_codegen_plan_test.py
git commit -m "feat: codegen front-end — optimize, validate, type inference"
```

---

### Task 7: Emitter and engine — thin vertical (scan, alias, filter, projection)

**Files:**
- Create: `sql_transform/_codegen.py`
- Test: `sql_transform/_codegen_test.py`

**Interfaces:**
- Consumes: `_codegen_plan` (Tasks 4–6), `_codegen_runtime` (Tasks 1–3).
- Produces: `CodegenFn(sql, row_tables, static_tables, output_model=None)` with `.infer(tables=None, **kwargs) -> list[BaseModel]`, `.output_model`, `.source` (the generated Python, for debugging). Re-exports `UnsupportedInCodegen`. Same duck-type as `sql_transform._interpreter.InferFn`.

The emitter walks the rel tree, passing a continuation for the innermost body, so each node contributes a loop or guard level. Column refs compile to a direct subscript on the loop variable — that resolution at compile time, not per row, is the point of the engine.

- [ ] **Step 1: Write the failing tests**

Create `sql_transform/_codegen_test.py`:

```python
"""End-to-end tests for the codegen engine.

Broad semantic parity is proven by the differential harness (tests/); these
cover the engine's own seams -- compilation, marshalling, output typing.
"""

import typing

import pyarrow as pa
import pytest
from pydantic import BaseModel

from sql_transform._codegen import CodegenFn, UnsupportedInCodegen


class Row(BaseModel):
    a: int
    b: float | None = None
    s: str = "x"


def test_projection_and_arithmetic():
    fn = CodegenFn("SELECT a + 1 AS x FROM t", {"t": Row}, {})
    assert [r.x for r in fn.infer({"t": [Row(a=1), Row(a=2)]})] == [2, 3]


def test_output_model_is_synthesized_with_inferred_types():
    fn = CodegenFn("SELECT a AS i, a / 2 AS q, s AS name FROM t", {"t": Row}, {})
    fields = fn.output_model.model_fields
    assert fields["i"].annotation is int
    assert fields["q"].annotation is int  # int / int stays int
    assert fields["name"].annotation is str


def test_nullable_column_yields_an_optional_output_field():
    fn = CodegenFn("SELECT b AS x FROM t", {"t": Row}, {})
    assert fn.output_model.model_fields["x"].annotation == typing.Optional[float]
    assert fn.infer({"t": [Row(a=1)]})[0].x is None


def test_where_filters_rows():
    fn = CodegenFn("SELECT a AS x FROM t WHERE a > 1", {"t": Row}, {})
    assert [r.x for r in fn.infer({"t": [Row(a=1), Row(a=2), Row(a=3)]})] == [2, 3]


def test_where_drops_null_and_non_true_predicates():
    fn = CodegenFn("SELECT a AS x FROM t WHERE b > 1.0", {"t": Row}, {})
    assert fn.infer({"t": [Row(a=1)]}) == []  # b is None -> predicate NULL -> dropped


def test_table_alias():
    fn = CodegenFn("SELECT z.a AS x FROM t AS z", {"t": Row}, {})
    assert [r.x for r in fn.infer({"t": [Row(a=7)]})] == [7]


def test_infer_accepts_kwargs_as_well_as_a_tables_dict():
    fn = CodegenFn("SELECT a AS x FROM t", {"t": Row}, {})
    assert [r.x for r in fn.infer(t=[Row(a=5)])] == [5]


def test_only_referenced_columns_are_read_from_the_row():
    fn = CodegenFn("SELECT a AS x FROM t", {"t": Row}, {})

    class Partial:
        a = 3  # no b/s at all; the engine must not touch them

    assert [r.x for r in fn.infer({"t": [Partial()]})] == [3]


def test_missing_attribute_is_a_clear_error():
    fn = CodegenFn("SELECT a AS x FROM t", {"t": Row}, {})

    class Empty:
        pass

    with pytest.raises(ValueError, match="missing attribute 'a'"):
        fn.infer({"t": [Empty()]})


def test_unknown_table_in_from_is_rejected():
    fn = CodegenFn("SELECT a AS x FROM t", {"t": Row}, {})
    with pytest.raises(ValueError, match="Unknown table"):
        fn.infer({"other": [Row(a=1)]})


def test_supplied_output_model_is_validated():
    class Good(BaseModel):
        x: float  # int is compatible with a declared float

    class MissingField(BaseModel):
        nope: int

    class Extra(BaseModel):
        x: int
        surplus: int

    CodegenFn("SELECT a AS x FROM t", {"t": Row}, {}, output_model=Good)
    with pytest.raises(ValueError, match="missing field 'x'"):
        CodegenFn("SELECT a AS x FROM t", {"t": Row}, {}, output_model=MissingField)
    with pytest.raises(ValueError, match="not produced by the query"):
        CodegenFn("SELECT a AS x FROM t", {"t": Row}, {}, output_model=Extra)


def test_incompatible_output_model_is_rejected():
    class Bad(BaseModel):
        x: str  # int is not compatible with a declared str

    with pytest.raises(ValueError, match="incompatible"):
        CodegenFn("SELECT a AS x FROM t", {"t": Row}, {}, output_model=Bad)


def test_int_division_by_zero_raises_value_error():
    fn = CodegenFn("SELECT a / 0 AS x FROM t", {"t": Row}, {})
    with pytest.raises(ValueError, match="division by zero"):
        fn.infer({"t": [Row(a=1)]})


def test_container_columns_are_deferred_not_silently_wrong():
    class Inner(BaseModel):
        x: int

    class WithStruct(BaseModel):
        s: Inner

    with pytest.raises(UnsupportedInCodegen):
        CodegenFn("SELECT s AS out FROM t", {"t": WithStruct}, {})


def test_generated_source_is_available_for_debugging():
    fn = CodegenFn("SELECT a AS x FROM t", {"t": Row}, {})
    assert "def _run(" in fn.source
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/_codegen_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sql_transform._codegen'`

- [ ] **Step 3: Write the implementation**

Create `sql_transform/_codegen.py`:

```python
"""Codegen serving engine — a codegen-based InferFn.

Same contract as the Rust InferFn (sql_transform._interpreter): built from the
post-fit rewritten __STATE__/__THIS__ SQL plus the row/static schemas, .infer()
returns validated Pydantic output rows. The difference is execution -- the plan
is compiled once into a cached Python function, so the per-row path is
straight-line Python over native values instead of an interpreter behind pyo3.

Column references resolve to a direct subscript at compile time rather than a
dict scan per row; that, and not the arithmetic, is where the win comes from.

ponytail: every operation emits a runtime call, so a statically-known int + int
still pays one. Specializing emission off infer_type is the obvious next win --
correctness first, with the differential harness as the net.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, create_model

from sql_transform import _codegen_plan as cp
from sql_transform import _codegen_runtime as rt
from sql_transform._codegen_plan import UnsupportedInCodegen

__all__ = ["CodegenFn", "UnsupportedInCodegen"]

_OPS = {
    "add": "rt.add",
    "sub": "rt.sub",
    "mul": "rt.mul",
    "div": "rt.div",
    "mod": "rt.mod",
    "eq": "rt.eq",
    "neq": "rt.neq",
    "lt": "rt.lt",
    "gt": "rt.gt",
    "lte": "rt.lte",
    "gte": "rt.gte",
    "and": "rt.and_",
    "or": "rt.or_",
}

_BUILTINS = {
    "upper": "rt.upper",
    "lower": "rt.lower",
    "trim": "rt.trim",
    "substr": "rt.substr",
    "substring": "rt.substr",
    "concat": "rt.concat",
    "abs": "rt.abs_",
    "round": "rt.round_",
    "coalesce": "rt.coalesce",
    "nullif": "rt.nullif",
}

_CASTS = {
    cp.STR: "rt.cast_str",
    cp.INT: "rt.cast_int",
    cp.FLOAT: "rt.cast_float",
    cp.BOOL: "rt.cast_bool",
}


class _Emitter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._n = 0

    def var(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def line(self, indent: int, text: str) -> None:
        self.lines.append("    " * indent + text)


def _emit_expr(e: Any, env: dict) -> str:
    if isinstance(e, cp.Column):
        var = env.get(e.table)
        if var is None:
            raise ValueError(f"Unknown table: {e.table}")
        return f"{var}[{e.name!r}]"
    if isinstance(e, cp.Literal):
        return repr(e.value)
    if isinstance(e, cp.BinaryOp):
        return f"{_OPS[e.op]}({_emit_expr(e.left, env)}, {_emit_expr(e.right, env)})"
    if isinstance(e, cp.Not):
        return f"rt.not_({_emit_expr(e.inner, env)})"
    if isinstance(e, cp.Cast):
        return f"{_CASTS[e.target]}({_emit_expr(e.expr, env)})"
    if isinstance(e, cp.Func):
        fn = _BUILTINS.get(e.name)
        if fn is None:
            raise ValueError(f"Unknown function: {e.name}")
        return f"{fn}({', '.join(_emit_expr(a, env) for a in e.args)})"
    raise UnsupportedInCodegen(f"cannot compile {type(e).__name__}")


def _emit_rel(node: Any, env: dict, ind: int, em: _Emitter, body) -> None:
    """Emit `node` as loop/guard levels, calling `body(env, indent)` to fill the
    innermost level. `env` maps each in-scope effective table name to the local
    holding its column dict."""
    if isinstance(node, cp.TableScan):
        v = em.var("_s")
        em.line(ind, f"for {v} in _tables[{node.table!r}]:")
        body({**env, node.table: v}, ind + 1)
    elif isinstance(node, cp.SubqueryAlias):
        real = cp.scan_name(node.input)

        def aliased(inner: dict, i: int) -> None:
            renamed = {k: v for k, v in inner.items() if k != real}
            renamed[node.alias] = inner[real]
            body(renamed, i)

        _emit_rel(node.input, env, ind, em, aliased)
    elif isinstance(node, cp.Filter):

        def filtered(inner: dict, i: int) -> None:
            em.line(i, f"if rt.truthy({_emit_expr(node.predicate, inner)}):")
            body(inner, i + 1)

        _emit_rel(node.input, env, ind, em, filtered)
    else:
        raise UnsupportedInCodegen(f"cannot compile {type(node).__name__}")


def compile_plan(plan: cp.Plan) -> tuple:
    em = _Emitter()
    em.line(0, "def _run(_tables, _lookups, _nullrows):")
    em.line(1, "_out = []")

    def project(env: dict, ind: int) -> None:
        items = ", ".join(f"{alias!r}: {_emit_expr(e, env)}" for alias, e in plan.projection)
        em.line(ind, f"_out.append({{{items}}})")

    _emit_rel(plan.input, {}, 1, em, project)
    em.line(1, "return _out")

    source = "\n".join(em.lines)
    namespace: dict = {"rt": rt}
    exec(compile(source, "<sql_transform.codegen>", "exec"), namespace)  # noqa: S102
    return namespace["_run"], source


class CodegenFn:
    """Codegen counterpart to the Rust InferFn — same constructor and infer()."""

    def __init__(
        self,
        sql: str,
        row_tables: dict,
        static_tables: dict,
        output_model: type[BaseModel] | None = None,
    ) -> None:
        plan = cp.build_plan(sql)
        # Lookup specs go unused until joins land (Task 9).
        plan, _ = cp.optimize(plan, set(static_tables))

        row_schemas = {n: cp.schema_from_pydantic(m) for n, m in row_tables.items()}
        static_schemas = {n: cp.schema_from_arrow(t) for n, t in static_tables.items()}
        validation = cp.validate_columns(plan, set(row_tables), row_schemas, static_schemas)
        schemas = validation.effective_schemas

        inferred = [(alias, cp.infer_type(e, schemas)) for alias, e in plan.projection]
        for alias, ft in inferred:
            if cp.is_container(ft.base):
                raise UnsupportedInCodegen(
                    f"column '{alias}' is a struct/list, which codegen does not support yet"
                )

        if output_model is None:
            self.output_model = create_model(
                "OutputRow",
                **{a: (cp.field_type_to_python(ft), ...) for a, ft in inferred},
            )
        else:
            _validate_output_model(output_model, inferred)
            self.output_model = output_model

        self._lookups: dict = {}
        self._nullrows: dict = {}
        self._row_table_columns = validation.row_table_columns
        self._referenced = cp.referenced_tables(plan.input)
        self._run, self.source = compile_plan(plan)

    def infer(self, tables: dict | None = None, **kwargs: list) -> list:
        merged = dict(tables or {})
        merged.update(kwargs)

        value_tables: dict = {}
        for table, rows in merged.items():
            columns = self._row_table_columns.get(table, [])
            out_rows = []
            for row_obj in rows:
                row = {}
                for col in columns:
                    try:
                        row[col] = getattr(row_obj, col)
                    except AttributeError as e:
                        raise ValueError(
                            f"Row for table '{table}' is missing attribute '{col}': {e}"
                        ) from e
                out_rows.append(row)
            value_tables[table] = out_rows

        for table in self._referenced:
            if table not in value_tables:
                raise ValueError(f"Unknown table in FROM clause: {table}")

        rows = self._run(value_tables, self._lookups, self._nullrows)
        return [self.output_model.model_validate(r) for r in rows]


def _validate_output_model(model: type[BaseModel], inferred: list) -> None:
    """Reject only what is provably wrong: a missing/extra field vs the
    projection, or a base-type mismatch compatible() cannot excuse. Nullability
    is never a build-time error (mirrors lib.rs validate_output_model)."""
    declared = cp.schema_from_pydantic(model)
    aliases = set()
    for alias, ft in inferred:
        aliases.add(alias)
        if alias not in declared:
            raise ValueError(f"output_model is missing field '{alias}' produced by the query")
        if not cp.compatible(ft.base, declared[alias].base):
            raise ValueError(
                f"output_model field '{alias}' is declared as a type incompatible "
                "with the query's inferred output"
            )
    extra = set(declared) - aliases
    if extra:
        raise ValueError(f"output_model declares fields not produced by the query: {sorted(extra)}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/_codegen_test.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
mise run fmt
git add sql_transform/_codegen.py sql_transform/_codegen_test.py
git commit -m "feat: codegen engine — emitter and CodegenFn thin vertical"
```

---

### Task 8: Wire codegen into the differential harness as a second backend

**Files:**
- Modify: `tests/differential.py:1-19` (module docstring + imports), `tests/differential.py:121-133` (`_run_infer`), `tests/differential.py:177-215` (`check`, `check_both_raise`)
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: `CodegenFn`, `UnsupportedInCodegen` from Task 7.
- Produces: `differential.BACKENDS` (dict name→runner), `differential.set_backend(name)`, and a `_backend` fixture that parametrizes every harness test module.

This is the real gate. No test call sites change: `_run_infer` is already engine-shaped, so a second backend only needs the same duck-type. Relational tests will **skip** on codegen until Task 9 lands joins — Task 10 then proves those skips are gone.

- [ ] **Step 1: Add the codegen runner and backend selection to the harness**

In `tests/differential.py`, replace the module docstring and imports (lines 1-19) with:

```python
"""Differential test harness for the serving engines.

`check(query, tables)` runs a query through DataFusion (the oracle) AND the
serving engine selected by the active backend, and asserts their output values
match. The backend is set per-test by the `_backend` fixture in conftest.py, so
every case here runs once per engine:

  * "rust"    — the Rust InferFn interpreter (sql_transform._interpreter)
  * "codegen" — the codegen engine (sql_transform._codegen)

Holding both to the same oracle is what makes them provably equivalent. Cases
touching surface a backend explicitly defers raise UnsupportedInCodegen and are
skipped loudly rather than passing silently.

Tests are native pytest parametrized decision tables (see test_diff_*.py). This
module is NOT collected by pytest (no test_ prefix / _test suffix).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import datafusion
import pyarrow as pa
import pytest

from sql_transform._codegen import CodegenFn, UnsupportedInCodegen
from sql_transform._interpreter import InferFn
from sql_transform._schema import synthesize_this_model
```

Replace `_run_infer` (lines 121-133) with:

```python
def _run_engine(engine: Any, query: str, tables: dict[str, Table]) -> list[dict]:
    row_models: dict[str, Any] = {}
    infer_rows: dict[str, list] = {}
    static_tables: dict[str, pa.Table] = {}
    for name, tbl in tables.items():
        if tbl.kind == "row":
            model = synthesize_this_model(tbl.schema)
            row_models[name] = model
            infer_rows[name] = [model(**r) for r in tbl.rows]
        else:
            static_tables[name] = pa.Table.from_pylist(tbl.rows, schema=tbl.schema)
    fn = engine(query, row_tables=row_models, static_tables=static_tables)
    return [r.model_dump() for r in fn.infer(infer_rows)]


def _run_infer(query: str, tables: dict[str, Table]) -> list[dict]:
    return _run_engine(InferFn, query, tables)


def _run_codegen(query: str, tables: dict[str, Table]) -> list[dict]:
    return _run_engine(CodegenFn, query, tables)


BACKENDS = {"rust": _run_infer, "codegen": _run_codegen}
_backend = "rust"


def set_backend(name: str) -> None:
    """Select which serving engine `check` exercises against the oracle.
    Driven by the `_backend` fixture in conftest.py."""
    global _backend
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; expected one of {sorted(BACKENDS)}")
    _backend = name


def _run_backend(query: str, tables: dict[str, Table]) -> list[dict]:
    return BACKENDS[_backend](query, tables)
```

- [ ] **Step 2: Point `check` and `check_both_raise` at the active backend**

In `tests/differential.py`, replace `check` and `check_both_raise` (lines 177-215) with:

```python
def check(
    query: str,
    tables: dict[str, Table],
    expect: list[dict] | None = None,
) -> None:
    """Run `query` through DataFusion (oracle) AND the active backend engine over
    the same typed tables; assert their output rows match (order-insensitive,
    float-tolerant, NULL-aware). If `expect` is given, also assert
    output == expect."""
    oracle = _run_datafusion(query, tables)
    try:
        actual = _run_backend(query, tables)
    except UnsupportedInCodegen as e:
        pytest.skip(f"{_backend} defers this surface: {e}")
    assert _rows_equal(actual, oracle), (
        f"{_backend} engine disagrees with DataFusion.\n  query: {query}\n"
        f"  {_backend}: {actual}\n  datafusion: {oracle}"
    )
    if expect is not None:
        assert _rows_equal(actual, expect), (
            f"Output does not match expected.\n  query: {query}\n"
            f"  actual:   {actual}\n  expected: {expect}"
        )


def check_both_raise(
    query: str,
    tables: dict[str, Table],
    match: str | None = None,
) -> None:
    """Assert BOTH DataFusion and the active backend reject `query` (at build or
    execution). If `match` is given, each engine's error message must contain
    that regex."""
    for runner in (_run_datafusion, _run_backend):
        try:
            runner(query, tables)
        except UnsupportedInCodegen as e:
            # A deferred surface is not a rejection -- don't let it pass as one.
            pytest.skip(f"{_backend} defers this surface: {e}")
        except Exception as e:  # noqa: BLE001 -- differential harness, any error counts
            if match is not None and not re.search(match, str(e)):
                raise AssertionError(
                    f"{runner.__name__} raised {e!r}, expected match {match!r}"
                ) from e
            continue
        raise AssertionError(f"{runner.__name__} did not raise for query: {query}")
```

- [ ] **Step 3: Add the backend fixture**

Create `tests/conftest.py`:

```python
"""Run every differential test once per serving engine.

The diff tests call differential.check(), which compares ONE engine against the
DataFusion oracle. Rather than touch 80 call sites, parametrize the modules that
use the harness over the available backends and let the fixture point check() at
each in turn -- so "rust" and "codegen" are each proven against the same oracle,
and appear as separate test IDs when one breaks.

autouse is load-bearing and NOT stylistic. A non-autouse fixture named by
`metafunc.fixturenames.append(...)` produces the parametrized IDs but is NEVER
INSTANTIATED -- measured 2026-07-17: the ID said "codegen" while the engine in
use was still "rust", so the whole suite ran the rust engine twice and reported a
green bar for an engine that never executed. autouse forces instantiation.
Modules that aren't parametrized have no request.param and fall through.
"""

from __future__ import annotations

import differential
import pytest

_HARNESS_MODULES = ("test_diff_", "test_differential")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if metafunc.module.__name__.startswith(_HARNESS_MODULES):
        metafunc.parametrize("_backend", list(differential.BACKENDS), indirect=True)


@pytest.fixture(autouse=True)
def _backend(request: pytest.FixtureRequest):
    param = getattr(request, "param", None)
    if param is None:  # not a harness module -- leave the default alone
        yield None
        return
    differential.set_backend(param)
    yield param
    differential.set_backend("rust")
```

- [ ] **Step 4: Prove the backend actually switches (do NOT skip or soften this)**

Counting tests is **not** a valid check. This exact wiring has already produced a fully fake green once: the IDs said `codegen`, the count doubled 106 → 186, everything passed, and the codegen engine never executed a single time. Only an assertion made from *inside* a test body proves which engine ran.

Create `tests/test_backend_wiring.py`:

```python
"""Guard the two-backend wiring itself.

A parametrized ID is not evidence the backend switched: a non-autouse fixture
yields correct-looking IDs while never running, so the suite silently exercises
one engine twice and reports green. This test fails in exactly that case.
"""

from __future__ import annotations

import differential


def test_diff_backend_fixture_actually_switches_the_engine(request):
    expected = request.node.callspec.params["_backend"]
    actual = differential._backend
    assert actual == expected, (
        f"backend wiring is broken: test ID says {expected!r} but the engine in "
        f"use is {actual!r} — the suite is testing one engine twice"
    )
```

Note the filename starts with `test_diff_`-adjacent naming *deliberately*: it must match `_HARNESS_MODULES` so it gets parametrized. `test_backend_wiring` does NOT match `("test_diff_", "test_differential")` — so add `"test_backend_wiring"` to `_HARNESS_MODULES` in `tests/conftest.py`:

```python
_HARNESS_MODULES = ("test_diff_", "test_differential", "test_backend_wiring")
```

Run: `uv run pytest tests/test_backend_wiring.py -v`
Expected: **2 tests, both PASS** — `[rust]` and `[codegen]`. If `[codegen]` FAILS with "engine in use is 'rust'", the fixture is not being instantiated: confirm `autouse=True` is present. Do not continue until both pass.

Then sanity-check the count as a *secondary* signal only:

Run: `uv run pytest tests/ -q --collect-only 2>&1 | tail -2`
Expected: 106 + 80 + 2 = **188** collected (80 harness tests now run twice, plus the 2 wiring tests).

- [ ] **Step 5: Run the suite and triage**

Run: `uv run pytest tests/ -q -rs`
Expected: all `rust` tests PASS. `codegen` tests: expression/type cases PASS; relational cases (joins) SKIP with "codegen defers this surface: cannot compile Join/CrossJoin/LookupJoin". `-rs` prints the skip reasons — read them and confirm every skip is a *join* or a *container*, not something codegen should already handle.

Any codegen FAILURE here is a real semantic divergence. Fix it in `_codegen_runtime.py` or `_codegen_plan.py` — the oracle is right.

- [ ] **Step 6: Commit**

```bash
mise run fmt
git add tests/differential.py tests/conftest.py
git commit -m "test: run the differential suite against both serving backends"
```

---

### Task 9: Emitter — cross joins, inner joins, lookup joins

**Files:**
- Modify: `sql_transform/_codegen.py` (`_emit_rel`, `CodegenFn.__init__`)
- Test: `sql_transform/_codegen_test.py` (append)

**Interfaces:**
- Consumes: Tasks 6–8. Uses `cp.LookupSpec` from Task 6 and `rt.key`/`rt.join_eq`/`rt.miss` from Tasks 1–3.
- Produces: no new public API — `CodegenFn` gains join support, and builds its lookup indexes from the `optimize` specs.

- [ ] **Step 1: Write the failing tests**

Append to `sql_transform/_codegen_test.py`:

```python
class Key(BaseModel):
    k: int
    a: int = 1


def _lookup_table():
    return pa.table({"k": pa.array([1, 2], type=pa.int64()), "v": pa.array([10.0, 20.0])})


def test_cross_join_is_a_cartesian_product():
    class L(BaseModel):
        a: int

    class R(BaseModel):
        b: int

    fn = CodegenFn("SELECT a AS x, b AS y FROM l CROSS JOIN r", {"l": L, "r": R}, {})
    out = fn.infer({"l": [L(a=1), L(a=2)], "r": [R(b=9)]})
    assert [(r.x, r.y) for r in out] == [(1, 9), (2, 9)]


def test_inner_join_matches_on_keys():
    class L(BaseModel):
        k: int
        a: int

    class R(BaseModel):
        k: int
        b: int

    fn = CodegenFn("SELECT a AS x, b AS y FROM l JOIN r ON l.k = r.k", {"l": L, "r": R}, {})
    out = fn.infer({"l": [L(k=1, a=1), L(k=2, a=2)], "r": [R(k=2, b=9)]})
    assert [(r.x, r.y) for r in out] == [(2, 9)]


def test_inner_join_never_matches_null_keys():
    class L(BaseModel):
        k: int | None
        a: int

    class R(BaseModel):
        k: int | None
        b: int

    fn = CodegenFn("SELECT a AS x, b AS y FROM l JOIN r ON l.k = r.k", {"l": L, "r": R}, {})
    assert fn.infer({"l": [L(k=None, a=1)], "r": [R(k=None, b=9)]}) == []


def test_lookup_join_binds_the_matching_static_row():
    fn = CodegenFn(
        "SELECT v AS x FROM t JOIN s ON t.k = s.k", {"t": Key}, {"s": _lookup_table()}
    )
    assert [r.x for r in fn.infer({"t": [Key(k=2)]})] == [20.0]


def test_inner_lookup_join_miss_raises_key_error():
    fn = CodegenFn(
        "SELECT v AS x FROM t JOIN s ON t.k = s.k", {"t": Key}, {"s": _lookup_table()}
    )
    with pytest.raises(KeyError, match="No row in static table 's'"):
        fn.infer({"t": [Key(k=99)]})


def test_left_lookup_join_miss_yields_nulls_and_a_nullable_output():
    fn = CodegenFn(
        "SELECT v AS x FROM t LEFT JOIN s ON t.k = s.k", {"t": Key}, {"s": _lookup_table()}
    )
    assert fn.output_model.model_fields["x"].annotation == typing.Optional[float]
    assert [r.x for r in fn.infer({"t": [Key(k=99)]})] == [None]


def test_lookup_join_keys_are_type_strict():
    # Value::Int(1) and Value::Float(1.0) hash differently, so a float key must
    # not match an int key row -- Python's 1 == 1.0 would wrongly match.
    class FloatKey(BaseModel):
        k: float

    fn = CodegenFn(
        "SELECT v AS x FROM t LEFT JOIN s ON t.k = s.k",
        {"t": FloatKey},
        {"s": _lookup_table()},
    )
    assert [r.x for r in fn.infer({"t": [FloatKey(k=1.0)]})] == [None]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/_codegen_test.py -v -k "join"`
Expected: FAIL with `UnsupportedInCodegen: cannot compile CrossJoin`

- [ ] **Step 3: Add the join branches to the emitter**

In `sql_transform/_codegen.py`, add these branches to `_emit_rel`, immediately before the final `else: raise UnsupportedInCodegen(...)`:

```python
    elif isinstance(node, cp.CrossJoin):

        def crossed(inner: dict, i: int) -> None:
            _emit_rel(node.right, inner, i, em, body)

        _emit_rel(node.left, env, ind, em, crossed)
    elif isinstance(node, cp.Join):

        def left_done(env_l: dict, i: int) -> None:
            def right_done(env_r: dict, j: int) -> None:
                conds = " and ".join(
                    f"rt.join_eq({_emit_expr(le, env_r)}, {_emit_expr(re, env_r)})"
                    for le, re in node.on
                )
                em.line(j, f"if {conds}:")
                body(env_r, j + 1)

            _emit_rel(node.right, env_l, i, em, right_done)

        _emit_rel(node.left, env, ind, em, left_done)
    elif isinstance(node, cp.LookupJoin):

        def looked_up(inner: dict, i: int) -> None:
            keys = ", ".join(f"rt.key({_emit_expr(k, inner)})" for k in node.keys)
            k = em.var("_k")
            h = em.var("_h")
            em.line(i, f"{k} = ({keys},)")
            em.line(i, f"{h} = _lookups[{node.table!r}].get({k})")
            em.line(i, f"if {h} is None:")
            if node.outer:
                em.line(i + 1, f"{h} = _nullrows[{node.table!r}]")
            else:
                em.line(i + 1, f"raise KeyError(rt.miss({node.table!r}, {k}))")
            body({**inner, node.table: h}, i)

        _emit_rel(node.input, env, ind, em, looked_up)
```

- [ ] **Step 4: Build the lookup indexes in the engine**

In `sql_transform/_codegen.py`, add this module-level function:

```python
def _build_index(table: Any, key_columns: list) -> tuple:
    """Index a static table by its type-tagged key tuple (mirrors lookup.rs).
    Also returns the all-NULL value row a LEFT lookup miss binds."""
    value_columns = [c for c in table.column_names if c not in key_columns]
    index = {}
    for row in table.to_pylist():
        key = tuple(rt.key(row[c]) for c in key_columns)
        index[key] = {c: row[c] for c in value_columns}
    return index, dict.fromkeys(value_columns)
```

Then in `CodegenFn.__init__`, capture the lookup specs — change:

```python
        # Lookup specs go unused until joins land (Task 9).
        plan, _ = cp.optimize(plan, set(static_tables))
```

to:

```python
        plan, specs = cp.optimize(plan, set(static_tables))
```

and replace the two empty-dict lines:

```python
        self._lookups: dict = {}
        self._nullrows: dict = {}
```

with:

```python
        self._lookups: dict = {}
        self._nullrows: dict = {}
        for spec in specs:
            table = static_tables.get(spec.static_table)
            if table is None:
                raise ValueError(
                    f"SQL references static table '{spec.static_table}' that was not provided"
                )
            self._lookups[spec.static_table], self._nullrows[spec.static_table] = _build_index(
                table, spec.key_columns
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest sql_transform/_codegen_test.py -v`
Expected: PASS

- [ ] **Step 6: Run the differential suite — the relational skips should be gone**

Run: `uv run pytest tests/ -q -rs`
Expected: all `rust` PASS; `codegen` PASS except container/UNNEST cases, which still SKIP. No codegen failures.

- [ ] **Step 7: Commit**

```bash
mise run fmt
git add sql_transform/_codegen.py sql_transform/_codegen_test.py
git commit -m "feat: codegen engine — cross, inner and lookup joins"
```

---

### Task 10: Pin the deferred surface so gaps can't hide

**Files:**
- Create: `tests/test_codegen_coverage.py`
- Modify: `docs/superpowers/specs/2026-07-17-codegen-inferfn-design.md` (status banner)

**Interfaces:**
- Consumes: everything above.
- Produces: a test asserting codegen skips *only* the deferred container surface.

A skip is invisible in a green bar. Without this, codegen could quietly stop supporting something and look fine. This test fails if the skip set grows.

- [ ] **Step 1: Write the failing test**

Create `tests/test_codegen_coverage.py`:

```python
"""Guard the codegen engine's deferred surface.

Skips are how the harness handles what codegen defers (containers/UNNEST). That
is fine only while the skip set is exactly the deferred surface and nothing has
quietly fallen out of coverage -- this test is what makes that true.
"""

from __future__ import annotations

import pytest
from differential import _run_codegen, rows, static

from sql_transform._codegen import UnsupportedInCodegen

# Every committed-surface shape. None of these may raise UnsupportedInCodegen.
_COMMITTED = [
    ("SELECT a AS x FROM t", {"t": rows({"a": "int"}, [{"a": 1}])}),
    ("SELECT a + 1 AS x FROM t", {"t": rows({"a": "int"}, [{"a": 1}])}),
    # NB: unary minus (`-a`, `-1`) and `||` are deliberately absent -- measured
    # 2026-07-17, the Rust engine REJECTS both while DataFusion evaluates them, so
    # they are not committed surface. See the spec's oracle-vs-Rust section.
    ("SELECT a / b AS x FROM t", {"t": rows({"a": "int", "b": "int"}, [{"a": 7, "b": 2}])}),
    ("SELECT a % b AS x FROM t", {"t": rows({"a": "int", "b": "int"}, [{"a": 7, "b": 2}])}),
    ("SELECT NOT (a > 1) AS x FROM t", {"t": rows({"a": "int"}, [{"a": 1}])}),
    ("SELECT a > 1 AND a < 5 AS x FROM t", {"t": rows({"a": "int"}, [{"a": 1}])}),
    ("SELECT UPPER(s) AS x FROM t", {"t": rows({"s": "str"}, [{"s": "a"}])}),
    ("SELECT LOWER(s) AS x FROM t", {"t": rows({"s": "str"}, [{"s": "A"}])}),
    ("SELECT TRIM(s) AS x FROM t", {"t": rows({"s": "str"}, [{"s": " a "}])}),
    ("SELECT SUBSTR(s, 1, 2) AS x FROM t", {"t": rows({"s": "str"}, [{"s": "abc"}])}),
    ("SELECT CONCAT(s, s) AS x FROM t", {"t": rows({"s": "str"}, [{"s": "a"}])}),
    ("SELECT ABS(a) AS x FROM t", {"t": rows({"a": "int"}, [{"a": -1}])}),
    ("SELECT ROUND(f) AS x FROM t", {"t": rows({"f": "float"}, [{"f": 1.5}])}),
    ("SELECT COALESCE(a, 0) AS x FROM t", {"t": rows({"a": "int?"}, [{"a": None}])}),
    ("SELECT NULLIF(a, 1) AS x FROM t", {"t": rows({"a": "int"}, [{"a": 1}])}),
    ("SELECT CAST(a AS VARCHAR) AS x FROM t", {"t": rows({"a": "int"}, [{"a": 1}])}),
    ("SELECT CAST(s AS BIGINT) AS x FROM t", {"t": rows({"s": "str"}, [{"s": "1"}])}),
    ("SELECT CAST(a AS DOUBLE) AS x FROM t", {"t": rows({"a": "int"}, [{"a": 1}])}),
    ("SELECT CAST(a AS BOOLEAN) AS x FROM t", {"t": rows({"a": "int"}, [{"a": 1}])}),
    ("SELECT a AS x FROM t WHERE a > 0", {"t": rows({"a": "int"}, [{"a": 1}])}),
    ("SELECT z.a AS x FROM t AS z", {"t": rows({"a": "int"}, [{"a": 1}])}),
    (
        "SELECT a AS x, b AS y FROM t CROSS JOIN u",
        {"t": rows({"a": "int"}, [{"a": 1}]), "u": rows({"b": "int"}, [{"b": 2}])},
    ),
    (
        "SELECT a AS x FROM t JOIN u ON t.k = u.k",
        {
            "t": rows({"k": "int", "a": "int"}, [{"k": 1, "a": 1}]),
            "u": rows({"k": "int"}, [{"k": 1}]),
        },
    ),
    (
        "SELECT v AS x FROM t JOIN s ON t.k = s.k",
        {
            "t": rows({"k": "int"}, [{"k": 1}]),
            "s": static({"k": "int", "v": "float"}, [{"k": 1, "v": 1.0}]),
        },
    ),
    (
        "SELECT v AS x FROM t LEFT JOIN s ON t.k = s.k",
        {
            "t": rows({"k": "int"}, [{"k": 9}]),
            "s": static({"k": "int", "v": "float"}, [{"k": 1, "v": 1.0}]),
        },
    ),
]


@pytest.mark.parametrize("query, tables", _COMMITTED, ids=lambda v: None)
def test_committed_surface_is_never_deferred(query, tables):
    """If this raises, codegen has silently dropped committed surface -- which
    the differential harness would otherwise report as a harmless skip."""
    try:
        _run_codegen(query, tables)
    except UnsupportedInCodegen as e:
        pytest.fail(f"committed surface must not be deferred: {query!r} raised {e}")


_DEFERRED = [
    ("SELECT unnest(l) AS x FROM t", {"t": rows({"l": "list[int]"}, [{"l": [1]}])}),
    ("SELECT s AS x FROM t", {"t": rows({"s": "struct{x:int}"}, [{"s": {"x": 1}}])}),
    ("SELECT l AS x FROM t", {"t": rows({"l": "list[int]"}, [{"l": [1]}])}),
]


@pytest.mark.parametrize("query, tables", _DEFERRED, ids=lambda v: None)
def test_deferred_surface_raises_rather_than_answering_wrongly(query, tables):
    with pytest.raises(UnsupportedInCodegen):
        _run_codegen(query, tables)
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_codegen_coverage.py -v`
Expected: PASS.

If a `_COMMITTED` case raises `UnsupportedInCodegen`, that is a real gap — fix the engine, do not delete the case. If a `_DEFERRED` case does *not* raise, codegen is answering where it should be refusing, which is worse than a gap: find out what it returned and whether it is correct.

- [ ] **Step 3: Verify the full suite on both backends**

Run: `mise run test`
Expected: all PASS. Skips only for the container/UNNEST cases on codegen.

Run: `uv run pytest tests/ -q -rs 2>&1 | grep -i skip`
Expected: every skip reason mentions a struct/list/unnest. Any other skip is a coverage hole — fix it, don't accept it.

- [ ] **Step 4: Update the spec status**

In `docs/superpowers/specs/2026-07-17-codegen-inferfn-design.md`, change the STATUS banner's first line from:

```markdown
> **STATUS: scope + front-end fork decided; plan written.** Front-end decided
```

to:

```markdown
> **STATUS: built — codegen engine shipped and proven equivalent on the committed
> surface (differential suite runs both backends). Containers/UNNEST deferred.**
> Front-end decided
```

- [ ] **Step 5: Commit**

```bash
mise run fmt
git add tests/test_codegen_coverage.py docs/superpowers/specs/2026-07-17-codegen-inferfn-design.md
git commit -m "test: pin the codegen engine's deferred surface"
```

---

---

### Task 11: Pin the Rust-engine bugs as xfail-on-rust differential tests

**Files:**
- Create: `tests/test_diff_rust_bugs.py`
- Modify: `tests/conftest.py` (add the `rust_bug` fixture)

**Interfaces:**
- Consumes: the `_backend` fixture (Task 8), `differential.check`.
- Produces: a `rust_bug(reason)` fixture that xfails the current test on the rust backend only.

Three cases where both engines run the query and return **different values**. Codegen matches the oracle and must PASS; rust is a known bug and is `xfail`-ed with `strict=True`, so the moment someone fixes the Rust engine the xfail flips to a failure and tells us to delete the marker. Per AmirHossein's process, the fix itself is the PM's ticket, not this plan's work.

- [ ] **Step 1: Add the `rust_bug` fixture**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def rust_bug(request):
    """Mark the current test xfail on the rust backend only.

    For cases where the Rust engine disagrees with the DataFusion oracle: codegen
    matches the oracle and passes, rust is a known bug tracked in BACKLOG.
    strict=True so that fixing the Rust engine turns the xpass into a failure --
    which is the reminder to delete the marker and the ticket.
    """

    def _mark(reason: str) -> None:
        if differential._backend == "rust":
            request.applymarker(pytest.mark.xfail(reason=reason, strict=True))

    return _mark
```

- [ ] **Step 2: Write the tests**

Create `tests/test_diff_rust_bugs.py`:

```python
"""Rust-engine bugs: cases where `infer` disagrees with `transform`.

Each of these violates the README's promise that both paths return identical
values -- a user gets a different answer at serving time than at batch time.
Codegen matches the DataFusion oracle and PASSES; the Rust engine is xfail-ed
(strict) pending its BACKLOG ticket. If one starts XPASSing, the Rust bug was
fixed: delete the rust_bug() call and close the ticket.

Surface Rust rejects outright (unary minus, `||`) is a different class of gap --
ticketed, but not testable here since both engines decline it.
"""

from __future__ import annotations

from differential import check, rows


def test_cast_float_to_varchar_keeps_the_decimal_point(rust_bug):
    rust_bug(
        "Rust bug: expr::display_value uses f64::to_string, rendering 1.0 as '1'; "
        "DataFusion gives '1.0'. transform/infer return different values."
    )
    check(
        "SELECT CAST(f AS VARCHAR) AS x FROM t",
        {"t": rows({"f": "float"}, [{"f": 1.0}])},
        expect=[{"x": "1.0"}],
    )


def test_round_of_an_int_returns_a_float(rust_bug):
    rust_bug(
        "Rust bug: eval_builtin's ROUND passes an int through unchanged, returning "
        "int 3; DataFusion returns float 3.0. transform/infer differ in output TYPE."
    )
    check(
        "SELECT ROUND(a) AS x FROM t",
        {"t": rows({"a": "int"}, [{"a": 3}])},
        expect=[{"x": 3.0}],
    )


def test_nullif_compares_numerically_across_int_and_float(rust_bug):
    rust_bug(
        "Rust bug: NULLIF uses Value's variant-tagged PartialEq, so Int(1) != "
        "Float(1.0) and it returns 1; DataFusion coerces and returns NULL."
    )
    check(
        "SELECT NULLIF(a, 1.0) AS x FROM t",
        {"t": rows({"a": "int"}, [{"a": 1}])},
        expect=[{"x": None}],
    )
```

- [ ] **Step 3: Run and verify the xfail/pass split**

Run: `uv run pytest tests/test_diff_rust_bugs.py -v -rxX`
Expected: **3 XFAIL** (`[rust]`) and **3 PASSED** (`[codegen]`). Any other combination is a real signal:
- `[codegen]` FAILS → codegen does not match the oracle; fix codegen, the oracle is right.
- `[rust]` XPASSES (reported as a failure, since `strict=True`) → the Rust bug is fixed; delete that `rust_bug()` call and tell the PM to close the ticket.

- [ ] **Step 4: Commit**

```bash
mise run fmt
git add tests/test_diff_rust_bugs.py tests/conftest.py
git commit -m "test: pin three Rust-engine/DataFusion divergences as xfail-on-rust"
```

- [ ] **Step 5: Tell the PM to open the BACKLOG tickets**

Per AmirHossein's standing process, message the PM session (do not edit BACKLOG directly — it is PM-owned). Report all five, noting each violates the README's `transform`/`infer` parity promise:

1. `CAST(<float> AS VARCHAR)` — `infer` renders `1.0` as `'1'`, `transform` gives `'1.0'`. (`expr::display_value` → `f64::to_string`.) Also affects `CONCAT`. Note `1e300` renders as 300 digits vs DataFusion's `'1e300'`.
2. `ROUND(<int>)` — `infer` returns int `3`, `transform` returns float `3.0`. Output **type** differs.
3. `NULLIF(1, 1.0)` — `infer` returns `1`, `transform` returns `NULL`. (Variant-tagged `Value` equality.)
4. Unary minus — `infer` rejects `-a` / `-1` with `Unsupported expression`; `transform` evaluates it.
5. `||` string concat — `infer` rejects with `Unsupported operator`; `transform` evaluates it.

State that 1–3 are pinned by `tests/test_diff_rust_bugs.py` (xfail-on-rust, strict) so a fix auto-surfaces, and that 4–5 are surface gaps with no test (both engines decline them).

---

## Done means

- `mise run test` green, with the differential suite running **both** engines against the DataFusion oracle — proven by `tests/test_backend_wiring.py`, **not** by the test count.
- Codegen skips **only** struct/list/UNNEST, and `tests/test_codegen_coverage.py` fails if that ever grows.
- The three Rust value-divergences are xfail-pinned (strict) and all five are ticketed with the PM.
- No `SQLTransform` default changed, no engine selection wired — those wait on the framing calls in the spec's "Open questions".

## Follow-ups (not this plan)

- Specialize emission off `infer_type` (a statically-known non-null `int + int` should emit `a + b`, not `rt.add(a, b)`) — the actual speed work, gated on a benchmark.
- Containers: struct/list/`FieldAccess`/`UNNEST` → full `InferFn` equivalence.
- Vectorized/columnar path; numpy-matrix output; `CASE WHEN`.
- Engine selection + default policy on `SQLTransform`.
