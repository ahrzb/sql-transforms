"""The unification rule, pinned per operator family (docs/specs/type-unification.md).

DuckDB unifies operands by casting each one's FINISHED result to the common
type; it never re-types an operand's own operands. So `c0 * c0` over TINYINT
overflows at 127 whatever wider operand sits beside it, in every family that
unifies (fuzz seed 5008 and 13 more). Each family is pinned on both backends:
the overflowing row traps on both engines, and in-range rows answer the same
values at the same output type.
"""

import sys
from pathlib import Path

import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity  # noqa: E402

SCHEMA = pa.schema(
    [
        pa.field("c0", pa.int8()),
        pa.field("c1", pa.int64()),
        pa.field("c2", pa.int16()),
        pa.field("x", pa.float64()),
        pa.field("s", pa.string()),
    ]
)


def _table(rows):
    return pa.Table.from_pylist(rows, schema=SCHEMA)


N = "c0 * c0"  # TINYINT: overflows at c0 = -128

# One entry per family that unifies operand types; the narrow operand is a
# computed TINYINT (or SMALLINT) expression beside a wider column or literal.
ARMS = {
    "CASE, wider THEN": f"CASE WHEN c1 > 100 THEN c1 ELSE {N} END",
    "CASE, wider ELSE": f"CASE WHEN c1 < 100 THEN {N} ELSE c1 END",
    "CASE, folded condition": f"CASE WHEN FALSE THEN c1 ELSE {N} END",
    "CASE, DOUBLE arm": f"CASE WHEN c1 < 100 THEN {N} ELSE x END",
    "CASE, literal arm": f"CASE WHEN c1 > 0 THEN {N} ELSE 1000 END",
    "CASE, SMALLINT": "CASE WHEN c1 > 100 THEN c1 ELSE c2 * c2 END",
    "CASE, two narrow widths": f"CASE WHEN c1 > 0 THEN {N} ELSE c2 END",
    "coalesce": f"coalesce({N}, c1)",
    "coalesce, literal": f"coalesce({N}, 1000)",
    "coalesce, negation": "coalesce(-c0, c1)",
    "ifnull": f"ifnull({N}, c1)",
    "nullif": f"nullif({N}, c1)",
    "greatest": f"greatest({N}, c1)",
    "greatest, literal": f"greatest({N}, 300)",
    "least": "least(c1, c0 + c0)",
    "IN, narrow subject": f"{N} IN (c1, 3)",
    "IN, narrow element": f"c1 IN ({N}, 3)",
    "IN, literal list": f"{N} IN (1000, 2)",
    "BETWEEN, narrow subject": f"{N} BETWEEN c1 AND 10",
    "BETWEEN, narrow bound": f"c1 BETWEEN {N} AND 10",
    "comparison": f"{N} = c1",
    "comparison, literal": f"{N} = 1000",
    "IS DISTINCT FROM": f"{N} IS DISTINCT FROM c1",
    "arithmetic": f"{N} + c1",
    "arithmetic, literal": f"{N} + 1000",
    "arithmetic, DOUBLE": f"{N} * x",
    "CAST": f"CAST({N} AS BIGINT)",
    "concatenation": f"{N} || s",
}


# Both backends, both DuckDB readings, output types by the campaign's rule.
@pytest.mark.parametrize("expr", ARMS.values(), ids=ARMS.keys())
def test_a_narrow_arm_overflows_at_its_own_width(expr):
    rows = [{"c0": -128, "c1": 5, "c2": -32768, "x": 1.5, "s": "a"}]
    assert_parity(
        f"SELECT {expr} AS o FROM __THIS__", _table(rows), trap="Overflow|out of range"
    )


@pytest.mark.parametrize("expr", ARMS.values(), ids=ARMS.keys())
def test_in_range_rows_answer_at_the_unified_width(expr):
    rows = [
        {"c0": 3, "c1": 5, "c2": 7, "x": 1.5, "s": "a"},
        {"c0": None, "c1": 9, "c2": None, "x": None, "s": None},
        {"c0": 11, "c1": None, "c2": -4, "x": -2.0, "s": "b"},
    ]
    assert_parity(f"SELECT {expr} AS o FROM __THIS__", _table(rows), expect="AGREE")
