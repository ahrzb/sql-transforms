"""Integer literals type like DuckDB's grammar types them.

An integer too big for INTEGER lexes as a numeric string, and the grammar
folds a unary minus into it: `-9223372036854775808` is the BIGINT minimum,
with spaces or parentheses in between too. Without the minus,
9223372036854775808 is HUGEINT, which confit does not serve.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity, table  # noqa: E402

ROWS = table({"b": "int", "i": "int32"}, [{"b": -9223372036854775808, "i": 3}])


@pytest.mark.parametrize(
    "expr",
    [
        "-9223372036854775808",
        "- 9223372036854775808",
        "-(9223372036854775808)",
        "b = -9223372036854775808",
        "17 IN (15, -9223372036854775808)",
        "i + -9223372036854775808",
        "-9223372036854775808 % 3",
        "-2147483648",
    ],
)
def test_the_bigint_minimum_literal_agrees(expr):
    assert_parity(f"SELECT {expr} AS o FROM __THIS__", ROWS, expect="AGREE")


def test_a_hugeint_literal_refuses_by_name():
    v = assert_parity(
        "SELECT 9223372036854775808 AS o FROM __THIS__", ROWS, expect="REFUSED"
    )
    assert "9223372036854775808" in v.detail


@pytest.mark.parametrize(
    "expr", ["- -9223372036854775808", "-(-(9223372036854775808))"]
)
def test_a_twice_negated_minimum_is_hugeint_and_refuses(expr):
    # DuckDB folds both minuses into the literal: 9223372036854775808 again.
    v = assert_parity(f"SELECT {expr} AS o FROM __THIS__", ROWS, expect="REFUSED")
    assert "HUGEINT" in v.detail
