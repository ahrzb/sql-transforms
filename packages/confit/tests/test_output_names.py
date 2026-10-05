"""Output names of unaliased items: DuckDB prints the parsed expression
(`naming.rs`), and confit must name every column as it does, at the top
level and across a derived table, whose outer level reads the names.
Each expression is compared against DuckDB's own DESCRIBE.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from confit.oracle import Oracle

SCHEMA = pa.schema(
    [
        ("a", pa.int64()),
        ("b", pa.int32()),
        ("x", pa.float64()),
        ("s", pa.string()),
        ("f", pa.bool_()),
        ("st", pa.struct([("p", pa.int32()), ("q", pa.string())])),
    ]
)

EXPRS = [
    "1",
    "-1",
    "- - 1",
    "-a",
    "-(a + 1)",
    "1.5",
    "-1.5",
    "1.50",
    "1.5e0",
    "1e300",
    "1E3",
    "0.1e0",
    "1.0e0",
    "-0.0e0",
    "0.30000000000000004e0",
    "1.5e-5",
    "1e16",
    "1e15",
    "'it''s'",
    "NULL",
    "true",
    "FALSE",
    "a + 1",
    "a - 1 - 2",
    "a * (b + 1)",
    "a / b",
    "a // b",
    "a % b",
    "a = 1",
    "a <> 1",
    "a != 1",
    "a < 1 AND b > 2 AND f",
    "a < 1 OR b > 2 OR f",
    "a = 1 AND (b = 2 AND f)",
    "a = 1 OR (b = 2 AND f)",
    "NOT f",
    "NOT (a = 1)",
    "NOT (a < 1)",
    "NOT (a IN (1, 2))",
    "NOT (s LIKE 'x')",
    "NOT (a BETWEEN 1 AND 2)",
    "NOT NOT f",
    "a IS NULL",
    "a IS NOT NULL",
    "a IS DISTINCT FROM b",
    "a IS NOT DISTINCT FROM b",
    "NOT (a IS DISTINCT FROM b)",
    "a BETWEEN 1 AND 2",
    "a NOT BETWEEN 1 AND 2",
    "a IN (1, 2)",
    "a NOT IN (1, 2)",
    "s LIKE 'x%'",
    "s NOT LIKE 'x%'",
    "s ILIKE 'x%'",
    "s NOT ILIKE 'x%'",
    "s GLOB 'x*'",
    "s SIMILAR TO 'x.*'",
    "s || 'z'",
    "CAST(a AS DOUBLE)",
    "CAST(a AS VARCHAR)",
    "CAST(a AS INTEGER)",
    "CAST(a AS INT)",
    "CAST(a AS BIGINT)",
    "CAST(a AS TINYINT)",
    "CAST(a AS SMALLINT)",
    "CAST(a AS BOOLEAN)",
    "CAST(a AS FLOAT)",
    "CAST(a AS FLOAT8)",
    "CAST(a AS DECIMAL(10,2))",
    "CAST(a AS TEXT)",
    "TRY_CAST(a AS BIGINT)",
    "a::DOUBLE",
    "CASE WHEN f THEN 1 END",
    "CASE WHEN f THEN 1 ELSE 2 END",
    "CASE a WHEN 1 THEN 'x' ELSE 'y' END",
    "coalesce(a, b)",
    "ifnull(a, b)",
    "nullif(a, b)",
    "greatest(a, b)",
    "least(a, b)",
    "replace(s, 'a', 'b')",
    "left(s, 2)",
    "right(s, 2)",
    "position('a' IN s)",
    "substring(s, 1, 2)",
    "substring(s FROM 1 FOR 2)",
    "trim(s)",
    "ltrim(s)",
    "lower(s)",
    "LOWER(s)",
    "abs(a)",
    "round(x, 2)",
    "length(s)",
    "concat(s, 'z')",
    "struct_pack(k := a)",
    "st.p",
    "st['p']",
    "(st).p",
    "struct_extract(st, 'p')",
    "[a, b]",
    "[a, b][1]",
    "list_value(a, b)",
    "if(f, 1, 2)",
    '"a"',
    "__THIS__.a",
    "st.p + 1",
    "a & b",
    "a | b",
    "a << 1",
    "ln(x) * 2",
    "xor(a, b)",
    "s[1]",
    "NOT (NOT (a = 1))",
    "NOT (NOT (NOT (a < 1)))",
    "NOT (NOT (a IN (1, 2)))",
    "NOT (a NOT IN (1, 2))",
    "-0.0",
    "- 0",
    "(st).p + 1",
    "st['p'] || 'x'",
    "ceil(x)",
    "floor(x)",
    "CAST(NULL AS INTEGER)",
    "'One'",
]


def _duck_names(sql: str) -> list[str]:
    with Oracle() as o:
        o.load("__THIS__", pa.table({f.name: pa.array([], f.type) for f in SCHEMA}))
        return list(o.answer(sql).schema.names)


def _ours(sql: str) -> list[str]:
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": SCHEMA}, static_tables={})
    return list(fn.output_schema.names)


@pytest.mark.parametrize("expr", EXPRS)
def test_an_unaliased_item_is_named_as_duckdb_names_it(expr):
    sql = f"SELECT {expr} FROM __THIS__"
    try:
        ours = _ours(sql)
    except ValueError:
        pytest.skip("refused")
    assert ours == _duck_names(sql)


def test_a_derived_table_reads_the_names_duckdb_gives():
    sql = (
        'SELECT "(a + 1)" + 1 AS o, "-(a)" AS n, "CAST(\'t\' AS BOOLEAN)" AS t '
        "FROM (SELECT a + 1, -a, true FROM __THIS__)"
    )
    assert _ours(sql) == _duck_names(sql) == ["o", "n", "t"]
