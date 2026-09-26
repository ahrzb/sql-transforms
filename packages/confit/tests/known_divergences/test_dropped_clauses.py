"""Clauses and modifiers parsed then silently dropped.

See README.md for what
belongs here (kept behaviour + its ground) versus in
../test_open_divergences.py (behaviour we intend to change).
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from confit import DuckDBInferFn

# ------------------------------------------- silently discarded clauses --
#
# A clause the frontend parses but does not look at — `select.qualify`, the
# FETCH or TOP spellings of LIMIT — would be dropped, so every input row would
# be emitted. Worse than a wrong row count: `one_row_blocker` would see no
# Filter node, so `shape='map'` — the exactly-one-row-out-per-row-in PROOF —
# would build and certify a query whose entire purpose is to drop rows.

_QUAL_ROWS = [(1, 1), (1, 2), (2, 5)]
QUAL_SCHEMA = pa.schema(
    [
        pa.field("k", pa.int64(), nullable=False),
        pa.field("ts", pa.int64(), nullable=False),
    ]
)


def _run(sql: str, schema: pa.Schema, rows: list[dict]) -> list[tuple]:
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={})
    return [tuple(r.values()) for r in fn.infer_rows(rows)]


# The resolution is REFUSAL, which is half the contract: match DuckDB or
# refuse by name. Ignoring the clause would be the third mode that is not
# supposed to exist.
#
# It holds as a class, not per instance: `refuse_unhandled_query` and
# `refuse_unhandled_select` destructure their AST node EXHAUSTIVELY, with no
# `..` pattern, so a clause added to sqlparser breaks the build instead of the
# answers (`Select::flavor` included).


@pytest.mark.parametrize(
    ("sql", "match"),
    [
        (
            "SELECT k, ts FROM __THIS__ "
            "QUALIFY row_number() OVER (PARTITION BY k ORDER BY ts DESC) = 1",
            "QUALIFY",
        ),
        ("SELECT k, ts FROM __THIS__ FETCH FIRST 1 ROWS ONLY", "FETCH"),
        ("SELECT TOP 1 k, ts FROM __THIS__", "TOP"),
        ("SELECT k, ts FROM __THIS__ LIMIT 1", "LIMIT"),
        ("SELECT k, ts FROM __THIS__ QUALIFY k > 1", "QUALIFY"),
    ],
)
def test_row_limiting_clauses_are_refused_not_dropped(sql, match):
    """Each of these would emit every input row if dropped. LIMIT is the
    control: it is refused by name, and the others are its synonyms."""
    with pytest.raises(ValueError, match=match):
        _run(sql, QUAL_SCHEMA, [{"k": k, "ts": t} for k, t in _QUAL_ROWS])


def test_ordinary_query_still_builds(oracle):
    """The exhaustive destructure refuses what it does not handle, so the
    risk is refusing an ordinary query."""
    sql = "SELECT k, ts FROM __THIS__ WHERE k = 1"
    got = _run(sql, QUAL_SCHEMA, [{"k": k, "ts": t} for k, t in _QUAL_ROWS])
    oracle.table("__THIS__", "k BIGINT, ts BIGINT", _QUAL_ROWS)
    assert got == oracle.execute(sql).fetchall()


# The SAME class for table names: a `TableFactor::Table { name, alias, .. }`
# pattern swallows every modifier sqlparser can hang off a table name, so
# `TABLESAMPLE 3 ROWS` would be dropped: DuckDB returns 3 rows, a dropping
# engine all 20 — under shape='map', whose one-row-out-per-row-in certificate
# the dropped clause satisfies.
#
# `plain_table` is the only `TableFactor::Table` pattern in the frontend, it
# destructures exhaustively, and all three relation positions call it.
#
# Of the remaining `..` patterns in sqlparser destructures, the only real
# modifiers are CAST's `array` and `format`, both refused below. The rest drop
# formatting-only fields (`Case`'s attached tokens, `Substring`'s
# `special`/`shorthand`) or sit in arms that already refuse the whole node.

_SAMPLE_ROW = pa.schema([pa.field("a", pa.int64(), nullable=False)])
_SAMPLE_STATIC = pa.table(
    {"c0": pa.array([1], pa.int64()), "v": pa.array([9], pa.int64())}
)


@pytest.mark.parametrize(
    "sql",
    [
        # the driving relation, a JOINed relation, and a comma-joined one:
        # three separate destructure sites, one shared refusal
        "SELECT a FROM __THIS__ TABLESAMPLE 3 ROWS",
        "SELECT s.v AS o FROM __THIS__ JOIN s TABLESAMPLE 1 ROWS ON a = s.c0",
        "SELECT s.v AS o FROM __THIS__, s TABLESAMPLE 1 ROWS WHERE a = s.c0",
    ],
)
def test_tablesample_is_refused_at_every_relation_position(sql):
    with pytest.raises(ValueError, match="TABLESAMPLE"):
        DuckDBInferFn(
            sql,
            row_tables={"__THIS__": _SAMPLE_ROW},
            static_tables={"s": _SAMPLE_STATIC},
        )


@pytest.mark.parametrize(
    ("sql", "match"),
    [
        ("SELECT a FROM __THIS__ WITH ORDINALITY", "WITH ORDINALITY"),
        ("SELECT CAST(a AS BIGINT ARRAY) AS o FROM __THIS__", "ARRAY"),
        ("SELECT CAST(a AS BIGINT FORMAT 'x') AS o FROM __THIS__", "FORMAT"),
    ],
)
def test_the_rest_of_the_audit_refuses_by_name(sql, match):
    """The other modifiers the two exhaustive destructures see. None is
    DuckDB syntax we serve."""
    with pytest.raises(ValueError, match=match):
        DuckDBInferFn(sql, row_tables={"__THIS__": _SAMPLE_ROW}, static_tables={})


def test_unmodified_relations_still_build():
    """Control for the exhaustive destructure: the bare spellings of all three
    relation positions still bind."""
    fn = DuckDBInferFn(
        "SELECT s.v AS o FROM __THIS__ JOIN s ON a = s.c0",
        row_tables={"__THIS__": _SAMPLE_ROW},
        static_tables={"s": _SAMPLE_STATIC},
    )
    assert fn.infer_rows([{"a": 1}]) == [{"o": 9}]


# A call-node modifier that DuckDB refuses -- OVER () on a scalar function,
# FILTER, IGNORE NULLS, WITHIN GROUP -- would, if dropped, serve the bare call
# where the oracle errors. One screen runs where every scalar call
# dispatches, builtin and udf alike.

_MOD_SCHEMA = pa.schema(
    [
        pa.field("k", pa.int64(), nullable=False),
        pa.field("s", pa.string(), nullable=False),
    ]
)


class _ModUdf:
    name = "mudf"
    takes = pa.schema([("x", pa.float64())])
    returns = pa.float64()

    def __call__(self, x):
        return (x,)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT abs(k) OVER () AS c FROM __THIS__",
        "SELECT abs(k) FILTER (WHERE TRUE) AS c FROM __THIS__",
        # both IGNORE NULLS spellings: in-paren parses to the argument
        # list's clauses, post-paren to the call node's null_treatment
        "SELECT upper(s IGNORE NULLS) AS c FROM __THIS__",
        "SELECT ltrim(lower(s) IGNORE NULLS) AS c FROM __THIS__",
        "SELECT coalesce(k, 1) OVER () AS c FROM __THIS__",
        "SELECT mudf(1.0e0) OVER () AS c FROM __THIS__",
        "SELECT mudf(1.0e0) FILTER (WHERE TRUE) AS c FROM __THIS__",
    ],
)
def test_scalar_call_modifiers_are_refused_not_dropped(sql):
    """DuckDB refuses each of these (CatalogException for OVER on a scalar,
    InvalidInput for FILTER, ParserException for IGNORE NULLS); serving the
    bare call would be a wrong answer."""
    with pytest.raises(ValueError, match="modifier|argument clauses"):
        DuckDBInferFn(
            sql,
            row_tables={"__THIS__": _MOD_SCHEMA},
            static_tables={},
            udfs=[_ModUdf()],
        )


def test_unmodified_scalar_calls_still_build():
    """The screen must not catch the bare spellings of the same calls."""
    fn = DuckDBInferFn(
        "SELECT abs(k) AS a, upper(s) AS u, mudf(1.0e0) AS m FROM __THIS__",
        row_tables={"__THIS__": _MOD_SCHEMA},
        static_tables={},
        udfs=[_ModUdf()],
    )
    rows = fn.infer_rows([{"k": -2, "s": "x"}])
    assert rows == [{"a": 2, "u": "X", "m": 1.0}]
