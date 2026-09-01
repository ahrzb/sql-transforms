"""Clauses and modifiers parsed then silently dropped.

Split out of test_known_divergences.py 2026-08-16; see README.md for what
belongs here (kept behaviour + its ground) versus in
../test_open_divergences.py (behaviour we intend to change).
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from confit import DuckDBInferFn

# ------------------------------------------- silently discarded clauses --
#
# The frontend validates `query.with`, `order_by`, `limit_clause`,
# `select.distinct`, `group_by` and `having` — and refuses LIMIT by name. It
# never looks at `select.qualify`, nor at the FETCH spelling of LIMIT. Those
# two clauses are parsed and dropped, so every input row is emitted.
#
# Worse than a wrong row count: `one_row_blocker` sees no Filter node, so
# `shape='map'` — the exactly-one-row-out-per-row-in PROOF — also builds and
# certifies a query whose entire purpose is to drop rows.
#
# Reproduced by hand 2026-08-08.

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


# FIXED 2026-08-08. The resolution is REFUSAL, which is half the
# contract: match DuckDB or refuse by name. Ignoring the clause was the third
# mode that is not supposed to exist.
#
# Fixed as a class, not as two instances: `refuse_unhandled_query` and
# `refuse_unhandled_select` destructure their AST node EXHAUSTIVELY, with no
# `..` pattern, so a clause added to sqlparser breaks the build instead of the
# answers. That immediately caught `Select::flavor`, which this audit had
# missed by hand.


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
    """Each of these silently emitted every input row before the exhaustive
    destructure landed. LIMIT is the control: it was always refused, and the
    others are its synonyms."""
    with pytest.raises(ValueError, match=match):
        _run(sql, QUAL_SCHEMA, [{"k": k, "ts": t} for k, t in _QUAL_ROWS])


def test_ordinary_query_still_builds(oracle):
    """The audit refuses by exhaustive destructure, so the risk is refusing
    something that used to work."""
    sql = "SELECT k, ts FROM __THIS__ WHERE k = 1"
    got = _run(sql, QUAL_SCHEMA, [{"k": k, "ts": t} for k, t in _QUAL_ROWS])
    oracle.table("__THIS__", "k BIGINT, ts BIGINT", _QUAL_ROWS)
    assert got == oracle.execute(sql).fetchall()


# 2026-08-16: the SAME class, leaking through the one struct the doctrine above
# was never applied to. `TableFactor::Table { name, alias, .. }` sat at three
# sites and swallowed every modifier sqlparser can hang off a table name, so
# `TABLESAMPLE 3 ROWS` was parsed and dropped: DuckDB returned 3 rows, we
# returned all 20 — again under shape='map', whose one-row-out-per-row-in
# certificate the dropped clause satisfied.
#
# Fixed the same way, as a class rather than as one field: `plain_table` is now
# the only `TableFactor::Table` pattern in the frontend, it destructures
# exhaustively, and all three relation positions call it. There is nothing left
# to forget.
#
# The follow-up audit — every remaining `..` in a sqlparser destructure — turned
# up exactly one more pair of real modifiers: CAST's `array` and `format`, both
# refused below. The rest drop formatting-only fields (`Case`'s attached tokens,
# `Substring`'s `special`/`shorthand`) or sit in arms that already refuse the
# whole node.

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
    """The other modifiers the two exhaustive destructures now see. Each was
    silently dropped before; none is DuckDB syntax we serve."""
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


# Fuzz campaign 2026-08-11, ~570 of 963 findings. A call-node
# modifier that DuckDB refuses -- OVER () on a scalar function, FILTER, IGNORE
# NULLS -- was parsed and silently DROPPED on the builtin path, so the bare
# call was served where the oracle errors. The udf path already screened
# FILTER / IGNORE NULLS / WITHIN GROUP (frontend review round) but not OVER;
# the builtin path screened nothing. One screen now runs where every scalar
# call dispatches.

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
        # list's clauses (already refused before this fix), post-paren to
        # the call node's null_treatment (silently dropped before it)
        "SELECT upper(s IGNORE NULLS) AS c FROM __THIS__",
        "SELECT ltrim(lower(s) IGNORE NULLS) AS c FROM __THIS__",
        "SELECT coalesce(k, 1) OVER () AS c FROM __THIS__",
        "SELECT mudf(1.0e0) OVER () AS c FROM __THIS__",
        "SELECT mudf(1.0e0) FILTER (WHERE TRUE) AS c FROM __THIS__",
    ],
)
def test_scalar_call_modifiers_are_refused_not_dropped(sql):
    """DuckDB refuses each of these (CatalogException for OVER on a scalar,
    InvalidInput for FILTER, ParserException for IGNORE NULLS); before the
    fix the engine built them and served the bare call."""
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
