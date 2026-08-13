"""The progression metric: corpus replay with a pinned scoreboard.

Three outcomes per statement, zero failures required:

- MARGINALIZED — accepted, and the training-set round-trip invariant holds
  bit-exactly against the corpus table.
- REFUSED — a named MarginalizeError.
- FAILED — anything else (a gate mismatch, an unexpected exception). Always
  a bug; the tests assert this set is empty.

The scoreboard pins are the progression record: widening a future loop means
editing them upward in a reviewable diff (Confit's 550/678, for
marginalization).

Two corpus halves: window queries **mined from DuckDB's own test suite**
(``duckdb/test/sql/window/*.test``, every ``query`` block referencing
``empsalary``, table renamed to ``__THIS__``, query-level ORDER BY — test
scaffolding, refused by design in a row-at-a-time context — stripped via the
AST during mining), replayed against the suite's own ten rows; and a curated
set covering every family from the three loops so far.
"""

import datetime

import pyarrow as pa
import pytest

from sql_transform import MarginalizeError, marginalize
from sql_transform._projection_test import gate

# The empsalary table exactly as DuckDB's window suite creates it.
_D = datetime.date
EMPSALARY = pa.table(
    {
        "depname": pa.array(
            [
                "develop",
                "sales",
                "personnel",
                "sales",
                "personnel",
                "develop",
                "develop",
                "sales",
                "develop",
                "develop",
            ],
            type=pa.string(),
        ),
        "empno": pa.array([10, 1, 5, 4, 2, 7, 9, 3, 8, 11], type=pa.int64()),
        "salary": pa.array(
            [5200, 5000, 3500, 4800, 3900, 4200, 4500, 4800, 6000, 5200],
            type=pa.int32(),
        ),
        "enroll_date": pa.array(
            [
                _D(2007, 8, 1),
                _D(2006, 10, 1),
                _D(2007, 12, 10),
                _D(2007, 8, 8),
                _D(2006, 12, 23),
                _D(2008, 1, 1),
                _D(2008, 1, 1),
                _D(2007, 8, 1),
                _D(2006, 10, 1),
                _D(2007, 8, 15),
            ],
            type=pa.date32(),
        ),
    }
)

# --- mined from duckdb/test/sql/window/*.test (DuckDB 1.5.5 clone) -----------

MINED = [
    # test_basic_window.test
    "SELECT depname, empno, salary, sum(salary) OVER (PARTITION BY depname ORDER BY empno) FROM __THIS__",
    "SELECT sum(salary) OVER (PARTITION BY depname ORDER BY salary) AS ss FROM __THIS__",
    "SELECT row_number() OVER (PARTITION BY depname ORDER BY salary) AS rn FROM __THIS__",
    "SELECT empno, first_value(empno) OVER (PARTITION BY depname ORDER BY empno) AS fv FROM __THIS__",
    "SELECT depname, empno, last_value(empno) OVER (PARTITION BY depname ORDER BY empno ASC ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS fv FROM __THIS__",
    "SELECT depname, salary, dense_rank() OVER (PARTITION BY depname ORDER BY salary) FROM __THIS__",
    "SELECT depname, salary, rank() OVER (PARTITION BY depname ORDER BY salary) FROM __THIS__",
    "SELECT depname, min(salary) OVER (PARTITION BY depname ORDER BY salary, empno) AS m1, max(salary) OVER (PARTITION BY depname ORDER BY salary, empno) AS m2, avg(salary) OVER (PARTITION BY depname ORDER BY salary, empno) AS m3 FROM __THIS__",
    "SELECT depname, stddev_pop(salary) OVER (PARTITION BY depname ORDER BY salary, empno) AS s FROM __THIS__",
    "SELECT depname, covar_pop(salary, empno) OVER (PARTITION BY depname ORDER BY salary, empno) AS c FROM __THIS__",
    # test_evil_window.test
    "SELECT depname, sum(sum(salary)) OVER (PARTITION BY depname ORDER BY salary) FROM __THIS__ GROUP BY depname, salary",
    "SELECT empno, sum((salary * 2)) OVER (PARTITION BY depname ORDER BY empno) FROM __THIS__",
    "SELECT empno, (2 * sum(salary) OVER (PARTITION BY depname ORDER BY empno)) FROM __THIS__",
    "SELECT depname, ((sum(salary) * 100.0000) / sum(sum(salary)) OVER (PARTITION BY depname ORDER BY salary)) AS revenueratio FROM __THIS__ GROUP BY depname, salary",
    # test_invalid_window.test
    "SELECT list(salary ORDER BY enroll_date, salary) OVER (PARTITION BY depname) FROM __THIS__",
    # test_nthvalue.test
    "SELECT depname, empno, nth_value(empno, 2) OVER (PARTITION BY depname ORDER BY empno ASC ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS fv FROM __THIS__",
    "SELECT depname, empno, nth_value(empno, NULL) OVER (PARTITION BY depname ORDER BY empno ASC ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS fv FROM __THIS__",
    "SELECT depname, empno, nth_value(NULL, 2) OVER (PARTITION BY depname ORDER BY empno ASC ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS fv FROM __THIS__",
    "SELECT depname, empno, nth_value(empno, CASE  WHEN (((empno % 3) = 1)) THEN (2) ELSE NULL END) OVER (PARTITION BY depname ORDER BY empno ASC ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS fv FROM __THIS__",
    'SELECT depname, empno, (1 + (empno % 3)) AS "offset", nth_value(empno, (1 + (empno % 3))) OVER (PARTITION BY depname ORDER BY empno ASC ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS fv FROM __THIS__',
    'SELECT depname, empno, (empno % 3) AS "offset", nth_value(empno, (empno % 3)) OVER (PARTITION BY depname ORDER BY empno ASC ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS fv FROM __THIS__',
    "SELECT depname, empno, nth_value(-1, 2) OVER (PARTITION BY depname ORDER BY empno ASC ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS fv FROM __THIS__",
]

MINED_SCOREBOARD = {"marginalized": 11, "refused": 11}

# --- curated: one entry per family, all three loops --------------------------

CURATED_MARGINALIZED = [
    # loop 1: per-partition aggregates
    "SELECT (salary - avg(salary) OVER (PARTITION BY depname)) / stddev_samp(salary) OVER (PARTITION BY depname) AS z FROM __THIS__",
    "SELECT salary - avg(salary) OVER () AS c, depname FROM __THIS__",
    "SELECT avg(salary) OVER (PARTITION BY depname, enroll_date) AS m FROM __THIS__",
    "SELECT median(salary) OVER (PARTITION BY depname) AS m, empno FROM __THIS__",
    "SELECT *, avg(salary) OVER (PARTITION BY depname) AS m FROM __THIS__",
    "SELECT salary + 1 AS s1, depname FROM __THIS__",
    # loop 2: running windows, frames, rank family, value functions
    "SELECT sum(salary) OVER (PARTITION BY depname ORDER BY enroll_date) AS run FROM __THIS__",
    "SELECT avg(salary) OVER (ORDER BY empno) AS run FROM __THIS__",
    "SELECT sum(salary) OVER (PARTITION BY depname ORDER BY empno RANGE BETWEEN 2 PRECEDING AND CURRENT ROW) AS r FROM __THIS__",
    "SELECT sum(salary) OVER (PARTITION BY depname ORDER BY empno GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW) AS g FROM __THIS__",
    "SELECT sum(salary) OVER (PARTITION BY depname ORDER BY empno ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS w FROM __THIS__",
    "SELECT rank() OVER (PARTITION BY depname ORDER BY salary) AS r FROM __THIS__",
    "SELECT percent_rank() OVER (ORDER BY salary DESC NULLS FIRST) AS pr FROM __THIS__",
    "SELECT cume_dist() OVER (PARTITION BY depname ORDER BY salary) AS cd FROM __THIS__",
    "SELECT first_value(empno) OVER (PARTITION BY depname ORDER BY salary) AS f FROM __THIS__",
    "SELECT last_value(empno) OVER (PARTITION BY depname ORDER BY salary) AS l FROM __THIS__",
    "SELECT nth_value(empno, 2) OVER (PARTITION BY depname ORDER BY salary) AS n2 FROM __THIS__",
    "SELECT avg(salary) FILTER (WHERE empno > 3) OVER (PARTITION BY depname) AS fa FROM __THIS__",
    "SELECT count(DISTINCT salary) OVER (PARTITION BY depname) AS cds FROM __THIS__",
    "SELECT string_agg(depname, '|' ORDER BY empno) OVER () AS sa FROM __THIS__",
    "SELECT first(empno) OVER (PARTITION BY depname) AS f FROM __THIS__",
    "SELECT array_agg(empno) OVER (PARTITION BY depname) AS arr FROM __THIS__",
    "SELECT quantile_cont(salary, 0.25) OVER (PARTITION BY depname) AS q FROM __THIS__",
    "SELECT bool_and(salary > 4000) OVER (PARTITION BY depname) AS ba FROM __THIS__",
    "SELECT corr(salary, empno) OVER (PARTITION BY depname) AS c FROM __THIS__",
    "SELECT mode(depname) OVER (PARTITION BY enroll_date) AS md FROM __THIS__",
    "SELECT avg(salary) OVER (PARTITION BY substr(depname, 1, 1)) AS m FROM __THIS__",
    "SELECT avg(salary) OVER (PARTITION BY depname ORDER BY empno % 3) AS m FROM __THIS__",
    "SELECT avg(salary) OVER w AS m FROM __THIS__ WINDOW w AS (PARTITION BY depname)",
    # loop 3: chains and scalar subqueries
    "WITH a AS (SELECT salary + 1 AS s1, depname FROM __THIS__) SELECT s1 * 2 AS s2, depname FROM a",
    "WITH c AS (SELECT salary - avg(salary) OVER () AS cs, depname FROM __THIS__) SELECT cs / stddev_samp(cs) OVER (PARTITION BY depname) AS z FROM c",
    "WITH a AS (SELECT salary - avg(salary) OVER () AS ca, depname FROM __THIS__), b AS (SELECT ca * 2 AS cb, depname FROM a) SELECT cb - avg(cb) OVER (PARTITION BY depname) AS m FROM b",
    "SELECT z + 1 AS z1 FROM (SELECT salary - avg(salary) OVER (PARTITION BY depname) AS z FROM __THIS__) AS sub",
    "WITH a(s2) AS (SELECT salary * 2 FROM __THIS__) SELECT s2 - avg(s2) OVER () AS c FROM a",
    "WITH a AS (SELECT * FROM __THIS__) SELECT salary - avg(salary) OVER () AS c FROM a",
    "SELECT salary / (SELECT max(salary) FROM __THIS__) AS r FROM __THIS__",
    "SELECT salary - (SELECT avg(salary) FROM __THIS__ WHERE empno > 3) AS d FROM __THIS__",
    "SELECT EXISTS(SELECT 1 FROM __THIS__ WHERE salary > 5500) AS any_high FROM __THIS__",
    "WITH a AS (SELECT salary / (SELECT max(salary) FROM __THIS__) AS r FROM __THIS__) SELECT r - avg(r) OVER () AS rc FROM a",
]

CURATED_REFUSED = [
    "SELECT salary FROM __THIS__ WHERE salary > 4000",
    "SELECT depname FROM __THIS__ GROUP BY depname",
    "SELECT salary FROM __THIS__ ORDER BY salary",
    "SELECT salary FROM __THIS__ LIMIT 3",
    "SELECT DISTINCT depname FROM __THIS__",
    "SELECT a.salary FROM __THIS__ a JOIN __THIS__ b ON true",
    "SELECT salary FROM __THIS__ UNION SELECT 1",
    "SELECT row_number() OVER (ORDER BY empno) FROM __THIS__",
    "SELECT ntile(4) OVER (ORDER BY salary) FROM __THIS__",
    "SELECT lag(salary) OVER (PARTITION BY depname ORDER BY empno) FROM __THIS__",
    "SELECT lead(salary) OVER (PARTITION BY depname ORDER BY empno) FROM __THIS__",
    "SELECT sum(salary) OVER (ORDER BY empno ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM __THIS__",
    "SELECT sum(salary) OVER (ORDER BY empno RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW) FROM __THIS__",
    "SELECT avg(salary) FROM __THIS__",
    "SELECT salary IN (SELECT salary FROM __THIS__) FROM __THIS__",
    "SELECT salary + 1 AS s, s * 2 FROM __THIS__",
    "SELECT (SELECT max(x) FROM other_table) FROM __THIS__",
]


# --- curated, schema-aware (loop 4): replayed with a declared this_schema ----

CURATED_SCHEMA = [
    "SELECT COLUMNS('.*name') FROM __THIS__",
    "SELECT * EXCLUDE (enroll_date) REPLACE (salary + 1 AS salary) FROM __THIS__",
    "SELECT * RENAME (depname AS dep) FROM __THIS__",
    "SELECT salary + 1 AS s2, s2 * 2 AS s4 FROM __THIS__",
    "WITH a AS (SELECT * EXCLUDE (empno) FROM __THIS__) SELECT salary - avg(salary) OVER (PARTITION BY depname) AS d FROM a",
]


def _outcome(sql: str) -> tuple[str, str]:
    try:
        marginalize(sql)
    except MarginalizeError as e:
        return "refused", str(e)
    try:
        gate(sql, EMPSALARY)
        return "marginalized", ""
    except Exception as e:  # corpus triage: anything non-named is a failure
        return "failed", f"{type(e).__name__}: {e}"


def test_mined_corpus_scoreboard():
    counts = {"marginalized": 0, "refused": 0}
    failures = []
    for sql in MINED:
        kind, detail = _outcome(sql)
        if kind == "failed":
            failures.append((sql, detail))
        else:
            counts[kind] += 1
    assert not failures, failures
    assert counts == MINED_SCOREBOARD


@pytest.mark.parametrize("sql", CURATED_MARGINALIZED, ids=lambda s: s[:56])
def test_curated_marginalizes(sql):
    kind, detail = _outcome(sql)
    assert kind == "marginalized", f"{kind}: {detail}"


@pytest.mark.parametrize("sql", CURATED_REFUSED, ids=lambda s: s[:56])
def test_curated_refuses(sql):
    kind, detail = _outcome(sql)
    assert kind == "refused", f"{kind}: {detail}"


@pytest.mark.parametrize("sql", CURATED_SCHEMA, ids=lambda s: s[:56])
def test_curated_schema_marginalizes(sql):
    from sql_transform._projection_test import gate

    try:
        marginalize(sql, list(EMPSALARY.column_names))
    except MarginalizeError as e:
        raise AssertionError(f"refused: {e}") from e
    gate(sql, EMPSALARY, schema=True)


def test_progression_totals():
    """The metric, in one place. Edit these pins when a loop widens support."""
    assert len(MINED) == 22
    assert MINED_SCOREBOARD["marginalized"] + MINED_SCOREBOARD["refused"] == 22
    assert len(CURATED_MARGINALIZED) == 39
    assert len(CURATED_REFUSED) == 17
    assert len(CURATED_SCHEMA) == 5
