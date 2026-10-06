"""Original corpus SQL and its distinct kept/migrated classifications.

The original query remains the numerical oracle for every computation.
Removing chain/schema sugar changes admission, not the required computation.
"""

import datetime

import pyarrow as pa
import pytest

from sql_transform import SQLProjection, TransformError
from sql_transform._marginal_projection_test import explicit_chain, gate

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
    # Ordinary scalar lateral aliases are now in CURATED_KEPT below.
    "SELECT (SELECT max(x) FROM other_table) FROM __THIS__",
]


# Original declared-schema SQL remains visible; explicit replacements follow.

CURATED_SCHEMA = [
    "SELECT COLUMNS('.*name') FROM __THIS__",
    "SELECT * EXCLUDE (enroll_date) REPLACE (salary + 1 AS salary) FROM __THIS__",
    "SELECT * RENAME (depname AS dep) FROM __THIS__",
    "SELECT salary + 1 AS s2, s2 * 2 AS s4 FROM __THIS__",
    "WITH a AS (SELECT * EXCLUDE (empno) FROM __THIS__) SELECT salary - avg(salary) OVER (PARTITION BY depname) AS d FROM a",
]


# The original accepted mined cases are required, not a count-based scoreboard.
# Bounded ROWS and position-dependent computations remain named refusals.
MINED_REFUSED = {
    MINED[2],
    MINED[4],
    MINED[10],
    MINED[13],
    *MINED[15:],
}
MINED_KEPT = [sql for sql in MINED if sql not in MINED_REFUSED]

# Chain admission was deliberately removed. Keep each original prefix intact
# in the explicit fit queries instead of fitting the upstream serving joins.
CURATED_MIGRATED = {
    sql: explicit_chain
    for sql in CURATED_MARGINALIZED
    if sql.startswith("WITH ") or " AS sub" in sql
}
CURATED_KEPT = [sql for sql in CURATED_MARGINALIZED if sql not in CURATED_MIGRATED]
CURATED_KEPT.append("SELECT salary + 1 AS s, s * 2 FROM __THIS__")

SCHEMA_EXPLICIT = [
    "SELECT depname FROM __THIS__",
    "SELECT depname, empno, salary + 1 AS salary FROM __THIS__",
    "SELECT depname AS dep, empno, salary, enroll_date FROM __THIS__",
    "SELECT salary + 1 AS s2, s2 * 2 AS s4 FROM __THIS__",
    "WITH a AS (SELECT depname, salary, enroll_date FROM __THIS__)"
    " SELECT salary - avg(salary) OVER (PARTITION BY depname) AS d FROM a",
]


def _outcome(sql: str, explicit: str | None = None) -> tuple[str, str]:
    try:
        if explicit is None:
            SQLProjection.marginalize(sql)
        else:
            SQLProjection(explicit)
    except TransformError as e:
        return "refused", str(e)
    try:
        gate(sql, EMPSALARY, explicit=explicit)
        return ("marginalized" if explicit is None else "migrated"), ""
    except Exception as e:  # a gate mismatch or unexpected exception is a bug
        return "failed", f"{type(e).__name__}: {e}"


@pytest.mark.parametrize("sql", MINED_KEPT, ids=lambda s: s[:56])
def test_original_accepted_mined_computations(sql):
    kind, detail = _outcome(sql)
    assert kind == "marginalized", f"{kind}: {detail}"


@pytest.mark.parametrize(
    "sql", [s for s in MINED if s in MINED_REFUSED], ids=lambda s: s[:56]
)
def test_mined_meaningful_refusals(sql):
    kind, detail = _outcome(sql)
    assert kind == "refused", f"{kind}: {detail}"


@pytest.mark.parametrize("sql", CURATED_KEPT, ids=lambda s: s[:56])
def test_curated_kept_original_syntax(sql):
    kind, detail = _outcome(sql)
    assert kind == "marginalized", f"{kind}: {detail}"


@pytest.mark.parametrize("sql", CURATED_MIGRATED, ids=lambda s: s[:56])
def test_curated_migrated_chain_computations(sql):
    with pytest.raises(TransformError):
        SQLProjection.marginalize(sql)
    kind, detail = _outcome(sql, explicit=CURATED_MIGRATED[sql](sql))
    assert kind == "migrated", f"{kind}: {detail}"


@pytest.mark.parametrize("sql", CURATED_REFUSED, ids=lambda s: s[:56])
def test_curated_refuses(sql):
    kind, detail = _outcome(sql)
    assert kind == "refused", f"{kind}: {detail}"


@pytest.mark.parametrize(
    "original,replacement",
    list(zip(CURATED_SCHEMA, SCHEMA_EXPLICIT, strict=True)),
    ids=[s[:56] for s in CURATED_SCHEMA],
)
def test_curated_schema_migrated_computations(original, replacement):
    explicit = (
        explicit_chain(replacement)
        if replacement.startswith("WITH ")
        else SQLProjection.marginalize(replacement).source
    )
    kind, detail = _outcome(original, explicit=explicit)
    assert kind == "migrated", f"{kind}: {detail}"
