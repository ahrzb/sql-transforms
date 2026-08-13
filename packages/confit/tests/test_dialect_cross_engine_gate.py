"""Dialect L3 gate: printed queries EXECUTE equivalently on the target engine.

Law L3 of 2026-08-13-dialect-logical-plan-design.md, executed live for
Spark: for every corpus statement the frontend admits and the Spark
printer prints, build the EQUIVALENT tables on both engines (DuckDB's data
shipped to Spark through Arrow, so widths and nullability survive), run
the original SQL on DuckDB and the printed SQL on Spark under the PINNED
config (ansi=true, UTC, local[1] — pins-dialect/spark-ansi.json), and
compare column names plus the row multiset (plan semantics are multisets,
design D4).

The v0 plan surface has no aggregates or windows, so there is NO epsilon
tier yet: every admitted statement is exact-tier and the comparison is
value equality. When float-accumulation aggregates land (phase 2+), their
outputs move to the tolerance comparison the design defines — extend
`rows_of`, do not weaken the exact tier.

Outcomes, three, as everywhere:

  match             -- identical names + row multiset across engines
  clean-unsupported -- frontend refusal (counted upstream by the L2 gate)
                       or a named Spark printer refusal
  FAIL              -- printed SQL errors on Spark, or results differ:
                       the gate requires zero

BigQuery runs through the same seam when credentials exist; without them
its leg SKIPS LOUDLY (design phase 4 — the remote gate is owed, not
forgotten). Set CONFIT_BIGQUERY_PROJECT to arm it.

The match floor is the measured count at introduction — raise it when the
surface grows, never lower it.
"""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from pathlib import Path

import duckdb
import pytest
from confit import _engine

CORPUS = Path(__file__).parent / "corpus" / "duckdb_mined.jsonl"

# Measured at introduction (see PR); a drop is a regression.
# Measured in CI (TASK-104 joins: 213 -> 260); a drop is a regression.
SPARK_MATCH_FLOOR = 260

PINNED_SPARK_CONFIG = {
    "spark.sql.ansi.enabled": "true",
    "spark.sql.session.timeZone": "UTC",
}


def cases():
    with CORPUS.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_duckdb(case) -> duckdb.DuckDBPyConnection | None:
    """Replay the case's setup; None when it needs the replay-only drop trick
    to fail differently than corpus replay would."""
    con = duckdb.connect()
    for stmt in case["setup"]:
        try:
            con.execute(stmt)
        except duckdb.CatalogException as e:
            m = re.match(r'\s*CREATE\s+TABLE\s+"?([A-Za-z_]\w*)"?', stmt, re.IGNORECASE)
            if m and "already exists" in str(e):
                con.execute(f'DROP TABLE "{m.group(1)}"')
                con.execute(stmt)
            else:
                raise
    return con


def catalog_of(con: duckdb.DuckDBPyConnection):
    tables = [
        r[0]
        for r in con.execute(
            "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main'"
        ).fetchall()
    ]
    return [
        (
            t,
            [
                (name, dtype, nullable == "YES")
                for name, dtype, nullable, *_ in con.execute(
                    f'DESCRIBE "{t}"'
                ).fetchall()
            ],
        )
        for t in tables
    ]


def norm(v):
    """One comparable form per value: NaN self-equal, ints/floats kept apart."""
    if isinstance(v, float) and math.isnan(v):
        return "NaN"
    return repr(v)


def rows_of(names, rows):
    return names, sorted(tuple(norm(v) for v in row) for row in rows)


def run_duckdb(con, sql):
    cur = con.execute(sql)
    names = [d[0] for d in cur.description]
    return rows_of(names, cur.fetchall())


@pytest.fixture(scope="module")
def spark():
    # The Spark leg is a REQUIRED gate: without pyspark it fails loudly
    # (review-confirmed blind spot: importorskip let every Spark-side
    # divergence pass green). CONFIT_ALLOW_NO_SPARK=1 is the explicit,
    # visible opt-out for environments that cannot run a JVM.
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        if os.environ.get("CONFIT_ALLOW_NO_SPARK") == "1":
            pytest.skip("CONFIT_ALLOW_NO_SPARK=1: Spark leg explicitly disabled")
        raise
    builder = SparkSession.builder.appName("dialect-l3-gate").master("local[1]")
    for k, v in PINNED_SPARK_CONFIG.items():
        builder = builder.config(k, v)
    session = builder.getOrCreate()
    yield session
    session.stop()


def ship_tables_to_spark(spark, con, catalog):
    """The 'equivalent tables' premise: the same rows, through Arrow, so
    integer widths, decimals, and nullability survive the trip."""
    for table, _cols in catalog:
        arrow_tbl = con.execute(f'SELECT * FROM "{table}"').to_arrow_table()
        try:
            df = spark.createDataFrame(arrow_tbl)
        except TypeError:  # older pyspark: no direct Arrow ingestion
            df = spark.createDataFrame(arrow_tbl.to_pandas())
        df.createOrReplaceTempView(table)


def run_spark(spark, sql):
    df = spark.sql(sql)
    names = list(df.columns)
    return rows_of(names, [tuple(r) for r in df.collect()])


def test_spark_execution_equivalence(spark):
    counts = {"match": 0, "clean-unsupported": 0}
    fails: list[str] = []

    for case in cases():
        con = build_duckdb(case)
        try:
            cat = catalog_of(con)
            try:
                plan_text = _engine.dialect_parse(case["sql"], cat)
                printed = _engine.dialect_print(plan_text, "spark", cat)
            except ValueError as e:
                if str(e).startswith("unsupported:"):
                    counts["clean-unsupported"] += 1
                    continue
                fails.append(f"{case['source']}: {case['sql']!r}: {e}")
                continue
            original = run_duckdb(con, case["sql"])
            ship_tables_to_spark(spark, con, cat)
            try:
                translated = run_spark(spark, printed)
            except Exception as e:
                fails.append(
                    f"{case['source']}: {printed!r}: spark error: "
                    f"{str(e).splitlines()[0][:160]}"
                )
                continue
            if original == translated:
                counts["match"] += 1
            else:
                fails.append(
                    f"{case['source']}: {case['sql']!r} -> {printed!r}: "
                    f"duckdb {original} != spark {translated}"
                )
        finally:
            con.close()

    total = counts["match"] + counts["clean-unsupported"] + len(fails)
    summary = (
        f"L3 spark gate: {counts['match']}/{total} match, "
        f"{counts['clean-unsupported']} clean-unsupported, {len(fails)} FAIL"
    )
    print(summary)
    # A warning survives `pytest -q` on green runs, so CI logs always carry
    # the measured count (the growth ladder needs it to ratchet floors).
    warnings.warn(summary, stacklevel=1)
    assert not fails, summary + "\n" + "\n".join(fails[:20])
    assert counts["match"] >= SPARK_MATCH_FLOOR, summary


# Review-confirmed divergences, forced in the printers - each scenario is
# data the mined corpus never contains, executed on both engines per
# commit. "Both engines error" is a matching outcome (same trap class);
# a value on one side and an error on the other is the divergence the
# forcing exists to prevent.
SYNTHETIC = [
    (
        "fdiv-ieee-zero-divisors",
        [
            "CREATE TABLE t(a DOUBLE, b DOUBLE)",
            "INSERT INTO t VALUES (1.0, 0.0), (-1.0, 0.0), (0.0, 0.0), (1.0, -0.0), "
            "(NULL, 0.0), (1.0, NULL), (1.0, 2.0), ('NaN'::DOUBLE, 0.0)",
        ],
        "SELECT a / b AS q FROM t",
    ),
    (
        "idiv-rem-zero-null",
        [
            "CREATE TABLE t(a INTEGER, b INTEGER)",
            "INSERT INTO t VALUES (1, 0), (7, -2), (-7, 2), (NULL, 3), (5, 2), (-7, 0)",
        ],
        "SELECT a // b AS q, a % b AS r FROM t",
    ),
    (
        "rem-int64-min-traps",
        [
            "CREATE TABLE t(a BIGINT, b BIGINT)",
            "INSERT INTO t VALUES (-9223372036854775808, -1)",
        ],
        "SELECT a % b AS r FROM t",
    ),
    (
        "idiv-int32-min-traps",
        [
            "CREATE TABLE t(a INTEGER, b INTEGER)",
            "INSERT INTO t VALUES (-2147483648, -1)",
        ],
        "SELECT a // b AS q FROM t",
    ),
    (
        "neg-int32-min-traps",
        ["CREATE TABLE t(a INTEGER)", "INSERT INTO t VALUES (-2147483648)"],
        "SELECT - a AS n FROM t",
    ),
    (
        "cast-double-int-rounds-half-even",
        [
            "CREATE TABLE t(x DOUBLE)",
            "INSERT INTO t VALUES (2.7), (2.5), (1.5), (-2.7), (-2.5), (0.5), (NULL)",
        ],
        "SELECT CAST(x AS INTEGER) AS s, TRY_CAST(x AS INTEGER) AS y FROM t",
    ),
    (
        "cast-double-int-overflow-traps",
        ["CREATE TABLE t(x DOUBLE)", "INSERT INTO t VALUES (1e300)"],
        "SELECT CAST(x AS INTEGER) AS s FROM t",
    ),
    (
        "cast-decimal-int-rounds-half-away",
        [
            "CREATE TABLE t(d DECIMAL(3,1))",
            "INSERT INTO t VALUES (2.5), (-2.5), (2.4), (-2.4), (NULL)",
        ],
        "SELECT CAST(d AS INTEGER) AS s FROM t",
    ),
    (
        "nan-total-order",
        [
            "CREATE TABLE t(x DOUBLE, y DOUBLE)",
            "INSERT INTO t VALUES ('NaN'::DOUBLE, 'NaN'::DOUBLE), "
            "('NaN'::DOUBLE, 5.0), (5.0, 'NaN'::DOUBLE), (5.0, 5.0), "
            "(NULL, 'NaN'::DOUBLE)",
        ],
        "SELECT x = y AS e, x <> y AS n, x < y AS l, x <= y AS le, x > y AS g, "
        "x >= y AS ge, x IS NOT DISTINCT FROM y AS d FROM t",
    ),
]


def run_or_error(fn, *args):
    try:
        return ("ok", fn(*args))
    except Exception as e:
        return ("error", str(e).splitlines()[0][:120])


def test_spark_synthetic_divergences(spark):
    fails = []
    for name, setup, sql in SYNTHETIC:
        con = duckdb.connect()
        try:
            for stmt in setup:
                con.execute(stmt)
            cat = catalog_of(con)
            plan_text = _engine.dialect_parse(sql, cat)
            printed = _engine.dialect_print(plan_text, "spark", cat)
            original = run_or_error(run_duckdb, con, sql)
            ship_tables_to_spark(spark, con, cat)
            translated = run_or_error(run_spark, spark, printed)
            outcomes_match = (
                original[0] == "error" and translated[0] == "error"
            ) or original == translated
            if not outcomes_match:
                fails.append(
                    f"{name}: {printed!r}: duckdb {original} != spark {translated}"
                )
        finally:
            con.close()
    assert not fails, "\n".join(fails)


@pytest.mark.skipif(
    not os.environ.get("CONFIT_BIGQUERY_PROJECT"),
    reason="design phase 4: the BigQuery leg of the L3 gate needs credentials "
    "(set CONFIT_BIGQUERY_PROJECT and provide ADC) - the printer ships "
    "documented-semantics until this runs",
)
def test_bigquery_execution_equivalence():
    raise NotImplementedError(
        "phase 4: wire the same seam as the Spark leg through the BigQuery "
        "client - ship tables via load_table_from_dataframe, run printed "
        "SQL, compare rows_of() outputs"
    )
