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
from pathlib import Path

import duckdb
import pytest
from confit import _engine

CORPUS = Path(__file__).parent / "corpus" / "duckdb_mined.jsonl"

# Measured at introduction (see PR); a drop is a regression.
SPARK_MATCH_FLOOR = 130

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
    findspark = pytest.importorskip(
        "pyspark.sql", reason="pyspark not installed - Spark leg of the L3 gate"
    )
    builder = findspark.SparkSession.builder.appName("dialect-l3-gate").master(
        "local[1]"
    )
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
    assert not fails, summary + "\n" + "\n".join(fails[:20])
    assert counts["match"] >= SPARK_MATCH_FLOOR, summary


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
