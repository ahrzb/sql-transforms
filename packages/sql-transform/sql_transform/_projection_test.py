"""SQLProjection tests: the DuckDB-vs-DuckDB differential gate.

The bulletproof property: for any accepted projection, running the original
SQL over the training table must be bit-exact with running the rewritten
``serving_sql`` joined against the fitted params — both executed by DuckDB,
the oracle. No inference code is involved anywhere.
"""

import inspect
import os
import random

import duckdb
import pyarrow as pa
import pytest

from sql_transform import MarginalizeError, SQLProjection

TRAIN = pa.table(
    {
        "country": ["US", "US", None, "DE", None, "DE", "FR"],
        "city": ["a", None, "a", "b", None, None, "c"],
        "age": [40.0, 30.0, None, 25.0, 50.0, None, 35.0],
        "fare": [7, 8, None, 9, 10, 11, None],
        "name": ["x", "y", "z", "w", "v", "u", "t"],
    }
)


def gate(sql: str, table: pa.Table = TRAIN) -> SQLProjection:
    """Assert original == marginalized under the oracle; returns the fitted p."""
    p = SQLProjection(sql).fit(table)
    con = duckdb.connect()
    try:
        # Both sides single-threaded: DuckDB's parallel window aggregation is
        # not bit-deterministic for floats, so the original text only has a
        # unique bit-answer at threads=1 — which is also how fit computes.
        con.execute("SET threads = 1")
        con.register("__THIS__", table)
        orig = con.execute(f"SELECT * FROM ({sql}) ORDER BY ALL").to_arrow_table()
        for name, params_table in p.params.items():
            con.register(name, params_table)
        rew = con.execute(
            f"SELECT * FROM ({p.serving_sql}) ORDER BY ALL"
        ).to_arrow_table()
    finally:
        con.close()
    assert orig.schema == rew.schema, f"\n{orig.schema}\n!=\n{rew.schema}"
    assert orig.equals(rew), f"\n{orig.to_pydict()}\n!=\n{rew.to_pydict()}"
    return p


def test_standard_scaler_with_null_keys_and_null_inputs():
    p = gate(
        "SELECT (age - avg(age) OVER (PARTITION BY country))"
        " / stddev_samp(age) OVER (PARTITION BY country) AS age_z FROM __THIS__"
    )
    (params,) = p.params.values()
    # NULL is a partition of its own and must be a real params row.
    assert None in params.column("country").to_pylist()


def test_global_and_keyed_windows_mixed():
    gate(
        "SELECT age - avg(age) OVER () AS c,"
        " fare - avg(fare) OVER (PARTITION BY country) AS f,"
        " name FROM __THIS__"
    )


def test_multi_key_partition_with_nulls_in_both_keys():
    gate("SELECT avg(age) OVER (PARTITION BY country, city) AS m FROM __THIS__")


def test_single_row_groups():
    # stddev_samp of a single-row group is NULL; must survive the join back.
    gate("SELECT stddev_samp(age) OVER (PARTITION BY name) AS s FROM __THIS__")


@pytest.mark.parametrize(
    "agg",
    [
        "avg(age)",
        "sum(fare)",
        "count(age)",
        "count(*)",
        "min(age)",
        "max(fare)",
        "stddev(age)",
        "stddev_pop(age)",
        "stddev_samp(age)",
        "var_pop(age)",
        "var_samp(age)",
        "variance(age)",
        "median(age)",
        "median(fare)",
    ],
)
def test_every_allowlisted_aggregate(agg):
    gate(f"SELECT {agg} OVER (PARTITION BY country) AS m, name FROM __THIS__")
    gate(f"SELECT {agg} OVER () AS g, name FROM __THIS__")


def test_expression_aggregate_arguments():
    gate("SELECT avg(age * 2 + fare) OVER (PARTITION BY country) AS m FROM __THIS__")


def test_quoted_and_unicode_identifiers():
    table = pa.table({"país": ["ES", "ES", None], "weird col": [1.0, 2.0, 3.0]})
    gate(
        'SELECT "weird col" - avg("weird col") OVER (PARTITION BY "país") AS z'
        " FROM __THIS__",
        table,
    )


def test_struct_field_access_passthrough():
    table = pa.table(
        {
            "s": [{"f": 1.0}, {"f": 2.0}, {"f": None}],
            "g": ["a", "a", "b"],
        }
    )
    gate("SELECT s.f - avg(s.f) OVER (PARTITION BY g) AS d FROM __THIS__", table)


def test_star_with_aggregate():
    gate("SELECT *, avg(age) OVER (PARTITION BY country) AS m FROM __THIS__")


def test_no_aggregates_identity():
    p = gate("SELECT age + 1 AS b, name FROM __THIS__")
    assert p.params == {}


def test_unaliased_outputs_keep_names():
    gate("SELECT age + 1, avg(age) OVER () FROM __THIS__")


def test_fuzz_differential():
    """Seeded random projections; MARGINALIZE_FUZZ_N deepens the run."""
    n = int(os.environ.get("MARGINALIZE_FUZZ_N", "25"))
    rng = random.Random(20260729)
    aggs = ["avg", "sum", "min", "max", "count", "stddev_samp", "median"]
    for _ in range(n):
        rows = rng.randrange(1, 40)
        # Explicit arrow types: an all-None pick must stay a typed column, not
        # degrade to pa.null() (which DuckDB coerces to INTEGER — degenerate).
        table = pa.table(
            {
                "k1": pa.array(
                    [rng.choice(["a", "b", "c", None]) for _ in range(rows)],
                    type=pa.string(),
                ),
                "k2": pa.array(
                    [rng.choice([1, 2, None]) for _ in range(rows)], type=pa.int64()
                ),
                "x": pa.array(
                    [rng.choice([rng.uniform(-9, 9), None]) for _ in range(rows)],
                    type=pa.float64(),
                ),
                "y": pa.array(
                    [rng.choice([rng.randrange(100), None]) for _ in range(rows)],
                    type=pa.int64(),
                ),
            }
        )
        exprs = []
        for i in range(rng.randrange(1, 4)):
            col = rng.choice(["x", "y"])
            keys = rng.sample(["k1", "k2"], k=rng.randrange(0, 3))
            over = f"OVER (PARTITION BY {', '.join(keys)})" if keys else "OVER ()"
            agg = rng.choice(aggs)
            arg = col if agg != "count" else rng.choice([col, "*"])
            template = rng.choice(
                [
                    f"{agg}({arg}) {over}",
                    f"{col} - {agg}({arg}) {over}",
                    f"{col} + 1",
                ]
            )
            exprs.append(f"{template} AS e{i}")
        gate(f"SELECT {', '.join(exprs)} FROM __THIS__", table)


# --- surface: the not-yet-implemented half stays honest ----------------------


def test_unfitted_params_access_raises():
    p = SQLProjection("SELECT avg(a) OVER () AS m FROM __THIS__")
    with pytest.raises(MarginalizeError, match="not fitted"):
        _ = p.params


def test_from_file(tmp_path):
    f = tmp_path / "q.sql"
    f.write_text("SELECT age + 1 AS b FROM __THIS__", encoding="utf-8")
    assert SQLProjection.from_file(str(f)).serving_sql


def test_template_input_is_a_later_loop():
    with pytest.raises(NotImplementedError, match="t-string"):
        SQLProjection(t"SELECT 1 AS x FROM __THIS__")


@pytest.mark.parametrize("method", ["infer", "infer_batch"])
def test_serving_methods_still_raise(method):
    p = SQLProjection("SELECT age + 1 AS b FROM __THIS__")
    with pytest.raises(NotImplementedError, match=method):
        getattr(p, method)(None)


@pytest.mark.parametrize("prop", ["backend", "boundary"])
def test_serving_properties_still_raise(prop):
    p = SQLProjection("SELECT age + 1 AS b FROM __THIS__")
    assert isinstance(inspect.getattr_static(SQLProjection, prop), property)
    with pytest.raises(NotImplementedError, match=prop):
        getattr(p, prop)


def test_signatures_are_stable():
    assert str(inspect.signature(SQLProjection.fit)) == (
        "(self, table: 'pa.Table', /, this_model: 'type[BaseModel] | None' = None)"
        " -> 'SQLProjection'"
    )
    assert str(inspect.signature(SQLProjection.infer)) == (
        "(self, row: 'dict[str, Any] | BaseModel', /) -> 'BaseModel'"
    )
    assert str(inspect.signature(SQLProjection.infer_batch)) == (
        "(self, rows: 'list[dict[str, Any] | BaseModel]', /) -> 'list[BaseModel]'"
    )
