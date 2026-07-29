"""SQLProjection tests: the training-set round-trip invariant.

The standing invariant for this loop and every one after it: fit + transform,
applied to the training set, must be bit-equal to running the original SQL
with ``__THIS__`` pointing at the training set. It is free — the training set
is the oracle input, so no expected values are written by hand. Today
"transform" is played by DuckDB executing ``serving_sql`` against the fitted
params; when ``infer``/``infer_batch`` land, the same assertion runs through
the real serving path and gates the wiring end-to-end.
"""

import inspect
import math
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
    # pyarrow equals says NaN != NaN; fall back to a NaN-aware (and signed-
    # zero-strict) recursive compare when the fast path disagrees.
    assert orig.equals(rew) or _same(orig.to_pylist(), rew.to_pylist()), (
        f"\n{orig.to_pydict()}\n!=\n{rew.to_pydict()}"
    )
    return p


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(v, b[k]) for k, v in a.items())
    return type(a) is type(b) and a == b


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


@pytest.mark.parametrize(
    "expr",
    [
        # running aggregates: order values join the key set
        "sum(fare) OVER (PARTITION BY country ORDER BY age)",
        "avg(age) OVER (ORDER BY fare)",
        "count(*) OVER (PARTITION BY country ORDER BY age)",
        # explicit RANGE / GROUPS frames with constant bounds
        "sum(age) OVER (PARTITION BY country ORDER BY fare"
        " RANGE BETWEEN 2 PRECEDING AND CURRENT ROW)",
        "sum(age) OVER (PARTITION BY country ORDER BY fare"
        " GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW)",
        # whole-partition frames are per-partition constants
        "sum(age) OVER (PARTITION BY country ORDER BY fare"
        " ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)",
        # rank family: functions of the order values
        "rank() OVER (PARTITION BY country ORDER BY age)",
        "dense_rank() OVER (PARTITION BY country ORDER BY age DESC NULLS FIRST)",
        "percent_rank() OVER (ORDER BY age)",
        "cume_dist() OVER (PARTITION BY country ORDER BY age)",
        # value functions
        "first_value(name) OVER (PARTITION BY country ORDER BY age)",
        "first_value(city IGNORE NULLS) OVER (PARTITION BY country ORDER BY age)",
        "last_value(name) OVER (PARTITION BY country ORDER BY age)",
        "nth_value(name, 2) OVER (PARTITION BY country ORDER BY age)",
        # FILTER / DISTINCT / ordered-argument aggregates
        "avg(age) FILTER (WHERE fare > 8) OVER (PARTITION BY country)",
        "count(DISTINCT city) OVER (PARTITION BY country)",
        "string_agg(name, ',') OVER (PARTITION BY country)",
        "string_agg(name, ',' ORDER BY age) OVER (PARTITION BY country)",
        # order-sensitive / formerly non-allowlisted aggregates
        "first(name) OVER (PARTITION BY country)",
        "array_agg(name) OVER (PARTITION BY country)",
        "quantile_cont(age, 0.25) OVER (PARTITION BY country)",
        "bool_and(age > 30) OVER (PARTITION BY country)",
        "corr(age, fare) OVER (PARTITION BY country)",
        "mode(city) OVER (PARTITION BY country)",
        # expression keys
        "sum(fare) OVER (PARTITION BY substr(country, 1, 1))",
        "avg(age) OVER (PARTITION BY country ORDER BY fare % 3)",
        "avg(age) OVER (PARTITION BY country, city ORDER BY fare, name)",
    ],
)
def test_widened_window_surface(expr):
    gate(f"SELECT {expr} AS m, name FROM __THIS__")


def test_named_window_is_inlined_by_the_parser():
    gate(
        "SELECT avg(age) OVER w AS m, name FROM __THIS__"
        " WINDOW w AS (PARTITION BY country)"
    )


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
            keys = rng.sample(["k1", "k2", "k2 % 2"], k=rng.randrange(0, 3))
            p = f"PARTITION BY {', '.join(keys)}" if keys else ""
            ovp = f"OVER ({p})"
            o = rng.choice(["y", "x", "y % 7"])
            po = f"OVER ({p + ' ' if p else ''}ORDER BY {o})"
            agg = rng.choice(aggs)
            arg = col if agg != "count" else rng.choice([col, "*"])
            template = rng.choice(
                [
                    f"{agg}({arg}) {ovp}",
                    f"{col} - {agg}({arg}) {ovp}",
                    f"{col} + 1",
                    # widened surface: running, frames, rank/value fns,
                    # FILTER/DISTINCT, order-sensitive aggregates
                    f"{agg}({arg}) {po}",
                    f"{agg}({arg}) OVER ({p + ' ' if p else ''}ORDER BY {o}"
                    " RANGE BETWEEN 2 PRECEDING AND CURRENT ROW)",
                    f"{agg}({arg}) OVER ({p + ' ' if p else ''}ORDER BY {o}"
                    " GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW)",
                    f"{agg}({arg}) OVER ({p + ' ' if p else ''}ORDER BY {o}"
                    " ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)",
                    f"rank() {po}",
                    f"dense_rank() {po}",
                    f"cume_dist() {po}",
                    f"first_value({col}) {po}",
                    f"last_value({col}) {po}",
                    f"nth_value({col}, 2) {po}",
                    f"{agg}(DISTINCT {arg.replace('*', col)}) {ovp}",
                    f"{agg}({arg}) FILTER (WHERE {o} > 1) {ovp}",
                    f"first({col}) {ovp}",
                    f"string_agg(k1, '|') {ovp}",
                    f"string_agg(k1, '|' ORDER BY {o}) {ovp}",
                    f"array_agg(k1) {ovp}",
                    f"quantile_cont({col}, 0.25) {ovp}",
                    f"bool_and({col} > 2) {ovp}",
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
