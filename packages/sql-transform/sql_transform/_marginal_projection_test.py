"""Exact original-SQL oracles for kept windows and explicit migrated chains.

Each fit carrier retains the complete original upstream computation. Serving
joins must never become a later fit input: that can change floating reduction
order. The seeded gate preserves the original draws, schemas, NaNs and signs.
"""

import math
import os
import random

import duckdb
import pyarrow as pa
import pytest

from sql_transform import FittedProjection, SQLProjection, TransformError
from sql_transform._ast import (
    FIT,
    THIS,
    _base_table,
    _deserialize,
    _parse,
    _statement,
    _subquery_ref,
)
from sql_transform._marginal import derive
from sql_transform._nodes import (
    BaseTable,
    ColumnRef,
    CteMap,
    Select,
    SubqueryRef,
    rebuild,
)

TRAIN = pa.table(
    {
        "country": ["US", "US", None, "DE", None, "DE", "FR"],
        "city": ["a", None, "a", "b", None, None, "c"],
        "age": [40.0, 30.0, None, 25.0, 50.0, None, 35.0],
        "fare": [7, 8, None, 9, 10, 11, None],
        "name": ["x", "y", "z", "w", "v", "u", "t"],
    }
)


def explicit_chain(sql: str, captured=None) -> str:
    """Test-only one-level derivation with original fit prefixes at each stage.

    Graft each derived program once. ``rebuild`` does not visit replacements,
    so an inserted stage's own FIT/THIS references keep their meaning.
    """
    scope = captured or {}

    def rebase(node):
        def replace(v):
            if isinstance(v, BaseTable) and v.table_name.upper() == THIS:
                return v.model_copy(update={"table_name": FIT})
            if isinstance(v, ColumnRef) and v.column_names[0].upper() == THIS:
                return v.model_copy(update={"column_names": [FIT, *v.column_names[1:]]})
            return None

        return rebuild(node, replace, deep=True)

    def lower(node):
        assert isinstance(node, Select), "only the original linear SELECT chains"
        source = node.from_table
        entries = node.cte_map.map
        aliases = []
        if isinstance(source, SubqueryRef):
            upstream = source.subquery.node
            aliases = source.column_name_alias
            source_alias = source.alias
        elif isinstance(source, BaseTable) and source.table_name.upper() != THIS:
            index = next(
                i
                for i, e in enumerate(entries)
                if e.key.lower() == source.table_name.lower()
            )
            entry = entries[index]
            upstream = entry.value.query.node
            upstream = upstream.model_copy(
                update={
                    "cte_map": CteMap(map=[*entries[:index], *upstream.cte_map.map]),
                }
            )
            aliases = entry.value.aliases
            source_alias = source.alias or source.table_name
        else:
            return (
                _parse(derive(_deserialize(_statement(node)), scope)).statements[0].node
            )

        serving = lower(upstream)
        local = node.model_copy(
            update={
                "cte_map": CteMap(map=[]),
                "from_table": _base_table(THIS, source_alias),
            }
        )
        derived = (
            _parse(derive(_deserialize(_statement(local)), scope)).statements[0].node
        )

        def graft(v):
            if isinstance(v, BaseTable) and v.table_name.upper() in (FIT, THIS):
                program = rebase(upstream) if v.table_name.upper() == FIT else serving
                return _subquery_ref(program, v.alias or v.table_name).model_copy(
                    update={"column_name_alias": aliases}
                )
            return None

        return rebuild(derived, graft, deep=True)

    return _deserialize(_statement(lower(_parse(sql).statements[0].node)))


def gate(
    sql: str,
    table: pa.Table = TRAIN,
    schema: bool = False,
    *,
    explicit: str | None = None,
    migrated: bool = False,
) -> FittedProjection:
    """Compare original SQL with the fitted artifact, not replacement SQL.

    ``schema`` is intentionally unused: the seeded gate still draws that bit.
    Explicit replacements are migrations, never claims of marginal admission.
    """
    if migrated:
        assert explicit is None
        explicit = explicit_chain(sql)
    p = (
        SQLProjection.marginalize(sql) if explicit is None else SQLProjection(explicit)
    ).fit(table)
    con = duckdb.connect()
    try:
        # Parallel window reductions do not have a unique floating bit-answer.
        con.execute("SET threads = 1")
        con.register("__THIS__", table)
        orig = con.execute(f"SELECT * FROM ({sql}) ORDER BY ALL").to_arrow_table()
        for name, params_table in p.params.items():
            con.register(name, params_table)
        rew = con.execute(f"SELECT * FROM ({p.sql}) ORDER BY ALL").to_arrow_table()
    finally:
        con.close()
    assert orig.schema == rew.schema, f"\n{orig.schema}\n!=\n{rew.schema}"
    assert _same(orig.to_pylist(), rew.to_pylist()), (
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
    # NULL is a partition of its own and must be a real params row.
    assert any(
        None in column.to_pylist()
        for params in p.params.values()
        for column in params.columns
    )


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
        "first_value(name) OVER (PARTITION BY country ORDER BY age)",
        "first_value(city IGNORE NULLS) OVER (PARTITION BY country ORDER BY age)",
        "last_value(name) OVER (PARTITION BY country ORDER BY age)",
        "nth_value(name, 2) OVER (PARTITION BY country ORDER BY age)",
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


# --- projection chains and scalar subqueries (loop 3) ------------------------


def test_cte_chain_flattening():
    gate(
        "WITH a AS (SELECT age + 1 AS b, name FROM __THIS__)"
        " SELECT b * 2 AS c, name FROM a",
        migrated=True,
    )


def test_nested_aggregation_dag():
    # Standardize, then aggregate the standardized values: the DAG case.
    gate(
        "WITH c AS (SELECT age - avg(age) OVER () AS cx, country FROM __THIS__)"
        " SELECT cx / stddev_samp(cx) OVER (PARTITION BY country) AS z FROM c",
        migrated=True,
    )


def test_three_level_chain():
    gate(
        "WITH a AS (SELECT age - avg(age) OVER () AS ca, country FROM __THIS__),"
        " b AS (SELECT ca * 2 AS cb, country FROM a)"
        " SELECT cb - avg(cb) OVER (PARTITION BY country) AS m FROM b",
        migrated=True,
    )


def test_derived_table_with_windows():
    gate(
        "SELECT z + 1 AS z1 FROM"
        " (SELECT (age - avg(age) OVER (PARTITION BY country)) AS z FROM __THIS__)"
        " AS sub",
        migrated=True,
    )


def test_upper_level_window_keys_on_projected_expression():
    gate(
        "WITH a AS (SELECT substr(country, 1, 1) AS c1, age FROM __THIS__)"
        " SELECT age - avg(age) OVER (PARTITION BY c1) AS m FROM a",
        migrated=True,
    )


def test_star_through_cte_with_windows():
    gate(
        "WITH a AS (SELECT name, age - avg(age) OVER () AS ca FROM __THIS__)"
        " SELECT * FROM a",
        migrated=True,
    )


def test_scalar_subquery():
    gate("SELECT age / (SELECT max(age) FROM __THIS__) AS r, name FROM __THIS__")


def test_scalar_subquery_with_where_and_group_by_inside():
    gate(
        "SELECT age - (SELECT avg(age) FROM __THIS__ WHERE fare > 8) AS d,"
        " (SELECT count(*) FROM (SELECT country FROM __THIS__ GROUP BY country))"
        " AS n_countries FROM __THIS__",
        explicit=(
            "SELECT age - (SELECT avg(age) FROM __FIT__ WHERE fare > 8) AS d,"
            " (SELECT count(*) FROM (SELECT country FROM __FIT__ GROUP BY country))"
            " AS n_countries FROM __THIS__"
        ),
    )


def test_exists_subquery():
    gate(
        "SELECT EXISTS(SELECT 1 FROM __THIS__ WHERE age > 45) AS any_old FROM __THIS__"
    )


def test_scalar_subquery_inside_cte():
    gate(
        "WITH a AS (SELECT age / (SELECT max(age) FROM __THIS__) AS r FROM __THIS__)"
        " SELECT r - avg(r) OVER () AS rc FROM a",
        migrated=True,
    )


# Original schema-aware computations, without the removed declaration API.


def test_kept_basic_windows_and_star():
    gate(
        "SELECT age - avg(age) OVER (PARTITION BY country) AS m FROM __THIS__",
        schema=True,
    )
    gate("SELECT * FROM __THIS__", schema=True)
    gate("SELECT *, avg(age) OVER () AS m FROM __THIS__", schema=True)


def test_migrated_columns_expansion():
    gate(
        "SELECT COLUMNS('c.*') FROM __THIS__",
        explicit="SELECT country, city FROM __THIS__",
    )
    gate(
        "SELECT COLUMNS('age|fare') FROM __THIS__",
        explicit="SELECT age, fare FROM __THIS__",
    )


def test_migrated_star_modifiers_through_cte():
    gate(
        "WITH a AS (SELECT * FROM __THIS__)"
        " SELECT * EXCLUDE (name) REPLACE (age + 1 AS age) FROM a",
        explicit="SELECT country, city, age + 1 AS age, fare FROM __THIS__",
    )
    gate(
        "WITH a AS (SELECT * EXCLUDE (city) FROM __THIS__)"
        " SELECT age - avg(age) OVER (PARTITION BY country) AS m FROM a",
        explicit=explicit_chain(
            "WITH a AS (SELECT country, age, fare, name FROM __THIS__)"
            " SELECT age - avg(age) OVER (PARTITION BY country) AS m FROM a"
        ),
    )


def test_migrated_star_rename():
    gate(
        "SELECT * RENAME (age AS years) FROM __THIS__",
        explicit="SELECT country, city, age AS years, fare, name FROM __THIS__",
    )


def test_kept_lateral_alias_column_precedence():
    # 'b' is not a column, so the alias applies: (age + 1) * 2.
    gate("SELECT age + 1 AS b, b * 2 AS c FROM __THIS__", schema=True)
    # 'fare' IS a column, so the column wins over the alias.
    gate("SELECT age + 1 AS fare, fare * 2 AS c FROM __THIS__", schema=True)


def test_migrated_struct_access_through_cte_column():
    table = pa.table({"s": [{"f": 1.0}, {"f": 2.0}, {"f": None}], "g": ["a", "a", "b"]})
    gate(
        "WITH a AS (SELECT s, g FROM __THIS__)"
        " SELECT s.f - avg(s.f) OVER (PARTITION BY g) AS d FROM a",
        table,
        migrated=True,
    )


def test_migrated_windows_over_columns_expansion():
    gate(
        "SELECT avg(fare) OVER (PARTITION BY country) AS m,"
        " * EXCLUDE (name) FROM __THIS__",
        explicit=SQLProjection.marginalize(
            "SELECT avg(fare) OVER (PARTITION BY country) AS m,"
            " country, city, age, fare FROM __THIS__"
        ).source,
    )


def test_explicit_column_selection_replaces_declared_schema():
    gate(
        "SELECT age, country FROM __THIS__",
        explicit="SELECT age, country FROM __THIS__",
    )


def test_inspectable_chain_preserves_original_computation():
    gate(
        "WITH c AS (SELECT age - avg(age) OVER () AS cx FROM __THIS__)"
        " SELECT stddev_samp(cx) OVER () AS s FROM c",
        migrated=True,
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
    n = int(os.environ.get("MARGINALIZE_FUZZ_N", "1500"))
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
        inner = f"SELECT {', '.join(exprs)} FROM __THIS__"
        schema = bool(rng.randrange(2))
        shape = rng.randrange(4)
        if shape == 0:
            gate(inner, table, schema=schema)
        elif shape == 1:
            outer = ", ".join(f"e{j} AS f{j}" for j in range(len(exprs)))
            original = f"WITH c AS ({inner}) SELECT {outer} FROM c"
            explicit = (
                f"WITH c AS ({SQLProjection.marginalize(inner).source})"
                f" SELECT {outer} FROM c"
            )
            gate(original, table, schema=schema, explicit=explicit)
        elif shape == 2:
            k_in = rng.choice(["PARTITION BY k1", "PARTITION BY k2", ""])
            k_out = rng.choice(["PARTITION BY k1", "PARTITION BY k2, k1", ""])
            agg2 = rng.choice(["avg", "sum", "median", "stddev_samp"])
            gate(
                f"WITH c AS (SELECT x - avg(x) OVER ({k_in}) AS e0, k1, k2"
                f" FROM __THIS__)"
                f" SELECT e0 - {agg2}(e0) OVER ({k_out}) AS g0 FROM c",
                table,
                schema=schema,
                migrated=True,
            )
        else:
            gate(
                f"SELECT x - (SELECT {rng.choice(['max', 'min', 'avg'])}(x)"
                f" FROM __THIS__) AS s0, {exprs[0]} FROM __THIS__",
                table,
                schema=schema,
            )


# Computations formerly hidden inside SQL-text goldens now use the oracle.
# The old carrier measurements found ulp drift from GROUP BY or solo-window
# fit queries. Keep the full original items and operator chain before picking.
NUMERICAL = pa.table(
    {
        "a": [1.0, 2.0, None, -0.0, 7.0],
        "b": [3.0, 4.0, 5.0, None, 8.0],
        "x": [2.0, -4.0, None, 0.0, 9.0],
        "y": [1.0, 8.0, 4.0, None, 5.0],
        "k": [1, 1, 2, None, 2],
        "o": [2, 1, 3, None, 1],
        "c": ["A", "a", "B", None, "b"],
    }
)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT __THIS__.age - avg(__THIS__.age) OVER () AS d FROM __THIS__",
        "SELECT (age - avg(age) OVER (PARTITION BY country))"
        " / stddev_samp(age) OVER (PARTITION BY country) AS age_z,"
        " avg(age) OVER (PARTITION BY country) AS mu,"
        " fare - avg(fare) OVER () AS fare_c FROM __THIS__",
    ],
    ids=lambda s: s[:56],
)
def test_old_golden_training_computations(sql):
    gate(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT avg(x) OVER (PARTITION BY a, b) AS m FROM __THIS__",
        "SELECT *, avg(a) OVER () AS m FROM __THIS__",
        "SELECT avg(a) OVER (PARTITION BY c) AS x,"
        " avg(b) OVER (PARTITION BY C) AS y FROM __THIS__",
        "SELECT sum(x) OVER (PARTITION BY k ORDER BY o) AS running FROM __THIS__",
        "SELECT rank() OVER (PARTITION BY k ORDER BY o) AS r FROM __THIS__",
        "SELECT first_value(x) OVER (PARTITION BY k ORDER BY o) AS f FROM __THIS__",
        "SELECT sum(x) OVER (PARTITION BY k ORDER BY o"
        " ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS s FROM __THIS__",
        "SELECT avg(x) OVER (PARTITION BY k % 2) AS m FROM __THIS__",
        "SELECT sum(x) OVER (PARTITION BY k ORDER BY o) AS s,"
        " rank() OVER (PARTITION BY k ORDER BY o) AS r FROM __THIS__",
        "SELECT sum(x) OVER (PARTITION BY k ORDER BY k) AS s FROM __THIS__",
        "SELECT rank() OVER (ORDER BY c COLLATE NOCASE) AS r FROM __THIS__",
        "SELECT x / (SELECT max(x) FROM __THIS__) AS xn FROM __THIS__",
        "SELECT EXISTS(SELECT 1 FROM __THIS__ WHERE x > 5) AS any_big FROM __THIS__",
        "SELECT x - (SELECT avg(x) FROM __THIS__ WHERE x > 0) AS d FROM __THIS__",
        "SELECT a + 1 AS b, b * 2 AS c FROM __THIS__",
        "SELECT a FROM __THIS__ AS x",
        "SELECT avg(a) OVER () AS __cf_x FROM __THIS__",
    ],
    ids=lambda s: s[:56],
)
def test_old_golden_numerical_computations(sql):
    explicit = (
        "WITH p AS (SELECT DISTINCT struct_pack(__cf_x := avg(a) OVER()) AS s "
        "FROM __FIT__) SELECT p.s.__cf_x FROM __THIS__ t LEFT JOIN p ON 1 = 1"
        if sql == "SELECT avg(a) OVER () AS __cf_x FROM __THIS__"
        else None
    )
    gate(sql, NUMERICAL, explicit=explicit)


@pytest.mark.parametrize(
    "sql",
    [
        "WITH a AS (SELECT x + 1 AS b FROM __THIS__) SELECT b * 2 AS c FROM a",
        "SELECT c FROM (SELECT x + 1 AS c FROM __THIS__) AS sub",
        "WITH a(z) AS (SELECT x + 1 FROM __THIS__) SELECT z FROM a",
        "WITH c AS (SELECT x - avg(x) OVER () AS cx FROM __THIS__)"
        " SELECT stddev_samp(cx) OVER () AS s FROM c",
        "WITH a AS (SELECT x + 1 AS b, y FROM __THIS__) SELECT * FROM a",
    ],
    ids=lambda s: s[:56],
)
def test_old_golden_migrated_chain_computations(sql):
    gate(sql, NUMERICAL, migrated=True)


def test_old_golden_migrated_expression_key_computation():
    table = NUMERICAL.set_column(
        NUMERICAL.schema.get_field_index("k"),
        "k",
        pa.array(["A", "a", "B", None, "b"]),
    )
    gate(
        "WITH a AS (SELECT lower(k) AS lk, x FROM __THIS__)"
        " SELECT avg(x) OVER (PARTITION BY lk) AS m FROM a",
        table,
        migrated=True,
    )


@pytest.mark.parametrize(
    "original,explicit",
    [
        ("SELECT COLUMNS('a.*') FROM __THIS__", "SELECT aa, ab FROM __THIS__"),
        (
            "SELECT * EXCLUDE (b) REPLACE (a + 1 AS a) RENAME (c AS z) FROM __THIS__",
            "SELECT a + 1 AS a, c AS z, aa, ab FROM __THIS__",
        ),
    ],
    ids=["columns", "modifiers"],
)
def test_old_golden_migrated_schema_computations(original, explicit):
    table = (
        NUMERICAL.select(["a", "b", "c"])
        .append_column("aa", pa.array([1, 2, 3, 4, 5]))
        .append_column("ab", pa.array([6, 7, 8, 9, 10]))
    )
    if original.startswith("SELECT COLUMNS"):
        table = table.select(["aa", "ab", "b"])
    gate(original, table, explicit=explicit)


def test_old_golden_migrated_struct_passthrough():
    table = pa.table({"s": [{"f": 1}, {"f": None}, {"f": 3}]})
    gate(
        "WITH a AS (SELECT s FROM __THIS__) SELECT s.f AS f FROM a",
        table,
        migrated=True,
    )


def test_old_golden_lateral_alias_without_an_input_column():
    gate("SELECT a + 1 AS b, b * 2 AS c FROM __THIS__", NUMERICAL.select(["a"]))


# Retain construct boundaries, not implementation names or exact old prose.
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a FROM __THIS__ WHERE a > 0",
        "SELECT a FROM __THIS__ GROUP BY a",
        "SELECT a FROM __THIS__ GROUP BY a HAVING a > 0",
        "SELECT a FROM __THIS__ QUALIFY row_number() OVER () = 1",
        "SELECT a FROM __THIS__ USING SAMPLE 10",
        "SELECT a FROM __THIS__ ORDER BY a",
        "SELECT a FROM __THIS__ LIMIT 5",
        "SELECT DISTINCT a FROM __THIS__",
        "SELECT a FROM __THIS__ JOIN s ON true",
        "SELECT a FROM other_table",
        "SELECT 1",
        "SELECT a FROM __THIS__ UNION SELECT 1",
        "SELECT 1; SELECT 2",
        "SELECT avg(a) FROM __THIS__",
        "SELECT sum(a) FROM __THIS__",
        "SELECT row_number() OVER () FROM __THIS__",
        "SELECT ntile(4) OVER (ORDER BY a) FROM __THIS__",
        "SELECT lag(a) OVER (PARTITION BY c ORDER BY o) FROM __THIS__",
        "SELECT lead(a) OVER (PARTITION BY c ORDER BY o) FROM __THIS__",
        "SELECT avg(a) OVER (ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM __THIS__",
        "SELECT sum(a) OVER (PARTITION BY c ORDER BY o"
        " ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) FROM __THIS__",
        "SELECT sum(a) OVER (ORDER BY o RANGE BETWEEN UNBOUNDED PRECEDING AND"
        " CURRENT ROW EXCLUDE CURRENT ROW) FROM __THIS__",
        "SELECT sum(a) OVER (ORDER BY o RANGE BETWEEN b PRECEDING AND"
        " CURRENT ROW) FROM __THIS__",
        "SELECT nth_value(a, b) OVER (PARTITION BY c ORDER BY o) FROM __THIS__",
        "SELECT avg(sum(a)) OVER () FROM __THIS__",
        "SELECT avg(a) FILTER (WHERE avg(b) > 0) OVER () FROM __THIS__",
        "SELECT avg(a) OVER (PARTITION BY (SELECT 1)) FROM __THIS__",
        "SELECT x IN (SELECT y FROM __THIS__) FROM __THIS__",
        "SELECT x = ANY(SELECT y FROM __THIS__) FROM __THIS__",
        "SELECT (SELECT max(y) FROM other) FROM __THIS__",
        "WITH a AS (SELECT 1) SELECT x FROM __THIS__",
        "WITH a AS (SELECT * FROM a) SELECT * FROM a",
        "WITH a AS (SELECT x AS n, y AS n FROM __THIS__) SELECT n FROM a",
        "WITH a AS (SELECT x FROM __THIS__) SELECT __THIS__.x FROM a",
        "WITH a AS (SELECT x + 1 AS s FROM __THIS__) SELECT s.f FROM a",
        "WITH a AS (SELECT * EXCLUDE (x) FROM __THIS__) SELECT y FROM a",
        "SELECT a FROM (SELECT b FROM __THIS__ WHERE b > 0)",
        "SELECT COLUMNS('a.*') FROM __THIS__",
        "SELECT COLUMNS(c -> c LIKE 'a%') FROM __THIS__",
        "SELECT COLUMNS('a.*') + 1 FROM __THIS__",
        "SELECT min(COLUMNS('a.*')) FROM __THIS__",
        "SELECT * EXCLUDE (nope) FROM __THIS__",
        "SELECT FROM WHERE",
    ],
    ids=lambda s: s[:56],
)
def test_retained_marginal_admission_refusals(sql):
    with pytest.raises(TransformError):
        SQLProjection.marginalize(sql)


def test_unbound_positional_parameter_has_a_normal_binding_refusal():
    fitted = SQLProjection("SELECT $1 + a FROM __THIS__").fit(NUMERICAL)
    with pytest.raises(duckdb.InvalidInputException, match="parameters"):
        fitted.transform(NUMERICAL)
