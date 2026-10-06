"""The walk must resolve names, not compare strings.

Four failures, one cause. Every one is loud, so C5 holds — but P7 does not:
three surfaced as raw DuckDB errors at fit, on queries ``run`` executes fine.

The general gate is the last test in this module: nothing ``run`` accepts may
die at ``fit`` with an error that is not ours. Each specific case below is a
row of that.
"""

import duckdb
import pyarrow as pa
import pytest
from sklearn.preprocessing import StandardScaler

from sql_transform import SQLTransform, TransformError, run
from sql_transform._program import Program
from sql_transform._udf import OrderSensitive, PythonTransform, PythonUDF

D = pa.table({"grp": ["a", "a", "b"], "price": [1.0, 3.0, 100.0]})
T = pa.table({"grp": ["a", "b"], "price": [7.0, 9.0]})


# ------------------------------------------------ _reads must follow a CTE


def test_a_subtree_reading_this_through_a_cte_is_not_frozen():
    """``_reads`` scanned for literal ``BASE_TABLE`` nodes, so a CTE reference
    hid ``__THIS__`` and the subtree was frozen — then died at fit, because
    ``__THIS__`` is not bound when parameters are computed."""
    t = SQLTransform(
        "WITH w AS (SELECT * FROM __THIS__) "
        "SELECT t.grp, (SELECT count(*) FROM w, (SELECT price FROM __FIT__) f) AS c "
        "FROM __THIS__ t"
    )
    assert t.fit(D).transform(D).to_pylist() == run(t, D).to_pylist()


def test_a_cte_reading_only_fit_still_freezes():
    """The fix must not cost the freezing it was meant to protect."""
    t = SQLTransform(
        "WITH w AS (SELECT avg(price) m FROM __FIT__) "
        "SELECT t.price / w.m AS z FROM __THIS__ t, w"
    )
    fitted = t.fit(D)
    assert "__FIT__" not in fitted.sql
    assert fitted.transform(D).to_pylist() == run(t, D).to_pylist()


def test_a_chain_of_ctes_propagates_what_it_reads():
    t = SQLTransform(
        "WITH a AS (SELECT * FROM __THIS__), b AS (SELECT * FROM a) "
        "SELECT t.grp, (SELECT count(*) FROM b, (SELECT price FROM __FIT__) f) AS c "
        "FROM __THIS__ t"
    )
    assert t.fit(D).transform(D).to_pylist() == run(t, D).to_pylist()


# ------------------------------------ an unqualified correlated reference


def test_an_unqualified_correlated_fit_reference_refuses_by_name():
    """``_correlation`` only inspected qualified references, so this was
    frozen and died with a raw ``BinderException``.

    Whether an unqualified name resolves inward or outward cannot be known
    before the data exists — ``__FIT__`` has no schema at construction — so
    unlike its qualified sibling this one is caught at fit. It is *named*,
    which is the part that matters.
    """
    # `cat` exists only in the outer relation, so DuckDB resolves it outward.
    # Had `__FIT__` carried a `cat` too it would bind inward and be correct —
    # which is exactly why this cannot be decided without the data.
    t = SQLTransform(
        "SELECT t.cat, (SELECT avg(f.price) FROM __FIT__ f WHERE f.grp = cat) AS m "
        "FROM (SELECT grp AS cat, price FROM __THIS__) t"
    )
    with pytest.raises(TransformError) as caught:
        t.fit(D)
    assert not isinstance(caught.value, duckdb.Error)
    assert "cat" in str(caught.value)


def test_an_unqualified_name_that_binds_inward_is_left_alone():
    """The same shape, except ``__FIT__`` has the column. DuckDB binds it
    locally, nothing is correlated, and the freeze is correct."""
    t = SQLTransform(
        "SELECT t.grp, (SELECT avg(f.price) FROM __FIT__ f WHERE f.grp = grp) AS m "
        "FROM __THIS__ t"
    )
    assert t.fit(D).transform(D).to_pylist() == run(t, D).to_pylist()


def test_the_qualified_form_is_lifted_because_we_can_see_it():
    """The other half of the pair. Being able to see the correlation without
    the data is what lets it become a keyed table rather than a refusal — the
    unqualified sibling above cannot be read either way until fit."""
    t = SQLTransform(
        "SELECT t.grp, (SELECT avg(price) FROM __FIT__ f WHERE f.grp = t.grp) AS m "
        "FROM __THIS__ t"
    )
    fitted = t.fit(D)
    assert "__FIT__" not in fitted.sql
    assert fitted.transform(T).to_pylist() == [
        {"grp": "a", "m": 2.0},
        {"grp": "b", "m": 100.0},
    ]


def test_an_ordinary_unqualified_column_is_untouched():
    """The canonical example in the guide is unqualified. Refusing on the mere
    presence of an unqualified name would refuse almost everything."""
    t = SQLTransform(
        "SELECT t.price / s.m AS z FROM __THIS__ t, "
        "(SELECT avg(price) m FROM __FIT__) s"
    )
    assert t.fit(D).transform(D).to_pylist() == run(t, D).to_pylist()


# ------------------------------------------------------- a recursive CTE


def test_a_recursive_cte_reading_fit_refuses_by_name():
    """``freeze`` hoisted the body into a standalone statement, but a recursive
    CTE's self-reference is bound by the enclosing entry key, not by anything
    inside the body — so nothing inside it can be lifted out.

    It used to be left live with the training set as the parameter. Refused
    now: ``(SELECT count(*) FROM __FIT__)`` needs one number and was shipping
    every row to get it."""
    with pytest.raises(TransformError, match="training set"):
        SQLTransform(
            "WITH RECURSIVE c(i) AS ("
            "  SELECT 1 UNION ALL"
            "  SELECT i+1 FROM c WHERE i < (SELECT count(*) FROM __FIT__)"
            ") SELECT t.grp, (SELECT count(*) FROM c) AS n FROM __THIS__ t"
        )


def test_a_plain_cte_named_like_its_own_body_still_freezes():
    t = SQLTransform(
        "WITH c AS (SELECT count(*) n FROM __FIT__) "
        "SELECT t.grp, c.n FROM __THIS__ t, c"
    )
    assert t.fit(D).transform(D).to_pylist() == run(t, D).to_pylist()


# ------------------------------------------------------ case-insensitivity


@pytest.mark.parametrize(
    "defined,used", [("Sales", "sales"), ("sales", "SALES"), ("SaLeS", "sAlEs")]
)
def test_cte_names_match_the_way_duckdb_matches_them(defined, used):
    """``ctes`` was a set of exact strings; DuckDB's binder is
    case-insensitive, so valid SQL was refused as an unknown free name."""
    t = SQLTransform(
        f"WITH {defined} AS (SELECT avg(price) m FROM __FIT__) "
        f"SELECT t.price / {used}.m AS z FROM __THIS__ t, {used}"
    )
    assert t.fit(D).transform(D).to_pylist() == run(t, D).to_pylist()


# ----------------------------------------------------------- the general gate


ACCEPTED = [
    "WITH w AS (SELECT * FROM __THIS__) "
    "SELECT t.grp, (SELECT count(*) FROM w, __FIT__) AS c FROM __THIS__ t",
    "WITH a AS (SELECT * FROM __THIS__), b AS (SELECT * FROM a) "
    "SELECT t.grp, (SELECT count(*) FROM b, __FIT__) AS c FROM __THIS__ t",
    "WITH Up AS (SELECT avg(price) m FROM __FIT__) "
    "SELECT t.price / up.m AS z FROM __THIS__ t, up",
    "WITH RECURSIVE c(i) AS ("
    "  SELECT 1 UNION ALL"
    "  SELECT i+1 FROM c WHERE i < (SELECT count(*) FROM __FIT__)"
    ") SELECT t.grp, (SELECT count(*) FROM c) AS n FROM __THIS__ t",
    "SELECT t.price / s.m AS z FROM __THIS__ t, (SELECT avg(price) m FROM __FIT__) s",
]


@pytest.mark.parametrize("sql", ACCEPTED)
def test_nothing_run_accepts_dies_at_fit_with_someone_elses_error(sql):
    """The property the four specific gates are instances of.

    A construct either serves or refuses *by our name*. A raw
    ``CatalogException`` or ``BinderException`` escaping from ``fit``, on a
    query ``run`` executes happily, is the model failing to know its own mind.

    Construction is inside the ``try`` because refusing there is the *better*
    outcome (P7), and two of these now do: a bare ``FROM __FIT__`` beside
    ``__THIS__`` would put the training set in the artifact.
    """
    try:
        t = SQLTransform(sql)
        assert t.fit(D).transform(D).to_pylist() == run(t, D).to_pylist()
    except TransformError as refusal:
        assert not isinstance(refusal, duckdb.Error)


RAW = (
    "WITH p AS (SELECT grp, sc_fit(price) AS iid FROM __FIT__ GROUP BY grp) "
    "SELECT sc_transform(p.iid, t.price).price AS z "
    "FROM __THIS__ t LEFT JOIN p ON t.grp IS NOT DISTINCT FROM p.grp"
)


def _identity(value):
    return value


def test_projection_only_python_admission_does_not_change_general_run():
    scaler = StandardScaler()
    scalar = PythonUDF("inc", _identity, pa.schema([("v", pa.int64())]), pa.int64())
    with pytest.raises(TransformError, match="only in SQLProjection"):
        SQLTransform(RAW, captured={"sc": scaler})
    with pytest.raises(TransformError, match="only in SQLProjection"):
        SQLTransform("SELECT inc(price) FROM __THIS__", captured={"inc": scalar})
    assert run(SQLTransform("SELECT price + 1 AS z FROM __THIS__"), T)[
        "z"
    ].to_pylist() == [8.0, 10.0]


def test_raw_replay_retains_only_author_captures_and_fits_one_scope_once():
    captured = {"sc": StandardScaler()}
    source = RAW.replace(
        "sc_transform(p.iid, t.price).price AS z",
        "sc_transform(p.iid, t.price).price AS z, "
        "sc_transform(p.iid, struct_pack(price := t.price + 2)).price AS q",
    )
    program = Program.compile(source, {}, captured=captured, row_udfs=True)
    replay = Program.compile(
        program.source, {}, captured=program.captured, row_udfs=True
    )
    assert program.captured is captured
    assert replay.captured is captured
    assert set(captured) == {"sc"}
    assert len(program.estimators) == 1
    fitted = program.fit(D)
    assert len(fitted.instances) == 2
    assert all(isinstance(udf, PythonTransform) for udf in fitted.udfs.values())
    for udf in fitted.udfs.values():
        assert all(fitted.instances[iid] is est for iid, est in udf.instances.items())
    assert fitted.transform(T).to_pylist() == [
        {"z": 5.0, "q": 7.0},
        {"z": -91.0, "q": -89.0},
    ]
    with pytest.raises(TransformError, match="Program.run"):
        program.run(D)


def test_separately_authored_raw_sources_have_separate_bindings_and_instances():
    program = Program.compile(
        "WITH p AS (SELECT sc_fit(price) iid FROM __FIT__), "
        "q AS (SELECT sc_fit(price) iid FROM __FIT__) "
        "SELECT sc_transform(p.iid, t.price).price a, "
        "sc_transform(q.iid, t.price).price b FROM __THIS__ t, p, q",
        {"sc": StandardScaler()},
        row_udfs=True,
    )
    assert len(program.estimators) == 2
    fitted = program.fit(D)
    assert len(fitted.instances) == 2
    assert len(fitted.udfs) == 2
    indices = [set(udf.instances) for udf in fitted.udfs.values()]
    assert not indices[0] & indices[1]
    rows = fitted.transform(T).to_pylist()
    assert all(row["a"] == row["b"] for row in rows)


def test_inline_raw_fit_is_bound_without_changing_public_run():
    program = Program.compile(
        "SELECT sc_transform((SELECT sc_fit(price) FROM __FIT__), price).price AS z "
        "FROM __THIS__",
        {"sc": StandardScaler()},
        row_udfs=True,
    )
    fitted = program.fit(pa.table({"price": [1.0, 3.0]}))
    assert fitted.transform(pa.table({"price": [4.0]})).to_pylist() == [{"z": 2.0}]
    assert all(
        pa.types.is_int64(table.schema.types[0]) for table in fitted.params.values()
    )


@pytest.mark.parametrize(
    "params",
    [
        "SELECT grp, sc_fit(price) iid, avg(price) m FROM __FIT__ GROUP BY grp",
        "SELECT grp, sc_fit(price) iid, sc_fit(price) other FROM __FIT__ GROUP BY grp",
        "SELECT grp, sc_fit(price) iid FROM __FIT__ WHERE price > 0 GROUP BY grp",
        "SELECT DISTINCT grp, sc_fit(price) iid FROM __FIT__ GROUP BY grp",
        "SELECT grp, sc_fit(price) iid FROM __FIT__ GROUP BY grp HAVING count(*) > 0",
        "SELECT grp, sc_fit(price) iid FROM __FIT__ GROUP BY grp ORDER BY grp",
        "SELECT grp, sc_fit(price) iid FROM __FIT__ GROUP BY grp LIMIT 1",
        "SELECT grp, sc_fit(price) iid "
        "FROM (SELECT * FROM __FIT__ WHERE price > 0) f GROUP BY grp",
        "SELECT grp, sc_fit(price) iid "
        "FROM (SELECT * FROM (SELECT * FROM __FIT__) f) q GROUP BY grp",
        "SELECT sc_fit(price) iid FROM (SELECT avg(price) price FROM __FIT__) f",
        "SELECT grp, sc_fit(price) iid "
        "FROM __FIT__ f JOIN __FIT__ q USING (grp,price) GROUP BY grp",
        "SELECT grp, sc_fit(price) iid FROM __THIS__ GROUP BY grp",
    ],
)
def test_noncanonical_raw_params_refuse_at_compile(params):
    with pytest.raises(TransformError):
        Program.compile(
            f"WITH p AS ({params}) SELECT sc_transform(p.iid, t.price).price z "
            "FROM __THIS__ t LEFT JOIN p ON t.grp IS NOT DISTINCT FROM p.grp",
            {"sc": StandardScaler()},
            row_udfs=True,
        )


@pytest.mark.parametrize(
    "iid",
    [
        "0",
        "p.iid + 0",
        "coalesce(p.iid, 0)",
        "struct_pack(type := 'sc', id := p.iid)",
        "t.price",
        "p.grp",
    ],
)
def test_raw_ids_cannot_be_arbitrary_expressions(iid):
    with pytest.raises(TransformError, match="canonical"):
        Program.compile(
            RAW.replace("p.iid, t.price", f"{iid}, t.price"),
            {"sc": StandardScaler()},
            row_udfs=True,
        )


@pytest.mark.parametrize(
    "sql",
    [
        "WITH f AS (SELECT * FROM __FIT__), p AS (SELECT sc_fit(price) iid FROM f) "
        "SELECT sc_transform(p.iid, t.price).price FROM __THIS__ t, p",
        "WITH p AS (SELECT sc_fit(price) iid FROM __FIT__), q AS (SELECT iid FROM p) "
        "SELECT sc_transform(q.iid, t.price).price FROM __THIS__ t, q",
        "SELECT sc_transform(sc_fit(price) OVER (), price).price FROM __THIS__",
        "SELECT sc_fit(price) OVER () AS iid FROM __THIS__",
        "SELECT sc_transform((SELECT sc_fit(price) FROM __FIT__),"
        " price).price FROM __FIT__",
        "SELECT sc_transform((SELECT sc_fit(struct_pack(v := sc(price))) "
        "FROM __FIT__), "
        "struct_pack(v := price)).v FROM __THIS__",
    ],
)
def test_raw_scope_passthrough_windows_and_fit_only_apply_refuse(sql):
    with pytest.raises(TransformError):
        Program.compile(sql, {"sc": StandardScaler()}, row_udfs=True)


def test_raw_bundle_fields_are_ordered_and_named():
    sql = (
        "WITH p AS (SELECT sc_fit(struct_pack(a := price, b := price + 1)) "
        "iid FROM __FIT__) "
        "SELECT sc_transform(p.iid, struct_pack(b := t.price, a := t.price)).a "
        "FROM __THIS__ t, p"
    )
    with pytest.raises(TransformError, match="match fit fields in order"):
        Program.compile(sql, {"sc": StandardScaler()}, row_udfs=True)
    with pytest.raises(TransformError, match="nested bundles"):
        Program.compile(
            RAW.replace(
                "sc_fit(price)", "sc_fit(struct_pack(price := struct_pack(v := price)))"
            ),
            {"sc": StandardScaler()},
            row_udfs=True,
        )


def test_order_sensitive_requires_argument_order_not_a_window_order():
    with pytest.raises(TransformError, match="in-call ORDER BY"):
        Program.compile(RAW, {"sc": OrderSensitive(StandardScaler())}, row_udfs=True)
    program = Program.compile(
        RAW.replace("sc_fit(price)", "sc_fit(price ORDER BY price DESC NULLS FIRST)"),
        {"sc": OrderSensitive(StandardScaler())},
        row_udfs=True,
    )
    assert program.fit(D).transform(T)["z"].to_pylist() == [5.0, -91.0]


@pytest.mark.parametrize("column", ["__cf_row", "__CF_FIT_ROW"])
def test_raw_fit_reserved_ordinals_refuse_before_numbering(column):
    program = Program.compile(RAW, {"sc": StandardScaler()}, row_udfs=True)
    with pytest.raises(TransformError, match="reserved ordinal"):
        program.fit(D.append_column(column, pa.array([1, 2, 3])))


def test_scalar_declared_integer_does_not_round_through_double():
    scalar = PythonUDF("exact", _identity, pa.schema([("v", pa.int64())]), pa.int64())
    program = Program.compile(
        "SELECT exact(v) v FROM __THIS__", {"exact": scalar}, row_udfs=True
    )
    data = pa.table({"v": pa.array([2**53 + 1, None], pa.int64())})
    output = program.fit(data).transform(data)
    assert output.schema == data.schema
    assert output.to_pylist() == data.to_pylist()


def test_scalar_case_fold_replay_keeps_author_capture_identity():
    scalar = PythonUDF("Exact", _identity, pa.schema([("v", pa.int64())]), pa.int64())
    captured = {"Exact": scalar}
    program = Program.compile(
        "SELECT EXACT(v) v FROM __THIS__", {}, captured=captured, row_udfs=True
    )
    replay = Program.compile(program.source, {}, captured=captured, row_udfs=True)
    assert replay.captured is captured
    assert set(captured) == {"Exact"}
    assert set(replay.udfs) == {"exact"}
    assert replay.fit(D).transform(pa.table({"v": [7]}))["v"].to_pylist() == [7]


def test_scalar_arity_and_sql_name_collisions_refuse_at_compile():
    scalar = PythonUDF("exact", _identity, pa.schema([("v", pa.int64())]), pa.int64())
    with pytest.raises(TransformError, match="expected 1"):
        Program.compile(
            "SELECT exact(v,v) FROM __THIS__", {"exact": scalar}, row_udfs=True
        )
    with pytest.raises(TransformError, match="collide case-insensitively"):
        Program.compile(
            "SELECT exact(v) FROM __THIS__",
            {"exact": scalar, "Exact": scalar},
            row_udfs=True,
        )
    with pytest.raises(TransformError, match="collides with a DuckDB builtin"):
        Program.compile("SELECT abs(v) FROM __THIS__", {"abs": scalar}, row_udfs=True)
    with pytest.raises(TransformError, match="object named"):
        Program.compile(
            "SELECT other(v) FROM __THIS__", {"other": scalar}, row_udfs=True
        )


@pytest.mark.parametrize(
    "expression",
    [
        "sc_transform(p.iid, t.price).price.other",
        "struct_extract(sc_transform(p.iid, t.price), t.grp)",
        "struct_extract_at(sc_transform(p.iid, t.price), 1)",
    ],
)
def test_raw_field_access_requires_one_literal_learned_field(expression):
    with pytest.raises(TransformError, match="field"):
        Program.compile(
            RAW.replace("sc_transform(p.iid, t.price).price", expression),
            {"sc": StandardScaler()},
            row_udfs=True,
        )


def test_raw_projected_source_keeps_sql_windows_and_independent_scalar_queries():
    program = Program.compile(
        "WITH p AS (SELECT grp, sc_fit(cv) iid FROM ("
        "SELECT grp, price - avg(price) OVER (PARTITION BY grp) "
        "+ (SELECT count(*) FROM __FIT__) * 0 AS cv FROM __FIT__"
        ") f GROUP BY grp) "
        "SELECT sc_transform(p.iid, struct_pack(cv := t.price)).cv z "
        "FROM __THIS__ t LEFT JOIN p ON t.grp IS NOT DISTINCT FROM p.grp",
        {"sc": StandardScaler()},
        row_udfs=True,
    )
    fitted = program.fit(D)
    assert fitted.transform(T)["z"].to_pylist() == [7.0, 9.0]
    assert all(
        "__cf_fit_row" not in table.column_names for table in fitted.params.values()
    )


def test_raw_sources_and_function_captures_preserve_case_folded_replay():
    captured = {"SC": StandardScaler()}
    program = Program.compile(RAW, {}, captured=captured, row_udfs=True)
    replay = Program.compile(program.source, {}, captured=captured, row_udfs=True)
    assert replay.captured is captured
    assert set(captured) == {"SC"}
    assert replay.fit(D).transform(T)["z"].to_pylist() == [5.0, -91.0]


@pytest.mark.parametrize(
    "value",
    [
        StandardScaler(),
        PythonUDF("exact", _identity, pa.schema([("v", pa.int64())]), pa.int64()),
    ],
)
def test_explicit_python_function_capture_cannot_be_used_as_a_relation(value):
    with pytest.raises(TransformError, match="not a relation"):
        Program.compile(
            "SELECT * FROM member", {}, captured={"member": value}, row_udfs=True
        )


def test_unaliased_raw_fit_id_keeps_the_authored_duckdb_output_name():
    program = Program.compile(
        "WITH p AS (SELECT sc_fit(price) FROM __FIT__) "
        'SELECT sc_transform(p."sc_fit(price)", t.price).price AS z FROM __THIS__ t, p',
        {"sc": StandardScaler()},
        row_udfs=True,
    )
    fitted = program.fit(pa.table({"price": [1.0, 3.0]}))
    assert fitted.transform(pa.table({"price": [4.0]})).to_pylist() == [{"z": 2.0}]


@pytest.mark.parametrize("group", ["g", "1"])
def test_raw_group_keys_can_use_their_output_alias_or_position(group):
    source = RAW.replace(
        "SELECT grp, sc_fit(price) AS iid FROM __FIT__ GROUP BY grp",
        f"SELECT grp AS g, sc_fit(price) AS iid FROM __FIT__ GROUP BY {group}",
    ).replace("p.grp", "p.g")
    program = Program.compile(source, {"sc": StandardScaler()}, row_udfs=True)
    assert program.fit(D).transform(T)["z"].to_pylist() == [5.0, -91.0]
