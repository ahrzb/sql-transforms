"""Executable twin of packages/confit/docs/known-limitations.md.

DELIBERATE limitations are asserted here: the SQL that hits each and the
named build-time rejection (or, for contract choices, the chosen behavior).
Coverage is partial, not total -- the divergence ledger names the gaps.
If an engine change lifts one of these, the test fails — update the doc in
the same commit. Section numbers mirror the document.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from confit.oracle import Oracle, Trap
from test_duckdb_interpreter import duck_check, static

T = pa.schema([pa.field("a", pa.int64(), nullable=False), pa.field("s", pa.string())])


def build(sql, statics=None):
    return DuckDBInferFn(sql, row_tables={"__THIS__": T}, static_tables=statics or {})


def rejects(sql, needle, statics=None):
    with pytest.raises(ValueError, match=needle):
        build(sql, statics)


# ---- 1. The specialization bargain ----------------------------------------


def test_non_constant_regex_pattern_rejects():
    # Regexes compile ONCE at prepare; per-row compilation is the opposite
    # of specialization.
    rejects("SELECT regexp_matches(s, s) FROM __THIS__", "non-constant regex")
    rejects(
        "SELECT regexp_replace(s, 'a', s) FROM __THIS__",
        "non-constant regexp_replace replacement",
    )


def test_static_tables_are_frozen_unique_key_maps():
    dup = static({"id": "int", "v": "int"}, [{"id": 1, "v": 1}, {"id": 1, "v": 2}])
    # Duplicate keys = 1:N multiplicity: rejected under the DEFAULT shapes,
    # served under the opt-in shape='many'.
    rejects(
        "SELECT v FROM __THIS__ JOIN d ON a = d.id",
        "duplicate map key",
        {"d": dup},
    )
    fn = DuckDBInferFn(
        "SELECT v FROM __THIS__ JOIN d ON a = d.id",
        row_tables={"__THIS__": T},
        static_tables={"d": dup},
        shape="many",
    )
    got = sorted(r["v"] for r in fn.infer_rows([{"a": 1, "s": None}]))
    assert got == [1, 2]
    # NULL VALUES serve (they ride as validity+payload pairs); only NULL
    # KEYS keep the drop rule, a NULL never equi-matching.
    withnull = static({"id": "int", "v": "int?"}, [{"id": 1, "v": None}])
    duck_check(
        "SELECT v FROM __THIS__ JOIN d ON a = d.id",
        {"a": "int", "s": "str?"},
        [{"a": 1, "s": None}],
        {"d": withnull},
    )


def test_dynamic_self_join_rejects():
    # Default shapes: the named rejection. Under shape='many' the batch
    # becomes the build side and ON self-joins SERVE.
    rejects(
        "SELECT t2.a FROM __THIS__ JOIN __THIS__ t2 ON __THIS__.a = t2.a",
        "dynamic table",
    )
    fn = DuckDBInferFn(
        "SELECT t2.a FROM __THIS__ JOIN __THIS__ t2 ON __THIS__.a = t2.a",
        row_tables={"__THIS__": T},
        static_tables={},
        shape="many",
    )
    got = sorted(
        r["a"] for r in fn.infer_rows([{"a": 1, "s": None}, {"a": 2, "s": None}])
    )
    assert got == [1, 2]
    # USING/NATURAL self-joins serve too (differentials in
    # test_duckdb_stageb_many.py); the default shapes still reject them.
    rejects("SELECT * FROM __THIS__ t1 JOIN __THIS__ t2 USING (a)", "dynamic table")


# ---- 2. Out of scope for row-serving --------------------------------------


@pytest.mark.parametrize(
    ("sql", "needle"),
    [
        ("SELECT sum(a) FROM __THIS__", "aggregation is not served"),
        ("SELECT a FROM __THIS__ GROUP BY a", "aggregation"),
        ("SELECT a FROM __THIS__ ORDER BY a", "ORDER BY"),
        ("SELECT a FROM __THIS__ LIMIT 5", "LIMIT"),
        ("SELECT DISTINCT a FROM __THIS__", "DISTINCT"),
        ("WITH c AS (SELECT 1) SELECT a FROM __THIS__", "common table"),
        ("SELECT a FROM __THIS__ UNION SELECT a FROM __THIS__", "UNION"),
        ("SELECT rowid FROM __THIS__", "rowid"),
        (
            "SELECT a FROM __THIS__ FULL OUTER JOIN x ON a = x.a",
            "join type",
        ),
    ],
)
def test_whole_relation_constructs_reject(sql, needle):
    rejects(sql, needle)


# ---- 3. Type-system boundaries ---------------------------------------------


def test_non_scalar_row_columns_reject():
    L = pa.schema([pa.field("xs", pa.list_(pa.int64()), nullable=False)])
    with pytest.raises(ValueError, match="non-scalar"):
        DuckDBInferFn(
            "SELECT xs FROM __THIS__", row_tables={"__THIS__": L}, static_tables={}
        )


def test_non_scalar_rejection_is_reference_time():
    # The rejection lands at REFERENCE, not at construction — an unreferenced
    # list/timestamp field does not block a scalar query, and star modifiers
    # can remove one. Referenced (incl. via *) keeps the named error.
    L = pa.schema(
        [
            pa.field("a", pa.int64(), nullable=False),
            pa.field("xs", pa.list_(pa.int64())),
        ]
    )
    DuckDBInferFn(
        "SELECT a FROM __THIS__", row_tables={"__THIS__": L}, static_tables={}
    )
    DuckDBInferFn(
        "SELECT * EXCLUDE (xs) FROM __THIS__",
        row_tables={"__THIS__": L},
        static_tables={},
    )
    for sql in ["SELECT xs FROM __THIS__", "SELECT * FROM __THIS__"]:
        with pytest.raises(ValueError, match="non-scalar"):
            DuckDBInferFn(sql, row_tables={"__THIS__": L}, static_tables={})


def test_struct_whole_value_rejects_but_fields_serve():
    # Structs of scalars serve AS FIELDS (flattened to lanes); the struct as
    # a whole value stays a named non-scalar rejection.
    M = pa.schema([pa.field("a", pa.struct([pa.field("i", pa.int64())]))])
    fn = DuckDBInferFn(
        "SELECT a.i FROM __THIS__",
        row_tables={"__THIS__": M},
        static_tables={},
    )
    assert fn.infer_rows([{"a": {"i": 5}}]) == [{"i": 5}]
    with pytest.raises(ValueError, match="whole value"):
        DuckDBInferFn(
            "SELECT a FROM __THIS__", row_tables={"__THIS__": M}, static_tables={}
        )
    with pytest.raises(ValueError, match="unsupported"):
        DuckDBInferFn(
            "SELECT a['i'] FROM __THIS__", row_tables={"__THIS__": M}, static_tables={}
        )


def test_list_valued_regexp_forms_reject():
    # Gated on list types, not on regex semantics.
    rejects("SELECT regexp_extract_all(s, 'a') FROM __THIS__", "list-valued")
    rejects("SELECT regexp_split_to_array(s, 'a') FROM __THIS__", "regexp_split")


def test_ubigint_static_payloads_reject():
    """Refused at the TYPE, not at the value's range.

    Riding a uint64 static on the i64 lane would catch only a payload past
    i64 — while every in-range one would emit int64 where DuckDB emits
    UINT64, a schema divergence with no refusal. Refusing the type refuses
    both, and names it."""
    big = pa.table({"id": pa.array([2**64 - 1], pa.uint64()), "v": [1]})
    rejects(
        "SELECT v FROM __THIS__ JOIN d ON a = d.id",
        "has type uint64",
        {"d": big},
    )


# ---- 4. Semantics descoped after measurement -------------------------------


@pytest.mark.parametrize(
    ("sql", "needle"),
    [
        # ^ IS pow, but sqlparser's precedence would compute the wrong tree.
        ("SELECT a ^ 2 FROM __THIS__", "precedence"),
        # Regex reject list: measured RE2 <-> rust-regex divergences.
        ("SELECT regexp_matches(s, 'a\\B') FROM __THIS__", "B in a regex"),
        ("SELECT regexp_matches(s, '\\Qab\\E') FROM __THIS__", "literal quoting"),
        ("SELECT regexp_matches(s, 'a*+') FROM __THIS__", "stacked"),
        (
            "SELECT regexp_matches(s, '(?P<n>a)(?P<n>b)') FROM __THIS__",
            "duplicate regex capture group",
        ),
        ("SELECT regexp_matches(s, 'a{1001}') FROM __THIS__", "repetition bound"),
        # Standing-fuzzer classes (pins-waveB/fuzzer-task54.json): each is
        # a measured silent-wrong-answer risk in rust-regex.
        ("SELECT regexp_matches(s, '(a)x\\1') FROM __THIS__", "backref"),
        ("SELECT regexp_matches(s, 'a?*') FROM __THIS__", "quantifi"),
        ("SELECT regexp_matches(s, 'a{2}*') FROM __THIS__", "quantifi"),
        ("SELECT regexp_matches(s, '(a{100}){20}') FROM __THIS__", "repetition"),
        ("SELECT regexp_matches(s, 'a{1, 3}') FROM __THIS__", "unsupported|parse"),
        ("SELECT regexp_matches(s, '[a--b]') FROM __THIS__", "unsupported"),
        ("SELECT regexp_matches(s, '[a-\\d]') FROM __THIS__", "Perl class endpoint"),
        ("SELECT regexp_matches(s, '(x){0}') FROM __THIS__", "unsupported"),
        ("SELECT regexp_matches(s, '^$') FROM __THIS__", "anchor-only"),
        # Seed-20260728 fuzzer classes: leading-$ prefix optimization bug;
        # RE2 program-size budget ('pattern too large' in DuckDB).
        ("SELECT regexp_matches(s, '$h') FROM __THIS__", "non-final position"),
        (
            "SELECT regexp_matches(s, '(\\p{L}){1,500}') FROM __THIS__",
            "program-size budget",
        ),
        # Not implemented in DuckDB itself.
        ("SELECT s SIMILAR TO 'a' ESCAPE 'x' FROM __THIS__", "escape"),
        # Exec-time conversion order (an empty input succeeds in DuckDB).
        ("SELECT a IN ('abc') FROM __THIS__", "non-numeric"),
        # Only bare COLUMNS as a select item is served.
        ("SELECT COLUMNS('a') + 1 FROM __THIS__", "COLUMNS"),
    ],
)
def test_measured_descopes_reject(sql, needle):
    rejects(sql, needle)


@pytest.mark.parametrize("qual", ["d.a", "__THIS__.a"])
def test_using_unmerge_exclude_rejects(qual):
    d = static({"a": "int", "v": "int"}, [{"a": 1, "v": 10}])
    # DuckDB UNMERGES the coalesced column under either qualifier (it comes
    # back at the right table's position with its values) — measured, not
    # modeled. The left-qualified form once served (w, v) where DuckDB
    # answers (w, a, v).
    rejects(
        f"SELECT * EXCLUDE ({qual}) FROM __THIS__ JOIN d USING (a)",
        "USING-merged",
        {"d": d},
    )


def test_unqualified_exclude_of_a_using_key_serves():
    d = static({"a": "int", "v": "int"}, [{"a": 1, "v": 10}])
    duck_check(
        "SELECT * EXCLUDE (a) FROM __THIS__ JOIN d USING (a)",
        {"a": "int", "s": "str?"},
        [{"a": 1, "s": "x"}, {"a": 2, "s": None}],
        {"d": d},
    )


@pytest.mark.parametrize(
    "target",
    [
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "HUGEINT",
        "UHUGEINT",
        "FLOAT",
        "REAL",
        "FLOAT4",
        "DECIMAL(3,1)",
        "NUMERIC",
        "INTERVAL",
        "DATE",
    ],
)
@pytest.mark.parametrize("form", ["CAST({e} AS {t})", "TRY_CAST({e} AS {t})"])
def test_a_cast_target_without_a_lane_refuses(target, form):
    # Computing these in the nearest lane served values DuckDB does not:
    # CAST(-1 AS UINTEGER) errors there, CAST(16777217 AS FLOAT) rounds to
    # 16777216, CAST(1.25 AS DECIMAL(3,1)) is 1.3, INTERVAL is no integer.
    rejects(f"SELECT {form.format(e='a', t=target)} FROM __THIS__", "CAST target type")


@pytest.mark.parametrize(
    "target",
    [
        "TINYINT",
        "INT1",
        "SMALLINT",
        "INT2",
        "SHORT",
        "INTEGER",
        "INT",
        "INT4",
        "SIGNED",
        "BIGINT",
        "INT8",
        "LONG",
        "DOUBLE",
        "FLOAT8",
        "VARCHAR",
        "TEXT",
        "STRING",
        "BOOLEAN",
        "BOOL",
    ],
)
def test_every_served_cast_spelling_matches_duckdb(target):
    # VARCHAR -> BOOLEAN is its own named refusal; the integer side covers it.
    y = "" if target.startswith("BOOL") else f", TRY_CAST(s AS {target}) AS y"
    duck_check(
        f"SELECT TRY_CAST(a AS {target}) AS x{y} FROM __THIS__",
        {"a": "int", "s": "str?"},
        [{"a": 1, "s": "7"}, {"a": 300, "s": "x"}, {"a": -40000, "s": None}],
    )


# ---- 5. Deliberate contract choices ----------------------------------------


def test_duplicate_names_use_duckdbs_boundary_rename():
    # Raw DuckDB keeps top-level duplicates; a dict cannot. We apply
    # DuckDB's OWN subquery/CTAS/.df() rename — not an invention.
    fn = DuckDBInferFn(
        "SELECT a, a AS a, a AS a_1 FROM __THIS__",
        row_tables={"__THIS__": T},
        static_tables={},
    )
    got = fn.infer_rows([{"a": 7, "s": None}])
    assert list(got[0].keys()) == ["a", "a_1", "a_1_1"]


def test_null_op_null_serves_with_measured_types():
    # The limitation is ONLY the context-free bare NULL; typed NULL ops
    # serve with the pinned result types.
    duck_check(
        "SELECT NULL + NULL AS s, NULL / NULL AS d FROM __THIS__",
        {"a": "int", "s": "str?"},
        [{"a": 1, "s": None}],
    )


# Schema qualifiers (divergence: schema-qualifiers). The registry is
# schema-less, so both directions of the divergence are pinned against the
# live oracle: a relation qualifier resolves by bare name where DuckDB refuses
# the unknown schema, and the refusals where DuckDB serves stay loud.
_D = pa.table({"id": pa.array([1], pa.int64()), "v": pa.array([10], pa.int64())})


def _oracle_answer(sql, row_schema, rows, statics=None):
    with Oracle() as o:
        o.load("__THIS__", pa.Table.from_pylist(rows, schema=row_schema))
        for name, t in (statics or {}).items():
            o.load(name, t)
        return o.try_answer(sql)


def test_a_relation_schema_qualifier_resolves_by_bare_name():
    sql = "SELECT v FROM __THIS__ LEFT JOIN s1.d ON a = id"
    assert build(sql, {"d": _D}).infer_rows([{"a": 1, "s": None}]) == [{"v": 10}]
    trap = _oracle_answer(sql, T, [{"a": 1, "s": None}], {"d": _D})
    assert isinstance(trap, Trap) and 'schema "s1" does not exist' in trap.message


def test_a_column_qualified_through_a_schema_qualified_relation_refuses():
    # DuckDB serves `d.v` over `JOIN main.d`; the registry names the relation
    # by its qualified spelling, so the bare qualifier misses -- loudly. The
    # 3-part `s1.d.v` refuses the same way (DuckDB also refuses it: no `s1`).
    sql = "SELECT d.v FROM __THIS__ JOIN main.d ON a = d.id"
    rejects(sql, "unknown table 'd'", {"d": _D})
    assert _oracle_answer(sql, T, [{"a": 1, "s": None}], {"d": _D}).to_pylist() == [
        {"v": 10}
    ]
    rejects(
        "SELECT s1.d.v FROM __THIS__ LEFT JOIN s1.d ON a = s1.d.id",
        "unknown table 'd'",
        {"d": _D},
    )


def test_a_schema_like_struct_path_takes_the_longer_parse_and_refuses():
    # `w.w.w` with table `w` and struct column `w{w}`: resolution is longest-
    # qualifier-first, so `w.w` binds as schema.table and `.w` as the whole
    # struct column -- refused by name, where DuckDB serves `column.field`.
    W = pa.schema([pa.field("w", pa.struct([("w", pa.int64())]))])
    sql = "SELECT w.w.w AS o FROM __THIS__ AS w"
    with pytest.raises(ValueError, match="struct column 'w' as a whole value"):
        DuckDBInferFn(sql, row_tables={"__THIS__": W}, static_tables={})
    assert _oracle_answer(sql, W, [{"w": {"w": 5}}]).to_pylist() == [{"o": 5}]


def test_rejections_are_build_time_and_named():
    # The load-bearing property: limits surface at CONSTRUCTION, before any
    # row is ever inferred — never silently at inference time.
    with pytest.raises(ValueError, match="unsupported"):
        DuckDBInferFn(
            "SELECT sum(a) FROM __THIS__",
            row_tables={"__THIS__": T},
            static_tables={},
        )


# ---- 6. How to read a rejection ------------------------------------------


def _width1_list_udf():
    class U:
        name = "u"
        takes = pa.schema([("x", pa.int64())])
        returns = pa.list_(pa.int64(), 1)

        def __call__(self, x):
            return (x,)

    return U()


def test_every_refusal_family_carries_a_documented_prefix():
    # Each refusal family under the prefix of its class; the family's
    # specific text rides along as a suffix.
    dup = pa.table(
        {"id": pa.array([1, 1], pa.int64()), "v": pa.array([1, 2], pa.int64())}
    )
    nn = pa.table(
        {"id": pa.array([1], pa.int64()), "v": pa.array([None], pa.int64())},
        schema=pa.schema(
            [("id", pa.int64()), pa.field("v", pa.int64(), nullable=False)]
        ),
    )
    cases = [
        ("SELECT v FROM __THIS__ JOIN d ON a = d.id", {"d": dup}, {}, "unsupported: "),
        ("SELECT v FROM __THIS__ JOIN d ON a = d.id", {"d": nn}, {}, "bind error: "),
        ("SELECT a FROM __THIS__ WHERE a > 0", {}, {"shape": "map"}, "unsupported: "),
        (
            "SELECT u(a) AS o FROM __THIS__",
            {},
            {"udfs": [_width1_list_udf()]},
            "bind error: ",
        ),
    ]
    for sql, statics, kw, prefix in cases:
        with pytest.raises(ValueError) as e:
            DuckDBInferFn(sql, row_tables={"__THIS__": T}, static_tables=statics, **kw)
        assert str(e.value).startswith(prefix), str(e.value)


@pytest.mark.parametrize(
    ("sql", "named"),
    [
        ("SELECT o FROM (SELECT a AS o FROM __THIS__) AS sub", "FROM a derived table"),
        (
            "SELECT a FROM __THIS__ RIGHT JOIN __THIS__ AS b ON true",
            "join type RIGHT JOIN",
        ),
        (
            "SELECT a FROM __THIS__ FULL JOIN __THIS__ AS b ON true",
            "join type FULL OUTER JOIN",
        ),
        ("SELECT (SELECT 1) FROM __THIS__", "expression a scalar subquery"),
        ("SELECT a IN (SELECT 1) FROM __THIS__", "IN \\(SELECT"),
        ("SELECT EXISTS (SELECT 1) FROM __THIS__", "EXISTS \\(SELECT"),
        ("SELECT DATE '2020-01-01' FROM __THIS__", "a typed literal"),
    ],
)
def test_refusals_name_the_construct_instead_of_echoing_it(sql, named):
    with pytest.raises(ValueError, match=named) as e:
        build(sql)
    # No echoed SQL and no Rust Debug dump of the AST.
    assert "SELECT 1" not in str(e.value).split("--")[0].replace("(SELECT ...)", "")
    assert "On(" not in str(e.value)


def test_an_all_null_case_over_a_trapping_condition_refuses():
    # DuckDB evaluates the conditions of an all-NULL CASE, so an overflow
    # there errors per row; serving the constant NULL would hide it.
    rejects(
        "SELECT CASE WHEN (a + 9223372036854775807) > 0 THEN NULL END AS o"
        " FROM __THIS__",
        "every branch is NULL, over a condition that can trap",
    )
