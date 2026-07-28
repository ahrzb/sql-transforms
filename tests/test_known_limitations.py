"""Executable twin of docs/known-limitations.md.

Every DELIBERATE limitation is asserted here: the SQL that hits it and the
named build-time rejection (or, for contract choices, the chosen behavior).
If an engine change lifts one of these, the test fails — update the doc in
the same commit. Section numbers mirror the document.
"""

from __future__ import annotations

import pytest
from pydantic import create_model
from test_duckdb_interpreter import duck_check, static

from sql_transform._interpreter import DuckDBInferFn

T = create_model("T", a=(int, ...), s=(str | None, None))


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
    # served under the opt-in shape='many' (stage B, TASK-59).
    rejects(
        "SELECT v FROM __THIS__ JOIN d ON a = d.id",
        "duplicate map key",
        {"d": dup},
    )
    fn = DuckDBInferFn(
        "SELECT v FROM __THIS__ JOIN d ON a = d.id",
        row_tables={"__THIS__": T},
        static_tables={"d": dup},
        output="dict",
        shape="many",
    )
    got = sorted(r["v"] for r in fn.infer({"__THIS__": [T(a=1)]}))
    assert got == [1, 2]
    # NULL VALUES serve since TASK-55 (they ride as validity+payload pairs);
    # only NULL keys keep the drop rule (a NULL never equi-matches).
    withnull = static({"id": "int", "v": "int?"}, [{"id": 1, "v": None}])
    duck_check(
        "SELECT v FROM __THIS__ JOIN d ON a = d.id",
        {"a": "int", "s": "str?"},
        [{"a": 1, "s": None}],
        {"d": withnull},
    )


def test_dynamic_self_join_rejects():
    rejects(
        "SELECT t2.a FROM __THIS__ JOIN __THIS__ t2 ON __THIS__.a = t2.a",
        "dynamic table",
    )


# ---- 2. Out of scope for row-serving --------------------------------------


@pytest.mark.parametrize(
    ("sql", "needle"),
    [
        ("SELECT sum(a) FROM __THIS__", "no aggregation"),
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
    L = create_model("L", xs=(list[int], ...))
    with pytest.raises(ValueError, match="non-scalar"):
        DuckDBInferFn(
            "SELECT xs FROM __THIS__", row_tables={"__THIS__": L}, static_tables={}
        )


def test_non_scalar_rejection_is_reference_time():
    # TASK-56: the rejection moved from construction to REFERENCE — an
    # unreferenced list/timestamp field no longer blocks a scalar query,
    # and star modifiers can remove one. Referenced (incl. via *) keeps
    # the named error.
    L = create_model("L", a=(int, ...), xs=(list[int] | None, None))
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
    # TASK-56: structs of scalars serve AS FIELDS (flattened to lanes);
    # the struct as a whole value stays a named non-scalar rejection.
    Inner = create_model("Inner", i=(int | None, None))
    M = create_model("M", a=(Inner | None, None))
    fn = DuckDBInferFn(
        "SELECT a.i FROM __THIS__",
        row_tables={"__THIS__": M},
        static_tables={},
        output="dict",
    )
    assert fn.infer({"__THIS__": [M(a=Inner(i=5))]}) == [{"i": 5}]
    with pytest.raises(ValueError, match="whole value"):
        DuckDBInferFn(
            "SELECT a FROM __THIS__", row_tables={"__THIS__": M}, static_tables={}
        )
    with pytest.raises(ValueError, match="unsupported"):
        DuckDBInferFn(
            "SELECT a['i'] FROM __THIS__", row_tables={"__THIS__": M}, static_tables={}
        )


def test_list_valued_regexp_forms_reject():
    # Gated on list types (wave C), not on regex semantics.
    rejects("SELECT regexp_extract_all(s, 'a') FROM __THIS__", "list-valued")
    rejects("SELECT regexp_split_to_array(s, 'a') FROM __THIS__", "regexp_split")


def test_ubigint_static_payloads_reject():
    import pyarrow as pa

    big = pa.table({"id": pa.array([2**64 - 1], pa.uint64()), "v": [1]})
    rejects(
        "SELECT v FROM __THIS__ JOIN d ON a = d.id",
        "outside BIGINT range",
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
        # TASK-54 fuzzer-found classes (pins-waveB/fuzzer-task54.json):
        # each was a measured silent-wrong-answer risk in rust-regex.
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
        # SQLNULL has no home type.
        ("SELECT NULL FROM __THIS__", "NULL literal"),
    ],
)
def test_measured_descopes_reject(sql, needle):
    rejects(sql, needle)


def test_using_unmerge_exclude_rejects():
    d = static({"a": "int", "v": "int"}, [{"a": 1, "v": 10}])
    # DuckDB UNMERGES the coalesced column here — measured, not modeled.
    rejects(
        "SELECT * EXCLUDE (d.a) FROM __THIS__ JOIN d USING (a)",
        "USING-merged",
        {"d": d},
    )


# ---- 5. Deliberate contract choices ----------------------------------------


def test_duplicate_names_use_duckdbs_boundary_rename():
    # Raw DuckDB keeps top-level duplicates; a typed model cannot. We apply
    # DuckDB's OWN subquery/CTAS/.df() rename — not an invention.
    fn = DuckDBInferFn(
        "SELECT a, a AS a, a AS a_1 FROM __THIS__",
        row_tables={"__THIS__": T},
        static_tables={},
        output="dict",
    )
    got = fn.infer({"__THIS__": [T(a=7)]})
    assert list(got[0].keys()) == ["a", "a_1", "a_1_1"]


def test_null_op_null_serves_with_measured_types():
    # The limitation is ONLY the context-free bare NULL; typed NULL ops
    # serve with the pinned result types.
    duck_check(
        "SELECT NULL + NULL AS s, NULL / NULL AS d FROM __THIS__",
        {"a": "int", "s": "str?"},
        [{"a": 1, "s": None}],
    )


def test_rejections_are_build_time_and_named():
    # The load-bearing property: limits surface at CONSTRUCTION, before any
    # row is ever inferred — never silently at inference time.
    with pytest.raises(ValueError, match="unsupported"):
        DuckDBInferFn(
            "SELECT sum(a) FROM __THIS__",
            row_tables={"__THIS__": T},
            static_tables={},
        )
