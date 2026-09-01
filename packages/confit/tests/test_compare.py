"""The comparison vocabulary, exercised on its own terms.

Each test here pins one claim the module makes about what "equal" means, so a
later edit that quietly weakens a comparison fails loudly instead of making
every leg that uses it weaker at once. The last test is the import ban that
keeps the module usable outside pytest: read off the source, because pytest is
in `sys.modules` by the time anything here runs.
"""

from __future__ import annotations

import ast
import math
import sys
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pytest
from confit import compare


def _dup_name_table() -> pa.Table:
    return pa.Table.from_arrays([pa.array([1, 2]), pa.array([3, 4])], names=["a", "a"])


def test_dedup_names_renames_left_to_right_case_insensitively():
    assert compare.dedup_names(["a", "a", "A", "b"]) == ["a", "a_1", "A_2", "b"]


def test_dedup_names_skips_a_generated_candidate_already_taken():
    assert compare.dedup_names(["a", "a_1", "a"]) == ["a", "a_1", "a_2"]


def test_rows_keeps_both_columns_that_to_pylist_collapses():
    t = _dup_name_table()
    assert t.to_pylist() == [{"a": 3}, {"a": 4}]  # the loss this guards
    assert compare.rows(t) == [{"a": 1, "a_1": 3}, {"a": 2, "a_1": 4}]


def test_rows_leaves_unique_names_alone():
    t = pa.table({"a": [1], "b": [2]})
    assert compare.rows(t) == [{"a": 1, "b": 2}]


def test_multiset_makes_nan_self_equal():
    nan = float("nan")
    assert compare.multiset([{"a": nan}]) == compare.multiset([{"a": nan}])
    assert nan != nan  # what `==` would have said


def test_multiset_keeps_signed_zero_distinct():
    assert compare.multiset([{"a": -0.0}]) != compare.multiset([{"a": 0.0}])
    assert -0.0 == 0.0  # what `==` would have said


@pytest.mark.parametrize(
    ("a", "b"),
    [(1, 1.0), (1, True), (1.0, True), (Decimal("0.50"), Decimal("0.5"))],
)
def test_multiset_keeps_equal_but_differently_typed_values_distinct(a, b):
    assert a == b  # what `==` would have said
    assert compare.multiset([{"v": a}]) != compare.multiset([{"v": b}])


def test_multiset_is_order_insensitive_where_sequence_is_not():
    rows = [{"a": 1}, {"a": 2}]
    flipped = list(reversed(rows))
    assert compare.multiset(rows) == compare.multiset(flipped)
    assert compare.sequence(rows) != compare.sequence(flipped)


def test_sequence_is_byte_identical_to_the_fuzzers_canonical_form():
    rows = [{"b": 2, "a": 1}, {"a": -0.0, "b": Decimal("0.50")}]
    assert compare.sequence(rows) == [
        sorted((k, repr(v)) for k, v in r.items()) for r in rows
    ]
    assert compare.multiset(rows) == sorted(compare.sequence(rows))


def test_assert_rows_default_accepts_reordered_rows():
    rows = [{"a": 1}, {"a": 2}]
    compare.assert_rows(rows, list(reversed(rows)))


def test_assert_rows_ordered_rejects_reordered_rows():
    rows = [{"a": 1}, {"a": 2}]
    with pytest.raises(AssertionError) as e:
        compare.assert_rows(rows, list(reversed(rows)), ordered=True)
    assert "unordered" not in str(e.value)
    assert "ordered" in str(e.value)


def test_assert_rows_message_names_the_axis_the_counts_and_the_context():
    with pytest.raises(AssertionError) as e:
        compare.assert_rows([{"a": 1}, {"a": 2}], [{"a": 1}], ctx="leg 3")
    msg = str(e.value)
    assert "unordered" in msg
    assert "2 rows" in msg and "1 row" in msg
    assert "leg 3" in msg


def test_assert_rows_unordered_message_locates_a_difference_in_canonical_order():
    """The unordered path pairs the two sides up AFTER canonicalizing, so the
    index it reports is that order's and not the caller's -- got's `3` is the
    caller's row 0 and the canonical row 1. The label has to say so, or the
    reader goes looking at the wrong input row."""
    got = [{"a": 3}, {"a": 1}]
    want = [{"a": 1}, {"a": 9}]
    with pytest.raises(AssertionError) as e:
        compare.assert_rows(got, want)
    msg = str(e.value)
    assert "unordered" in msg
    assert "canonical row 1" in msg
    assert "{'a': 3}" in msg and "{'a': 9}" in msg


def test_assert_rows_message_locates_the_first_differing_row():
    got = [{"a": 1}, {"a": 2}, {"a": 3}]
    want = [{"a": 1}, {"a": 9}, {"a": 3}]
    with pytest.raises(AssertionError) as e:
        compare.assert_rows(got, want, ordered=True)
    msg = str(e.value)
    assert "row 1" in msg and "row 0" not in msg
    assert "{'a': 2}" in msg and "{'a': 9}" in msg


@pytest.mark.parametrize(
    ("a", "b", "property_name"),
    [
        (Decimal("0.50"), Decimal("0.5"), "Decimal scale"),
        (-0.0, 0.0, "signed zero"),
        (float("nan"), Decimal("NaN"), "NaN"),
    ],
)
def test_assert_rows_message_names_the_property_that_caught_it(a, b, property_name):
    with pytest.raises(AssertionError) as e:
        compare.assert_rows([{"v": a}], [{"v": b}])
    assert property_name in str(e.value)


def test_assert_rows_message_truncates_a_long_run_of_differences():
    got = [{"a": i} for i in range(60)]
    want = [{"a": i + 100} for i in range(60)]
    with pytest.raises(AssertionError) as e:
        compare.assert_rows(got, want, ordered=True)
    msg = str(e.value)
    assert msg.count("  row ") <= 20
    assert "40 more differing rows" in msg


def test_assert_rows_passes_silently_when_equal():
    assert compare.assert_rows([{"a": float("nan")}], [{"a": float("nan")}]) is None


def test_assert_rows_close_accepts_a_one_ulp_difference():
    assert (
        compare.assert_rows_close([{"c": 1.0}], [{"c": math.nextafter(1.0, 2.0)}])
        is None
    )


def test_assert_rows_close_rejects_two_ulps_at_max_ulp_one():
    two = math.nextafter(math.nextafter(1.0, 2.0), 2.0)
    with pytest.raises(AssertionError) as e:
        compare.assert_rows_close([{"c": 1.0}], [{"c": two}], ctx="cbrt leg")
    msg = str(e.value)
    assert "2 ulp" in msg
    assert "cbrt leg" in msg
    assert "row 0" in msg and "c:" in msg
    compare.assert_rows_close([{"c": 1.0}], [{"c": two}], max_ulp=2)


def test_assert_rows_close_makes_nan_self_equal():
    nan = float("nan")
    assert compare.assert_rows_close([{"c": nan}], [{"c": nan}]) is None
    assert nan != nan  # what `==` would have said


def test_assert_rows_close_still_rejects_signed_zero():
    """The bit patterns of -0.0 and 0.0 are 2**63 apart, so the tolerance leg
    keeps the property the repr leg keeps -- out of the encoding, not out of a
    special case, and at any tolerance a caller would plausibly ask for."""
    with pytest.raises(AssertionError) as e:
        compare.assert_rows_close([{"c": -0.0}], [{"c": 0.0}], max_ulp=1000)
    assert str(2**63) in str(e.value)


@pytest.mark.parametrize("dropped_from", ["got", "want"])
def test_assert_rows_close_fails_on_a_key_missing_from_either_side(dropped_from):
    both, one = {"a": 1.0, "b": 2.0}, {"a": 1.0}
    got, want = (one, both) if dropped_from == "got" else (both, one)
    with pytest.raises(AssertionError) as e:
        compare.assert_rows_close([got], [want])
    msg = str(e.value)
    assert "b:" in msg and "missing" in msg


def test_assert_rows_close_falls_back_to_repr_off_the_float_pairs():
    compare.assert_rows_close([{"v": "x"}], [{"v": "x"}])
    with pytest.raises(AssertionError):
        compare.assert_rows_close([{"v": 1}], [{"v": 1.0}], max_ulp=1000)


def test_assert_rows_close_is_positional_where_assert_rows_is_not():
    rows = [{"c": 1.0}, {"c": 2.0}]
    compare.assert_rows(rows, list(reversed(rows)))
    with pytest.raises(AssertionError):
        compare.assert_rows_close(rows, list(reversed(rows)))


def test_assert_rows_close_message_truncates_a_long_run_of_differences():
    got = [{"c": float(i)} for i in range(60)]
    want = [{"c": float(i + 100)} for i in range(60)]
    with pytest.raises(AssertionError) as e:
        compare.assert_rows_close(got, want)
    msg = str(e.value)
    assert msg.count("  row ") <= 20
    assert "40 more differing rows" in msg


def test_assert_rows_close_reports_a_row_count_mismatch():
    with pytest.raises(AssertionError) as e:
        compare.assert_rows_close([{"c": 1.0}], [], ctx="leg 3")
    msg = str(e.value)
    assert "1 row" in msg and "0 rows" in msg and "leg 3" in msg


def test_assert_schema_accepts_an_equal_schema():
    s = pa.schema([pa.field("a", pa.int64())])
    assert compare.assert_schema(s, pa.schema([pa.field("a", pa.int64())])) is None


@pytest.mark.parametrize(
    ("want_field", "attribute"),
    [
        (pa.field("z", pa.int64()), "name"),
        (pa.field("b", pa.int32()), "type"),
        (pa.field("b", pa.int64(), nullable=False), "nullable"),
    ],
)
def test_assert_schema_names_the_first_differing_field_and_attribute(
    want_field, attribute
):
    got = pa.schema([pa.field("a", pa.int64()), pa.field("b", pa.int64())])
    want = pa.schema([pa.field("a", pa.int64()), want_field])
    with pytest.raises(AssertionError) as e:
        compare.assert_schema(got, want, ctx="output")
    msg = str(e.value)
    assert "field 1" in msg
    assert attribute in msg
    assert "output" in msg
    assert "field 0" not in msg


def test_assert_schema_reports_a_length_difference():
    got = pa.schema([pa.field("a", pa.int64())])
    want = pa.schema([pa.field("a", pa.int64()), pa.field("b", pa.int64())])
    with pytest.raises(AssertionError) as e:
        compare.assert_schema(got, want)
    assert "1 field" in str(e.value) and "2 field" in str(e.value)


def test_compare_imports_stdlib_and_pyarrow_only():
    """Read off the source, not `sys.modules`: pytest and duckdb are both
    already imported by the time this test runs. The fuzzer's canonicalizers
    ARE this module's -- fuzz/oracle.py imports `dedup_names`, `multiset` and
    `sequence` from here -- and a campaign runs outside pytest, so neither may
    be made to pay for a test-only or an oracle-only dependency."""
    tree = ast.parse(Path(compare.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative imports: this module stands alone"
            roots.add((node.module or "").split(".")[0])
    assert roots.isdisjoint({"pytest", "duckdb", "fuzz", "tests"})
    assert roots - {"pyarrow"} <= sys.stdlib_module_names
    # And pyarrow is reached only through annotations, so nothing third-party
    # is imported at RUN time. A module-level `import pyarrow` means someone
    # added a real use; move it back out of the TYPE_CHECKING block if so.
    assert "pyarrow" not in {
        a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names
    }


def test_import_confit_stays_lean():
    """`confit.compare` is a tool for comparison sites, not part of the
    serving surface -- importing the package must not drag it in."""
    tree = ast.parse(
        Path(compare.__file__).with_name("__init__.py").read_text(encoding="utf-8")
    )
    assert "compare" not in {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    } | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
