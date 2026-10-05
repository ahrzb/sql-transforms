"""Equivalent spellings agree (fuzz/metamorphic.py), plus the fixes it found.

The oracle campaign compares confit with DuckDB one spelling at a time.
These checks compare confit with itself across spellings that DuckDB treats
as the same query, so an over-refusal that only one spelling hits shows up
even where DuckDB has nothing to diverge from. The deep run is
`python -m fuzz.metamorphic --n 2000`; this is the gate-sized slice.
"""

import sys
from pathlib import Path

import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz import metamorphic  # noqa: E402
from fuzz.parity import assert_parity  # noqa: E402

N = 300


def test_spellings_agree():
    counts, findings = metamorphic.run(range(N))
    assert not findings, "\n".join(f.line() for f in findings)
    # A rewrite that never applies checks nothing.
    for name in metamorphic.REWRITES:
        assert counts[(name, "checked")] > 0, name


# ------------------------------------------------ struct-leaf join keys
# Found by lane-sub/lane-paren/lane-fn: a static struct leaf spelled any
# way but the dot path was not recognized as an equality key, so a
# multi-row static table refused as keyless.

ROW = pa.schema([pa.field("a", pa.int64())])
D = pa.table(
    {
        "k": pa.array(
            [{"f": 1, "g": 10}, {"f": 2, "g": 20}, {"f": None, "g": 30}],
            pa.struct([("f", pa.int64()), ("g", pa.int64())]),
        ),
        "v": ["one", "two", "none"],
    }
)


@pytest.mark.parametrize(
    "key",
    ["d.k.f", "d.k['f']", "(d.k).f", "struct_extract(d.k, 'f')", "k['f']", "(k).f"],
)
def test_every_spelling_of_a_struct_leaf_is_a_join_key(key):
    sql = f"SELECT a, v FROM __THIS__ LEFT JOIN d ON a = {key}"
    rows = pa.table({"a": pa.array([1, 2, 3, None], pa.int64())})
    assert_parity(sql, rows, statics={"d": D}, expect="AGREE")


@pytest.mark.parametrize("key", ["d['k']['f']", "(d).k.f", "struct_extract(d, 'k').f"])
def test_a_relation_read_as_a_struct_is_its_column(key):
    # DuckDB reads a relation name no column shares as its row struct, so
    # `d['k']['f']` is `d.k.f` there, a join key like every spelling above.
    sql = f"SELECT a, v FROM __THIS__ LEFT JOIN d ON a = {key}"
    rows = pa.table({"a": pa.array([1, 2, 3, None], pa.int64())})
    assert_parity(sql, rows, statics={"d": D}, expect="AGREE")


# ------------------------------------------------- found by the nightly run
# (issue ahrzb/sql-transforms#303, window 1000000..1004999)

T = pa.table(
    {
        "a": pa.array([1, 2], pa.int64()),
        "from": pa.array([{"f0": 1}, {"f0": 2}], pa.struct([("f0", pa.int64())])),
    }
)
DIM = pa.table({"id": pa.array([1], pa.int64()), "v": ["x"]})


@pytest.mark.parametrize("qual", ['"d"', '"D"', "d", "D"])
def test_a_quoted_wildcard_qualifier_names_its_relation(qual):
    # `"d".*` kept its quotes and matched no relation (quote-all rewrite).
    sql = f"SELECT {qual}.* FROM __THIS__ LEFT JOIN d ON a = d.id"
    assert_parity(sql, T.select(["a"]), statics={"d": DIM}, expect="AGREE")


@pytest.mark.parametrize(
    "sql, kind",
    [
        # DuckDB's grammar will not start a column reference with a reserved
        # keyword; confit built `(from).f0` (lane-paren rewrite).
        ("SELECT (from).f0 AS x FROM __THIS__", "REFUSED"),
        ("SELECT from.f0 AS x FROM __THIS__", "REFUSED"),
        ('SELECT "from".f0 AS x FROM __THIS__', "AGREE"),
        ("SELECT __THIS__.from.f0 AS x FROM __THIS__", "AGREE"),
    ],
)
def test_a_reserved_keyword_starts_no_column_reference(sql, kind):
    v = assert_parity(sql, T, expect=kind)
    if kind == "REFUSED":
        assert v.oracle == "rejects", v
