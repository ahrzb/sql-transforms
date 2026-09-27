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
from confit import DuckDBInferFn

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


def test_a_relation_read_as_a_struct_stays_refused():
    # DuckDB reads a relation name as its row struct, so `d['k']['f']` is
    # `d.k.f` there. confit does not serve relation-as-struct reads in any
    # spelling; the key recognition above must not turn `d['k']` into the
    # column `d.k` by accident.
    sql = "SELECT a FROM __THIS__ LEFT JOIN d ON a = d['k']['f']"
    with pytest.raises(ValueError, match="column 'd' does not exist"):
        DuckDBInferFn(sql, row_tables={"__THIS__": ROW}, static_tables={"d": D})
