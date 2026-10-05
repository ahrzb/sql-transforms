"""The parity harness (fuzz/parity.py) and the probe CLI (fuzz/probe.py).

`assert_parity` is only worth using if it is at least as strict as the
hand-rolled helpers it replaces, so these pin that it catches what they
caught: a wrong output name, a trap on one side only, a trap with another
message, and a refusal where agreement was expected.
"""

import sys
from pathlib import Path

import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz import gen, parity, probe  # noqa: E402

T = pa.table({"a": pa.array([1, None], pa.int64()), "s": ["1", None]})


def test_the_campaign_vocabulary_round_trips():
    for seed in range(0, 400, 7):
        c = gen.gen(seed)
        t = parity.table(c.row_schema, c.rows)
        assert {f.name: parity.spec_of(f) for f in t.schema} == c.row_schema


def test_types_outside_the_vocabulary_are_named():
    with pytest.raises(TypeError, match="outside the campaign vocabulary"):
        parity.case(
            "SELECT u FROM __THIS__", pa.table({"u": pa.array([1], pa.uint64())})
        )


def test_agreement_passes_and_returns_the_verdict():
    v = parity.assert_parity("SELECT a + 1 AS b FROM __THIS__", T, expect="AGREE")
    assert v.kind == "AGREE"


def _tick():
    import itertools

    from confit import ExternFunction

    n = itertools.count()
    return ExternFunction(
        "tick",
        pa.schema([("a", pa.int64())]),
        pa.int64(),
        lambda a: (next(n),),
        side_effects=True,
    )


@pytest.mark.parametrize(
    "sql, kw, message",
    [
        # A udf that counts its calls answers differently on each engine.
        ("SELECT tick(a) AS b FROM __THIS__", {"udfs": [_tick()]}, "DIVERGE_VALUE"),
        (
            "SELECT a + 1 AS b FROM __THIS__",
            {"trap": "Overflow"},
            "expected AGREE_TRAP",
        ),
        (
            "SELECT a + 9223372036854775807 AS b FROM __THIS__",
            {"trap": "division"},
            "confit: ",
        ),
        ("SELECT a FROM __THIS__ QUALIFY TRUE", {}, "REFUSED"),
    ],
)
def test_what_the_harness_catches(sql, kw, message):
    rows = pa.table({"a": pa.array([1], pa.int64())})
    with pytest.raises(AssertionError, match=message):
        parity.assert_parity(sql, rows, **kw)


def test_the_probe_reports_each_query(capsys):
    code = probe.main(
        [
            "SELECT a + 1 AS b FROM __THIS__",
            "SELECT a + 1 AS b FROM __THIS__ QUALIFY TRUE",
            "--table",
            '__THIS__::int32={"a": [1, null]}',
        ]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "ok AGREE" in out and "!! REFUSED" in out
    assert "confit: [DataType(int32)] [{'b': 2}, {'b': None}]" in out
