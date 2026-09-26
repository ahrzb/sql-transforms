"""The pin corpus carries one uniform header per file, and it is current.

`scripts/pin_corpus.py header` derives each `_pin` header from the file and
git; this re-derives it and compares, so a pin edited without refreshing its
header, or a new pin without one, fails here. `committed` is left out: it
comes from git history, which a shallow CI checkout does not have.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _script():
    path = ROOT / "scripts" / "pin_corpus.py"
    spec = importlib.util.spec_from_file_location("pin_corpus", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PC = _script()
FILES = PC.pin_files()


def test_the_corpus_is_where_the_script_looks():
    assert len(FILES) >= 53


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(PC.PINS).as_posix())
def test_every_pin_file_opens_with_a_current_header(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    assert next(iter(d)) == "_pin", "the header is the first key"
    have = {k: v for k, v in d["_pin"].items() if k != "committed"}
    want = {k: v for k, v in PC.header(path, d).items() if k != "committed"}
    assert have == want, "run: uv run python scripts/pin_corpus.py header"
    assert d["_pin"]["schema"] == PC.SCHEMA
    assert d["_pin"]["committed"]


def test_every_under_determined_token_names_real_fields():
    for path in FILES:
        d = json.loads(path.read_text(encoding="utf-8"))
        for v in d["_pin"].get("varies", []):
            assert v["mark"] == "unspecified" or v["mark"].startswith("by:"), v
            assert v["note"], v
            assert PC.resolve(d, v["at"]), f"{path.name}: {v['at']} names nothing"


def test_the_token_marks_the_platform_dependent_nan_sign():
    d = json.loads((PC.PINS / "pins-wave3/math_tail.json").read_text("utf-8"))
    (v,) = [v for v in d["_pin"]["varies"] if v["mark"] == "by:platform"]
    assert PC.resolve(d, v["at"]) == [["7ff8000000000000"] * 4] * 2


@pytest.mark.parametrize(
    ("sql", "want"),
    [
        ("SELECT 1", ["SELECT 1"]),
        ("SELECT 1; SELECT 2;", ["SELECT 1", "SELECT 2"]),
        (
            "SELECT sum(x) FROM a\nSELECT sum(x) FROM b",
            ["SELECT sum(x) FROM a", "SELECT sum(x) FROM b"],
        ),
        ("SELECT uuid() AS v  -- observed = str(type)", ["SELECT uuid() AS v"]),
    ],
)
def test_a_pin_query_splits_into_its_statements(sql, want):
    assert PC.statements(sql) == want


def test_the_drift_manifest_names_real_pins_and_this_reference():
    m = json.loads(PC.DRIFT.read_text(encoding="utf-8"))
    import duckdb

    assert m["_meta"]["duckdb"] == duckdb.__version__
    assert m["_meta"]["replayed"] == len(m["answers"])
    for key in m["answers"]:
        rel, ptr = key.split("#")
        d = json.loads((PC.PINS / rel).read_text(encoding="utf-8"))
        assert PC.resolve(d, ptr), key


def test_a_replayable_pin_answers_and_a_prose_table_does_not():
    assert PC.answer([], ["SELECT 1 + 1 AS v"]) == ["['INTEGER'] [(2,)]"]
    assert PC.answer([], ["SELECT x FROM table_only_in_prose"]) is None


def test_a_converted_pin_reproduces_what_it_recorded():
    """A typed input_repr turned into CREATE + INSERT must rebuild the table
    the capture saw: every converted pin with a recorded result_repr answers
    it exactly (the floats go through text, so -0.0 keeps its sign)."""
    m = json.loads(PC.REPLAY.read_text(encoding="utf-8"))
    checked = 0
    for key, v in m["setups"].items():
        if v["from"] != "input_repr":
            continue
        rel, ptr = key.split("#")
        d = json.loads((PC.PINS / rel).read_text(encoding="utf-8"))
        entry = PC.resolve(d, ptr.rsplit("/", 1)[0])[0]
        if entry.get("result_repr") is None:
            continue
        (got,) = PC.answer(v["setup"], PC.statements(entry["sql"]))[:1]
        rows = got.split("] ", 1)[1]
        env = {"__builtins__": {}, "nan": float("nan"), "inf": float("inf")}
        vals = [r[0] for r in eval(rows, env)]  # noqa: S307 — our own repr
        rec = entry["result_repr"]
        assert repr(vals) == rec or (len(vals) == 1 and repr(vals[0]) == rec), key
        checked += 1
    assert checked >= 160


def test_every_pin_is_replayed_or_inventoried_with_a_reason():
    m = json.loads(PC.REPLAY.read_text(encoding="utf-8"))
    total = sum(
        1 for p in FILES for _ in PC.pin_queries(json.loads(p.read_text("utf-8")))
    )
    assert sum(m["_meta"]["counts"].values()) == total
    assert all(k.startswith("pins-") and "#/" in k for k in m["setups"])
