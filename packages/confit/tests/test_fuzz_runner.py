"""The runner's side of the worker protocol, driven by stand-in workers.

A REFUSED verdict line is followed by one more line (`fuzz.oracle.refusal_json`):
the report-only DuckDB reading, which the runner reads under its own budget.
Stand-ins answer every even seed AGREE and every odd seed REFUSED, then
complete, hang in or die in that reading, so each branch of the runner is
driven directly: no engine, no DuckDB, no real seed.

A verdict line marked `"bracket": "pending"` is followed by the whole verdict,
which the runner reads under the bracket's own budget. Bracket stand-ins send
AGREE before the bracket and DIVERGE_OPT after it, and on odd seeds hang in or
die in the bracket reading.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz import runner  # noqa: E402

STAND_IN = r"""
import json, sys, time
mode = sys.argv[1]
doom = int(mode[len("doomed"):]) if mode.startswith("doomed") else 0
for count, line in enumerate(sys.stdin, 1):
    if count == doom:
        sys.exit(9)  # killed an instant ago: this seed never runs
    seed = int(line)
    kind = "REFUSED" if seed % 2 else "AGREE"
    verdict = {"seed": seed, "sql": "", "kind": kind, "klass": "k", "detail": "",
               "tags": [], "triples": []}
    print(json.dumps(verdict), flush=True)
    if kind == "REFUSED":
        if mode == "hang":
            time.sleep(60)
        if mode == "die":
            sys.exit(3)
        print(json.dumps({"oracle": "serves"}), flush=True)
"""


BRACKET_STAND_IN = r"""
import json, sys, time
mode = sys.argv[1]
for line in sys.stdin:
    seed = int(line)
    verdict = {"seed": seed, "sql": "", "kind": "AGREE", "klass": "", "detail": "",
               "tags": [], "triples": []}
    print("@@phase oracle", file=sys.stderr, flush=True)
    print(json.dumps({**verdict, "bracket": "pending"}), flush=True)
    if seed % 2:
        if mode == "legs":
            print("@@phase confit:legs", file=sys.stderr, flush=True)
        if mode in ("hang", "legs"):
            time.sleep(60)
        if mode == "die":
            sys.exit(3)
    print(json.dumps({**verdict, "kind": "DIVERGE_OPT", "klass": "k"}), flush=True)
"""


def _stand_in(monkeypatch, *modes: str, script: str = STAND_IN) -> list:
    """Every worker the runner spawns is a stand-in running `script`, the
    n-th in `modes[n]` (the last mode repeats); returns the list the spawned
    processes are appended to."""
    spawned = []

    def spawn():
        mode = modes[min(len(spawned), len(modes) - 1)]
        err = tempfile.TemporaryFile()
        proc = subprocess.Popen(  # noqa: S603 — fixed argv
            [sys.executable, "-c", script, mode],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
        )
        spawned.append(proc)
        return proc, err

    monkeypatch.setattr(runner, "_spawn", spawn)
    return spawned


def _campaign(tmp_path, n=4):
    # 1 s is far more than a stand-in needs between its two lines, and all a
    # hanging one costs.
    results = runner.campaign(
        0,
        n,
        1,
        timeout=10.0,
        out=tmp_path / "f.jsonl",
        report_timeout=1.0,
        bracket_timeout=1.0,
    )
    return {r["seed"]: r for r in results}


@pytest.mark.parametrize(
    ("mode", "outcome", "workers"),
    [("complete", "serves", 1), ("hang", "over-budget", 3), ("die", "died", 3)],
)
def test_a_refusal_reading_costs_at_most_its_outcome(
    tmp_path, monkeypatch, mode, outcome, workers
):
    """Completed, the reading's line fills in the outcome. Outrun or died in,
    the worker is replaced and the verdict stays REFUSED: never a gated
    TIMEOUT or PANIC. Either way the next seed is still answered."""
    spawned = _stand_in(monkeypatch, mode)
    got = _campaign(tmp_path)
    assert sorted(got) == [0, 1, 2, 3]
    assert [got[s]["kind"] for s in range(4)] == ["AGREE", "REFUSED"] * 2
    assert got[1]["oracle"] == got[3]["oracle"] == outcome
    assert "oracle" not in got[0]
    assert runner.gate(list(got.values()), None) == []
    assert len(spawned) == workers


@pytest.mark.parametrize(("race", "doomed"), [(1, "doomed2"), (3, "doomed3")])
def test_a_worker_the_timer_fired_on_is_replaced_even_with_its_line(
    tmp_path, monkeypatch, race, doomed
):
    """The timer can fire between `readline` returning a whole line and
    `cancel`: on a verdict line (read 1: seed 0's) or on a refusal's report
    line (read 3: seed 1's). The line counts, and the worker is replaced.
    The stand-in plays the kill in flight: it dies on the next seed it
    takes, and that seed would be blamed for a PANIC it never caused."""
    spawned = _stand_in(monkeypatch, doomed, "complete")
    real, reads = runner._read, []

    def racy(proc, timeout):
        line, fired = real(proc, timeout)
        reads.append(line)
        return line, fired or len(reads) == race

    monkeypatch.setattr(runner, "_read", racy)
    got = _campaign(tmp_path)
    assert [got[s]["kind"] for s in range(4)] == ["AGREE", "REFUSED"] * 2
    assert got[1]["oracle"] == got[3]["oracle"] == "serves"
    assert runner.gate(list(got.values()), None) == []
    assert len(spawned) == 2


@pytest.mark.parametrize(
    ("mode", "outcome", "workers"),
    [("complete", None, 1), ("hang", "over-budget", 3), ("die", "died", 3)],
)
def test_a_bracket_reading_costs_at_most_its_label(
    tmp_path, monkeypatch, mode, outcome, workers
):
    """Completed, the bracket's line replaces the baseline verdict. Outrun or
    died in, in DuckDB, the worker is replaced and the baseline verdict
    stays, marked with why: never a gated TIMEOUT or PANIC."""
    spawned = _stand_in(monkeypatch, mode, script=BRACKET_STAND_IN)
    got = _campaign(tmp_path)
    assert sorted(got) == [0, 1, 2, 3]
    late = "DIVERGE_OPT" if outcome is None else "AGREE"
    assert [got[s]["kind"] for s in range(4)] == ["DIVERGE_OPT", late] * 2
    assert [got[s].get("bracket") for s in range(4)] == [None, outcome] * 2
    assert runner.gate(list(got.values()), None) == []
    assert len(spawned) == workers


def test_a_worker_lost_in_confit_after_the_baseline_verdict_is_blamed(
    tmp_path, monkeypatch
):
    """The boundary legs can run after the bracket (when only the bracket
    makes a verdict that needs them). A worker lost there is confit's, and
    is blamed like any other."""
    _stand_in(monkeypatch, "legs", script=BRACKET_STAND_IN)
    got = _campaign(tmp_path)
    assert [got[s]["kind"] for s in range(4)] == ["DIVERGE_OPT", "TIMEOUT"] * 2
    assert got[1]["klass"] == got[3]["klass"] == "timeout:confit"
    assert runner.gate(list(got.values()), None) == ["2 TIMEOUT"]


def test_an_unfinished_bracket_is_reported_with_its_reason(tmp_path, capsys):
    results = [
        {"seed": s, "kind": "AGREE", "klass": "", "detail": "", "tags": []}
        for s in range(8)
    ]
    for r in results[:6]:
        r["bracket"] = "over-budget"
    results[6]["bracket"] = "died"
    runner.report(results, tmp_path / "f.jsonl")
    out = capsys.readouterr().out
    assert (
        "over-budget    6  the reading ran past --bracket-timeout: "
        "seeds 0, 1, 2, 3, 4, ..."
    ) in out
    assert "died           1  the worker died in the reading: seeds 6" in out


def test_an_unfinished_reading_is_reported_with_its_reason(tmp_path, capsys):
    results = [
        {"seed": 1, "kind": "REFUSED", "klass": "x", "detail": "", "tags": []},
        {"seed": 2, "kind": "REFUSED", "klass": "x", "detail": "", "tags": []},
    ]
    results[0]["oracle"], results[1]["oracle"] = runner.UNFINISHED
    runner.report(results, tmp_path / "f.jsonl")
    out = capsys.readouterr().out
    assert "over-budget    1  the reading ran past --report-timeout" in out
    assert "died           1  the worker died in the reading" in out
