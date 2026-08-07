"""DuckDB 1.5.5: registering an ``.arrow()`` result deadlocks its own connection.

``DuckDBPyConnection.arrow()`` returns a lazy ``pyarrow.RecordBatchReader``,
not a materialised ``pa.Table``. Register that reader back into the connection
that produces it, then scan it, and the scan never returns — no error, no
timeout, no output. That is C5's "no third mode" violated at the API level:
neither serving nor refusing. Observed once it was worse still, returning zero
rows and reporting success.

Measured 2026-08-07 on duckdb 1.5.5 / pyarrow 25.0.0 / CPython 3.14, six for
six across process restarts. The reader is fine when its producer is a
*different* connection, and ``to_arrow_table()`` is fine on the same one —
that second spelling is the workaround, pinned below, and is what the nesting
oracle in the datamodel spec must use to materialise a member.

The repro runs in a child process because a deadlocked DuckDB scan cannot be
interrupted from Python; a thread would wedge interpreter exit.

Strict xfail: it flips to XPASS the day DuckDB materialises, raises, or times
out, and that is the day to delete the workaround.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# The healthy spelling finishes in ~1s including interpreter start.
_TIMEOUT = 10.0
_EXPECTED = "[(20.0,), (40.0,), (60.0,)]"


def _repro(fetch: str) -> str:
    return (
        "import duckdb, pyarrow as pa\n"
        "con = duckdb.connect()\n"
        'con.register("src", pa.table({"price": [10.0, 20.0, 30.0]}))\n'
        f'con.register("out", con.execute("SELECT price * 2 AS z '
        f'FROM src").{fetch}())\n'
        'print(con.execute("SELECT * FROM out ORDER BY 1").fetchall())\n'
    )


def _run(fetch: str) -> str:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", _repro(fetch)],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=True,
    ).stdout.strip()


def test_to_arrow_table_round_trips_through_register():
    """The workaround. Not xfailed — the implementation depends on it."""
    assert _run("to_arrow_table") == _EXPECTED


@pytest.mark.xfail(
    strict=True,
    reason="DuckDB 1.5.5 hangs forever scanning a RecordBatchReader that was "
    "registered back into the connection producing it; .arrow() returns a "
    "reader, not a table. Use .to_arrow_table(). See module docstring.",
)
def test_arrow_result_round_trips_through_register():
    assert _run("arrow") == _EXPECTED
