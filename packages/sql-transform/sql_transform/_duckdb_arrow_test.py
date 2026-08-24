"""DuckDB 1.5.5: a lazy Arrow reader does not survive being registered back.

``.arrow()`` returns a ``pyarrow.RecordBatchReader``, not a materialised
``pa.Table``. Register that reader into the connection that produced it, scan
it, and it does not work — but *how* it fails is not stable:

    con = duckdb.connect()                     # one connection, not two
    con.register("src", pa.table({"price": [10.0, 20.0, 30.0]}))
    con.register("out", con.execute("SELECT price * 2 AS z FROM src").arrow())
    con.execute("SELECT * FROM out").fetchall()

Measured 2026-08-07 on duckdb 1.5.5 / pyarrow 25.0.0 / CPython 3.14, five
runs of that exact script: **three hung with no output and no error, two
returned zero rows and reported success.** The reader spellings
``con.sql(...).arrow()`` and ``fetch_record_batch()`` hung on every attempt.

The silent-empty outcome is the one that matters — C5's "no third mode"
violated at the API level, since it neither serves nor refuses. An earlier
version of this file called the whole thing a deadlock; that was overstated
from a run of three where it happened to hang each time.

This is *not* a cross-connection problem. One connection, and no module-level
``duckdb.execute`` (which would be the default connection, i.e. a second one).
Feeding a connection its own undrained result is the self-reference at issue.

Working spellings, both pinned below: ``to_arrow_table()``, and draining the
reader into a table yourself. The model uses the first and never calls
``.arrow()`` at all.

The repro runs in a child process because a wedged DuckDB scan cannot be
interrupted from Python; a thread would hang interpreter exit.

Strict xfail, and it stays strict under either failure mode: a hang trips the
timeout and an empty result trips the comparison. It flips to XPASS the day
DuckDB materialises, drains, or raises.
"""

import subprocess
import sys

import pytest

# The budget is a HANG DETECTOR, not a performance bound -- the bug this
# file exists for wedges forever, and the timeout is what turns that into a
# failure instead of a wedged suite. It must therefore survive a LOADED
# machine: measured 2026-08-19, interpreter start + `import duckdb` alone
# takes 4.1-7.7s with 2x-cores of process burners (the healthy spelling's
# real work is ~1s on top), and the two observed flakes ran inside full
# suites at 3-6x normal wall time (TASK-123). 120s is >15x the worst
# measured cold start; a genuine hang still fails, just slower -- a price
# paid only when the bug actually regresses. Deliberately NO retry: an
# INTERMITTENT hang must not be able to pass on its second try.
_TIMEOUT = 120.0
_EXPECTED = "[(20.0,), (40.0,), (60.0,)]"

_DRAIN = (
    "r = con.execute('SELECT price * 2 AS z FROM src').arrow()\n"
    "obj = pa.Table.from_batches(list(r), r.schema)\n"
)


def _repro(fetch: str) -> str:
    body = (
        _DRAIN
        if fetch == "drain"
        else f'obj = con.execute("SELECT price * 2 AS z FROM src").{fetch}()\n'
    )
    return (
        "import duckdb, pyarrow as pa\n"
        "con = duckdb.connect()\n"
        'con.register("src", pa.table({"price": [10.0, 20.0, 30.0]}))\n'
        + body
        + 'con.register("out", obj)\n'
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


@pytest.mark.parametrize("fetch", ["to_arrow_table", "drain"])
def test_the_working_spellings(fetch):
    """Not xfailed: the model depends on ``to_arrow_table``."""
    assert _run(fetch) == _EXPECTED


@pytest.mark.parametrize("fetch", ["arrow", "fetch_record_batch"])
@pytest.mark.xfail(
    strict=True,
    reason="DuckDB 1.5.5: a RecordBatchReader registered back into the "
    "connection that produced it either hangs or scans as zero rows. Use "
    "to_arrow_table(). See the module docstring for the measurements.",
)
def test_a_lazy_reader_round_trips_through_register(fetch):
    assert _run(fetch) == _EXPECTED
