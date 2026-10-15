"""Worker loop: seeds on stdin, verdict JSON per line on stdout.

Runs one case per line so the parent can blame the in-flight seed when this
process dies (rust panic, abort, unbounded recursion) or hangs.

A REFUSED verdict is followed by one more line: `fuzz.oracle.refusal_json`,
the report-only DuckDB reading of the refused query. It comes second so the
verdict is out before DuckDB runs at all, and the parent reads it under a
budget of its own (`fuzz.runner`).

A case that DuckDB reads sends its verdict twice. The first line is the
verdict without the bracket, marked `"bracket": "pending"`, sent before the
optimizer-on reading runs; the second is the whole verdict. The parent reads
the second line under a budget of its own too: a slow bracket reading costs
only its label.

A line `timeout <seed>` asks for the witness of a seed whose oracle reading
ran past the time limit in an earlier worker: one line,
`fuzz.oracle.timeout_json`.

A worker runs thousands of cases, so it keeps one oracle database for all of
them (`fuzz.oracle.reuse_oracle`).
"""

from __future__ import annotations

import json
import sys

# Before the heavy imports: a worker killed while still starting up is the
# harness's time, not DuckDB's or ours (see fuzz.oracle.PHASE_MARK).
print("@@phase harness:startup", file=sys.stderr, flush=True)

from .oracle import (  # noqa: E402
    refusal_json,
    reuse_oracle,
    run_case_json,
    timeout_json,
)


def main() -> None:
    reuse_oracle()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line.startswith("timeout "):
            print(json.dumps(timeout_json(int(line.split()[1]))), flush=True)
            continue
        seed = int(line)
        print("@@phase harness:gen", file=sys.stderr, flush=True)
        out = run_case_json(seed, report=False, baseline=_pending)
        print(json.dumps(out), flush=True)
        if out["kind"] == "REFUSED":
            print(json.dumps(refusal_json(seed)), flush=True)


def _pending(out: dict) -> None:
    """The verdict without the bracket, before the bracket reading runs."""
    print(json.dumps({**out, "bracket": "pending"}), flush=True)


if __name__ == "__main__":
    main()
