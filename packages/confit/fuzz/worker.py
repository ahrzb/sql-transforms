"""Worker loop: seeds on stdin, verdict JSON per line on stdout.

Runs one case per line so the parent can blame the in-flight seed when this
process dies (rust panic, abort, unbounded recursion) or hangs.
"""

from __future__ import annotations

import json
import sys

from .oracle import run_case_json


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        print(json.dumps(run_case_json(int(line))), flush=True)


if __name__ == "__main__":
    main()
