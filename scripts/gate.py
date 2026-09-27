"""The specializer loop's definition of done: cargo test + pytest, one exit code.

`cargo test` needs python3.dll on PATH at runtime (the crate links libpython for
tests; see the note in Cargo.toml). Run under `uv run` so sys.base_prefix is the
uv-managed CPython that owns that DLL.

pytest runs on GATE_WORKERS xdist workers (default 4, the CI runner's cores;
0 runs serially). The suite is parallel-safe: every test builds its own
engine and oracle, and none shares files.
"""

import os
import subprocess
import sys

env = os.environ.copy()
env["PATH"] = sys.base_prefix + os.pathsep + env["PATH"]

workers = os.environ.get("GATE_WORKERS", "4")
pytest = [sys.executable, "-m", "pytest", "-q"]
if workers != "0":
    pytest += ["-n", workers]

for cmd in (["cargo", "test"], pytest):
    print(f"gate: {' '.join(cmd)}", flush=True)
    # noqa justification: fixed argv, no untrusted input.
    result = subprocess.run(cmd, env=env)  # noqa: S603
    if result.returncode != 0:
        sys.exit(result.returncode)
print("gate: green")
