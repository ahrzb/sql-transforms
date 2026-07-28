"""Rebuild the native `_engine` extension before tests when it's stale.

After a checkout / rebase / history rewrite the Rust source can be newer than
the built `.pyd`, so the suite silently runs OLD native code -- this once
produced 14 phantom identifier-folding failures until a manual rebuild
(TASK-33). `ensure_native_built()` runs a cheap mtime compare at conftest import
-- BEFORE anything imports `confit`, which eagerly loads the native module --
and shells `maturin develop` only when stale; an up-to-date build is a no-op.

Locate the `.pyd` on disk, NOT via importlib.find_spec: resolving the submodule
would import the parent package and load the stale native module into the
process before we get a chance to rebuild it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from shutil import which

_PACKAGE = Path(__file__).resolve().parent.parent
_SRC = _PACKAGE / "src"
_PKG = _PACKAGE / "confit"


def _newest_rs_mtime(src_dir: Path) -> float:
    return max((p.stat().st_mtime for p in src_dir.glob("**/*.rs")), default=0.0)


def _pyd_path(pkg_dir: Path) -> Path | None:
    # maturin names it `_engine.pyd` (Windows) or `_engine*.so` (posix).
    for pattern in ("_engine*.pyd", "_engine*.so"):
        hits = sorted(pkg_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def is_stale(src_dir: Path, pyd: Path | None) -> bool:
    """True if the native build is missing, or any Rust source is newer than it.
    Equal mtimes count as fresh (the build is no older than the source)."""
    if pyd is None or not pyd.exists():
        return True
    return _newest_rs_mtime(src_dir) > pyd.stat().st_mtime


def _maturin() -> str | None:
    exe = "maturin.exe" if sys.platform == "win32" else "maturin"
    in_venv = Path(sys.executable).parent / exe
    return str(in_venv) if in_venv.exists() else which("maturin")


def ensure_native_built() -> None:
    """Rebuild `_engine` via maturin iff a `src/*.rs` is newer than the built
    extension. No-op when the build is up to date.

    ponytail: no lock -- under `pytest -n` (xdist) each worker could race the
    rebuild. The repo runs plain `uv run pytest`, so serialize only if xdist is
    ever adopted.
    """
    if not is_stale(_SRC, _pyd_path(_PKG)):
        return
    maturin = _maturin()
    if maturin is None:
        print(
            "native guard: src/*.rs is newer than the built _engine, but "
            "maturin was not found -- skipping rebuild (confit may be stale).",
            file=sys.stderr,
        )
        return
    print(
        "native guard: src/*.rs newer than _engine -- rebuilding "
        "(maturin develop, DEBUG profile; benchmarks will refuse this build "
        "-- rerun with --release before timing)...",
        file=sys.stderr,
    )
    # noqa justification: maturin is the venv's own executable and _PACKAGE
    # derives from __file__ -- no untrusted input.
    subprocess.run([maturin, "develop"], cwd=_PACKAGE, check=True)  # noqa: S603
