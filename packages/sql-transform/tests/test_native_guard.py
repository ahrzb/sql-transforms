"""Unit tests for the native-staleness check (TASK-33).

Only the pure mtime-compare is tested here; the maturin subprocess glue in
ensure_native_built() is thin and verified live (touch a src/*.rs, watch the
suite rebuild).
"""

import os

from _native_guard import is_stale


def _touch(path, mtime):
    path.write_text("x")
    os.utime(path, (mtime, mtime))


def test_stale_when_pyd_missing(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _touch(src / "lib.rs", 1000)
    assert is_stale(src, None) is True
    assert is_stale(src, tmp_path / "nonexistent.pyd") is True


def test_fresh_when_pyd_newer_than_all_rs(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _touch(src / "lib.rs", 1000)
    _touch(src / "expr.rs", 1500)
    pyd = tmp_path / "_interpreter.pyd"
    _touch(pyd, 2000)  # built after the newest source
    assert is_stale(src, pyd) is False


def test_stale_when_any_rs_newer_than_pyd(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    pyd = tmp_path / "_interpreter.pyd"
    _touch(pyd, 2000)
    _touch(src / "lib.rs", 1000)  # older, fine
    _touch(src / "expr.rs", 2500)  # newer than the build -> stale
    assert is_stale(src, pyd) is True


def test_equal_mtime_is_not_stale(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    pyd = tmp_path / "_interpreter.pyd"
    _touch(src / "lib.rs", 2000)
    _touch(pyd, 2000)  # tie -> not stale (built no earlier than source)
    assert is_stale(src, pyd) is False


def test_nested_rs_is_seen(tmp_path):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    pyd = tmp_path / "_interpreter.pyd"
    _touch(pyd, 2000)
    _touch(src / "sub" / "deep.rs", 2500)  # newer, nested
    assert is_stale(src, pyd) is True
