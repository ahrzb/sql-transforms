"""A missing native build must not silently become a passing test run."""

import pytest
from _native_guard import ensure_native_built


def test_missing_build_without_maturin_stops_validation(monkeypatch, tmp_path):
    monkeypatch.setattr("_native_guard._PKG", tmp_path)
    monkeypatch.setattr("_native_guard._maturin", lambda: None)

    with pytest.raises(RuntimeError):
        ensure_native_built()
