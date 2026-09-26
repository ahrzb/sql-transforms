"""Confit depends on nothing above it.

sql-transform builds on confit (it hands fitted models to `udfs=` and serves
through `DuckDBInferFn`); the arrow never runs the other way. That holds for
the whole package tree -- library, tests and fuzzer -- so confit can be
tested and shipped on its own: the tree protocol is exercised with models
built straight into Arrow (tests/test_tree_predict.py, fuzz/trees.py), the
UDF protocol with duck-typed objects, and sklearn parity is sql-transform's
gate. Read off the sources, like the raw-connection ban in test_oracle.py.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
ROOTS = ("confit", "tests", "fuzz")
BANNED = ("sql_transform", "sklearn")


def _sources():
    for root in ROOTS:
        yield from sorted((PACKAGE / root).rglob("*.py"))


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


@pytest.mark.parametrize(
    "path", list(_sources()), ids=lambda p: p.relative_to(PACKAGE).as_posix()
)
def test_no_confit_source_imports_upward(path):
    bad = [m for m in _imports(path) if m.split(".")[0] in BANNED]
    assert not bad, f"{path.name} imports {bad}: confit must not depend on them"


def test_the_manifest_declares_no_upward_dependency():
    text = (PACKAGE / "pyproject.toml").read_text(encoding="utf-8")
    assert "sql-transform" not in text and "sql_transform" not in text
    assert "scikit-learn" not in text
