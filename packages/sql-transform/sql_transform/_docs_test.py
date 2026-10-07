"""Run the pycon examples in the package spec."""

import doctest
from pathlib import Path

import pytest

SPEC = Path(__file__).resolve().parents[1] / "spec"
# A missing spec folder must fail, not leave an empty parameter set.
PAGES = sorted(SPEC.rglob("*.md")) or [SPEC / "README.md"]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.relative_to(SPEC).as_posix())
def test_spec_examples(page):
    result = doctest.testfile(
        str(page),
        module_relative=False,
        optionflags=doctest.ELLIPSIS,
        encoding="utf-8",
    )
    assert result.failed == 0
