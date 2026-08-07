"""The guide is executable.

`docs/sql-transform-guide.md` is run as a doctest, so every number printed in
it is one this suite produced. A guide whose examples are merely plausible is
worse than none — it teaches a surface that does not exist.
"""

import doctest
import pathlib

import pytest

GUIDE = pathlib.Path(__file__).resolve().parents[4] / "docs" / "sql-transform-guide.md"


def test_the_guide_is_where_it_says_it_is():
    assert GUIDE.is_file(), GUIDE


def test_every_example_in_the_guide_runs():
    failed, attempted = doctest.testfile(
        str(GUIDE),
        module_relative=False,
        optionflags=doctest.ELLIPSIS | doctest.IGNORE_EXCEPTION_DETAIL,
        verbose=False,
    )
    assert attempted > 20, f"only {attempted} examples — the guide lost its teeth"
    if failed:
        pytest.fail(f"{failed}/{attempted} guide examples do not match their output")
