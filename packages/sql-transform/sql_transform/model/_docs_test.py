"""The guide is executable.

`docs/sql-transform-guide/` is run as a doctest, page by page, so every number
printed in it is one this suite produced. A guide whose examples are merely
plausible is worse than none — it teaches a surface that does not exist.

Each page runs in its own namespace, which is the point: a reader who lands on
one page must be able to run what is on it. A page that quietly depends on a
name defined three pages back fails here rather than on them.
"""

import doctest
import pathlib

import pytest

GUIDE = pathlib.Path(__file__).resolve().parents[4] / "docs" / "sql-transform-guide"
PAGES = sorted(GUIDE.glob("*.md"))

FLAGS = doctest.ELLIPSIS | doctest.IGNORE_EXCEPTION_DETAIL


def test_the_guide_is_where_it_says_it_is():
    assert GUIDE.is_dir(), GUIDE
    assert (GUIDE / "README.md").is_file(), "the index is missing"


def test_every_page_is_listed_in_the_index():
    """A page nobody links to is a page nobody reads."""
    index = (GUIDE / "README.md").read_text(encoding="utf-8")
    missing = [p.name for p in PAGES if p.name != "README.md" and p.name not in index]
    assert not missing, f"not linked from README.md: {missing}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_example_on_the_page_runs(page):
    failed, attempted = doctest.testfile(
        str(page), module_relative=False, optionflags=FLAGS, verbose=False
    )
    if failed:
        pytest.fail(f"{page.name}: {failed}/{attempted} examples do not match")


def test_the_guide_as_a_whole_still_has_teeth():
    total = sum(
        doctest.testfile(
            str(p), module_relative=False, optionflags=FLAGS, verbose=False
        ).attempted
        for p in PAGES
    )
    assert total > 80, f"only {total} examples across the guide — it lost its teeth"
