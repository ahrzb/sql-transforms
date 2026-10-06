"""Execute the current authoring contract's examples."""

import doctest
from pathlib import Path

CONTRACT = Path(__file__).resolve().parents[1] / "docs" / "contract.md"


def test_contract_examples():
    result = doctest.testfile(
        str(CONTRACT),
        module_relative=False,
        optionflags=doctest.ELLIPSIS,
    )
    assert result.failed == 0
