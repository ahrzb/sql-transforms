"""The surface is the contract: it must exist, and it must refuse honestly.

These tests are deliberately about shape, not behaviour. They fail if a method
is dropped or renamed, and they fail if one quietly starts returning something
instead of raising -- which is the failure mode a stub package invites.
"""

from __future__ import annotations

import inspect

import pytest

from sql_transform import SQLTransform

# `from_file` is a classmethod, so `cls` is already bound off the class object.
SIGNATURES = {
    "__init__": ["self", "sql"],
    "from_file": ["path"],
    "fit": ["self", "table", "this_model"],
    "infer": ["self", "row"],
    "infer_batch": ["self", "rows"],
}


@pytest.mark.parametrize(("name", "params"), SIGNATURES.items())
def test_surface_keeps_its_signature(name, params):
    sig = inspect.signature(getattr(SQLTransform, name))
    assert list(sig.parameters) == params


def test_constructing_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="rebuilt on confit"):
        SQLTransform("SELECT 1 AS x FROM __THIS__")


@pytest.mark.parametrize("name", ["backend", "boundary"])
def test_properties_are_declared_and_raise(name):
    prop = getattr(SQLTransform, name)
    assert isinstance(prop, property), f"{name} must stay a property"
    with pytest.raises(NotImplementedError):
        prop.fget(object.__new__(SQLTransform))


@pytest.mark.parametrize(
    ("name", "args"),
    [("fit", (None,)), ("infer", ({},)), ("infer_batch", ([],)), ("from_file", ("x",))],
)
def test_methods_raise_not_implemented(name, args):
    # Bypass __init__ (which also raises) to reach each method independently.
    obj = object.__new__(SQLTransform)
    target = getattr(SQLTransform, name) if name == "from_file" else getattr(obj, name)
    with pytest.raises(NotImplementedError, match="no implementation yet"):
        target(*args)
