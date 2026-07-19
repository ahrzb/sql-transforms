"""TASK-32: native struct detection must probe the CLASS, not the instance.

`from_pyobject_typed` decides "is this struct-shaped?" for a non-dict value by
reading `model_fields`. Reading it off a pydantic INSTANCE is deprecated in
Pydantic 2.11 (PydanticDeprecatedSince211) and REMOVED in 3.0 -- on v3 the
instance probe returns false and native misclassifies struct-typed values,
silently wrong. A plain transform==infer parity test passes on 2.x either way
(the deprecated access still works and only warns), so the meaningful assertion
is that NO PydanticDeprecatedSince211 warning fires when a struct value -- which
arrives at infer as a validated nested-model INSTANCE -- flows through infer.
"""

import warnings

from differential import _run_infer, rows


def test_struct_infer_probes_class_not_instance():
    # `s` is declared struct-typed, so the row model carries it as a nested
    # submodel; at infer the value is a submodel INSTANCE -- exactly the path
    # that hit `obj.hasattr("model_fields")` on the instance.
    tables = {"t": rows({"s": "struct{x:int,y:int}"}, [{"s": {"x": 1, "y": 2}}])}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = _run_infer("SELECT s AS r FROM t", tables)
    assert out == [{"r": {"x": 1, "y": 2}}]
    offenders = [w for w in rec if w.category.__name__ == "PydanticDeprecatedSince211"]
    assert not offenders, [str(w.message) for w in offenders]
