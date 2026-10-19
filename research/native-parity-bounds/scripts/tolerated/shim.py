"""Environment shim: this venv's Python is 3.14.0rc2, whose
typing._eval_type lacks the `prefer_fwd_module` kwarg pydantic 2.13 passes,
so `import sql_transform` fails at class creation. Drop the kwarg."""
import typing

_orig = typing._eval_type


def _eval_type(*a, prefer_fwd_module=None, **k):  # noqa: ARG001
    return _orig(*a, **k)


typing._eval_type = _eval_type
