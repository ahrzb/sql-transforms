"""Environment shim (NOT a repo change): this venv runs CPython 3.14.0rc2,
whose typing._eval_type lacks the `prefer_fwd_module` kwarg pydantic 2.13
passes; drop it so sql_transform imports."""
import typing

_orig = typing._eval_type


def _eval_type(*a, prefer_fwd_module=None, **k):
    return _orig(*a, **k)


typing._eval_type = _eval_type
