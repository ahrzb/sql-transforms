# independent env shim: CPython 3.14.0rc2 typing._eval_type lacks prefer_fwd_module
import typing
_o = typing._eval_type
def _e(*a, prefer_fwd_module=None, **k):
    return _o(*a, **k)
typing._eval_type = _e
