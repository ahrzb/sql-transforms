"""The catalog's fixture generator, loaded from the repo's source text without
importing the sql_transform package (whose __init__ fails to import under this
venv's CPython 3.14.0rc2 + pydantic). PythonTransform is loaded from _udf.py
alone. The generator functions are exec'd verbatim from catalog_test.py."""

from __future__ import annotations

import ast
import functools
import importlib.util
import math
import os
import random
import warnings
from typing import Any

import numpy as np
import pyarrow as pa
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline

REPO = "/home/user/sql-transforms/packages/sql-transform/sql_transform"

_spec = importlib.util.spec_from_file_location("sqlt_udf", f"{REPO}/_udf.py")
udf = importlib.util.module_from_spec(_spec)
import sys as _sys
_sys.modules['sqlt_udf'] = udf
_spec.loader.exec_module(udf)
PythonTransform = udf.PythonTransform

_WANT = {
    "VOCAB", "UNSEEN", "EDGES", "SMALL", "MAX_LANES",
    "_fit_matrix", "_step", "_runs", "_draw", "_value", "_rows",
}
_src = open(os.environ.get("FIXTURE_SRC", f"{REPO}/native/catalog_test.py")).read()
_tree = ast.parse(_src)
_nodes = []
for node in _tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in _WANT:
        _nodes.append(node)
    elif isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in _WANT for t in node.targets
    ):
        _nodes.append(node)
ns: dict[str, Any] = dict(
    np=np, pa=pa, random=random, warnings=warnings, math=math, os=os,
    functools=functools, Any=Any, Pipeline=Pipeline,
    ColumnTransformer=ColumnTransformer, FeatureUnion=FeatureUnion,
    PythonTransform=PythonTransform,
)
exec(compile(ast.Module(body=_nodes, type_ignores=[]), "catalog_test_extract", "exec"), ns)
_step = ns["_step"]
_rows = ns["_rows"]
assert {"_step", "_rows"} <= set(ns)
