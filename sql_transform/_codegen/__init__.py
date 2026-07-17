"""Codegen serving engine — a codegen-based `InferFn`.

`runtime` = value semantics, `plan` = sqlglot front-end (parse/optimize/validate/
type-infer), `engine` = emitter + `CodegenFn`. Public surface re-exported here so
`from sql_transform._codegen import CodegenFn` keeps working.
"""

from sql_transform._codegen.engine import CodegenFn
from sql_transform._codegen.plan import UnsupportedInCodegen

__all__ = ["CodegenFn", "UnsupportedInCodegen"]
