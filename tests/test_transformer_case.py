"""TASK-31: a transformer reference nested inside a CASE branch resolves.

`resolve_transformers` (src/lib.rs) recurses through the expression tree to
rewrite each `foo(...)` transformer call into an `Expr::Transform`. Its CASE arm
has a *silent* failure mode: if it ever stopped recursing into every branch, a
transformer call in a non-first branch would simply never be rewritten, and the
engine would then fail (or worse, silently mishandle) at eval time.

The guard here puts the transformer in a NON-first WHEN arm AND in the ELSE
(both non-first branches, exercised by the arms loop beyond index 0 and by the
separate default recursion). If either recursion regressed, the g=2 row (arm 1)
or the g=3 row (ELSE) would raise "Unknown function __tfm_0__" instead of
matching the DataFusion oracle -- so this test genuinely FAILS on the
regression, not just passes on the happy path.

Transformers are a native-only feature (codegen has no transformer support and
`SQLTransform` runs on the native InferFn), so parity is asserted native-vs-oracle;
the codegen side is covered by asserting it defers the construct loudly rather
than mishandling it silently.
"""

import pandas as pd
import pyarrow as pa
import pytest
from sklearn.preprocessing import StandardScaler
from test_diff_transformer_callout import _parity

from sql_transform._codegen import CodegenFn, UnsupportedInCodegen
from sql_transform._schema import synthesize_this_model

# Same known false-positive warning as the other transformer tests: both the
# oracle UDF and the native infer path emit "X does not have valid feature names".
pytestmark = pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names:UserWarning"
)

# A transformer in a non-first WHEN arm (g=2) and in the ELSE (g=3); a plain
# named_struct in the first arm (g=1) so the transformer is genuinely not the
# first branch. All three branches are struct{age: float}, so the CASE type-checks.
_SQL = (
    "SELECT CASE "
    "WHEN g = 1 THEN named_struct('age', 0.0) "
    "WHEN g = 2 THEN __tfm_0__(named_struct('age', age)) "
    "ELSE __tfm_0__(named_struct('age', age)) "
    "END AS s FROM __THIS__"
)


def test_transformer_in_nonfirst_case_branch_parity_native():
    # Rows hit each branch: g=1 -> plain arm, g=2 -> transformer in arm 1,
    # g=3 -> transformer in ELSE. If resolve_transformers stopped recursing into
    # any branch, g=2 or g=3 would raise instead of matching the oracle.
    train = pd.DataFrame({"g": [1, 2, 3], "age": [10.0, 20.0, 30.0]})
    sc = StandardScaler().fit(train[["age"]])
    schema = pa.schema([("age", pa.float64())])
    _parity(_SQL, pa.Table.from_pandas(train), sc, schema, schema)


def test_transformer_in_case_defers_on_codegen():
    # Codegen has no transformer support and already defers named_struct; the
    # construct must defer loudly (UnsupportedInCodegen), never silently mishandle.
    model = synthesize_this_model(pa.schema([("g", pa.int64()), ("age", pa.float64())]))
    with pytest.raises(UnsupportedInCodegen):
        CodegenFn(_SQL, row_tables={"__THIS__": model}, static_tables={})
