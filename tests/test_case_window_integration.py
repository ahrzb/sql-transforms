"""CASE composing with window-aggregate freezing (fit -> freeze -> infer).

A window aggregate inside a CASE is frozen at fit and looked up per row at
infer, exactly like any other window agg; CASE just wraps the resulting
state-column reference. This pins that the two compose (transform ==
infer_batch) through SQLTransform's native infer path -- reachable now that
native supports CASE (TASK-27). Deferred here from TASK-30, whose codegen CASE
was not reachable through SQLTransform.infer_batch (which is native-backed).
"""

import pyarrow as pa
from differential import _rows_equal

from sql_transform import SQLTransform


def test_case_over_window_agg_transform_infer_parity():
    train = pa.table({"g": ["a", "a", "b"], "x": [1.0, 3.0, 10.0]})
    sql = (
        "SELECT CASE WHEN x > AVG(x) OVER (PARTITION BY g) THEN 'above' "
        "ELSE 'below' END AS c FROM __THIS__"
    )
    t = SQLTransform(sql).fit(train)
    transform_out = t.transform(train).to_pylist()
    infer_out = [r.model_dump() for r in t.infer_batch(train.to_pylist())]
    assert _rows_equal(transform_out, infer_out), (transform_out, infer_out)
    # g='a': AVG=2 -> x=1 below, x=3 above; g='b': AVG=10 -> x=10 below.
    assert sorted(r["c"] for r in transform_out) == ["above", "below", "below"]
