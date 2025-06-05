import re
from typing import Any

import pyarrow as pa
import pytest

from sql_transform.transformer import SQLTransformer

# Create a PyArrow table with 5 rows for a hypothetical classification use case
data = {
    "feature1": [5.1, 4.9, 4.7, 4.6, 5.0],
    "feature2": [3.5, 3.0, 3.2, 3.1, 3.6],
    "feature3": [1.4, 1.4, 1.3, 1.5, 1.4],
    "feature4": [0.2, 0.2, 0.2, 0.2, 0.2],
    "class": ["A", "B", "A", "A", "B"],
}
table = pa.table(data)


def _test_case(
    query: str, **expected_columns: list[float] | pa.Array | pa.ChunkedArray
) -> tuple[str, dict[str, list[Any]]]:
    query = re.sub(r"\n\s+", " ", query)
    expected = {
        name: value if isinstance(value, list) else value.to_pylist()
        for name, value in expected_columns.items()
    }
    return (query, expected)


@pytest.mark.parametrize(
    "sql,expected_columns",
    [
        _test_case(
            "SELECT feature1 as feature1 FROM data",
            feature1=[5.1, 4.9, 4.7, 4.6, 5.0],
        ),
        _test_case(
            "SELECT feature1 - avg(feature1) as x FROM data",
            x=[0.24, 0.04, -0.16, -0.26, 0.14],
        ),
        _test_case(
            """
            SELECT
                feature1 as feature1,
                avg(feature1) as avg_feature1,
                avg(feature2) as avg_feature2,
                avg(feature1) over (partition by class) as avg_feature1_by_class,
                avg(feature2) over (partition by class) as avg_feature2_by_class
            FROM data
            """,
            feature1=[5.1, 4.9, 4.7, 4.6, 5.0],
            avg_feature1=[4.86, 4.86, 4.86, 4.86, 4.86],
            avg_feature2=[3.28, 3.28, 3.28, 3.28, 3.28],
            avg_feature1_by_class=[4.8, 4.95, 4.8, 4.8, 4.95],
            avg_feature2_by_class=[
                3.266666666666667,
                3.3,
                3.266666666666667,
                3.266666666666667,
                3.3,
            ],
        ),
    ],
)
def test_transformer(sql: str, expected_columns: dict[str, list[Any]]):
    transformer = SQLTransformer(sql)
    transformer.fit(table)
    result = transformer.transform(table)
    expected = pa.table(expected_columns)

    # Sort both tables by the first column to ensure consistent ordering
    first_col = list(expected_columns.keys())[0]
    result_sorted = result.sort_by(first_col)
    expected_sorted = expected.sort_by(first_col)

    for column in expected_columns.keys():
        result_values = result_sorted[column].to_pylist()
        expected_values = expected_sorted[column].to_pylist()
        assert result_values == pytest.approx(expected_values), (
            f"Column {column} values do not match"
        )
    assert set(result.column_names) == set(expected_columns.keys())
