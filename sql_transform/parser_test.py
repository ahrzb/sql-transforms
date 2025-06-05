import pytest

from sql_transform.parser import (
    AggregateFunction,
    AggregationRef,
    ApplyFunction,
    ColumnRef,
    Query,
    WindowSpecification,
    parse,
)


@pytest.mark.parametrize(
    "sql,expected",
    [
        (
            """
            SELECT 
                feature1 as feature1,
                feature2 as feature2
            FROM data
            """,
            Query(
                columns={
                    "feature1": ColumnRef("feature1"),
                    "feature2": ColumnRef("feature2"),
                },
            ),
        ),
        (
            """
            SELECT
                AVG(feature1) as avg_feature1
            FROM data
            """,
            Query(
                columns={
                    "avg_feature1": AggregationRef(0, "feature1"),
                },
                aggregations={
                    AggregationRef(0, "feature1"): AggregateFunction(
                        "avg", [ColumnRef("feature1")], over=WindowSpecification()
                    )
                },
            ),
        ),
        (
            """
            SELECT
                AVG(feature1) over (partition by feature2) as avg_feature1
            FROM data
            """,
            Query(
                columns={
                    "avg_feature1": AggregationRef(0, "feature1"),
                },
                aggregations={
                    AggregationRef(0, "feature1"): AggregateFunction(
                        "avg",
                        [ColumnRef("feature1")],
                        over=WindowSpecification(partition_by=[ColumnRef("feature2")]),
                    )
                },
            ),
        ),
        (
            """
            SELECT
                STDDEV(feature1 - AVG(feature1) over (partition by feature2))
                    over (partition by feature2)
                    AS feature1_normalized
            FROM data
            """,
            Query(
                columns={
                    "feature1_normalized": AggregationRef(1, "sub_feature1_feature1"),
                },
                aggregations={
                    AggregationRef(0, "feature1"): AggregateFunction(
                        "avg",
                        [ColumnRef("feature1")],
                        over=WindowSpecification(partition_by=[ColumnRef("feature2")]),
                    ),
                    AggregationRef(1, "sub_feature1_feature1"): AggregateFunction(
                        "stddev",
                        [
                            ApplyFunction(
                                "-",
                                [ColumnRef("feature1"), AggregationRef(0, "feature1")],
                            ),
                        ],
                        over=WindowSpecification(partition_by=[ColumnRef("feature2")]),
                    ),
                },
            ),
        ),
    ],
)
def test_parse(sql, expected):
    assert parse(sql) == expected
