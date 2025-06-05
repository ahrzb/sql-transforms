from typing import Self

import datafusion
import pyarrow as pa

import sql_transform.parser
from sql_transform.parser import AggregateFunction, Expression


class SQLTransformer:
    def __init__(self, sql: str):
        self.ctx = datafusion.SessionContext()
        self.query = sql_transform.parser.parse(sql=sql)
        self.aggs: list[tuple[AggregateFunction, datafusion.DataFrame]] = []

    def fit(self, data: pa.Table) -> Self:
        df = self.ctx.from_arrow(data)

        self.partition_keys: dict[Expression, str] = {}
        for agg in self.query.aggregations.values():
            for partition in agg.over.partition_by:
                if partition in self.partition_keys:
                    continue
                self.partition_keys[partition] = (
                    f"${partition.hint_name()}_partition{len(self.partition_keys)}"
                )

        for agg_ref, agg in self.query.aggregations.items():
            agg_df = df.aggregate(
                [
                    p.datafusion().alias(f"_{self.partition_keys[p]}")
                    for p in agg.over.partition_by
                ],
                agg.datafusion().alias(agg_ref.name),
            )
            self.aggs.append((agg, agg_df))

        return self

    def transform(self, data: pa.Table) -> pa.Table:
        df = self.ctx.from_arrow(data)

        df = df.with_columns(
            [
                expr.datafusion().alias(name)
                for expr, name in self.partition_keys.items()
            ]
        )

        for agg, aggdf in self.aggs:
            df = df.join(
                aggdf,
                left_on=[self.partition_keys[p] for p in agg.over.partition_by],
                right_on=[f"_{self.partition_keys[p]}" for p in agg.over.partition_by],
                how="left",
            ).drop(*[f"_{self.partition_keys[p]}" for p in agg.over.partition_by])

        return df.select(
            *[col.datafusion().alias(name) for name, col in self.query.columns.items()]
        ).to_arrow_table()
