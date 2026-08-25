"""Join-semantics probe backing joins.json (TASK-104).

Run: uv run python probe_joins.py
"""

import duckdb

con = duckdb.connect()
con.execute("CREATE TABLE t1 (a INT, b VARCHAR)")
con.execute("INSERT INTO t1 VALUES (1,'x'), (2,'y'), (NULL,'n')")
con.execute("CREATE TABLE t2 (a INT, c DOUBLE)")
con.execute("INSERT INTO t2 VALUES (1,10.0), (3,30.0), (NULL,99.0)")
con.execute("CREATE TABLE t3 (d INT); INSERT INTO t3 VALUES (7)")


def show(sql):
    try:
        cur = con.execute(sql)
        print(
            sql,
            "\n  cols:",
            [d[0] for d in cur.description],
            "\n  rows:",
            sorted(cur.fetchall(), key=repr),
        )
    except Exception as e:  # noqa: BLE001 - a probe records error classes too
        print(sql, "\n  ERROR:", type(e).__name__, str(e)[:140])


show("SELECT * FROM t1 JOIN t2 USING (a)")
show("SELECT * FROM t1 LEFT JOIN t2 USING (a)")
show("SELECT * FROM t1 RIGHT JOIN t2 USING (a)")
show("SELECT * FROM t1 FULL JOIN t2 USING (a)")
show("SELECT a FROM t1 RIGHT JOIN t2 USING (a)")
show("SELECT a FROM t1 FULL JOIN t2 USING (a)")
show("SELECT t1.a, t2.a FROM t1 FULL JOIN t2 USING (a)")
show("SELECT * FROM t1 NATURAL JOIN t2")
show("SELECT * FROM t1 JOIN t2 ON t1.a = t2.a")
show("SELECT a FROM t1 JOIN t2 ON t1.a = t2.a")
show("SELECT * FROM t1, t2 WHERE t1.a = t2.a")
show("SELECT * FROM t1 JOIN t2 USING (a) WHERE a > 0")
show("SELECT * FROM t1 AS x JOIN t1 AS x ON true")
show("SELECT * FROM t1 JOIN t2")
show("SELECT * FROM t1 NATURAL JOIN t3")
show("SELECT * FROM t1, t3 JOIN t2 ON t1.a = t2.a")
