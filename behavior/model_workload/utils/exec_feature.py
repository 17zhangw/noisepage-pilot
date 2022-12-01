import copy
import json
from tqdm import tqdm
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from psycopg.rows import dict_row
from pandas.api import types as pd_types

from behavior import OperatingUnit
from behavior.model_workload.utils import OpType


INDEX_EXEC_FEATURES = [
    ("query_order", "bigint"),
    ("statement_timestamp", "bigint"),
    ("unix_timestamp", "float8"),
    ("optype", "int"),
    ("txn", "int"),
    ("target", "text"),
    ("num_modify_tuples", "int"),
    ("num_select_tuples", "int"),
    ("num_inserts", "int"),
    ("num_extend", "int"),
    ("num_split", "int"),
]

INDEX_EXEC_FEATURES_QUERY = """
SELECT  s.query_order,
        s.statement_timestamp,
        s.unix_timestamp,
        s.optype,
        s.txn,
        s.target,
        MAX(s.num_modify_tuples) as num_modify_tuples,
        MAX(s.num_select_tuples) as num_select_tuples,
        MAX(s.num_inserts) as num_inserts,
        MAX(s.num_extend) as num_extend,
        MAX(s.num_split) as num_split
FROM (
    SELECT
        a.query_order,
        a.query_id,
        a.statement_timestamp,
        a.unix_timestamp,
        a.optype,
        a.txn,
        COALESCE(b.target_idx_insert, b.target_idx_scan, b.target_idx_scan_table, b.target) AS target,
        SUM(CASE b.comment WHEN 'ModifyTableInsert' THEN 1 WHEN 'ModifyTableUpdate' THEN b.counter8 WHEN 'ModifyTableDelete' THEN b.counter5 ELSE 0 END) OVER w AS num_modify_tuples,
        SUM(CASE b.comment WHEN 'IndexScan' THEN b.counter0 WHEN 'IndexOnlyScan' THEN b.counter0 ELSE 0 END) OVER w AS num_select_tuples,
        SUM(CASE b.comment WHEN 'ModifyTableIndexInsert' THEN b.counter1 ELSE 0 END) OVER w AS num_extend,
        SUM(CASE b.comment WHEN 'ModifyTableIndexInsert' THEN b.counter2 ELSE 0 END) OVER w as num_split,
        SUM(CASE b.comment WHEN 'ModifyTableIndexInsert' THEN 1 ELSE 0 END) OVER w as num_inserts
        FROM {work_prefix}_mw_queries_args a
        LEFT JOIN LATERAL (SELECT * FROM {work_prefix}_mw_queries b WHERE a.query_order = b.query_order AND b.plan_node_id != -1) b ON a.query_order = b.query_order
        WHERE a.target = '{target}'
        WINDOW w AS (PARTITION BY a.query_order, b.payload)
) s
WHERE position(',' in s.target) = 0
GROUP BY s.query_order, s.statement_timestamp, s.unix_timestamp, s.optype, s.txn, s.target;
"""


TABLE_EXEC_FEATURES = [
    ("query_order", "bigint"),
    ("statement_timestamp", "bigint"),
    ("unix_timestamp", "float8"),
    ("optype", "int"),
    ("txn", "int"),
    ("target", "text"),
    ("num_modify_tuples", "int"),
    ("num_select_tuples", "int"),
    ("num_extend", "int"),
    ("num_hot", "int"),
    ("num_defrag", "int")
]


# The use of MAX() is really funky but that's because postgres doesn't have an understanding
# of a first/any aggregate. In reality, the numbers computed are static across the window
# because of the inner query. So we can just take any value, as far as i know.
# (famous last words)
#
# num_select_tuples is an awkward one. This is intuitively *every* tuple that is
# touched regardless of what purpose it is for. For instance, if you update 100
# tuples, num_select_tuples = 100 and num_modify_tuples = 100.
#
# FIXME(BITMAP): Account for BitmapIndex/HeapScan in num_select_tuples/num_defrag
TABLE_EXEC_FEATURES_QUERY = """
	SELECT
		s.query_order,
		s.statement_timestamp,
		s.unix_timestamp,
		s.optype,
		s.txn,
		s.target,
		MAX(s.num_modify_tuples) as num_modify_tuples,
		MAX(s.num_select_tuples) as num_select_tuples,
		MAX(s.num_extend) as num_extend,
		MAX(s.num_hot) as num_hot,
		MAX(s.num_defrag) as num_defrag
	FROM (
		SELECT
			a.query_order,
			a.query_id,
			a.statement_timestamp,
			a.unix_timestamp,
			a.optype,
			a.txn,
            b.comment,
			COALESCE(b.target_idx_scan_table, b.target) AS target,

			SUM(CASE b.comment
			    WHEN 'ModifyTableInsert' THEN 1
			    WHEN 'ModifyTableUpdate' THEN b.counter8
			    WHEN 'ModifyTableDelete' THEN b.counter5
			    ELSE 0 END) OVER w AS num_modify_tuples,

			SUM(CASE b.comment
			    WHEN 'IndexScan' THEN b.counter0
			    WHEN 'IndexOnlyScan' THEN b.counter0
			    WHEN 'SeqScan' THEN b.counter0
			    ELSE 0 END) OVER w AS num_select_tuples,

			SUM(CASE b.comment
			    WHEN 'ModifyTableInsert' THEN b.counter4
			    WHEN 'ModifyTableUpdate' THEN b.counter4
			    ELSE 0 END) OVER w AS num_extend,

			SUM(CASE b.comment
			    WHEN 'ModifyTableUpdate' THEN b.counter8 - b.counter1
			    ELSE 0 END) OVER w AS num_hot,

			SUM(CASE b.comment
			    WHEN 'IndexScan' THEN b.counter3
			    WHEN 'IndexOnlyScan' THEN b.counter3
			    WHEN 'SeqScan' THEN b.counter1
			    ELSE 0 END) OVER w AS num_defrag

        FROM {work_prefix}_mw_queries_args a
        LEFT JOIN LATERAL (SELECT * FROM {work_prefix}_mw_queries b WHERE a.query_order = b.query_order AND b.plan_node_id != -1) b ON a.query_order = b.query_order
        WHERE a.target = '{target}'
        WINDOW w AS (PARTITION BY a.query_order, b.target_idx_scan_table)
    ) s
    WHERE s.comment != 'ModifyTableIndexInsert' AND position(',' in s.target) = 0
    GROUP BY s.query_order, s.statement_timestamp, s.unix_timestamp, s.optype, s.txn, s.target;
"""


def build_table_exec(logger, connection, work_prefix, tables):
    logger.info("Building execution statistics.")
    with connection.transaction():
        sql = f"CREATE UNLOGGED TABLE {work_prefix}_mw_stats ("
        sql += ",".join([f"{k} {v}" for k, v in TABLE_EXEC_FEATURES])
        sql += ") WITH (autovacuum_enabled = OFF)"
        connection.execute(sql)

        sql = f"CREATE UNLOGGED TABLE {work_prefix}_mw_stats_index ("
        sql += ",".join([f"{k} {v}" for k, v in INDEX_EXEC_FEATURES])
        sql += ") WITH (autovacuum_enabled = OFF)"
        connection.execute(sql)

        for t in tables:
            sql = f"INSERT INTO {work_prefix}_mw_stats " + TABLE_EXEC_FEATURES_QUERY.format(work_prefix=work_prefix, target=t)
            logger.info("%s", sql)
            c = connection.execute(sql)
            logger.info("Finished executing: %s", c.rowcount)

        for t in tables:
            sql = f"INSERT INTO {work_prefix}_mw_stats_index " + INDEX_EXEC_FEATURES_QUERY.format(work_prefix=work_prefix, target=t)
            logger.info("%s", sql)
            c = connection.execute(sql)
            logger.info("Finished executing: %s", c.rowcount)


def construct_query_window_stats(logger, connection, work_prefix, tbls, tbl_index_map, window_index_map, buckets):
    tbl_ks = {}
    agg_stats = [
        "num_modify_tuples",
        "num_select_tuples",
        "num_extend",
        "num_hot",
        "num_defrag",
    ]

    agg_index_stats = [
        "num_modify_tuples",
        "num_select_tuples",
        "num_inserts",
        "num_extend",
        "num_split",
    ]

    ops = [
        ("num_insert_tuples", OpType.INSERT.value),
        ("num_update_tuples", OpType.UPDATE.value),
        ("num_delete_tuples", OpType.DELETE.value),
    ]

    qs = [
        ("num_select_queries", OpType.SELECT.value),
        ("num_insert_queries", OpType.INSERT.value),
        ("num_update_queries", OpType.UPDATE.value),
        ("num_delete_queries", OpType.DELETE.value),
    ]

    with connection.cursor() as cursor:
        for tbl in tbls:
            logger.info("Computing data keys for %s", tbl)

            # We have to truncate the first value off. This is because width_bucket() has the dynamic that anything less
            # than the 1st element returns index 0 (which is marked by the first entry of window_index_map[tbl].
            #
            # The 0th window of window_index_map spans all queries between window_index_map[tbl].time[0] and .time[1].
            # Which means we need width_bucket() to return 0 for QOs betweeen time[0] and time[1].
            query_orders = [str(i) for i in window_index_map[tbl].query_order.values[1:]]

            sql_format = """
                SELECT  width_bucket(query_order, ARRAY[{query_orders}]) as window_index, {agg_stats}
                FROM {work_prefix}_mw_stats{index}
                WHERE target = '{tbl}'
                GROUP BY window_index {aux_group}
            """

            def execute_sql(sql):
                records = []
                result = cursor.execute(sql)
                for r in result:
                    # This actually belongs to a window that we can use.
                    if window_index_map[tbl].iloc[r[0]].true_window_index != -1:
                        records.append(list(r))
                return records

            sql = sql_format.format(
                work_prefix=work_prefix,
                query_orders=",".join(query_orders),
                agg_stats=",".join(
                    [f"SUM({k})" for k in agg_stats] +
                    [f"SUM(CASE optype WHEN {v} THEN num_modify_tuples ELSE 0 END) AS {f}" for f, v in ops] +
                    [f"SUM(CASE optype WHEN {v} THEN 1 ELSE 0 END) AS {f}" for f, v in qs]),
                index="",
                tbl=tbl,
                aux_group="")
            tbl_ks[tbl] = pd.DataFrame(execute_sql(sql), columns=["window_index"] + agg_stats + [f for f, _ in ops] + [f for f, _ in qs])

            idx_frames = []
            tbl_index = [tbl] + tbl_index_map[tbl]
            for idx in tbl_index:
                index = sql_format.format(
                    work_prefix=work_prefix,
                    query_orders=",".join(query_orders),
                    agg_stats=",".join(
                        ["target", "MIN(statement_timestamp) as statement_timestamp"] +
                        [f"SUM({k})" for k in agg_index_stats] +
                        [f"SUM(CASE optype WHEN {v} THEN num_modify_tuples ELSE 0 END) AS {f}" for f, v in ops] +
                        [f"SUM(CASE optype WHEN {v} THEN 1 ELSE 0 END) AS {f}" for f, v in qs]),
                    index="_index",
                    tbl=idx,
                    aux_group=", target")

                record = execute_sql(index)
                if len(record) > 0:
                    idx_frames.append(pd.DataFrame(record, columns=["window_index", "target", "start_timestamp"] + agg_index_stats + [f for f, _ in ops] + [f for f, _ in qs]))
            if len(idx_frames) > 0:
                tbl_ks[tbl + "_index"] = pd.concat(idx_frames, ignore_index=True)

    return tbl_ks
