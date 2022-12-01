import os
import glob
import psycopg
from tqdm import tqdm
import pandas as pd
from plumbum import cli
from pathlib import Path
import numpy as np
import logging

from behavior import BENCHDB_TO_TABLES
from behavior.model_workload.utils.data_cardest import load_initial_data, compute_data_change_frames
from behavior.model_workload.utils.workload_analysis import workload_analysis

from behavior.model_workload.utils import keyspace_metadata_read, keyspace_metadata_output
from behavior.model_workload.utils.exec_feature import build_table_exec
from behavior.model_workload.utils.data_cardest import compute_underspecified
from behavior.model_workload.utils.eval_analysis import load_eval_windows
from behavior.utils.process_raw_stats import crunch_raw_data, diff_data
from behavior.utils.process_raw_plans import read_all_plans


logger = logging.getLogger("workload_analyze")


QSS_STATS_COLUMNS = [
    ("query_id", "bigint"),
    ("generation", "integer"),
    ("db_id", "integer"),
    ("pid", "integer"),
    ("statement_timestamp", "bigint"),
    ("unix_timestamp", "float8"),
    ("plan_node_id", "int"),
    ("left_child_node_id", "int"),
    ("right_child_node_id", "int"),
    ("optype", "int"),
    ("elapsed_us", "float8"),
    ("counter0", "float8"),
    ("counter1", "float8"),
    ("counter2", "float8"),
    ("counter3", "float8"),
    ("counter4", "float8"),
    ("counter5", "float8"),
    ("counter6", "float8"),
    ("counter7", "float8"),
    ("counter8", "float8"),
    ("counter9", "float8"),
    ("blk_hit", "integer"),
    ("blk_miss", "integer"),
    ("blk_dirty", "integer"),
    ("blk_write", "integer"),
    ("startup_cost", "float8"),
    ("total_cost", "float8"),
    ("payload", "bigint"),
    ("txn", "bigint"),
    ("comment", "text"),
    ("query_text", "text"),
    ("num_rel_refs", "float8"),
    ("target", "text"),
    ("target_idx_scan_table", "text"),
    ("target_idx_scan", "text"),
    ("target_idx_insert", "text"),
]


def load_raw_data(connection, workload_only, work_prefix, input_dir, plans_df, indexoid_table_map, indexoid_name_map):
    # Crunch and instantiate the raw {work_prefix}_mw_raw table.
    crunch_raw_data(logger, connection, input_dir, work_prefix, workload_only, False, plans_df, QSS_STATS_COLUMNS, indexoid_table_map, indexoid_name_map)

    if workload_only:
        connection.execute(f"ALTER TABLE {work_prefix}_mw_raw RENAME TO {work_prefix}_mw_diff")
        connection.execute(f"VACUUM ANALYZE {work_prefix}_mw_diff")
    else:
        logger.info("Creating the diff database table now.")
        diff_data(logger, connection, work_prefix, "mw_diff", "raw", QSS_STATS_COLUMNS, indexes=True)

        logger.info("Now executing a vacuum analyze on diff table.")
        connection.execute(f"VACUUM ANALYZE {work_prefix}_mw_diff")
        connection.execute(f"DROP TABLE {work_prefix}_raw")


def load_workload(connection, work_prefix, input_dir, pg_qss_plans, workload_only, wa):
    table_attr_map = wa.table_attr_map
    table_keyspace_map = wa.table_keyspace_map
    indexoid_table_map = wa.indexoid_table_map
    indexoid_name_map = wa.indexoid_name_map
    query_template_map = wa.query_template_map

    # Load all the raw data into the database.
    logger.info("Loading the raw data.")
    load_raw_data(connection, workload_only, work_prefix, input_dir, pg_qss_plans, indexoid_table_map, indexoid_name_map)

    logger.info("Loading queries with query order.")
    with connection.transaction():
        query = f"CREATE UNLOGGED TABLE {work_prefix}_mw_queries WITH (autovacuum_enabled = OFF) AS "
        query += f"select *, dense_rank() over (order by statement_timestamp, pid) query_order from {work_prefix}_mw_diff order by query_order;"
        connection.execute(query)
        connection.execute(f"CREATE INDEX {work_prefix}_mw_queries_0 ON {work_prefix}_mw_queries (query_order, plan_node_id) INCLUDE (comment, target_idx_scan, target_idx_scan_table)")
        connection.execute(f"DROP TABLE {work_prefix}_mw_diff")
    connection.execute(f"VACUUM ANALYZE {work_prefix}_mw_queries")

    max_arg = 0
    for q, v in query_template_map.items():
        for _, (_, a) in v.items():
            if a.startswith("arg") and int(a[3:]) > max_arg:
                max_arg = int(a[3:])

    logger.info("Creating materialized view of the deconstructed arguments.")
    with connection.transaction():
        # Construct the temporary arguments table in parallel.
        assert max_arg > 0
        query = """
            CREATE UNLOGGED TABLE {work_prefix}_mw_queries_args_temp AS SELECT q.*, {trims}
            FROM {work_prefix}_mw_queries q, LATERAL (
                SELECT array_agg(i.arg) as args FROM (
                    SELECT (regexp_matches(comment, '(\$\w+) = (\''(?:[^\'']*(?:\''\'')?[^\'']*)*\'')', 'g'))[2] as arg
                ) i
            ) n
            WHERE q.plan_node_id = -1 ORDER BY q.query_order, q.plan_node_id
        """.format(work_prefix=work_prefix, trims=",".join([f"TRIM(n.args[{i+1}], '''') as arg{i+1}" for i in range(max_arg)]))

        logger.info("Executing SQL: %s", query)
        connection.execute(query)

        # Make the real table partitioned from the temporary table.
        query = f"CREATE UNLOGGED TABLE {work_prefix}_mw_queries_args (LIKE {work_prefix}_mw_queries_args_temp) PARTITION BY LIST (target)"
        connection.execute(query)

        for target, _ in pg_qss_plans.groupby(by=["target"]):
            # Attempt to normalize the string out.
            norm_target = target.replace(",", "_")
            connection.execute(f"CREATE UNLOGGED TABLE {work_prefix}_mw_queries_args_{norm_target} PARTITION OF {work_prefix}_mw_queries_args FOR VALUES IN ('{target}') WITH (autovacuum_enabled = OFF)")
        connection.execute(f"CREATE UNLOGGED TABLE {work_prefix}_mw_queries_args_default PARTITION OF {work_prefix}_mw_queries_args DEFAULT WITH (autovacuum_enabled = OFF)")
        connection.execute(f"INSERT INTO {work_prefix}_mw_queries_args SELECT * FROM {work_prefix}_mw_queries_args_temp")
        connection.execute(f"DROP TABLE {work_prefix}_mw_queries_args_temp")

        query = f"CREATE INDEX {work_prefix}_mw_queries_args_0 ON {work_prefix}_mw_queries_args (query_order, plan_node_id) INCLUDE ("
        query += ",".join([f"arg{i+1}" for i in range(max_arg)]) + ")"
        connection.execute(query)

    logger.info("Finished loading queries in query order.")


def analyze_workload(benchmark, input_dir, workload_only, psycopg2_conn, work_prefix,
                     load_raw, load_data, load_deltas, load_hits, load_exec_stats, load_windows):
    assert psycopg2_conn is not None
    tables = BENCHDB_TO_TABLES[benchmark]

    with psycopg.connect(psycopg2_conn, autocommit=True, prepare_threshold=None) as connection:
        if load_raw:
            wa, pg_qss_plans = workload_analysis(connection, input_dir, workload_only, tables)
            keyspace_metadata_output(input_dir, wa)

            # Analyze and populate the workload.
            load_workload(connection, work_prefix, input_dir, pg_qss_plans, workload_only, wa)
        else:
            wa = keyspace_metadata_read(input_dir)[0]

        if load_data:
            logger.info("Loading the initial data to be manipulated.")
            load_initial_data(logger, connection, workload_only, work_prefix, input_dir, wa.table_attr_map, wa.table_keyspace_map)

        if load_deltas:
            logger.info("Computing data change frames.")
            compute_data_change_frames(logger, connection, work_prefix, wa)

        if load_hits:
            # We need the raw plans to load the hits correctly.
            plans = read_all_plans(input_dir)
            logger.info("Computing data access frames.")
            compute_underspecified(logger, connection, work_prefix, wa, plans)

        if load_exec_stats:
            logger.info("Computing statistics features")
            build_table_exec(logger, connection, work_prefix, list(set(wa.query_table_map.values())))

        if load_windows:
            logger.info("Loading windows for workload analysis.")
            max_arg = 0
            for q, v in wa.query_template_map.items():
                for _, (_, a) in v.items():
                    if a.startswith("arg") and int(a[3:]) > max_arg:
                        max_arg = int(a[3:])
            tbls = [t for t in wa.table_attr_map if len(wa.table_attr_map[t]) > 0]
            load_eval_windows(logger, connection, work_prefix, max_arg, tbls, list(wa.query_table_map.values()))


class AnalyzeWorkloadCLI(cli.Application):
    benchmark = cli.SwitchAttr("--benchmark", str, mandatory=True, help="Benchmark that should be analyzed.",)
    dir_workload_input = cli.SwitchAttr("--dir-workload-input", str, mandatory=True, help="Path to the folder containing the workload input.",)
    workload_only = cli.SwitchAttr("--workload-only", str, help="Whether the input contains only the workload stream.",)
    psycopg2_conn = cli.SwitchAttr("--psycopg2-conn", str, mandatory=True, help="Psycopg2 connection that should be used.",)
    work_prefix = cli.SwitchAttr("--work-prefix", str, mandatory=True, help="Prefix to use for working with the database.",)
    load_raw = cli.Flag("--load-raw", default=False, help="Whether to load the raw data or not.",)
    load_data = cli.Flag("--load-initial-data", default=False, help="Whether to load the initial data or not.",)
    load_deltas = cli.Flag("--load-deltas", default=False, help="Whether to load the deltas or not.",)
    load_hits = cli.Flag("--load-hits", default=False, help="Whether to load hits or not.",)
    load_exec_stats = cli.Flag("--load-exec-stats", default=False, help="Whether to load exec stats or not.",)
    load_windows = cli.Flag("--load-windows", default=False, help="Whether to load windows or not.",)

    def main(self):
        b_parts = self.benchmark.split(",")
        input_parts = self.dir_workload_input.split(",")
        for i in range(len(input_parts)):
            logger.info("Processing %s (%s, %s)", input_parts[i], b_parts[i], self.workload_only)

            file_handler = logging.FileHandler(Path(input_parts[i]) / "output.log", mode="a")
            file_handler.propagate = False
            logger.addHandler(file_handler)

            analyze_workload(b_parts[i],
                             Path(input_parts[i]),
                             (self.workload_only == "True"),
                             self.psycopg2_conn,
                             self.work_prefix,
                             self.load_raw,
                             self.load_data,
                             self.load_deltas,
                             self.load_hits,
                             self.load_exec_stats,
                             self.load_windows)

            logger.removeHandler(file_handler)


if __name__ == "__main__":
    AnalyzeWorkloadCLI.run()
