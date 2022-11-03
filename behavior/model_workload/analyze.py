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
from behavior.utils.process_pg_state_csvs import postgres_julian_to_unix
from behavior.model_workload.utils.keyspace_feature import build_table_exec
from behavior.model_workload.utils.data_cardest import compute_underspecified
from behavior.model_ous.extract_ous import construct_diff_sql


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

DIFFERENCE_COLUMNS = [
    "elapsed_us",
    "blk_hit",
    "blk_miss",
    "blk_dirty",
    "blk_write",
    "startup_cost",
    "total_cost",
]

def load_raw_data(connection, workload_only, work_prefix, input_dir, plans_df, indexoid_table_map, indexoid_name_map):
    with open(f"/tmp/{work_prefix}_stats.csv", "w") as f:
        write_header = True
        for chunk in tqdm(pd.read_csv(input_dir / "pg_qss_stats.csv", chunksize=8192*1000)):
            mask = chunk.query_id.isin(plans_df.query_id)
            chunk = chunk[mask]
            if workload_only:
                # If we're workload only, ensure we only load the root nodes.
                chunk = chunk[chunk.plan_node_id == -1]

            chunk.set_index(keys=["statement_timestamp"], drop=True, append=False, inplace=True)
            chunk.sort_index(axis=0, inplace=True)
            initial = chunk.shape[0]

            chunk = pd.merge_asof(chunk, plans_df, left_index=True, right_index=True, by=["query_id", "generation", "db_id", "pid", "plan_node_id"], allow_exact_matches=True)
            chunk.reset_index(drop=False, inplace=True)
            # We don't actually want to drop. Just set their child plans to -1.
            # chunk.drop(chunk[chunk.total_cost.isna()].index, inplace=True)
            chunk.fillna(value={"left_child_node_id": -1, "right_child_node_id": -1}, inplace=True)
            # Ensure that we don't have magical explosion.
            assert chunk.shape[0] <= initial

            chunk["target_idx_insert"] = None
            mask = chunk.comment == "ModifyTableIndexInsert"
            chunk.loc[mask, "target"] = chunk[mask].payload.apply(lambda x: indexoid_table_map[x] if x in indexoid_table_map else None)
            chunk.loc[mask, "target_idx_insert"] = chunk[mask].payload.apply(lambda x: indexoid_name_map[x] if x in indexoid_name_map else None)
            chunk["unix_timestamp"] = postgres_julian_to_unix(chunk.statement_timestamp)

            chunk = chunk[[t[0] for t in QSS_STATS_COLUMNS]]
            if chunk.shape[0] == 0:
                continue

            chunk["left_child_node_id"] = chunk.left_child_node_id.astype(int)
            chunk["right_child_node_id"] = chunk.right_child_node_id.astype(int)
            chunk["optype"] = chunk.optype.astype('Int32')
            chunk.to_csv(f, header=write_header, index=False)
            write_header = False

    logger.info("Creating the raw database table now.")
    with connection.transaction():
        create_stats_sql = f"CREATE UNLOGGED TABLE {work_prefix}_mw_raw (" + ",".join([f"{tup[0]} {tup[1]}" for tup in QSS_STATS_COLUMNS]) + ")"
        connection.execute(create_stats_sql)
        connection.execute(f"COPY {work_prefix}_mw_raw FROM '/tmp/{work_prefix}_stats.csv' WITH (FORMAT csv, HEADER true, NULL '')")

    if workload_only:
        connection.execute(f"ALTER TABLE {work_prefix}_mw_raw RENAME TO {work_prefix}_mw_diff")
        connection.execute(f"VACUUM ANALYZE {work_prefix}_mw_diff")
    else:
        logger.info("Creating the diff database table now.")
        with connection.transaction():
            diff_sql = f"CREATE UNLOGGED TABLE {work_prefix}_mw_diff WITH (autovacuum_enabled = OFF) AS "
            diff_sql += construct_diff_sql(work_prefix, "mw_raw", QSS_STATS_COLUMNS, DIFFERENCE_COLUMNS, constrain_child=True)
            connection.execute(diff_sql)
            connection.execute(f"CREATE INDEX {work_prefix}_mw_diff_0 ON {work_prefix}_mw_diff (query_id, db_id, pid, statement_timestamp)")
            connection.execute(f"CREATE INDEX {work_prefix}_mw_diff_1 ON {work_prefix}_mw_diff (query_id, db_id, pid, statement_timestamp, plan_node_id)")

        logger.info("Now executing a vacuum analyze on diff table.")
        connection.execute(f"VACUUM ANALYZE {work_prefix}_mw_diff")
        connection.execute(f"DROP TABLE {work_prefix}_mw_raw")


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
        connection.execute(f"CREATE INDEX {work_prefix}_mw_queries_0 ON {work_prefix}_mw_queries (query_order, plan_node_id)")
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
        query = f"CREATE UNLOGGED TABLE {work_prefix}_mw_queries_args_temp AS SELECT *, "
        query += ",".join([f"TRIM(TRIM(NULLIF(split_part(split_part(comment, ',', {i+1}), '=', 2), '')), '''') as arg{i+1}" for i in range(max_arg)])
        query += f" FROM {work_prefix}_mw_queries WHERE plan_node_id = -1 ORDER BY query_order, plan_node_id"
        logger.info("Executing SQL: %s", query)
        connection.execute(query)

        # Make the real table partitioned from the temporary table.
        query = f"CREATE UNLOGGED TABLE {work_prefix}_mw_queries_args (LIKE {work_prefix}_mw_queries_args_temp) PARTITION BY LIST (target)"
        connection.execute(query)

        for target, _ in pg_qss_plans.groupby(by=["target"]):
            # Attempt to normalize the string out.
            target = target.replace(",", "_")
            connection.execute(f"CREATE UNLOGGED TABLE {work_prefix}_mw_queries_args_{target} PARTITION OF {work_prefix}_mw_queries_args FOR VALUES IN ('{target}') WITH (autovacuum_enabled = OFF)")
        connection.execute(f"CREATE UNLOGGED TABLE {work_prefix}_mw_queries_args_default PARTITION OF {work_prefix}_mw_queries_args DEFAULT WITH (autovacuum_enabled = OFF)")
        connection.execute(f"INSERT INTO {work_prefix}_mw_queries_args SELECT * FROM {work_prefix}_mw_queries_args_temp")
        connection.execute(f"DROP TABLE {work_prefix}_mw_queries_args_temp")

        query = f"CREATE INDEX {work_prefix}_mw_queries_args_0 ON {work_prefix}_mw_queries_args (query_order, plan_node_id) INCLUDE ("
        query += ",".join([f"arg{i+1}" for i in range(max_arg)]) + ")"
        connection.execute(query)

        query = f"CREATE INDEX {work_prefix}_mw_queries_args_1 ON {work_prefix}_mw_queries_args (query_order, unix_timestamp)"
        connection.execute(query)

    logger.info("Finished loading queries in query order.")


def analyze_workload(benchmark, input_dir, workload_only, psycopg2_conn, work_prefix,
                     load_raw, load_data, load_deltas, load_hits, load_exec_stats):
    assert psycopg2_conn is not None
    tables = BENCHDB_TO_TABLES[benchmark]

    with psycopg.connect(psycopg2_conn, autocommit=True, prepare_threshold=None) as connection:
        if load_raw:
            wa, pg_qss_plans = workload_analysis(connection, input_dir, tables)
            keyspace_metadata_output(input_dir, wa)

            # Analyze and populate the workload.
            load_workload(connection, work_prefix, input_dir, pg_qss_plans, workload_only, wa)
        else:
            wa = keyspace_metadata_read(input_dir)[0]

        if load_data:
            logger.info("Loading the initial data to be manipulated.")
            load_initial_data(logger, connection, work_prefix, input_dir, wa.table_attr_map, wa.table_keyspace_map)

        if load_deltas:
            logger.info("Computing data change frames.")
            compute_data_change_frames(logger, connection, work_prefix, wa)

        if load_hits:
            # We need the raw plans to load the hits correctly.
            plans = pd.read_csv(f"{input_dir}/pg_qss_plans.csv")
            logger.info("Computing data access frames.")
            compute_underspecified(logger, connection, work_prefix, wa, plans)

        if load_exec_stats:
            logger.info("Computing statistics features")
            build_table_exec(logger, connection, work_prefix, list(set(wa.query_table_map.values())))


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

    def main(self):
        b_parts = self.benchmark.split(",")
        input_parts = self.dir_workload_input.split(",")
        for i in range(len(input_parts)):
            logger.info("Processing %s (%s, %s)", input_parts[i], b_parts[i], self.workload_only)
            analyze_workload(b_parts[i],
                             Path(input_parts[i]),
                             (self.workload_only == "True"),
                             self.psycopg2_conn,
                             self.work_prefix,
                             self.load_raw,
                             self.load_data,
                             self.load_deltas,
                             self.load_hits,
                             self.load_exec_stats)


if __name__ == "__main__":
    AnalyzeWorkloadCLI.run()
