from tqdm import tqdm
import glob
import os
import multiprocessing as mp
import pandas as pd
import numpy as np
from pathlib import Path
from behavior.utils.process_pg_state_csvs import postgres_julian_to_unix


# Function to process a chunk consistently and extract metadata.
def process_chunk(work_prefix, workload_only, extract_ous, chunk, plans_df, usecols, indexoid_table_map, indexoid_name_map):
    pid = mp.current_process().pid
    output = Path(f"/tmp/{work_prefix}_stats{pid}.csv")
    write_header = not output.exists()

    chunk = chunk[(chunk.query_id != 0) & (chunk.statement_timestamp != 0)]
    mask = chunk.query_id.isin(plans_df.query_id)
    chunk = chunk[mask]
    if chunk.shape[0] == 0:
        return

    if workload_only:
        # If we're workload only, ensure we only load the root nodes.
        chunk = chunk[chunk.plan_node_id == -1]

    noplans = None
    if extract_ous:
        # Don't care about the -1 plan nodes.
        chunk = chunk[chunk.plan_node_id != -1]

        # For OUs, separate the no plan and plans.
        noplans = chunk[chunk.plan_node_id < 0].copy()
        noplans["left_child_node_id"] = -1;
        noplans["right_child_node_id"] = -1
        chunk = chunk[chunk.plan_node_id >= 0]

    if chunk["query_id"].dtype != "int64":
        # Convert to int64 if needed.
        chunk = chunk.copy()
        chunk["query_id"] = chunk.query_id.astype(np.int64)

    chunk.set_index(keys=["statement_timestamp"], drop=True, append=False, inplace=True)
    chunk.sort_index(axis=0, inplace=True)
    initial = chunk.shape[0]

    chunk = pd.merge_asof(chunk, plans_df, left_index=True, right_index=True, by=["query_id", "generation", "db_id", "pid", "plan_node_id"], allow_exact_matches=True)
    chunk.reset_index(drop=False, inplace=True)
    if extract_ous:
        # Assert that we've fully matched.
        chunk = chunk[~chunk.total_cost.isna()]

    # We don't actually want to drop. Just set their child plans to -1.
    chunk.fillna(value={"left_child_node_id": -1, "right_child_node_id": -1}, inplace=True)
    # Ensure that we don't have magical explosion.
    assert chunk.shape[0] <= initial

    if extract_ous:
        # Extract OUs.
        chunk = pd.concat([chunk, noplans], ignore_index=True)
    else:
        # Get the target / target index insert.
        chunk["target_idx_insert"] = None
        mask = chunk.comment == "ModifyTableIndexInsert"
        chunk.loc[mask, "target"] = chunk[mask].payload.apply(lambda x: indexoid_table_map[x] if x in indexoid_table_map else None)
        chunk.loc[mask, "target_idx_insert"] = chunk[mask].payload.apply(lambda x: indexoid_name_map[x] if x in indexoid_name_map else None)
    chunk["unix_timestamp"] = postgres_julian_to_unix(chunk.statement_timestamp)

    chunk = chunk[[t[0] for t in usecols]]
    if chunk.shape[0] == 0:
        return

    # Switch some datatypes.
    chunk["left_child_node_id"] = chunk.left_child_node_id.astype(int)
    chunk["right_child_node_id"] = chunk.right_child_node_id.astype(int)
    if "optype" in chunk:
        chunk["optype"] = chunk.optype.astype('Int32')
    chunk.to_csv(output, header=write_header, index=False, mode="a" if not write_header else "w")


# usecols is [("name", "type")....]
def crunch_raw_data(logger, connection, input_dir, work_prefix, workload_only, extract_ous, plans_df, usecols, indexoid_table_map=None, indexoid_name_map=None):
    # First delete all the old data.
    stats = glob.glob(f"/tmp/{work_prefix}_stats*.csv")
    [os.remove(s) for s in stats]

    num_processes = 4
    logger.info("Crunching through all the raw data now.")
    with mp.Pool(num_processes) as pool:
        stats_files = [f for f in glob.glob(f"{input_dir}/stats.*/pg_qss_stats_*.csv")]
        results = []
        for stats_file in tqdm(stats_files, leave=False):
            for chunk in tqdm(pd.read_csv(stats_file, chunksize=8192*1000), leave=False):
                results.append(pool.apply_async(process_chunk, [work_prefix, workload_only, extract_ous, chunk, plans_df, usecols, indexoid_table_map, indexoid_name_map]))

        for res in tqdm(results, leave=False):
            res.get()

        pool.close()
        pool.join()

    stats = glob.glob(f"/tmp/{work_prefix}_stats*.csv")
    with connection.transaction():
        create_raw = """CREATE UNLOGGED TABLE {work_prefix}_raw ({cols});""".format(work_prefix=work_prefix, cols=",".join([f"{n} {t}" for n, t in usecols]))
        connection.execute(create_raw)
        for stat in stats:
            # Load each stat file.
            logger.info(f"Loading {stat} into the database now.")
            connection.execute(f"COPY {work_prefix}_raw FROM '{stat}' WITH (FORMAT csv, HEADER true)")

        # Create the indexes that might be useful.
        connection.execute(f"CREATE INDEX {work_prefix}_raw_0 ON {work_prefix}_raw (query_id, db_id, pid, statement_timestamp)")
        connection.execute(f"CREATE UNIQUE INDEX {work_prefix}_raw_1 ON {work_prefix}_raw (query_id, db_id, pid, statement_timestamp, plan_node_id) WHERE plan_node_id >= 0")
        connection.execute(f"CREATE INDEX {work_prefix}_raw_2 ON {work_prefix}_raw (query_id, db_id, pid, statement_timestamp, plan_node_id)")

    connection.execute(f"VACUUM ANALYZE {work_prefix}_raw")

    # Remove all the stats files.
    [os.remove(s) for s in stats]


def _construct_diff_sql(work_prefix, raw, all_columns):
    DIFFERENCE_COLUMNS = [
        "elapsed_us",
        "blk_hit",
        "blk_miss",
        "blk_dirty",
        "blk_write",
        "startup_cost",
        "total_cost",
    ]

    # Columns we are selecting.
    sel_columns = []
    for k, _ in all_columns:
        if k not in DIFFERENCE_COLUMNS:
            sel_columns.append(f"r1.{k}")
        else:
            sel_columns.append(f"GREATEST(r1.{k} - COALESCE(r2.{k}, 0) - COALESCE(r3.{k}, 0), 0) as {k}")

    return """
        SELECT {sel_columns} FROM {work_prefix}_{raw} r1
        LEFT JOIN LATERAL (
            SELECT * FROM {work_prefix}_{raw} r2
                    WHERE r1.query_id = r2.query_id
                      AND r1.db_id = r2.db_id
                      AND r1.statement_timestamp = r2.statement_timestamp
                      AND r1.pid = r2.pid
                      AND r1.left_child_node_id = r2.plan_node_id
                      AND r2.plan_node_id >= 0
        ) r2 ON r1.query_id = r2.query_id
         AND r1.db_id = r2.db_id
         AND r1.statement_timestamp = r2.statement_timestamp
         AND r1.pid = r2.pid
         AND r1.left_child_node_id = r2.plan_node_id
         AND r1.plan_node_id >= 0

         LEFT JOIN LATERAL (
            SELECT * FROM {work_prefix}_{raw} r3
                    WHERE r1.query_id = r3.query_id
                      AND r1.db_id = r3.db_id
                      AND r1.statement_timestamp = r3.statement_timestamp
                      AND r1.pid = r3.pid
                      AND r1.right_child_node_id = r3.plan_node_id
                      AND r3.plan_node_id >= 0
        ) r3 ON r1.query_id = r3.query_id
         AND r1.db_id = r3.db_id
         AND r1.statement_timestamp = r3.statement_timestamp
         AND r1.pid = r3.pid
         AND r1.right_child_node_id = r3.plan_node_id
         AND r1.plan_node_id >= 0
    WHERE r1.plan_node_id >= 0;
    """.format(sel_columns=",".join(sel_columns), work_prefix=work_prefix, raw=raw)


# Perform differencing on the raw data.
def diff_data(logger, connection, work_prefix, diff, raw, all_columns, indexes=False, partitions=None):
    with connection.transaction():
        queries = [
            ("Extension Query", "CREATE EXTENSION IF NOT EXISTS pg_prewarm"),
            ("Prewarm Query", f"SELECT * FROM pg_prewarm('{work_prefix}_{raw}_0')"),
        ]

        if partitions is not None:
            diff_sql = f"CREATE UNLOGGED TABLE {work_prefix}_{diff} (" + ",".join([f"{k} {v}" for k, v in all_columns]) + ") PARTITION BY LIST (comment)"
            connection.execute(diff_sql)
            for partition in partitions:
                connection.execute(f"CREATE UNLOGGED TABLE {work_prefix}_{diff}_{partition} PARTITION OF {work_prefix}_{diff} FOR VALUES IN ('{partition}') WITH (autovacuum_enabled = OFF)")
            connection.execute(f"CREATE UNLOGGED TABLE {work_prefix}_{diff}_default PARTITION OF {work_prefix}_{diff} DEFAULT WITH (autovacuum_enabled = OFF)")

            diff_sql = f"INSERT INTO {work_prefix}_{diff} " + _construct_diff_sql(work_prefix, raw, all_columns)
            queries.append(("Difference Query", diff_sql))
        else:
            diff_sql = f"CREATE UNLOGGED TABLE {work_prefix}_{diff} WITH (autovacuum_enabled = OFF) AS "
            diff_sql += _construct_diff_sql(work_prefix, raw, all_columns)
            queries.append(("Difference Query", diff_sql))

        diff_sql = """INSERT INTO {work_prefix}_{diff} SELECT {cols} FROM {work_prefix}_{raw} WHERE plan_node_id < 0""".format(work_prefix=work_prefix, diff=diff, raw=raw, cols=",".join([t[0] for t in all_columns]))
        queries.append(("Insert Query", diff_sql))
        for qname, query in queries:
            logger.info("Executing %s", qname)
            connection.execute(query)

        if indexes:
            connection.execute(f"CREATE INDEX {work_prefix}_mw_diff_0 ON {work_prefix}_mw_diff (query_id, db_id, pid, statement_timestamp)")
            connection.execute(f"CREATE INDEX {work_prefix}_mw_diff_1 ON {work_prefix}_mw_diff (query_id, db_id, pid, statement_timestamp, plan_node_id)")
