import gc
import pickle
import glob
from tqdm import tqdm
import psycopg
from psycopg import Rollback
import pandas as pd
from pandas.api import types as pd_types
from pathlib import Path
import shutil
import numpy as np
import logging
from datetime import datetime
from sqlalchemy import create_engine
from plumbum import cli
from behavior import BENCHDB_TO_TABLES
from behavior.model_workload.utils import keyspace_metadata_read
from behavior.model_workload.utils.keyspace_feature import construct_keyspaces
from behavior.model_workload.utils.exec_feature import construct_query_window_stats
from behavior.utils.process_pg_state_csvs import process_time_pg_class

logger = logging.getLogger("exec_feature_synthesis")

##################################################################################
# Window generation and saving.
##################################################################################

def save_bucket_keys_to_output(output_dir):
    def save_df_return_none(tbl, df):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df["key_dist"] = [",".join(map(str, l)) for l in df.key_dist]
        df.to_feather(f"{output_dir}/{tbl}.feather")
        return None

    return save_df_return_none


def fetch_window_indexes(input_dir, slice_window_target, generate_vacuum_flag):
    vacuum = "flag" if generate_vacuum_flag else "dropwindow"
    subdir = f"windows/windows_{slice_window_target}_{vacuum}"
    assert (Path(input_dir) / subdir).exists()

    window_index_map = {}
    for t in glob.glob(f"{input_dir}/{subdir}/*.feather"):
        tbl = pd.read_feather(t)
        window_index_map[Path(t).stem.split(".")[0]] = tbl
    return window_index_map


# Tables describes the base tables of the workload.
# all_targets captures all "targets" (this holds the multi-join targets..).
def build_window_indexes(connection, work_prefix, input_dir, wa, buckets, slice_window, generate_vacuum_flag):
    tables = [t for t in wa.table_attr_map.keys()]
    all_targets = list(set(wa.query_table_map.values()))

    # This is when the autovacuum executed.
    pg_stat_user_tables = pd.read_csv(f"{input_dir}/pg_stat_user_tables.csv")
    pg_stat_user_tables = pg_stat_user_tables[~pg_stat_user_tables.last_autovacuum.isnull()]
    pg_stat_user_tables["autovacuum_unix_timestamp"] = pd.to_datetime(pg_stat_user_tables.last_autovacuum).map(pd.Timestamp.timestamp)

    with connection.transaction() as tx:
        # Get the query order ranges.
        with connection.transaction():
            r = [r for r in connection.execute(f"SELECT min(query_order), max(query_order) FROM {work_prefix}_mw_queries_args")][0]
            min_qo, max_qo = r[0], r[1]

        connection.execute(f"CREATE INDEX {work_prefix}_mw_queries_args_time ON {work_prefix}_mw_queries_args (query_order, unix_timestamp)")
        slices = slice_window.split(",")
        for window_slice in slices:
            # Check that we haven't processed it before.
            vacuum = "flag" if generate_vacuum_flag else "dropwindow"
            keys_subdir = f"keys/keys_{window_slice}_{vacuum}"
            windows_subdir = f"windows/windows_{window_slice}_{vacuum}"
            if (Path(input_dir) / windows_subdir / "done").exists():
                continue

            window_index_map = {}
            if window_slice == "query":
                for tbl in tables:
                    logger.info("Computing window index map for table: %s", tbl)
                    # Get all the sample points from the CSV file.
                    sample = pd.read_csv(f"{input_dir}/{tbl}.csv")
                    sample["true_window_index"] = sample.index
                    sample["time"] = (sample.time / float(1e6))

                    # Get a frame representing the autovacuum times.
                    autovacuum_times = []
                    substat = pg_stat_user_tables[pg_stat_user_tables.relname == tbl]
                    if substat.shape[0] > 0:
                        autovacuum_times = [v for v in substat.autovacuum_unix_timestamp.value_counts().index]
                        wipe_frame = pd.DataFrame([{"time": autovac, "true_window_index": -1} for autovac in autovacuum_times])

                    if not generate_vacuum_flag and wipe_frame.shape[0] > 0:
                        # Inject a window_index that is -1. window_index = -1 are tuples that get discarded! woo-hoo.
                        sample = pd.concat([sample, wipe_frame], ignore_index=True)
                        sample.sort_values(by=["time"], ignore_index=True, inplace=True)

                    sample["window_index"] = sample.index

                    # Find the possible space of targets that we have to consult.
                    consider_tbls = [t for t in all_targets if tbl in t]
                    clause = "(" + " or ".join([f"target = '{t}'" for t in consider_tbls]) + ")"

                    # This SQL is awkward. But the insight here is that instead of [t1, t2, t3] as providing the bounds, we want to use query_order.
                    # Since width_bucket() uses the property that if x = t1, it returns [t1, t2] bucket. So in principle, we want to find the first
                    # query *AFTER* t1 so it'll still act as the correct exclusive bound.
                    sql = "UNNEST(ARRAY[" + ",".join([str(i) for i in sample.window_index.values.tolist()]) + "], "
                    sql += "ARRAY[" + ",".join([str(i) for i in sample.time.values.tolist()]) + "]) as x(window_index, time)"
                    sql = f"SELECT window_index, time, b.query_order FROM {sql}, "
                    sql += f"LATERAL (SELECT MIN(query_order) as query_order FROM {work_prefix}_mw_queries_args WHERE unix_timestamp > time AND {clause}) b ORDER BY window_index"
                    c = connection.execute(sql)
                    tups = [(tup[0], tup[2]) for tup in c]

                    sample.set_index(keys=["window_index"], inplace=True)
                    sample.loc[[r[0] for r in tups], "query_order"] = [r[1] for r in tups]
                    sample.drop(sample[sample.query_order.isna()].index, inplace=True)
                    # Convert query_order back into an integral type.
                    sample["query_order"] = sample.query_order.astype(int)
                    # We now have a window index stream.
                    sample.reset_index(drop=False, inplace=True)
                    assert sample.query_order.is_monotonic_increasing

                    if generate_vacuum_flag:
                        # We now build the indicators for whether VACUUM ran during the interval or not.
                        sample["vacuum"] = 0
                        if substat.shape[0] > 0:
                            assert wipe_frame.shape[0] > 0
                            assert sample.time.is_monotonic_increasing
                            slots = sample.time.searchsorted(wipe_frame.time.values, side="right")
                            sample.loc[slots, "vacuum"] = 1

                    window_index_map[tbl] = sample
            else:
                # In this case, just fragment everything.
                window_index_map = {t: pd.DataFrame(range(min_qo, max_qo, int(window_slice)), columns=["query_order"]) for t in tables}

            # Get all the "keyspace" descriptor features.
            (Path(input_dir) / keys_subdir).mkdir(parents=True, exist_ok=True)
            callback = save_bucket_keys_to_output(Path(input_dir) / keys_subdir)
            tbls = [t for t in wa.table_attr_map.keys() if not (Path(input_dir) / keys_subdir / f"{t}.feather").exists()]
            construct_keyspaces(logger, connection, work_prefix, tbls, wa.table_attr_map, window_index_map, buckets, callback_fn=callback)

            # Save the windows.
            (Path(input_dir) / windows_subdir).mkdir(parents=True, exist_ok=True)
            for t, v in window_index_map.items():
                v.to_feather(f"{input_dir}/{windows_subdir}/{t}.feather")

            open(f"{input_dir}/{windows_subdir}/done", "w").close()

        # Let's rollback the index.
        raise Rollback(tx)

    logger.info("Finished computing window index map: %s", datetime.now())
    return window_index_map

##################################################################################
# Generate the execution page features.
##################################################################################

def __gen_exec_features(input_dir, connection, work_prefix, wa, buckets, generate_vacuum_flag):
    vacuum = "flag" if generate_vacuum_flag else "dropwindow"
    if (Path(input_dir) / "exec_features_{vacuum}/done").exists():
        return

    # Compute the window frames.
    logger.info("Computing window frames.")
    logger.info("Starting at %s", datetime.now())
    window_index_map = fetch_window_indexes(input_dir, "query", generate_vacuum_flag)

    # Get all the data space features.
    if not (Path(input_dir) / f"exec_features_{vacuum}/done").exists():
        (Path(input_dir) / f"exec_features_{vacuum}/").mkdir(parents=True, exist_ok=True)
        table_indexname_map = {t: [wa.indexoid_name_map[idxoid] for idxoid, idxt in wa.indexoid_table_map.items() if idxt == t] for t in wa.table_attr_map.keys()}
        data_ks = construct_query_window_stats(logger, connection, work_prefix, wa.table_attr_map.keys(), table_indexname_map, window_index_map, buckets)
        for tbl, df in data_ks.items():
            df.to_feather(f"{input_dir}/exec_features_{vacuum}/{tbl}.feather")
        open(f"{input_dir}/exec_features_{vacuum}/done", "w").close()

    logger.info("Finished at %s", datetime.now())

##################################################################################
# Generate the data page features.
##################################################################################

DATA_PAGES_COLUMNS = [
    "window_bucket",
    "start_timestamp",
    "target",
    "comment",
    "num_queries",
    "total_blks_hit",
    "total_blks_miss",
    "total_blks_requested",
    "total_blks_affected",
    "total_tuples_touched",
    "reltuples",
    "relpages"
]


# FIXME(BITMAP): We currently don't support Bitmap in total_tuples_touched.
DATA_PAGES_QUERY = """
SELECT
    s.window_bucket,
    s.start_timestamp,
    s.target,
    s.comment,
    s.num_queries,
    s.total_blks_hit,
    s.total_blks_miss,
    s.total_blks_requested,
    s.total_blks_affected,
    s.total_tuples_touched,
    f.reltuples,
    f.relpages FROM
(SELECT
    COALESCE(target_idx_scan, target_idx_scan_table, target) as target,
    comment,
    MIN(unix_timestamp) as start_timestamp,
    COUNT(DISTINCT statement_timestamp) as num_queries,
    SUM(blk_hit) as total_blks_hit,
    SUM(blk_miss) as total_blks_miss,
    SUM(blk_hit + blk_miss) as total_blks_requested,
    SUM(blk_dirty + blk_write) as total_blks_affected,
    SUM(CASE comment
            WHEN 'Agg' THEN counter0
            WHEN 'NestLoop' THEN counter1
            WHEN 'SeqScan' THEN counter0
            WHEN 'IndexScan' THEN counter0
            WHEN 'IndexOnlyScan' THEN counter0
            WHEN 'ModifyTableInsert' THEN 1
            WHEN 'ModifyTableUpdate' THEN counter8
            WHEN 'ModifyTableDelete' THEN counter5
            WHEN 'TupleARInsertTriggers' THEN counter0
            WHEN 'TupleARUpdateTriggers' THEN counter0
            WHEN 'TupleARDeleteTriggers' THEN counter0
        END) as total_tuples_touched,
    width_bucket(query_order, array[{values}]) as window_bucket
FROM {work_prefix}_mw_queries
WHERE comment IN (
    'Agg',
    'NestLoop',
    'SeqScan',
    'IndexScan',
    'IndexOnlyScan',
    'ModifyTableInsert',
    'ModifyTableUpdate',
    'ModifyTableDelete',
    'TupleARInsertTriggers',
    'TupleARUpdateTriggers',
    'TupleARDeleteTriggers'
)
GROUP BY COALESCE(target_idx_scan, target_idx_scan_table, target), comment, window_bucket) s,

LATERAL (
    SELECT * FROM {work_prefix}_mw_tables
    WHERE s.target = {work_prefix}_mw_tables.relname AND s.start_timestamp >= {work_prefix}_mw_tables.unix_timestamp
    ORDER BY {work_prefix}_mw_tables.unix_timestamp DESC LIMIT 1
) f
WHERE s.total_tuples_touched > 0 AND s.num_queries > 0 {addt_filter};
"""

def __gen_data_page_features(input_dir, engine, connection, work_prefix, wa, slice_window, generate_vacuum_flag):
    try:
        with engine.begin() as alchemy:
            # Load the pg_class table.
            time_tables, _ = process_time_pg_class(pd.read_csv(f"{input_dir}/pg_class.csv"))
            time_tables.to_sql(f"{work_prefix}_mw_tables", alchemy, index=False)

        # Build the index.
        with connection.transaction():
            connection.execute(f"CREATE INDEX {work_prefix}_mw_tables_0 ON {work_prefix}_mw_tables (relname, unix_timestamp)")
    except Exception as e:
        logger.info("Exception: %s", e)

    # This is so we can compute multiple slices at once.
    slices = slice_window.split(",")
    vacuum = "flag" if generate_vacuum_flag else "dropwindow"
    for slice_fragment in slices:
        if (Path(input_dir) / f"data_page_{slice_fragment}_{vacuum}/done").exists():
            continue

        (Path(input_dir) / f"data_page_{slice_fragment}_{vacuum}").mkdir(parents=True, exist_ok=True)
        logger.info("Computing data page information for slice: %s", slice_fragment)
        tables = wa.table_attr_map.keys()
        window_index_map = fetch_window_indexes(input_dir, slice_fragment, generate_vacuum_flag)

        for t, v in window_index_map.items():
            if not (isinstance(v, range) or isinstance(v, list)):
                # Assume it is a data frame then.
                v = v.query_order.values

            # Truncate off the first value; this is because we want queries that span [t=0, t=1] to be assigned window 0.
            sql = DATA_PAGES_QUERY.format(work_prefix=work_prefix, values=",".join([str(i) for i in v[1:]]), addt_filter=f" AND (s.target = '{t}')")
            logger.info("Executing SQL: %s", sql)
            result = connection.execute(sql)
            logger.info("Extracted data returned %s", result.rowcount)
            result = [r for r in result]
            if len(result) > 0:
                pd.DataFrame(result, columns=DATA_PAGES_COLUMNS).to_feather(f"{input_dir}/data_page_{slice_fragment}_{vacuum}/data_{t}.feather")
        open(f"{input_dir}/data_page_{slice_fragment}_{vacuum}/done", "w").close()

##################################################################################
# Collect the inputs from the database.
##################################################################################

def collect_inputs(input_dir, workload_only, psycopg2_conn,
                   work_prefix, buckets,
                   slice_window, offcpu_logwidth, gen_exec_features,
                   gen_data_page_features,
                   generate_vacuum_flag):

    # Read in the input keyspace metadata.
    wa = keyspace_metadata_read(input_dir)[0]
    engine = create_engine(psycopg2_conn)
    with psycopg.connect(psycopg2_conn, autocommit=True, prepare_threshold=None) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")

        # Construct all the useful window indexes.
        build_window_indexes(connection, work_prefix, input_dir, wa, buckets, slice_window, generate_vacuum_flag)

        # Generate the execution features.
        if gen_exec_features:
            __gen_exec_features(input_dir, connection, work_prefix, wa, buckets, generate_vacuum_flag)

        # Generate the data page features.
        if gen_data_page_features:
            __gen_data_page_features(input_dir, engine, connection, work_prefix, wa, slice_window, generate_vacuum_flag)


class ExecFeatureSynthesisCLI(cli.Application):
    dir_workload_input = cli.SwitchAttr(
        "--dir-workload-input",
        str,
        mandatory=True,
        help="Path to the folder containing the workload input.",
    )

    workload_only = cli.SwitchAttr(
        "--workload-only",
        str,
        help="Whether the input contains only the workload stream.",
    )

    psycopg2_conn = cli.SwitchAttr(
        "--psycopg2-conn",
        str,
        help="Psycopg2 connection that should be used.",
    )

    work_prefix = cli.SwitchAttr(
        "--work-prefix",
        str,
        mandatory=True,
        help="Prefix to use for working with the database.",
    )

    buckets = cli.SwitchAttr(
        "--buckets",
        int,
        default=10,
        help="Number of buckets to use for input data.",
    )

    offcpu_logwidth = cli.SwitchAttr(
        "--offcpu-logwidth",
        int,
        default=31,
        help="Log Width of the off cpu elapsed us axis.",
    )

    slice_window = cli.SwitchAttr(
        "--slice-window",
        str,
        default="1000",
        help="Slice window to use for data.",
    )

    gen_exec_features = cli.Flag(
        "--gen-exec-features",
        default=False,
        help="Whether to generate exec features data.",
    )

    gen_data_page_features = cli.Flag(
        "--gen-data-page-features",
        default=False,
        help="Whether to generate data page features data.",
    )

    generate_vacuum_flag = cli.Flag(
        "--generate-vacuum-flag",
        default=False,
        help="Whether to generate the vacuum flag instead of dropping the window.",
    )

    def main(self):
        pd.options.display.max_colwidth = 0
        input_parts = self.dir_workload_input.split(",")
        for i in range(len(input_parts)):
            logger.info("Processing %s (%s)", input_parts[i], self.workload_only)
            file_handler = logging.FileHandler(Path(input_parts[i]) / "output.log", mode="a")
            file_handler.propagate = False
            logger.addHandler(file_handler)

            collect_inputs(Path(input_parts[i]),
                           (self.workload_only == "True"),
                           self.psycopg2_conn,
                           self.work_prefix,
                           self.buckets,
                           self.slice_window,
                           self.offcpu_logwidth,
                           self.gen_exec_features,
                           self.gen_data_page_features,
                           self.gen_concurrency_features,
                           self.generate_vacuum_flag)

            logger.removeHandler(file_handler)


if __name__ == "__main__":
    ExecFeatureSynthesisCLI.run()

