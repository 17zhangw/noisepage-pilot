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
from behavior.model_workload.utils.keyspace_feature import construct_keyspaces, construct_query_window_stats
from behavior.utils.process_pg_state_csvs import process_time_pg_class

logger = logging.getLogger("exec_feature_synthesis")


DATA_PAGES_COLUMNS = [
    "window_bucket",
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


# For this, we classify each "trigger" as part of TupleAR* as being a single
# tuple. This is a gross simplification, however fetching a trigger metadata
# is a consistent operation (there is no "branching"-ness to it).
DATA_PAGES_QUERY = """
SELECT
    s.window_bucket,
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
    COALESCE(target_idx_scan_table, target) as target,
    comment,
    MIN(unix_timestamp) as start_timestamp,
    COUNT(DISTINCT statement_timestamp) as num_queries,
    SUM(blk_hit) as total_blks_hit,
    SUM(blk_miss) as total_blks_miss,
    SUM(blk_hit + blk_miss) as total_blks_requested,
    SUM(blk_dirty + blk_write) as total_blks_affected,
    SUM(CASE comment WHEN 'SeqScan' THEN counter0
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
GROUP BY COALESCE(target_idx_scan_table, target), comment, window_bucket) s,

LATERAL (
    SELECT * FROM {work_prefix}_mw_tables
    WHERE s.target = {work_prefix}_mw_tables.relname AND s.start_timestamp >= {work_prefix}_mw_tables.unix_timestamp
    ORDER BY {work_prefix}_mw_tables.unix_timestamp DESC LIMIT 1
) f
WHERE s.total_tuples_touched > 0 AND s.num_queries > 0;
"""


CONCURRENCY_COLUMNS = [
    "window_bucket",
    "target",
    "optype",
    "num_queries",
    "total_blks_hit",
    "total_blks_miss",
    "total_blks_requested",
    "total_blks_affected",
    "reltuples",
    "relpages"
]


# This is a simplification of the query above, in that we only look at information
# relevant for the computation of concurrency analysis. We assume that each plan
# already has the blocks counts available.
CONCURRENCY_QUERY = """
SELECT
    s.window_bucket,
    s.target,
    s.optype,
    s.num_queries,
    s.total_blks_hit,
    s.total_blks_miss,
    s.total_blks_requested,
    s.total_blks_affected,
    f.reltuples,
    f.relpages FROM
(SELECT
    etarget as target,
    optype,
    MIN(unix_timestamp) as start_timestamp,
    COUNT(DISTINCT statement_timestamp) as num_queries,
    SUM(blk_hit) as total_blks_hit,
    SUM(blk_miss) as total_blks_miss,
    SUM(blk_hit + blk_miss) as total_blks_requested,
    SUM(blk_dirty + blk_write) as total_blks_affected,
    width_bucket(query_order, array[{values}]) as window_bucket
FROM {work_prefix}_mw_queries, LATERAL unnest(string_to_array(target, ',')) etarget
WHERE plan_node_id = -1
GROUP BY etarget, optype, window_bucket) s,

LATERAL (
    SELECT * FROM {work_prefix}_mw_tables
    WHERE s.target = {work_prefix}_mw_tables.relname AND s.start_timestamp >= {work_prefix}_mw_tables.unix_timestamp
    ORDER BY {work_prefix}_mw_tables.unix_timestamp DESC LIMIT 1
) f
WHERE s.num_queries > 0;
"""


def save_bucket_keys_to_output(output_dir):
    def save_df_return_none(tbl, df):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df["key_dist"] = [",".join(map(str, l)) for l in df.key_dist]
        df.to_feather(f"{output_dir}/{tbl}.feather")
        return None

    return save_df_return_none


# Tables describes the base tables of the workload.
# all_targets captures all "targets" (this holds the multi-join targets..).
def build_window_indexes(connection, work_prefix, input_dir, tables, all_targets):
    window_index_map = {}
    if (Path(input_dir) / "exec_features/windows/done").exists():
        for t in glob.glob(f"{input_dir}/exec_features/windows/*.feather"):
            tbl = pd.read_feather(t)
            window_index_map[Path(t).stem.split(".")[0]] = tbl
        return window_index_map

    # This is when the autovacuum executed.
    pg_stat_user_tables = pd.read_csv(f"{input_dir}/pg_stat_user_tables.csv")
    pg_stat_user_tables = pg_stat_user_tables[~pg_stat_user_tables.last_autovacuum.isnull()]
    pg_stat_user_tables["autovacuum_unix_timestamp"] = pd.to_datetime(pg_stat_user_tables.last_autovacuum).map(pd.Timestamp.timestamp)

    with connection.transaction() as tx:
        connection.execute(f"CREATE INDEX {work_prefix}_mw_queries_args_time ON {work_prefix}_mw_queries_args (unix_timestamp)")
        for tbl in tables:
            logger.info("Computing window index map for table: %s", tbl)
            sample = pd.read_csv(f"{input_dir}/{tbl}.csv")
            sample["true_window_index"] = sample.index
            sample["time"] = (sample.time / float(1e6))

            # Inject a window_index that is -1
            # window_index = -1 are tuples that get discarded! woo-hoo.
            # FIXME(VACUUM): We currently don't have a way of encapsulating vacuum.
            # So we take the setup where we "drop" the window. To correctly capture VACUUM, it might be simpler
            # to capture the *impact* of vacuum as opposed to vacuum itself. Otherwise, we'll probably need
            # a "backward"-work model where we feed the deltas in the prior window to predict elapsed time.

            autovacuum_times = []
            substat = pg_stat_user_tables[pg_stat_user_tables.relname == tbl]
            if substat.shape[0] > 0:
                # Get the times when autovacuum ran.
                autovacuum_times = [v for v in substat.autovacuum_unix_timestamp.value_counts().index]

            if len(autovacuum_times) > 0:
                wipe_frame = pd.DataFrame([{"time": autovac, "true_window_index": -1} for autovac in autovacuum_times])
                sample = pd.concat([sample, wipe_frame], ignore_index=True)
                sample.sort_values(by=["time"], ignore_index=True, inplace=True)

            sample["window_index"] = sample.index

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
            sample.reset_index(drop=False, inplace=True)
            assert sample.query_order.is_monotonic_increasing
            window_index_map[tbl] = sample

        # Let's rollback the index.
        raise Rollback(tx)

    (Path(input_dir) / "exec_features" / "windows").mkdir(parents=True, exist_ok=True)
    for t, v in window_index_map.items():
        v.to_feather(f"{input_dir}/exec_features/windows/{t}.feather")
    open(f"{input_dir}/exec_features/windows/done", "w").close()
    logger.info("Finished computing window index map: %s", datetime.now())
    return window_index_map


def __gen_exec_features(input_dir, connection, work_prefix, wa, buckets):
    if (Path(input_dir) / "exec_features/keys/done").exists():
        return

    # Compute the window frames.
    logger.info("Computing window frames.")
    logger.info("Starting at %s", datetime.now())
    window_index_map = build_window_indexes(connection, work_prefix, input_dir, [t for t in wa.table_attr_map.keys()], list(set(wa.query_table_map.values())))

    # Get all the "keyspace" descriptor features.
    if not (Path(input_dir) / "exec_features" / "keys" / "done").exists():
        output_dir = Path(input_dir) / "exec_features/keys"
        callback = save_bucket_keys_to_output(output_dir)
        tbls = [t for t in wa.table_attr_map.keys() if not (Path(output_dir) / f"{t}.feather").exists()]
        construct_keyspaces(logger, connection, work_prefix, tbls, wa.table_attr_map, window_index_map, buckets, data_hist=True, callback_fn=callback)
        open(f"{input_dir}/exec_features/keys/done", "w").close()

    # Get all the data space features.
    if not (Path(input_dir) / "exec_features/data/done").exists():
        (Path(input_dir) / "exec_features/data/").mkdir(parents=True, exist_ok=True)
        data_ks = construct_query_window_stats(logger, connection, work_prefix, wa.table_attr_map.keys(), window_index_map, buckets)
        for tbl, df in data_ks.items():
            df.to_feather(f"{input_dir}/exec_features/data/{tbl}.feather")
        open(f"{input_dir}/exec_features/data/done", "w").close()

    logger.info("Finished at %s", datetime.now())


def __gen_data_page_features(input_dir, engine, connection, work_prefix, wa, buckets, slice_window):
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

    # Get the query order ranges.
    with connection.transaction():
        r = [r for r in connection.execute(f"SELECT min(query_order), max(query_order) FROM {work_prefix}_mw_queries_args")][0]
        min_qo, max_qo = r[0], r[1]

    # This is so we can compute multiple slices at once.
    slices = slice_window.split(",")
    for slice_fragment in slices:
        if (Path(input_dir) / f"data_page_{slice_fragment}/done").exists():
            continue

        logger.info("Computing data page information for slice: %s", slice_fragment)
        tables = wa.table_attr_map.keys()
        values = range(min_qo, max_qo, int(slice_fragment))
        window_index_map = {t: values for t in tables}

        # Get the keyspace affected.
        if not (Path(input_dir) / f"data_page_{slice_fragment}/keys/done").exists():
            # Generate keyspaces that separate SELECT/INSERT/UPDATE/DELETE.
            output_dir = Path(input_dir) / f"data_page_{slice_fragment}/keys"
            callback = save_bucket_keys_to_output(output_dir)
            tbls = [t for t in tables if not (Path(output_dir) / f"{t}.feather").exists()]
            construct_keyspaces(logger, connection, work_prefix, tbls, wa.table_attr_map, window_index_map, buckets, data_hist=False, callback_fn=callback)
            open(f"{input_dir}/data_page_{slice_fragment}/keys/done", "w").close()

        # Truncate off the first value; this is because we want queries that span [t=0, t=1] to be assigned window 0.
        sql = DATA_PAGES_QUERY.format(work_prefix=work_prefix, values=",".join([str(i) for i in values[1:]]))
        logger.info("Executing SQL: %s", sql)
        result = connection.execute(sql)
        logger.info("Extracted data returned %s", result.rowcount)
        pd.DataFrame([r for r in result], columns=DATA_PAGES_COLUMNS).to_feather(f"{input_dir}/data_page_{slice_fragment}/data.feather")
        open(f"{input_dir}/data_page_{slice_fragment}/done", "w").close()


def __gen_concurrency_features(input_dir, engine, connection, work_prefix, wa, buckets, concurrency_steps, offcpu_logwidth):
    if (Path(input_dir) / "concurrency/done").exists():
        return

    step = [int(i) for i in concurrency_steps.split(",")]
    data = pd.read_csv(f"{input_dir}/histograms.csv")
    if data.pid.nunique() > 1:
        # This is to handle the awkward case where the collector isn't captured.
        data = data[data.pid != data.iloc[0].pid]
    mpi = data.pid.nunique()

    try:
        with engine.begin() as alchemy:
            # Load the pg_class table.
            time_tables, _ = process_time_pg_class(pd.read_csv(f"{input_dir}/pg_class.csv"))
            time_tables.to_sql(f"{work_prefix}_mw_tables", alchemy, index=False)

        # Build the index.
        with connection.transaction():
            connection.execute(f"CREATE INDEX {work_prefix}_mw_tables_0 ON {work_prefix}_mw_tables (relname, unix_timestamp)")
    except Exception as e:
        logger.info("Exception encountered: %s", e)

    if not (Path(input_dir) / "concurrency/data_done").exists():
        # FIXME(TIME): In principle, this should be done per PID and stats accumulated per PID.
        # But we are already aggregating over all PIDs anyways so I think this should be fine.
        time_us = data[(data.pid == data.iloc[0].pid) & (data.elapsed_slice == 0)][["window_index", "time"]]
        time_us["time"] = time_us.time / 1.0e6
        time_us.sort_values(by=["window_index"], inplace=True)
        time_us["query_order"] = np.nan

        # Get the query order ranges.
        with connection.transaction() as tx:
            # Create a temporary index that will be destroyed.
            connection.execute(f"CREATE INDEX ON {work_prefix}_mw_queries_args (unix_timestamp)")

            # The times mark the end of a window. So we find the corresponding largest query_order that is before the end of the window.
            qo = """
                SELECT nr - 1 AS window_index, a.query_order FROM unnest(array[{times}]) WITH ORDINALITY AS u(elem, nr),
                LATERAL (
                    SELECT query_order FROM {work_prefix}_mw_queries_args
                    WHERE {work_prefix}_mw_queries_args.unix_timestamp <= elem
                    ORDER BY {work_prefix}_mw_queries_args.unix_timestamp DESC
                    LIMIT 1
                ) a ORDER BY window_index
            """.format(times=",".join([str(t) for t in time_us.time.values]), work_prefix=work_prefix)
            result = connection.execute(qo)
            r = [r for r in result]

            time_us.set_index(keys=["window_index"], drop=True, inplace=True)
            time_us.loc[[r[0] for r in r], "query_order"] = [r[1] for r in r]
            time_us.reset_index(drop=False, inplace=True)
            time_us = time_us[~time_us.query_order.isna()]

            for s in step:
                logger.info("Computing query access patterns under step: %s", s)
                # Segment the query orders based on the step size.
                #
                # We prepend 1 because 1 is the first valid "query_order". This is also a function of how
                # the first valid insert_version for a data slot is 1.
                qos_range = [1] + [time_us.query_order[i] for i in range(s - 1, time_us.shape[0], s)]

                # Force the end so we can then later drop everything that happens afterwards.
                if qos_range[-1] != time_us.iloc[-1].query_order:
                    qos_range.append(time_us.iloc[-1].query_order)
                qos = [str(i) for i in qos_range]

                # Get the keyspace affected.
                output_dir = Path(input_dir) / f"concurrency/step{s}/keys"
                callback = save_bucket_keys_to_output(output_dir)
                tbls = [t for t in wa.table_attr_map.keys() if not (Path(output_dir) / f"{t}.feather").exists()]
                window_index_map = {t:qos for t in tbls}
                construct_keyspaces(logger, connection, work_prefix, tbls, wa.table_attr_map, window_index_map, buckets, data_hist=False, callback_fn=callback)

                # once again, we truncate off the first value so we can get a "window 0" (since all values < first element is window 0).
                # in this case, we just strip off our dummy 0 that was added to mutate endpoints into corresponding start points.
                logger.info("Extracting the concurrency features under step: %s", s)
                sql = CONCURRENCY_QUERY.format(work_prefix=work_prefix, values=",".join(qos[1:]))
                result = connection.execute(sql)
                df = pd.DataFrame([r for r in result], columns=CONCURRENCY_COLUMNS)
                df["mpi"] = mpi
                df["step"] = s
                df.to_feather(f"{input_dir}/concurrency/step{s}/data.feather")
                del df
            raise Rollback(tx)

        open(f"{input_dir}/concurrency/data_done", "w").close()

    # Now we start generating the fits and curves.
    def convert(n):
        if n in ["window_index", "elapsed_slice", "pid", "time"]:
            return n
        elif n == "0_1":
            return "delay_1"
        else:
            return "delay_" + str(round(np.log2(float(n.split("_")[1]))))

    data.rename(convert, axis="columns", inplace=True)
    data.drop(columns=["time"], inplace=True)
    for key, df in data.groupby(by=["pid", "elapsed_slice"]):
        # Adjust so each "window" has its own data.
        df.sort_values(by=["window_index"], ascending=False)
        ind = [f"delay_{i}" for i in range(1, offcpu_logwidth + 1)]
        data.loc[df.index, ind] = df[ind] - df[ind].shift(fill_value=0)

    max_elapsed_slice = np.max(data.elapsed_slice)
    max_window_index = np.max(data.window_index)
    window_aggs = data.drop(columns=["pid"]).groupby(by=["elapsed_slice", "window_index"]).sum()
    assert window_aggs.index.shape[0] == (max_window_index + 1) * (max_elapsed_slice + 1)
    frame_tuples = []

    # Step the windows in this sequence.
    for s in step:
        logger.info("Computing distribution windows with step function: %s", s)
        if (Path(input_dir) / f"concurrency/step{s}/frame.feather").exists():
            continue

        with tqdm(total=(max_elapsed_slice + 1) * (max_window_index + 1)) as t:
            for sl in range(0, max_elapsed_slice + 1):
                window_index = 0
                l = [l for l in range(s - 1, max_window_index + 1, s)]
                for idx in l:
                    sf = window_aggs[(sl, idx-s+1):(sl, idx)]
                    values = sf.sum()[0:offcpu_logwidth].values.astype(float)
                    total = values.sum()
                    if total != 0:
                        values /= total
                        frame_tuples.append({
                            "mpi": mpi,
                            "step": s,
                            "window_index": window_index,
                            "elapsed_slice": sl,
                            "elapsed_slice_queries": total,
                            "targets": values,
                        })
                    window_index += 1
                    t.update(1)

        pd.DataFrame(frame_tuples).to_feather(f"{input_dir}/concurrency/step{s}/frame.feather")
        frame_tuples = []

    open(f"{input_dir}/concurrency/done", "w").close()


def collect_inputs(input_dir, workload_only, psycopg2_conn, work_prefix, buckets, concurrency_steps, slice_window, offcpu_logwidth, gen_exec_features, gen_data_page_features, gen_concurrency_features):
    wa = keyspace_metadata_read(input_dir)[0]

    engine = create_engine(psycopg2_conn)
    with psycopg.connect(psycopg2_conn, autocommit=True, prepare_threshold=None) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
        if gen_exec_features:
            __gen_exec_features(input_dir, connection, work_prefix, wa, buckets)

        if gen_data_page_features:
            __gen_data_page_features(input_dir, engine, connection, work_prefix, wa, buckets, slice_window)

        if gen_concurrency_features:
            __gen_concurrency_features(input_dir, engine, connection, work_prefix, wa, buckets, concurrency_steps, offcpu_logwidth)


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

    concurrency_steps = cli.SwitchAttr(
        "--steps",
        str,
        default="1",
        help="Summarizations of the concurrency features.",
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

    gen_concurrency_features = cli.Flag(
        "--gen-concurrency-features",
        default=False,
        help="Whether to generate concurrency features data.",
    )

    def main(self):
        pd.options.display.max_colwidth = 0
        input_parts = self.dir_workload_input.split(",")
        for i in range(len(input_parts)):
            logger.info("Processing %s (%s)", input_parts[i], self.workload_only)
            collect_inputs(Path(input_parts[i]),
                           (self.workload_only == "True"),
                           self.psycopg2_conn,
                           self.work_prefix,
                           self.buckets,
                           self.concurrency_steps,
                           self.slice_window,
                           self.offcpu_logwidth,
                           self.gen_exec_features,
                           self.gen_data_page_features,
                           self.gen_concurrency_features)


if __name__ == "__main__":
    ExecFeatureSynthesisCLI.run()

