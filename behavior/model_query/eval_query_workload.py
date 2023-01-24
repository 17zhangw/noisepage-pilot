import json
import os
import joblib
from tqdm import tqdm
import logging
import glob
import shutil
import psycopg
from psycopg import Rollback
import numpy as np
import pandas as pd
from pathlib import Path
from plumbum import cli

from behavior.datagen.pg_collector_utils import SettingType, _parse_field
from behavior.utils.process_pg_state_csvs import postgres_julian_to_unix
from behavior.model_workload.utils import keyspace_metadata_read
from behavior.model_workload.utils.keyspace_feature import construct_keyspaces
from behavior.model_query.utils.query_ous import generate_query_ous_window
from behavior.model_query.utils.buffer_ous import instantiate_buffer_page_reqs, compute_buffer_page_features, compute_buffer_access_features
from behavior.model_query.utils.table_state import (
    initial_trigger_metadata,
    initial_table_feature_state,
    refresh_table_fillfactor,
    resolve_index_feature_state,
    compute_table_exec_features,
    compute_index_exec_features,
    compute_next_window_state,
    compute_next_index_window_state,
)

from behavior.model_workload.utils import OpType
from behavior.model_query.utils import OUGenerationContext, load_ou_models, load_model
from behavior.model_query.utils.worker import process_window_ous


logger = logging.getLogger(__name__)

##################################################################################
# Scrape the knob state.
##################################################################################

def scrape_knobs(connection, ougc):
    knobs = {
        "autovacuum": SettingType.BOOLEAN,
        "autovacuum_max_workers": SettingType.INTEGER,
        "autovacuum_naptime": SettingType.INTEGER_TIME,
        "autovacuum_vacuum_threshold": SettingType.INTEGER,
        "autovacuum_vacuum_insert_threshold": SettingType.INTEGER,
        "autovacuum_vacuum_scale_factor": SettingType.FLOAT,
        "autovacuum_vacuum_insert_scale_factor": SettingType.FLOAT,
        "shared_buffers": SettingTYpe.BYTES,
    }

    for knobname, knobtype in knobs:
        knobvalue = [r for r in connection.execute(f"SHOW {knobnmae}")][0][0]
        ougc.knobs[knobname] = _parse_field(knobtype, knobvalue)

##################################################################################
# Generate the keyspace features.
##################################################################################

def generate_qos_from_ts(connection, work_prefix, timestamps):
    with connection.transaction() as tx:
        # This SQL is awkward. But the insight here is that instead of [t1, t2, t3] as providing the bounds, we want to use query_order.
        # Since width_bucket() uses the property that if x = t1, it returns [t1, t2] bucket. So in principle, we want to find the first
        # query *AFTER* t1 so it'll still act as the correct exclusive bound.
        sql = "UNNEST(ARRAY[" + ",".join([str(i) for i in timestamps]) + "], "
        sql += "ARRAY[" + ",".join([str(i) for i in range(len(timestamps))]) + "]) as x(time, window_index)"
        sql = f"SELECT window_index, time, b.query_order FROM {sql}, "
        sql += f"LATERAL (SELECT query_order FROM {work_prefix}_mw_eval_analysis WHERE query_order >= 1 and unix_timestamp > time ORDER BY unix_timestamp, query_order ASC LIMIT 1) b ORDER BY window_index"
        c = connection.execute(sql)
        tups = [(tup[0], tup[2]) for tup in c]

        # Let's rollback the index.
        raise Rollback(tx)
    return tups


def populate_keyspace_features(workload_conn, workload_analysis_prefix, output_dir, ougc, qos_windows, histogram_width):
    # Populate the keyspace features.
    if not (Path(output_dir) / "scratch/keyspaces/done").exists():
        def save_bucket_keys_to_output(output):
            def save_df_return_none(tbl, df):
                Path(output).mkdir(parents=True, exist_ok=True)
                df["key_dist"] = [",".join(map(str, l)) for l in df.key_dist]
                df.to_feather(f"{output}/{tbl}.feather")
                return None
            return save_df_return_none

        window_index_map = {t: [i for i in qos_windows] for t in ougc.tables}
        tables = [t for t in ougc.tables if not (Path(output_dir) / f"scratch/keyspaces/{t}.feather").exists()]
        callback = save_bucket_keys_to_output(Path(output_dir) / "scratch/keyspaces")
        construct_keyspaces(logger, workload_conn, workload_analysis_prefix, tables, ougc.table_attr_map, window_index_map, histogram_width, callback_fn=callback)
        open(f"{output_dir}/scratch/keyspaces/done", "w").close()

    # Load the table keyspace features.
    ougc.table_keyspace_features = {}
    for t in ougc.tables:
        if (Path(output_dir) / f"scratch/keyspaces/{t}.feather").exists():
            ougc.table_keyspace_features[t] = pd.read_feather(f"{output_dir}/scratch/keyspaces/{t}.feather")

##################################################################################
# Compute the DDL Changes
##################################################################################

def advance_ddl_change(ddl_changes, target_conn, ougc, current_qo):
    # If it is time to execute a DDL statement, then execute the DDL statement.
    for (ddl_qo, ddl_stmt) in ddl_changes:
        if current_qo < ddl_qo:
            break

        if current_qo == ddl_qo:
            # Reload the index feature state to acquire newly created indexes.
            # This will also catch any deleted indexes.
            target_conn.execute(ddl_stmt)

            # Refresh any table fillfactor related information.
            refresh_table_fillfactor(target_conn, ougc)

            # FIXME(INDEX): We are under the assumption that we don't need to compute keyspaces.
            # This is because we currently don't have INDEX keyspaces. We still need to refresh
            # the relevant metadata.
            resolve_index_feature_state(target_conn, ougc)

            # Refresh any relevant settings.
            scrape_knobs(target_conn, ougc)


def compute_ddl_changes(dir_data, workload_prefix, workload_conn):
    if not Path(f"{dir_data}/pg_qss_ddl.csv").exists():
        return []

    # Get all the relevant DDL changes that we care about.
    ddl = pd.read_csv(f"{dir_data}/pg_qss_ddl.csv")
    ddl = ddl[(ddl.command == "AlterTableOptions") | (ddl.command == "CreateIndex") | (ddl.command == "DropIndex")]
    ddl["unix_timestamp"] = postgres_julian_to_unix(ddl.statement_timestamp)
    ddl.reset_index(drop=True, inplace=True)
    if ddl.shape[0] == 0:
        return []

    # Get the query_order equivalent timestamp for each DDL change.
    sql = """
        SELECT ord - 1, b.query_order FROM UNNEST(ARRAY[{args}]) WITH ORDINALITY AS x(time, ord),
        LATERAL (SELECT MAX(query_order) as query_order FROM {prefix}_mw_queries_args WHERE unix_timestamp < time) b
    """.format(
        args=",".join([str(s) for s in ddl.unix_timestamp.values]),
        prefix=workload_prefix)

    result = [(t[0], t[1]) for t in workload_conn.execute(sql)]
    ddl["query_order"] = np.isnan
    ddl.loc[[t[0] for t in result], "query_order"] = [t[1] for t in result]
    ddl.drop(ddl[ddl.query_order.isna()].index, inplace=True)

    steps = []
    for d in ddl.itertuples():
        steps.append((d.query_order, d.query))
    return steps

##################################################################################
# Populate the table feature states.
##################################################################################

def populate_table_feature_states(workload_conn, target_conn, workload_analysis_prefix,
                                  output_dir, forward_state, estimate_vacuum, use_vacuum_flag,
                                  ougc, ddl_changes, tables, qos_windows):
    logging.root.setLevel(logging.WARN)

    valid_tbls = [t for t in tables if (Path(output_dir) / f"scratch/keyspaces/{t}.feather").exists()]
    if not (Path(output_dir) / "scratch/states/done").exists():
        # Table execution feature frame inputs.
        exec_feature_dfs = []
        index_exec_feature_dfs = []
        # Table state feature frame inputs.
        exec_state_dfs = []
        index_exec_state_dfs = []
        # BPM feature frame inputs.
        bpm_dfs = []

        (Path(output_dir) / "scratch/states").mkdir(parents=True, exist_ok=True)
        for i, current_qo in tqdm(enumerate(qos_windows), total=len(qos_windows)): #, leave=False):
            if i != len(qos_windows) - 1:
                upper_qo = qos_windows[i+1]
            else:
                # There is no more at the end.
                break

            # See if we need to advance the DDL state of the world.
            advance_ddl_change(ddl_changes, target_conn, ougc, current_qo)

            # This SQL query is designed to compute what happens in a frame.
            # Hopefully much faster than we can do in memory only through a dataframe.
            state_forward_sql = """
                SELECT starget, SUM(is_insert) as num_insert_tuples, SUM(hits) as num_select_tuples,
                       SUM(hits * is_update) as num_update_tuples, SUM(hits * is_delete) as num_delete_tuples,
                       SUM(is_insert + hits * is_update + hits * is_delete) as num_modify_tuples,
                       SUM(is_select) as num_select_queries,
                       SUM(is_insert) as num_insert_queries,
                       SUM(is_update) as num_update_queries,
                       SUM(is_delete) as num_delete_queries
                FROM (
                    SELECT starget, is_insert, is_update, is_delete, is_select,
                           (CASE starget {clauses} ELSE 0 END) as hits
                    FROM (
                        SELECT string_to_table(target, ',') as starget,
                         (CASE optype WHEN {select_val} THEN 1 ELSE 0 END) as is_select,
			 (CASE optype WHEN {insert_val} THEN 1 ELSE 0 END) as is_insert,
			 (CASE optype WHEN {update_val} THEN 1 ELSE 0 END) as is_update,
			 (CASE optype WHEN {delete_val} THEN 1 ELSE 0 END) as is_delete,
                         {tbl_hits}
                        FROM {prefix}_mw_eval_analysis
                        WHERE query_order >= {lower_qo} AND query_order < {upper_qo}
                        ) c
                    ) b GROUP BY starget ORDER BY starget
            """.format(select_val=OpType.SELECT.value,
                       insert_val=OpType.INSERT.value,
                       update_val=OpType.UPDATE.value,
                       delete_val=OpType.DELETE.value,
                       prefix=workload_analysis_prefix,
                       tbl_hits=",\n".join([f"{t}_hits" for t in valid_tbls]),
                       clauses="\n".join([f"WHEN '{t}' THEN {t}_hits" for t in valid_tbls]),
                       lower_qo=current_qo,
                       upper_qo=upper_qo)
            tbl_summaries = [{
                "target": r[0],
                "num_insert_tuples": r[1],
                "num_select_tuples": r[2],
                "num_update_tuples": r[3],
                "num_delete_tuples": r[4],
                "num_modify_tuples": r[5],
                "num_select_queries": r[6],
                "num_insert_queries": r[7],
                "num_update_queries": r[8],
                "num_delete_queries": r[9],
            } for r in workload_conn.execute(state_forward_sql)]

            # Get the table execution features using the window.
            ret_df = compute_table_exec_features(ougc, tbl_summaries, i, output_df=True)
            if ret_df is not None:
                exec_feature_dfs.append(ret_df)

            # Get the index execution features.
            ret_df = compute_index_exec_features(ougc, i, output_df=True)
            if ret_df is not None:
                index_exec_feature_dfs.append(ret_df)

            # Populate the buffer pages/OU used during this window.
            ret_df = instantiate_buffer_page_reqs(ougc, i)
            if ret_df is not None:
                bpm_dfs.append(ret_df)

            # TODO: Compute the vacuum flag here...!

            # We need to save the state *FROM* before forwarding.
            prior_image = ougc.save_state(i, current_qo, upper_qo)
            joblib.dump(prior_image, f"{output_dir}/scratch/states/{i}.gz")

            # Now consider forwarding the state.
            if not forward_state:
                # TODO: Incorporate the feedback from the vacuum above.
                # First compute the new index state since we want the old state still.
                ret_df = compute_next_index_window_state(ougc, i, output_df=True)
                if ret_df is not None:
                    index_exec_state_dfs.append(ret_df)

                # Compute the next stats incarnation
                ret_df = compute_next_window_state(ougc, i, output_df=True)
                if ret_df is not None:
                    exec_state_dfs.append(ret_df)

        if len(exec_feature_dfs) > 0:
            pd.concat(exec_feature_dfs, ignore_index=True).to_feather(f"{output_dir}/scratch/eval_TFM.feather")
        if len(index_exec_feature_dfs) > 0:
            pd.concat(index_exec_feature_dfs, ignore_index=True).to_feather(f"{output_dir}/scratch/eval_IFM.feather")
        if len(exec_state_dfs) > 0:
            pd.concat(exec_state_dfs, ignore_index=True).to_feather(f"{output_dir}/scratch/eval_TSM.feather")
        if len(index_exec_state_dfs) > 0:
            pd.concat(index_exec_state_dfs, ignore_index=True).to_feather(f"{output_dir}/scratch/eval_ISM.feather")
        if len(bpm_dfs) > 0:
            pd.concat(bpm_dfs, ignore_index=True).to_feather(f"{output_dir}/scratch/eval_BPM.feather")
        open(f"{output_dir}/scratch/states/done", "w").close()

##################################################################################
# Main
##################################################################################

def main(workload_analysis_conn, target_db_conn, workload_analysis_prefix,
         input_dir, session_sql, ou_models_path,
         query_feature_granularity_queries,
         time_slice_interval, approx_stats,
         estimate_vacuum, use_vacuum_flag,
         table_feature_model_cls, table_feature_model_path,
         table_state_model_cls, table_state_model_path,
         index_feature_model_cls, index_feature_model_path,
         index_state_model_cls, index_state_model_path,
         buffer_page_model_cls, buffer_page_model_path,
         buffer_access_model_cls, buffer_access_model_path,
         histogram_width,
         forward_state,
         use_plan_estimates,
         num_cpus, num_threads,
         output_dir):

    args = {
        "forward_state": forward_state,
        "workload_analysis_conn": workload_analysis_conn,
        "target_db_conn": target_db_conn,
        "workload_analysis_prefix": workload_analysis_prefix,
        "input_dir": input_dir,
        "session_sql": session_sql,
        "ou_models_path": ou_models_path,
        "query_feature_granularity_queries": query_feature_granularity_queries,
        "time_slice_interval": time_slice_interval,
        "approx_stats": approx_stats,
        "estimate_vacuum": estimate_vacuum,
        "use_vacuum_flag": use_vacuum_flag,
        "table_feature_model_cls": table_feature_model_cls,
        "table_feature_model_path": table_feature_model_path,
        "table_state_model_cls": table_state_model_cls,
        "table_state_model_path": table_state_model_path,
        "index_feature_model_cls": index_feature_model_cls,
        "index_feature_model_path": index_feature_model_path,
        "index_state_model_cls": index_state_model_cls,
        "index_state_model_path": index_state_model_path,
        "buffer_page_model_cls": buffer_page_model_cls,
        "buffer_page_model_path": buffer_page_model_path,
        "histogram_width": histogram_width,
        "use_plan_estimates": use_plan_estimates,
        "output_dir": output_dir,
    }

    logger.info("Beginning eval_query_workload evaluation of %s", input_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/config", "w") as f:
        f.write(json.dumps({k: str(v) for k, v in args.items()}, indent=2))

    # Create a context object for invoking sub functions.
    ougc = OUGenerationContext()
    ougc.table_feature_model = load_model(table_feature_model_path, table_feature_model_cls)
    ougc.table_state_model = load_model(table_state_model_path, table_state_model_cls)
    ougc.buffer_page_model = load_model(buffer_page_model_path, buffer_page_model_cls)
    ougc.index_feature_model = load_model(index_feature_model_path, index_feature_model_cls)
    ougc.index_state_model = load_model(index_state_model_path, index_state_model_cls)
    # Start loading the OUGenerationContext with metadata.
    wa = keyspace_metadata_read(input_dir)[0]
    ougc.tables = list(wa.table_attr_map.keys())
    ougc.table_attr_map = wa.table_attr_map

    with psycopg.connect(workload_analysis_conn) as workload_conn:
        with psycopg.connect(target_db_conn) as target_conn:
            with target_conn.transaction() as txn:
                target_conn.execute("SET qss_capture_enabled = OFF")
                target_conn.execute("SET plan_cache_mode = 'force_generic_plan'")
                target_conn.execute("CREATE EXTENSION IF NOT EXISTS pgstattuple")
                target_conn.execute("CREATE EXTENSION IF NOT EXISTS qss")

            # Get the DDL changes and the initial shared_buffers state.
            scrape_knobs(target_conn, ougc)

            # Load the initial state into the OUGenerationContext.
            initial_trigger_metadata(target_conn, ougc)
            initial_table_feature_state(target_conn, ougc, approx_stats)
            resolve_index_feature_state(target_conn, ougc)

            # Get the query order ranges.
            with workload_conn.transaction():
                r = [r for r in workload_conn.execute(f"SELECT min(query_order), max(query_order) FROM {workload_analysis_prefix}_mw_queries_args")][0]
                min_qo, max_qo = r[0], r[1]

                r = [r for r in workload_conn.execute(f"SELECT min(unix_timestamp), max(unix_timestamp) FROM {workload_analysis_prefix}_mw_eval_analysis")][0]
                min_ts, max_ts = r[0], r[1]

                r = [r for r in workload_conn.execute(f"SELECT a.attname FROM pg_attribute a, pg_class c WHERE a.attrelid = c.oid AND c.relname = '{workload_analysis_prefix}_mw_queries_args'")]
                r = [int(t[0][3:]) for t in r if t[0].startswith("arg")]
                max_arg = max(r)

            # Identify the relevant window boundaries.
            ddl_changes = compute_ddl_changes(input_dir, workload_analysis_prefix, workload_conn)
            if time_slice_interval > 0:
                timestamps = []
                while min_ts < max_ts:
                    timestamps.append(min_ts)
                    min_ts += time_slice_interval
                timestamps.append(max_ts)
                timestamps = timestamps[1:]
                qos_windows = generate_qos_from_ts(workload_conn, workload_analysis_prefix, timestamps)
                qos_windows = set([q[1] for q in qos_windows]) | set([q[0] for q in ddl_changes]) | set([min_qo, max_qo + 1])
                qos_windows = sorted(list(qos_windows))
            else:
                qos_windows = set(range(min_qo, max_qo, query_feature_granularity_queries)) | set([q[0] for q in ddl_changes]) | set([max_qo + 1])
                qos_windows = sorted(list(qos_windows))

            # Populate keyspace features.
            populate_keyspace_features(workload_conn, workload_analysis_prefix, output_dir, ougc, qos_windows, histogram_width)

            # Populate all the table feature states.
            populate_table_feature_states(workload_conn, target_conn, workload_analysis_prefix, output_dir, forward_state, estimate_vacuum, use_vacuum_flag, ougc, ddl_changes, wa.table_attr_map.keys(), qos_windows)

    key_fn = lambda k: int(k.split("/")[-1].split(".")[0])
    (Path(output_dir) / "scratch/frames").mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "scratch/ous").mkdir(parents=True, exist_ok=True)
    options = sorted(glob.glob(f"{output_dir}/scratch/states/*.gz"), key=key_fn)
    if not (Path(output_dir) / "scratch/ous/done").exists():
        with psycopg.connect(target_db_conn) as target_conn:
            with joblib.parallel_backend(backend="loky", inner_max_num_threads=num_threads):
                with joblib.Parallel(n_jobs=num_cpus, verbose=10) as parallel:
                    def dispatch_range(last_i, i):
                        # This is a check to make sure we only process that which we need to.
                        opts = [opt for opt in options[last_i:i] if not Path(f"{output_dir}/scratch/frames/{key_fn(opt)}.feather").exists()]
                        parallel(joblib.delayed(process_window_ous)(args, i) for i in opts)

                    last_i = 0
                    for i, current_qo in enumerate(qos_windows):
                        # If it is time to execute a DDL statement, then execute the DDL statement.
                        for (ddl_qo, ddl_stmt) in ddl_changes:
                            if current_qo == ddl_qo:
                                break

                            # Process all the windows up until where the DDL statement manifests.
                            dispatch_range(last_i, i)

                            # Manifest the DDL change.
                            target_conn.execute(ddl_stmt)
                            last_i = i

                    dispatch_range(last_i, i)
        open(f"{output_dir}/scratch/ous/done", "w").close()

    # The output directory is the resolved directory.
    if (Path(output_dir) / "evals").exists():
        os.remove(f"{output_dir}/evals")
    os.symlink(f"{output_dir}/scratch/frames", f"{output_dir}/evals")


class EvalQueryWorkloadCLI(cli.Application):
    num_cpus = cli.SwitchAttr(
        "--num-cpus",
        int,
        default=1,
        help="Number of CPUs to process OUs in parallel.",
    )

    num_threads = cli.SwitchAttr(
        "--num-threads",
        int,
        default=1,
        help="Number of threads per parallel process."
    )

    session_sql = cli.SwitchAttr(
        "--session-sql",
        Path,
        mandatory=False,
        help="Path to list of SQL statements taht should be executed prior to EXPLAIN.",
    )

    input_dir = cli.SwitchAttr(
        "--input-dir",
        Path,
        mandatory=True,
        help="Folder that contains the input data.",
    )

    output_dir = cli.SwitchAttr(
        "--output-dir",
        Path,
        mandatory=True,
        help="Folder to output evaluations to.",
    )

    ou_models_path = cli.SwitchAttr(
        "--ou-models",
        Path,
        mandatory=True,
        help="Folder that contains all the OU models.",
    )

    query_feature_granularity_queries = cli.SwitchAttr(
        "--query-feature-granularity-queries",
        int,
        default=1000,
        help="Granularity of window slices for table feature, buffer page, and buffer access model in (# queries).",
    )

    time_slice_interval = cli.SwitchAttr(
        "--time-slice-interval",
        int,
        default=0,
        help="Zero to disable otherwise slice queries by time.",
    )

    table_feature_model_cls =cli.SwitchAttr(
        "--table-feature-model-cls",
        str,
        default="AutoMLTableFeatureModel",
        help="Name of the table feature model class that should be instantiated.",
    )

    table_feature_model_path = cli.SwitchAttr(
        "--table-feature-model-path",
        Path,
        default=None,
        help="Path to the table feature model that should be loaded.",
    )

    table_state_model_cls =cli.SwitchAttr(
        "--table-state-model-cls",
        str,
        default="AutoMLTableStateModel",
        help="Name of the table state model class that should be instantiated.",
    )

    table_state_model_path = cli.SwitchAttr(
        "--table-state-model-path",
        Path,
        default=None,
        help="Path to the table state model that should be loaded.",
    )

    buffer_page_model_path = cli.SwitchAttr(
        "--buffer-page-model-path",
        Path,
        default=None,
        help="Path to the buffer page model that should be loaded.",
    )

    buffer_page_model_cls = cli.SwitchAttr(
        "--buffer-page-model-cls",
        str,
        default="AutomLBufferPageModel",
        help="Name of the buffer page model class that should be instantiated.",
    )

    index_feature_model_cls = cli.SwitchAttr(
        "--index-feature-model-cls",
        str,
        default="AutoMLIndexFeatureModel",
        help="Name of the index feature model class that should be instantiated.",
    )

    index_feature_model_path = cli.SwitchAttr(
        "--index-feature-model-path",
        Path,
        default=None,
        help="Path to the index feature model that should be loaded.",
    )

    index_state_model_cls = cli.SwitchAttr(
        "--index-state-model-cls",
        str,
        default="AutoMLIndexFeatureModel",
        help="Name of the index state model class that should be instantiated.",
    )

    index_state_model_path = cli.SwitchAttr(
        "--index-state-model-path",
        Path,
        default=None,
        help="Path to the index state model that should be loaded.",
    )

    workload_analysis_conn = cli.SwitchAttr(
        "--workload-analysis-conn",
        mandatory=True,
        help="Connection string to workload analysis database.",
    )

    target_db_conn = cli.SwitchAttr(
        "--target-db-conn",
        mandatory=True,
        help="COnnection string to target database.",
    )

    workload_analysis_prefix = cli.SwitchAttr(
        "--workload-analysis-prefix",
        mandatory=True,
        help="Prefix that we should use for looking at the workload analysis database.",
    )

    histogram_width = cli.SwitchAttr(
        "--histogram-width",
        int,
        default=10,
        help="Number of buckets (or histogram width to accomodate into)."
    )

    forward_state = cli.Flag(
        "--forward-state",
        default=False,
        help="Whether to forward initial state forwards or not."
    )

    use_plan_estimates = cli.Flag(
        "--use-plan-estimates",
        default=False,
        help="Whether to use postgres plan estimates as opposed to row execution feature.",
    )

    approx_stats = cli.Flag(
        "--approx-stats",
        default=False,
        help="Whether to use pgstattuple_approx or not.",
    )

    estimate_vacuum = cli.Flag(
        "--estimate-vacuum",
        default=False,
        help="Whether to perform analytical vacuum estimation or not.",
    )

    use_vaccum_flag = cli.Flag(
        "--use-vacuum-flag",
        default=False,
        help="Whether to use the vacuum flag with the table state model.",
    )

    def main(self):
        main(self.workload_analysis_conn, self.target_db_conn,
             self.workload_analysis_prefix,
             self.input_dir,
             self.session_sql,
             self.ou_models_path,
             self.query_feature_granularity_queries,
             self.time_slice_interval,
             self.approx_stats,
             self.estimate_vacuum,
             self.use_vacuum_flag,
             self.table_feature_model_cls,
             self.table_feature_model_path,
             self.table_state_model_cls,
             self.table_state_model_path,
             self.index_feature_model_cls,
             self.index_feature_model_path,
             self.index_state_model_cls,
             self.index_state_model_path,
             self.buffer_page_model_cls,
             self.buffer_page_model_path,
             self.histogram_width,
             self.forward_state,
             self.use_plan_estimates,
             self.num_cpus, self.num_threads,
             self.output_dir)


if __name__ == "__main__":
    EvalQueryWorkloadCLI.run()
