import json
import os
import joblib
from tqdm import tqdm
import logging
import glob
import shutil
import psycopg
import numpy as np
import pandas as pd
from pathlib import Path
from plumbum import cli

from behavior.datagen.pg_collector_utils import SettingType, _parse_field
from behavior.utils.process_pg_state_csvs import postgres_julian_to_unix
from behavior.model_workload.utils import keyspace_metadata_read
from behavior.model_workload.utils.keyspace_feature import construct_keyspaces
from behavior.model_query.utils.query_ous import generate_query_ous_window
from behavior.model_query.utils.buffer_ous import compute_buffer_page_features, compute_buffer_access_features
from behavior.model_query.utils.table_state import (
    initial_trigger_metadata,
    initial_table_feature_state,
    refresh_table_fillfactor,
    resolve_index_feature_state,
    compute_table_exec_features,
    compute_next_window_state,
)

from behavior.model_workload.utils import OpType
from behavior.model_query.utils import OUGenerationContext, load_ou_models, load_model
from behavior.model_query.utils.worker import process_window_ous, process_window_concurrency


logger = logging.getLogger(__name__)

##################################################################################
# Generate the keyspace features.
##################################################################################

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
            ougc.shared_buffers = [r for r in target_conn.execute("SHOW shared_buffers")][0][0]
            ougc.shared_buffers = _parse_field(SettingType.BYTES, ougc.shared_buffers)


def compute_ddl_changes(dir_data, workload_prefix, workload_conn):
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

def populate_table_feature_states(workload_conn, target_conn, workload_analysis_prefix, output_dir, forward_state, ougc, ddl_changes, tables, qos_windows):
    valid_tbls = [t for t in tables if (Path(output_dir) / f"scratch/keyspaces/{t}.feather").exists()]
    if not (Path(output_dir) / "scratch/states/done").exists():
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
                       SUM(is_insert + hits * is_update + hits * is_delete) as num_modify_tuples
                FROM (
                    SELECT starget, is_insert, is_update, is_delete,
                           (CASE starget {clauses} ELSE 0 END) as hits
                    FROM (
                        SELECT string_to_table(target, ',') as starget,
			 (CASE optype WHEN {insert_val} THEN 1 ELSE 0 END) as is_insert,
			 (CASE optype WHEN {update_val} THEN 1 ELSE 0 END) as is_update,
			 (CASE optype WHEN {delete_val} THEN 1 ELSE 0 END) as is_delete,
                         {tbl_hits}
                        FROM {prefix}_mw_eval_analysis
                        WHERE query_order >= {lower_qo} AND query_order < {upper_qo}
                        ) c
                    ) b GROUP BY starget ORDER BY starget
            """.format(insert_val=OpType.INSERT.value, update_val=OpType.UPDATE.value, delete_val=OpType.DELETE.value,
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
                "num_modify_tuples": r[5]
            } for r in workload_conn.execute(state_forward_sql)]

            # Get the table execution features using the window.
            compute_table_exec_features(ougc, tbl_summaries, i)
            prior_image = ougc.save_state(i, current_qo, upper_qo)

            if not forward_state:
                # Compute the next stats incarnation
                compute_next_window_state(ougc)

            # We need to save the state *FROM* before forwarding.
            joblib.dump(prior_image, f"{output_dir}/scratch/states/{i}.gz")
        open(f"{output_dir}/scratch/states/done", "w").close()

##################################################################################
# Main
##################################################################################

def main(workload_analysis_conn, target_db_conn, workload_analysis_prefix,
         input_dir, session_sql, ou_models_path,
         query_feature_granularity_queries,
         table_feature_model_cls, table_feature_model_path,
         buffer_page_model_cls, buffer_page_model_path,
         buffer_access_model_cls, buffer_access_model_path,
         concurrency_model_cls, concurrency_model_path,
         concurrency_mpi,
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
        "table_feature_model_cls": table_feature_model_cls,
        "table_feature_model_path": table_feature_model_path,
        "buffer_page_model_cls": buffer_page_model_cls,
        "buffer_page_model_path": buffer_page_model_path,
        "buffer_access_model_cls": buffer_access_model_cls,
        "buffer_access_model_path": buffer_access_model_path,
        "concurrency_model_cls": concurrency_model_cls,
        "concurrency_model_path": concurrency_model_path,
        "concurrency_mpi": concurrency_mpi,
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
    ougc.concurrency_model = load_model(args["concurrency_model_path"], args["concurrency_model_cls"])
    # Start loading the OUGenerationContext with metadata.
    wa = keyspace_metadata_read(input_dir)[0]
    ougc.tables = list(wa.table_attr_map.keys())
    ougc.table_attr_map = wa.table_attr_map

    with psycopg.connect(workload_analysis_conn) as workload_conn:
        with psycopg.connect(target_db_conn) as target_conn:
            # Get the DDL changes and the initial shared_buffers state.
            ougc.shared_buffers = [r for r in target_conn.execute("SHOW shared_buffers")][0][0]
            ougc.shared_buffers = _parse_field(SettingType.BYTES, ougc.shared_buffers)

            # Load the initial state into the OUGenerationContext.
            initial_trigger_metadata(target_conn, ougc)
            initial_table_feature_state(target_conn, ougc)
            resolve_index_feature_state(target_conn, ougc)

            # Get the query order ranges.
            with workload_conn.transaction():
                r = [r for r in workload_conn.execute(f"SELECT min(query_order), max(query_order) FROM {workload_analysis_prefix}_mw_queries_args")][0]
                min_qo, max_qo = r[0], r[1]

                r = [r for r in workload_conn.execute(f"SELECT a.attname FROM pg_attribute a, pg_class c WHERE a.attrelid = c.oid AND c.relname = '{workload_analysis_prefix}_mw_queries_args'")]
                r = [int(t[0][3:]) for t in r if t[0].startswith("arg")]
                max_arg = max(r)

            # Identify the relevant window boundaries.
            ddl_changes = compute_ddl_changes(input_dir, workload_analysis_prefix, workload_conn)
            qos_windows = set(range(min_qo, max_qo, query_feature_granularity_queries)) | set([q[0] for q in ddl_changes]) | set([max_qo + 1])
            qos_windows = sorted(list(qos_windows))

            # Populate keyspace features.
            populate_keyspace_features(workload_conn, workload_analysis_prefix, output_dir, ougc, qos_windows, histogram_width)

            # Populate all the table feature states.
            populate_table_feature_states(workload_conn, target_conn, workload_analysis_prefix, output_dir, forward_state, ougc, ddl_changes, wa.table_attr_map.keys(), qos_windows)

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

    if ougc.concurrency_model is None:
        # The output directory is the resolved directory.
        if (Path(output_dir) / "evals").exists():
            os.remove(f"{output_dir}/evals")
        os.symlink(f"{output_dir}/scratch/frames", f"{output_dir}/evals")
    else:
        (Path(output_dir) / "evals").mkdir(parents=True, exist_ok=True)
        # FIXME(TIME_SLICE): There is a reality where we should slice based on elapsed time.
        # However, that would require finding a way to estimate the number of concurrent
        # terminals in a time slice. It's possible that if you're given a [start_time]
        # you can compute the # overlapping...
        assert concurrency_mpi is not None

        # Generate the state/frame pairs and assert them.
        key_fn = lambda x: int(x.split("/")[-1].split(".")[0])
        states = sorted(glob.glob(f"{output_dir}/scratch/states/*.gz"), key=key_fn)
        options = sorted(glob.glob(f"{output_dir}/scratch/frames/*.feather"), key=key_fn)
        operations = [(s, o) for s, o in zip(states, options) if not Path(f"{output_dir}/scratch/concurrency/{key_fn(s)}.feather").exists()]
        for s, o in operations:
            assert key_fn(s) == key_fn(o)

        with joblib.parallel_backend(backend="loky", inner_max_num_threads=num_threads):
            with joblib.Parallel(n_jobs=num_cpus, verbose=10) as parallel:
                parallel(joblib.delayed(process_window_concurrency)(args, s, o) for s, o in operations)

        if (Path(output_dir) / "evals").exists():
            os.remove(f"{output_dir}/evals")
        os.symlink(f"{output_dir}/scratch/concurrency", f"{output_dir}/evals")


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

    table_feature_model_cls =cli.SwitchAttr(
        "--table-feature-model-cls",
        str,
        default="TableFeatureModel",
        help="Name of the table feature model class that should be instantiated.",
    )

    table_feature_model_path = cli.SwitchAttr(
        "--table-feature-model-path",
        Path,
        default=None,
        help="Path to the table feature model that should be loaded.",
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
        default="BufferPageModel",
        help="Name of the buffer page model class that should be instantiated.",
    )

    buffer_access_model_cls = cli.SwitchAttr(
        "--buffer-access-model-cls",
        str,
        default="BufferAccessModel",
        help="Name of the buffer access model class that should be instantiated.",
    )

    buffer_access_model_path = cli.SwitchAttr(
        "--buffer-access-model-path",
        Path,
        default=None,
        help="Path to the buffer access model that should be loaded.",
    )

    concurrency_model_cls = cli.SwitchAttr(
        "--concurrency-model-cls",
        str,
        default="ConcurrencyModel",
        help="Name of the concurrency model class that should be instantiated.",
    )

    concurrency_model_path = cli.SwitchAttr(
        "--concurrency-model-path",
        Path,
        default=None,
        help="Path to the concurrency model that should be loaded.",
    )

    concurrency_mpi = cli.SwitchAttr(
        "--concurrency-mpi",
        int,
        default=None,
        help="MPI of concurrency to use for evaluation.",
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

    def main(self):
        main(self.workload_analysis_conn, self.target_db_conn,
             self.workload_analysis_prefix,
             self.input_dir,
             self.session_sql,
             self.ou_models_path,
             self.query_feature_granularity_queries,
             self.table_feature_model_cls,
             self.table_feature_model_path,
             self.buffer_page_model_cls,
             self.buffer_page_model_path,
             self.buffer_access_model_cls,
             self.buffer_access_model_path,
             self.concurrency_model_cls,
             self.concurrency_model_path,
             self.concurrency_mpi,
             self.histogram_width,
             self.forward_state,
             self.use_plan_estimates,
             self.num_cpus, self.num_threads,
             self.output_dir)


if __name__ == "__main__":
    EvalQueryWorkloadCLI.run()
