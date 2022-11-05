import math
import random
import pglast
from tqdm import tqdm
import logging
import glob
import shutil
import gc
import re
import psycopg
import copy
from psycopg.rows import dict_row
from distutils import util
import json
from datetime import datetime
import numpy as np
import itertools
import functools
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
from pathlib import Path
from plumbum import cli

from behavior import OperatingUnit, Targets, BENCHDB_TO_TABLES
from behavior.datagen.pg_collector_utils import _parse_field, KNOBS
from behavior.utils.evaluate_ou import evaluate_ou_model
from behavior.utils.prepare_ou_data import prepare_index_input_data
from behavior.model_workload.utils import keyspace_metadata_read
from behavior.utils.process_pg_state_csvs import (
    postgres_julian_to_unix,
    process_time_pg_stats,
    process_time_pg_attribute,
    process_time_pg_index,
    process_time_pg_class,
    build_time_index_metadata
)
import behavior.model_workload.models as model_workload_models
from behavior.model_workload.utils.keyspace_feature import construct_keyspaces
from behavior.model_query.utils.query_ous import generate_query_ous_window
from behavior.model_query.utils.table_state import (
    initial_trigger_metadata,
    initial_table_feature_state,
    refresh_table_fillfactor,
    initial_index_feature_state,
    compute_table_exec_features,
)


class OUGenerationContext:
    tables = None
    table_attr_map = None
    table_keyspace_features = None
    table_feature_state = None
    trigger_info_map = None
    oid_table_map = None
    index_feature_state = None
    indexoid_table_map = None
    table_indexoid_map = None
    shared_buffers = None

    ou_models = None
    table_feature_model = None
    buffer_page_model = None
    buffer_access_model = None
    concurrency_model = None

    def save_state(self, window_index):
        return {
            "window_index": window_index,
            "table_attr_map": table_attr_map,
            "table_feature_state": table_feature_state.copy(),
            "trigger_info_map": trigger_info_map,
            "oid_table_map": oid_table_map,
            "index_feature_state": index_feature_state.copy(),
            "indexoid_table_map": indexoid_table_map,
            "table_indexoid_map": table_indexoid_map,
            "shared_buffers": shared_buffers,
        }

    def restore_state(self, state):
        self.table_attr_map = state["table_attr_map"]
        self.table_feature_state = state["table_feature_state"]
        self.trigger_info_map = state["trigger_info_map"]
        self.oid_table_map = state["oid_table_map"]
        self.index_feature_state = state["index_feature_state"]
        self.indexoid_table_map = state["indexoid_table_map"]
        self.table_indexoid_map = state["table_indexoid_map"]
        self.shared_buffers = state["shared_buffers"]


logger = logging.getLogger(__name__)

##################################################################################
# Attach metadata to query operating units
##################################################################################

def prepare_metadata(target_conn, states_dir):
    with target_conn.cursor(row_factory=dict_row) as cursor:
        # Now prep pg_attribute.
        result = [r for r in cursor.execute("SELECT * FROM pg_attribute")]
        for r in result:
            r["time"] = 0.0
        time_pg_attribute = process_time_pg_attribute(pd.DataFrame(result))

        # FIXME(STATS): We assume that pg_stats doesn't change over time. Or more precisely, we know that the
        # attributes from pg_stats that we care about don't change significantly over time (len / key type).
        # If we start using n_distinct/correlation, then happy trials!
        result = [r for r in cursor.execute("SELECT * FROM pg_stats")]
        for r in result:
            r["time"] = 0.0
        time_pg_stats = process_time_pg_stats(pd.DataFrame(result))
        time_pg_stats.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
        time_pg_stats.sort_index(axis=0, inplace=True)

        # Extract all the relevant settings that we care about.
        cursor.execute("SHOW ALL;")
        pg_settings = {}
        for record in cursor:
            setting_name = record["name"]
            if setting_name in KNOBS:
                # Map a pg_setting name to the setting value.
                setting_type = KNOBS[setting_name]
                setting_str = record["setting"]
                pg_settings[setting_name] = _parse_field(setting_type, setting_str)

        # Now let's produce the pg_class entries.
        key_fn = lambda x: int(x.split(".")[0])
        states = sorted(glob.glob(f"{states_dir}/*.gz"), key=key_fn)

        result_cls = [r for r in cursor.execute("SELECT c.* FROM pg_class c, pg_namespace n WHERE c.relnamespace = n.oid AND n.nspname = 'public'")]
        result_idx = [r for r in cursor.execute("SELECT i.*, c.relname FROM pg_index i, pg_class c WHERE i.indexrelid = c.oid")]
        pg_class_total = []
        pg_index_total = []
        pg_settings_total = []
        for state in states:
            window_i_state = joblib.load(state)
            table_feature_state = window_i_state["table_feature_state"]
            index_feature_state = window_i_state["index_feature_state"]

            slice_num = key_fn(state)
            for record in result_cls:
                if "relname" in record and (record["relname"] in table_feature_state):
                    relname = record["relname"]
                    record["reltuples"] = table_feature_state[relname]["approx_tuple_count"]
                    record["relpages"] = table_feature_state[relname]["num_pages"]
                    record["time"] = (slice_num) * 1.0 * 1e6
                    pg_class_total.append(copy.deepcopy(record))

            for record in result_idx:
                if record["relname"] in table_feature_state and record["indexrelid"] in index_feature_state:
                    entry = copy.deepcopy(record)
                    entry.pop("relname")
                    entry["reltuples"] = index_feature_state[record["indexrelid"]]["approx_tuple_count"]
                    entry["relpages"] = index_feature_state[record["indexrelid"]]["num_pages"]
                    entry["time"] = (slice_num) * 1.0 * 1e6
                    pg_index_total.append(entry)

            # Update the KNOBs
            pg_settings["time"] = slice_num * 1.0 * 1e6
            pg_settings["unix_timestamp"] = slice_num
            pg_settings["shared_buffers"] = window_i_state["shared_buffers"]
            pg_settings_total.append(copy.deepcopy(pg_settings))

    pg_settings = pd.DataFrame(pg_settings_total)
    pg_settings.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    pg_settings.sort_index(axis=0, inplace=True)

    # Prepare all the augmented catalog data in timestamp order.
    process_tables, process_idxs = process_time_pg_class(pd.DataFrame(pg_class_total))
    process_pg_index = process_time_pg_index(pd.DataFrame(pg_index_total))
    time_pg_index = build_time_index_metadata(process_pg_index, process_tables.copy(deep=True), process_idxs, time_pg_attribute)
    return process_tables, time_pg_index, time_pg_stats, pg_settings


def attach_metadata_ous(ous_keyed, process_tables, process_index, process_stats, process_settings):
    # These are the INDEX OUs that require metadata augmentation.
    for index_ou in [OperatingUnit.IndexScan, OperatingUnit.IndexOnlyScan, OperatingUnit.ModifyTableIndexInsert]:
        if index_ou.name not in ous_keyed:
            continue

        column = {
            OperatingUnit.IndexOnlyScan: "IndexOnlyScan_indexid",
            OperatingUnit.IndexScan: "IndexScan_indexid",
            OperatingUnit.ModifyTableIndexInsert: "ModifyTableIndexInsert_indexid"
        }[index_ou]

        data = ous_keyed[index_ou.name]

        # This is super confusing but essentially "time" is the raw time. `unix_timestamp` is the time adjusted to
        # the correct unix_timestamp seconds. Essentially we want to join the unix_timestamp to window_slice
        # which is how the code is setup.
        data["window_index"] = data.window_index.astype(np.float)
        data.set_index(keys=["window_index"], drop=True, append=False, inplace=True)
        data.sort_index(axis=0, inplace=True)

        settings_col = process_settings.columns[0]
        data = pd.merge_asof(data, process_settings, left_index=True, right_index=True, allow_exact_matches=True)
        # This guarantees that all the settings are matched up.
        assert data[settings_col].isna().sum() == 0

        data = pd.merge_asof(data, process_index, left_index=True, right_index=True, left_by=[column], right_by=["indexrelid"], allow_exact_matches=True)
        # This guarantees that all the indexes are matched up.
        assert data.indexrelid.isna().sum() == 0

        indkey_atts = [key for key in data.columns if "indkey_attname_" in key]
        for idx, indkey_att in enumerate(indkey_atts):
            left_by = ["table_relname", indkey_att]
            right_by = ["tablename", "attname"]
            data = pd.merge_asof(data, process_stats, left_index=True, right_index=True, left_by=left_by, right_by=right_by, allow_exact_matches=True)

            # Rename the key and drop the other useless columns.
            data.drop(labels=["tablename", "attname"], axis=1, inplace=True)
            remapper = {column:f"indkey_{column}_{idx}" for column in process_stats.columns}
            data.rename(columns=remapper, inplace=True)

        # Purify the index data.
        data = prepare_index_input_data(data)
        data.reset_index(drop=True, inplace=True)
        ous_keyed[index_ou.name] = data

    process_tables.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    process_tables.sort_index(axis=0, inplace=True)
    for augment in [OperatingUnit.ModifyTableInsert, OperatingUnit.ModifyTableUpdate]:
        if augment.name not in ous_keyed:
            continue

        data = ous_keyed[augment.name]
        data["window_index"] = data.window_index.astype(np.float)
        data.set_index(keys=["window_index"], drop=True, append=False, inplace=True)
        data.sort_index(axis=0, inplace=True)

        data = pd.merge_asof(data, process_tables, left_index=True, right_index=True, left_by=["ModifyTable_target_oid"], right_by=["oid"], allow_exact_matches=True)
        assert data.oid.isna().sum() == 0
        data.reset_index(drop=False, inplace=True)
        ous_keyed[augment.name] = data

##################################################################################
# Evaluation of query plans
##################################################################################

def evaluate_query_ous(ougc, evals_dir, query_plans_dir, ous_dir, process_tbls, time_pg_index, time_pg_stats, pg_settings):
    start_window = 0
    files = sorted(glob.glob(f"{evals_dir}/resolved/*.gz"), key=lambda x: int(x.split(".")[0]))
    if len(files) > 0:
        start_window = int(Path(files[-1]).stem.split(".")[0])

    while True:
        if not Path(f"{query_plans_dir}/{start_window}.gz").exists():
            # We are done now.
            break

        query_frame = joblib.load(f"{query_plans_dir}/{start_window}.gz")

        # Shuffle OU dict into per OU-type
        ous = joblib.load(f"{ous_dir}/{start_window}.gz")
        ous_keyed = {o.name: [] for o in OperatingUnit}
        for ou in ous:
            ous_keyed[ou.node_type].append(ou)
        del ous
        ous_keyed = {o: pd.DataFrame(ous_keyed[o]) for o in ous_keyed}
        ous_evals = []

        # Attach metadata on demand.
        attach_metadata_ous(ous_keyed, process_tbls, time_pg_index, time_pg_stats, pg_settings)

        for ou_name, df in ous_keyed.items():
            if ou_name not in ougc.ou_models:
                # If we don't have the model for the particular OU, we just predict 0.
                df["pred_elapsed_us"] = 0
                # Set a bit in [error_missing_model]
                df["error_missing_model"] = 1
            else:
                df = evaluate_ou_model(ougc.models[ou_name], None, None, eval_df=df, return_df=True, output=False)
                df["error_missing_model"] = 0

                if OperatingUnit[ou_type] == OperatingUnit.IndexOnlyScan or OperatingUnit[ou_type] == OperatingUnit.IndexScan:
                    prefix = "IndexOnlyScan" if OperatingUnit[ou_type] == OperatingUnit.IndexOnlyScan else "IndexScan"
                    df["pred_elapsed_us"] = df.pred_elapsed_us * df[f"{prefix}_num_outer_loops"]
            df.to_feather(f"{evals_dir}" / ou_name / f"{start_window}.feather")

            columns = ["query_id", "query_order", "pred_elapsed_us", "error_missing_model"]
            if "total_blks_requested" in df:
                columns += ["total_blks_requested"]
            ous_evals.append(df[columns])

        ous_evals = pd.concat(ous_evals, ignore_index=True)
        query_evals = ous_evals.groupby(["query_id", "query_order"]).sum()
        query_evals.reset_index(drop=False, inplace=True)
        query_evals.sort_values(by=["query_id", "query_order"], inplace=True, ignore_index=True)
        query_evals.set_index(keys=["query_id", "query_order"], inplace=True)

        query_frame.set_index(keys=["query_id", "query_order"], inplace=True)
        query_frame = query_frame.join(query_evals, how="inner")
        query_frame.reset_index(drop=False, inplace=True)
        assert np.sum(query_frame.pred_elapsed_us.isna()) == 0
        joblib.dump(query_frame, f"{evals_dir}/resolved/{start_window}.gz")

##################################################################################
# Mind-trick postgres about what's actually in the tables.
##################################################################################

def implant_stats_to_postgres(target_conn, ougc):
    target_conn.execute("SELECT qss_clear_stats()", prepare=False)
    def implant_stat(name, data):
        relpages = int(math.ceil(data["table_len"] / 8192))
        reltuples = data["approx_tuple_count"]

        # FIXME(INDEX): This is a back of the envelope estimation for the index height.
        height = 0
        if data["tuple_len_avg"] > 0:
            fanout = (8192 / data["tuple_len_avg"])
            height = math.ceil(np.log(relpages) / np.log(fanout))

        # FIXME(STATS): Should we try and fake the histogram?
        query = f"SELECT qss_install_stats('{name}', {relpages}, {reltuples}, {height})"
        target_conn.execute(query, prepare=False)

    for tbl, tbl_state in ougc.table_feature_state.items():
        implant_stat(tbl, tbl_state)

    for _, idx_state in ougc.index_feature_state.items():
        implant_stat(idx_state["indexname"], idx_state)

##################################################################################
# Compute the DDL Changes
##################################################################################

def compute_ddl_changes(dir_data, workload_prefix, workload_conn):
    ddl = pd.read_csv(f"{dir_data}/pg_qss_ddl.csv")
    ddl = ddl[(ddl.command == "AlterTableOptions") | (ddl.command == "CreateIndex") | (ddl.command == "DropIndex")]
    ddl["unix_timestamp"] = postgres_julian_to_unix(ddl.statement_timestamp)
    ddl.reset_index(drop=True, inplace=True)
    if ddl.shape[0] == 0:
        return []

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
# Load the Models
##################################################################################

def load_ou_models(path):
    model_dict = {}
    for model_path in path.rglob('*.pkl'):
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        model_dict[model.ou_name] = model
    return model_dict


def load_model(path, name):
    if path is None:
        return None

    model_cls = getattr(model_workload_models, name)
    return model_cls.load_model(path)


def main(workload_conn, target_conn, workload_analysis_prefix,
         input_dir, session_sql, ou_models_path,
         query_feature_granularity_queries,
         table_feature_model_path, buffer_page_model_path,
         buffer_access_model_path, concurrency_model_path,
         concurrency_granularity_sec,
         concurrency_mpi,
         histogram_width,
         output_dir):

    logger.info("Beginning eval_query_workload evaluation of %s", input_dir)
    target_conn.execute("SET qss_capture_enabled = OFF")
    target_conn.execute("SET plan_cache_mode = 'force_generic_plan'")
    target_conn.execute("CREATE EXTENSION IF NOT EXISTS pgstattuple")
    if session_sql is not None and session_sql.exists():
        with open(session_sql, "r") as f:
            for line in f:
                target_conn.execute(line)

    ougc = OUGenerationContext()
    ougc.ou_models = load_ou_models(ou_models_path)
    ougc.table_feature_model = load_model(table_feature_model_path, "TableFeatureModel")
    ougc.buffer_page_model = load_model(buffer_page_model_path, "BufferPageModel")
    ougc.buffer_access_model = load_model(buffer_access_model_path, "BufferAccessModel")
    ougc.concurrency_model = load_model(concurrency_model_path, "ConcurrencyModel")

    # Get the query order ranges.
    with workload_conn.transaction():
        r = [r for r in workload_conn.execute(f"SELECT min(query_order), max(query_order) FROM {workload_analysis_prefix}_mw_queries_args")][0]
        min_qo, max_qo = r[0], r[1]

        r = [r for r in workload_conn.execute(f"SELECT a.attname FROM pg_attribute a, pg_class c WHERE a.attrelid = c.oid AND c.relname = '{workload_analysis_prefix}_mw_queries_args'")]
        r = [int(t[0][3:]) for t in r if t[0].startswith("arg")]
        max_arg = max(r)

    wa = keyspace_metadata_read(input_dir)[0]
    ougc.tables = wa.table_attr_map.keys()
    ougc.table_attr_map = wa.table_attr_map

    shared_buffers = [r for r in target_conn.execute("SHOW shared_buffers")][0][0]
    ddl_changes = compute_ddl_changes(input_dir, workload_analysis_prefix, workload_conn)
    qos_windows = set(range(min_qo, max_qo, query_feature_granularity_queries)) | set([q[0] for q in ddl_changes])
    qos_windows = sorted(list(qos_windows))

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
        construct_keyspaces(logger, workload_conn, workload_analysis_prefix, tables, wa.table_attr_map, window_index_map, histogram_width, callback_fn=callback)
        open(f"{output_dir}/scratch/keyspaces/done", "w").close()

    # Load the table keyspace features.
    ougc.table_keyspace_features = {}
    for t in ougc.tables:
        if (Path(output_dir) / f"scratch/keyspaces/{t}.feather").exists():
            ougc.table_keyspace_features[t] = pd.read_feather(f"{output_dir}/scratch/keyspaces/{t}.feather")

    initial_trigger_metadata(target_conn, ougc)
    initial_table_feature_state(target_conn, ougc)
    initial_index_feature_state(target_conn, ougc)

    # Get all the query plans in this window and the hits count.
    joins = []
    valid_tbls = [t for t in ougc.tables if (Path(output_dir) / f"scratch/keyspaces/{t}.feather").exists()]
    for tbl in valid_tbls:
        joins.append(f"LEFT JOIN LATERAL (SELECT COUNT(1) as hits FROM {workload_analysis_prefix}_{tbl}_hits {tbl} WHERE {tbl}.query_order = a.query_order) {tbl} ON true")
    query_columns = ["query_order", "query_id", "statement_timestamp", "optype", "query_text", "target", "elapsed_us"] 
    query_columns += [f"arg{i}" for i in range(1, max_arg+1)]
    current_qo = 1

    # Handle state save/restore.
    scratch = Path(output_dir) / "scratch/states"
    def accumulate_state_tuple(query_frame, window_i_state, query_ous):
        wi = window_i_state["window_index"]
        joblib.dump(window_i_state, f"{output_dir}/scratch/states/{wi}.gz")
        joblib.dump(query_frame, f"{output_dir}/scratch/frames/{wi}.gz")
        joblib.dump(query_ous, f"{output_dir}/scratch/ous/{wi}.gz")

    files = sorted(glob.glob(f"{output_dir}/scratch/states/*.gz"), key=lambda x: int(x.split(".")[0]))
    if len(files) > 0:
        window_i_state = joblib.load(files[-1])
        current_qo = window_i_state["window_index"] + 1
        ougc.restore_state(window_i_state)

    for i, current_qo in enumerate(qos_windows):
        upper_qo = current_qo
        if i != len(qos_windows) - 1:
            upper_qo = qos_windows[i+1]

        # If it is time to execute a DDL statement, then execute the DDL statement.
        for (ddl_qo, ddl_stmt) in ddl_changes:
            if current_qo < ddl_qo:
                break

            if current_qo == ddl_qo:
                # Reload the index feature state to acquire newly created indexes.
                # This will also catch any deleted indexes.
                target_conn.execute(ddl_stmt)
                refresh_table_fillfactor(target_conn, ougc)

                # FIXME(INDEX): We are under the assumption that we don't need to compute keyspaces.
                # But we possibly might have to refresh for the keyspaces.
                initial_index_feature_state(target_conn, ougc)
                ougc.shared_buffers = [r for r in target_conn.execute("SHOW shared_buffers")][0][0]

        # Reset and Implant the stats.
        implant_stats_to_postgres(target_conn, ougc)

        # Get this window's queries.
        sql = """
            SELECT {tbl_outputs}
            FROM {prefix}_mw_queries_args a
            {joins}
            WHERE a.query_order >= {lower_qo} AND a.query_order < {upper_qo} ORDER BY a.query_order
        """.format(prefix=workload_analysis_prefix,
                   tbl_outputs=",".join([f"a.{c}" for c in query_columns] + [f"{t}.hits as \"{t}_hits\"" for t in valid_tbls]),
                   joins="\n".join(joins),
                   lower_qo=current_qo,
                   upper_qo=upper_qo)
        query_plans = pd.DataFrame([r for r in workload_conn.execute(sql)], columns=query_columns + [f"{t}_hits" for t in valid_tbls])

        # Get the table execution features using the window.
        compute_table_exec_features(ougc, query_plans, i)
        window_i_state = ougc.save_state(i)

        # Generate the OUs for the queries.
        query_ous = generate_query_ous_window(logger, target_conn, ougc, i, query_plans, output_dir)

        # Pass to buffer page and buffer access model to get buffer hits/misses
        compute_buffer_page_features(ougc, query_ous)
        compute_buffer_access_features(ougc, query_ous, i, query_plans.shape[0])

        # Compute the next stats incarnation
        compute_next_window_state(ougc, query_ous)

        # Accumulate and save the state.
        accumulate_state_tuple(query_plans, window_i_state, query_ou)

    # Create all the metadata frames from the incremental state.
    process_tbls, time_pg_index, time_pg_stats, pg_settings = prepare_metadata(target_conn, f"{output_dir}/scratch/states/")

    # Join metadata state on-demand, inference, produce query results
    evaluate_query_ous(ougc, f"{output_dir}/scratch/evals", f"{output_dir}/scratch/frames", f"{output_dir}/scratch/ous", process_tbls, time_pg_index, time_pg_stats, pg_settings)

    if ougc.concurrency_model is not None:
        assert concurrency_mpi is not None
        # TODO slice based on time to get query orders? alternative use a given MPI and just take the windows we are given
        key = lambda x: int(x.split(".")[0])
        window_index = 0
        files = sorted(glob.glob(f"{output_dir}/scratch/concurrency/*.gz"), key=key)
        if len(files) > 0:
            window_index = key(files[-1]) + 1

        while window_index < len(qos_windows):
            window_state = joblib.load(f"{output_dir}/scratch/states/{window_index}.gz")
            query_frame = joblib.load(f"{output_dir}/scratch/evals/resolved/{key(window_file)}.gz")

            outputs = ougc.concurrency_model.inference(
                window_index,
                concurrency_mpi,
                query_frame,
                window_state["table_feature_state"],
                window_state["table_attr_map"],
                ougc.table_keyspace_features)

            query_frame = ougc.concurrency_model.bias(outputs, query_frame)
            joblib.dump(query_frame, f"{output_dir}/scratch/concurrency/{window_index}.gz")
            window_index += 1
    else:
        # The output directory is the resolved directory.
        os.symlink(f"{output_dir}/evals", f"{output_dir}/scratch/evals/resolved")


class EvalQueryWorkloadCLI(cli.Application):
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

    buffer_access_model_path = cli.SwitchAttr(
        "--buffer-access-model-path",
        Path,
        default=None,
        help="Path to the buffer access model that should be loaded.",
    )

    concurrency_model_path = cli.SwitchAttr(
        "--concurrency-model-path",
        Path,
        default=None,
        help="Path to the concurrency model that should be loaded.",
    )

    concurrency_granularity_sec = cli.SwitchAttr(
        "--concurrency-granularity-secs",
        float,
        default=1,
        help="Granularity of concurrency window in seconds.",
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

    def main(self):
        with psycopg.connect(self.workload_analysis_conn, autocommit=True) as workload_conn:
            with psycopg.connect(self.target_db_conn, autocommit=True) as target_conn:
                main(workload_conn, target_conn,
                     self.workload_analysis_prefix,
                     self.input_dir,
                     self.session_sql,
                     self.ou_models_path,
                     self.query_feature_granularity_queries,
                     self.table_feature_model_path,
                     self.buffer_page_model_path,
                     self.buffer_access_model_path,
                     self.concurrency_model_path,
                     self.concurrency_granularity_sec,
                     self.concurrency_mpi,
                     self.histogram_width,
                     self.output_dir)


if __name__ == "__main__":
    EvalQueryWorkloadCLI.run()
