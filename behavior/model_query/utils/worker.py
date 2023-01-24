import os
import copy
import math
import psycopg
from psycopg.rows import dict_row
import joblib
from pathlib import Path
import pandas as pd
import numpy as np

from behavior import OperatingUnit
from behavior.model_query.utils import OUGenerationContext, load_ou_models, load_model
from behavior.model_query.utils.query_ous import generate_query_ous_window
from behavior.model_query.utils.buffer_ous import compute_buffer_page_features, compute_buffer_access_features
from behavior.datagen.pg_collector_utils import _parse_field, KNOBS

from behavior.utils.evaluate_ou import evaluate_ou_model
from behavior.utils.prepare_ou_data import prepare_index_input_data
from behavior.utils.process_pg_state_csvs import (
    process_time_pg_attribute,
    process_time_pg_stats,
    process_time_pg_class,
    process_time_pg_index,
    build_time_index_metadata,
)

##################################################################################
# Attach metadata to query operating units
##################################################################################

def prepare_metadata(target_conn, state):
    with target_conn.cursor(row_factory=dict_row) as cursor:
        # Now prep pg_attribute.
        result = [r for r in cursor.execute("SELECT * FROM pg_attribute", prepare=False)]
        for r in result:
            r["time"] = 0.0
        time_pg_attribute = process_time_pg_attribute(pd.DataFrame(result))

        # FIXME(STATS): We assume that pg_stats doesn't change over time. Or more precisely, we know that the
        # attributes from pg_stats that we care about don't change significantly over time (len / key type).
        # If we start using n_distinct/correlation, then happy trials!
        result = [r for r in cursor.execute("SELECT * FROM pg_stats", prepare=False)]
        for r in result:
            r["time"] = 0.0
        time_pg_stats = process_time_pg_stats(pd.DataFrame(result))
        time_pg_stats.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
        time_pg_stats.sort_index(axis=0, inplace=True)

        # Extract all the relevant settings that we care about.
        cursor.execute("SHOW ALL;", prepare=False)
        pg_settings = {}
        for record in cursor:
            setting_name = record["name"]
            if setting_name in KNOBS:
                # Map a pg_setting name to the setting value.
                setting_type = KNOBS[setting_name]
                setting_str = record["setting"]
                pg_settings[setting_name] = _parse_field(setting_type, setting_str)

        # Now let's produce the pg_class and pg_index entries.
        table_feature_state = state.table_feature_state
        index_feature_state = state.index_feature_state
        result_cls = [r for r in cursor.execute("SELECT c.* FROM pg_class c, pg_namespace n WHERE c.relnamespace = n.oid AND n.nspname = 'public'", prepare=False)]
        result_idx = [r for r in cursor.execute("SELECT i.*, c.relname FROM pg_index i, pg_class c WHERE i.indrelid = c.oid", prepare=False)]
        pg_class_total = []
        pg_index_total = []
        pg_settings_total = []

        # Produce the pg_class entries.
        for record in result_cls:
            if "relname" in record and (record["relname"] in table_feature_state):
                relname = record["relname"]
                record["reltuples"] = table_feature_state[relname]["tuple_count"]
                record["relpages"] = table_feature_state[relname]["num_pages"]
                record["time"] = (state.window_index) * 1.0 * 1e6
                pg_class_total.append(copy.deepcopy(record))

            if record["oid"] in index_feature_state:
                record["reltuples"] = index_feature_state[record["oid"]]["tuple_count"]
                record["relpages"] = index_feature_state[record["oid"]]["num_pages"]
                record["time"] = (state.window_index) * 1.0 * 1e6
                pg_class_total.append(copy.deepcopy(record))

        # Produce the pg_index entries.
        for record in result_idx:
            if record["relname"] in table_feature_state and record["indexrelid"] in index_feature_state:
                entry = copy.deepcopy(record)
                entry.pop("relname")
                # Install the stats that we are convinced should be correct.
                entry["reltuples"] = index_feature_state[record["indexrelid"]]["tuple_count"]
                entry["relpages"] = index_feature_state[record["indexrelid"]]["num_pages"]
                entry["time"] = (state.window_index) * 1.0 * 1e6
                pg_index_total.append(entry)

        # Update the KNOBs
        pg_settings["time"] = state.window_index * 1.0 * 1e6
        pg_settings["unix_timestamp"] = state.window_index * 1.0
        for knob, value in state.knobs.items():
            pg_settings[knob] = value
        pg_settings_total.append(copy.deepcopy(pg_settings))

    # Construct the dataframes and process them accordingly.
    pg_settings = pd.DataFrame(pg_settings_total)
    pg_settings.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    pg_settings.sort_index(axis=0, inplace=True)

    pg_index_total = pd.DataFrame(pg_index_total)
    pg_class_total = pd.DataFrame(pg_class_total)

    # Prepare all the augmented catalog data in timestamp order.
    process_tables, process_idxs = process_time_pg_class(pg_class_total)
    process_pg_index = process_time_pg_index(pg_index_total)
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
        data["window_index"] = data.window_index.astype(float)
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
        data = prepare_index_input_data(data, nodrop=True, separate_indkey_features=None)
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
# Mind-trick postgres about what's actually in the tables.
##################################################################################

def implant_stats_to_postgres(target_conn, ougc):
    target_conn.execute("SELECT qss_clear_stats()", prepare=False)
    def implant_stat(name, data):
        relpages = int(data["num_pages"])
        reltuples = data["tuple_count"]

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
# Control routine for processing an OU.
##################################################################################

ou_state = None
target_conn = None
workload_conn = None

def process_window_ous(args, state_file):
    global ou_state
    global target_conn
    global workload_conn
    output_dir = args["output_dir"]

    # Create the OUGenerationContext
    if ou_state is None:
        ou_state = OUGenerationContext()
        ou_state.ou_models = load_ou_models(args["ou_models_path"])
        ou_state.table_feature_model = load_model(args["table_feature_model_path"], args["table_feature_model_cls"])
        ou_state.buffer_page_model = load_model(args["buffer_page_model_path"], args["buffer_page_model_cls"])
        ou_state.restore_state(joblib.load(state_file))

        # Load the table keyspace features. This can all be loaded only once.
        ou_state.table_keyspace_features = {}
        for t in ou_state.tables:
            if (Path(output_dir) / f"scratch/keyspaces/{t}.feather").exists():
                ou_state.table_keyspace_features[t] = pd.read_feather(f"{output_dir}/scratch/keyspaces/{t}.feather")
    else:
        ou_state.restore_state(joblib.load(state_file))

    if target_conn is None:
        target_conn = psycopg.connect(args["target_db_conn"], prepare_threshold=None)
        target_conn.execute("SET qss_capture_enabled = OFF")
        target_conn.execute("SET plan_cache_mode = 'force_generic_plan'")
        target_conn.execute("CREATE EXTENSION IF NOT EXISTS pgstattuple")
        target_conn.execute("CREATE EXTENSION IF NOT EXISTS qss")
        if args["session_sql"] is not None and args["session_sql"].exists():
            session_sql = args["session_sql"]
            with open(session_sql, "r") as f:
                for line in f:
                    target_conn.execute(line)

    workload_analysis_prefix = args["workload_analysis_prefix"]
    if workload_conn is None:
        workload_conn = psycopg.connect(args["workload_analysis_conn"], prepare_threshold=None)

    # Reset and Implant the stats.
    implant_stats_to_postgres(target_conn, ou_state)

    # Get the relevant plans for the window.
    window_i = ou_state.window_index
    current_qo = ou_state.current_qo
    upper_qo = ou_state.upper_qo
    query_plans = workload_conn.execute(f"SELECT * FROM {workload_analysis_prefix}_mw_eval_analysis WHERE query_order >= {current_qo} AND query_order < {upper_qo}", prepare=False)
    columns = [c.name for c in query_plans.description]
    query_plans = pd.DataFrame([r for r in query_plans], columns=columns)

    # Generate the OUs for the queries.
    use_plan_estimates = args["use_plan_estimates"]
    query_ous = generate_query_ous_window(target_conn, ou_state, window_i, query_plans, use_plan_estimates, output_dir)

    # Pass to buffer page and buffer access model to get buffer hits/misses
    compute_buffer_page_features(ou_state, query_ous)
    compute_buffer_access_features(ou_state, query_ous, window_i, query_plans.shape[0])

    # Group all the OUs based on Type.
    ous_keyed = {o.name: [] for o in OperatingUnit}
    for ou in query_ous:
        ous_keyed[ou["node_type"]].append(ou)
    del query_ous
    # Prune out the empty ones.
    ous_keyed = {o: pd.DataFrame(ous_keyed[o]) for o in ous_keyed}
    ous_keyed = {o: v for o, v in ous_keyed.items() if v.shape[0] > 0}

    # Attach metadata on demand.
    process_tbls, time_pg_index, time_pg_stats, pg_settings = prepare_metadata(target_conn, ou_state)
    attach_metadata_ous(ous_keyed, process_tbls, time_pg_index, time_pg_stats, pg_settings)

    # Generate the prediction results for this set of OUs.
    ous_evals = []
    for ou_name, df in ous_keyed.items():
        if ou_name not in ou_state.ou_models:
            # If we don't have the model for the particular OU, we just predict 0.
            df["pred_elapsed_us"] = 0
                # Set a bit in [error_missing_model]
            df["error_missing_model"] = 1
        else:
            df = evaluate_ou_model(ou_state.ou_models[ou_name], None, None, eval_df=df, return_df=True, output=False)
            df["error_missing_model"] = 0

            if OperatingUnit[ou_name] == OperatingUnit.IndexOnlyScan or OperatingUnit[ou_name] == OperatingUnit.IndexScan:
                prefix = "IndexOnlyScan" if OperatingUnit[ou_name] == OperatingUnit.IndexOnlyScan else "IndexScan"
                df["pred_elapsed_us"] = df.pred_elapsed_us * df[f"{prefix}_num_outer_loops"]

        (Path(output_dir) / f"scratch/ous/{ou_name}").mkdir(parents=True, exist_ok=True)
        df.to_feather(f"{output_dir}/scratch/ous/{ou_name}/{window_i}.feather")
        columns = ["query_id", "query_order", "pred_elapsed_us", "error_missing_model"]
        if "total_blks_requested" in df:
            columns += ["total_blks_requested"]
        ous_evals.append(df[columns])

    # Construct the query level predictions.
    ous_evals = pd.concat(ous_evals, ignore_index=True)
    query_evals = ous_evals.groupby(["query_id", "query_order"]).sum()
    query_evals.reset_index(drop=False, inplace=True)
    query_evals.sort_values(by=["query_id", "query_order"], inplace=True, ignore_index=True)
    query_evals.set_index(keys=["query_id", "query_order"], inplace=True)
    # Resolve the query level predictions with their original frame.
    query_plans.set_index(keys=["query_id", "query_order"], inplace=True)
    query_plans = query_plans.join(query_evals, how="inner")
    query_plans.reset_index(drop=False, inplace=True)
    assert np.sum(query_plans.pred_elapsed_us.isna()) == 0
    query_plans.to_feather(f"{output_dir}/scratch/frames/{window_i}.feather.tmp")
    # Atomic rename to indicate done.
    os.rename(f"{output_dir}/scratch/frames/{window_i}.feather.tmp", f"{output_dir}/scratch/frames/{window_i}.feather")
