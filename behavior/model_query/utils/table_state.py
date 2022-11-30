import numpy as np
import torch
import re
from math import floor
from psycopg.rows import dict_row
from behavior import OperatingUnit
from behavior.model_workload.utils import OpType
from behavior.model_workload.models.table_feature_model import MODEL_WORKLOAD_TARGETS
from behavior.model_workload.models.table_state_model import STATE_WORKLOAD_TARGETS
from behavior.model_query.utils.query_ous import mutate_index_state_will_extend


def initial_trigger_metadata(target_conn, ougc):
    """
    Get all the relevant trigger information from the target database.
    This is all the information about the relevant table names and attributes.
    """
    with target_conn.cursor(row_factory=dict_row) as cursor:
        result = cursor.execute("""
            SELECT t.oid as "pg_trigger_oid", t.tgfoid, c.contype, c.confrelid, c.confupdtype, c.confdeltype, c.conkey, c.confkey, c.conpfeqop
            FROM pg_trigger t, pg_constraint c
            JOIN pg_namespace n ON c.connamespace = n.oid and n.nspname = 'public'
            WHERE t.tgconstraint = c.oid
        """, prepare=False)

        triggers = {x["pg_trigger_oid"]: x for x in result}

        result = cursor.execute("""
            SELECT * FROM pg_attribute
            JOIN pg_class c ON pg_attribute.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid and n.nspname = 'public'
        """, prepare=False)

        atts = {(x["attrelid"], x["attnum"]): x for x in result}
        atttbls = [x for (x, _) in atts.keys()]

        for _, trigger in triggers.items():
            attnames = []
            if trigger["confrelid"] in atttbls:
                for attnum in trigger["confkey"]:
                    attname = atts[(trigger["confrelid"], attnum)]["attname"]
                    attnames.append(attname)
            trigger["attnames"] = attnames
    ougc.trigger_info_map = triggers


def initial_table_feature_state(target_conn, ougc, approx_stats):
    """
    Compute the initial state information from the target database.
    This is the state that we pass forward to the table features model.
    And also warp into future states.
    """
    table_feature_state = {}
    oid_table_map = {}

    # Here we assume that the "initial" state with which populate_data() was invoked with
    # and the state at which the query stream was captured is roughly consistent (along
    # with the state at which we are assessing pgstattuple_approx) to some degree.
    with target_conn.cursor(row_factory=dict_row) as cursor:
        for tbl in ougc.tables:
            func = "pgstattuple_approx" if approx_stats else "pgstattuple"
            result = [r for r in cursor.execute(f"SELECT * FROM {func}('{tbl}')", prepare=False)][0]
            pgc_record = [r for r in cursor.execute(f"SELECT * FROM pg_class where relname = '{tbl}'", prepare=False)][0]

            ff = 1.0
            if pgc_record["reloptions"] is not None:
                for record in pgc_record["reloptions"]:
                    for key, value in re.findall(r'(\w+)=(\w*)', record):
                        if key == "fillfactor":
                            ff = float(value) / 100.0
                            break

            oid_table_map[pgc_record["oid"]] = tbl
            table_feature_state[tbl] = {
                "num_pages": result["table_len"] / 8192.0,
                "free_percent": result["approx_free_percent"] if approx_stats else result["free_percent"],
                "dead_tuple_percent": result["dead_tuple_percent"],
                "tuple_count": result["approx_tuple_count"] if approx_stats else result["tuple_count"],
                "tuple_len_avg": (result["approx_tuple_len"] / result["approx_tuple_count"]) if approx_stats else (result["tuple_len"] / result["tuple_count"]),
                "target_ff": ff,
            }
    ougc.table_feature_state = table_feature_state
    ougc.oid_table_map = oid_table_map


def resolve_index_feature_state(target_conn, ougc):
    """
    Build information about all the indexes in the target database.
    This rolls forward the prior state if it already exists, otherwise
    constructs and installs the new state.
    """
    new_index_feature_state = {}
    new_indexoid_table_map = {}
    new_table_indexoid_map = {}
    with target_conn.cursor(row_factory=dict_row) as cursor:
        for tbl in ougc.tables:
            sql = """
                SELECT indexrelid, i.relname as "indrelname", t.relname as "tblname"
                FROM pg_index, pg_class i, pg_class t
                WHERE pg_index.indrelid = t.oid AND pg_index.indexrelid = i.oid
                  AND t.relname = '{tbl}'
            """.format(tbl=tbl)
            pgi_record = [r for r in cursor.execute(sql)]
            for pgi_rec in pgi_record:
                indexrelid = pgi_rec["indexrelid"]
                indname = pgi_rec["indrelname"]
                tblname = pgi_rec["tblname"]
                if ougc.index_feature_state is None or indexrelid not in ougc.index_feature_state:
                    result = [r for r in cursor.execute(f"SELECT * FROM pgstattuple('{indname}')", prepare=False)][0]
                    new_index_feature_state[indexrelid] = {
                        "indexname": indname,
                        "num_pages": result["table_len"] / 8192.0,
                        "tuple_count": result["tuple_count"],
                        "tuple_len_avg": 0.0 if result["tuple_count"] == 0 else result["tuple_len"] / result["tuple_count"],
                        "num_inserts": 0,
                    }
                else:
                    new_index_feature_state[indexrelid] = ougc.index_feature_state[indexrelid]

                if tblname not in new_table_indexoid_map:
                    new_table_indexoid_map[tblname] = []
                new_table_indexoid_map[tblname].append(indexrelid)
                new_indexoid_table_map[indexrelid] = tblname
    ougc.index_feature_state = new_index_feature_state
    ougc.indexoid_table_map = new_indexoid_table_map
    ougc.table_indexoid_map = new_table_indexoid_map


def refresh_table_fillfactor(target_conn, ougc):
    """
    Get the new fill factor setting for all the tables.
    """
    with target_conn.cursor(row_factory=dict_row) as cursor:
        for tbl in ougc.tables:
            if tbl not in ougc.table_feature_state:
                continue

            pgc_record = [r for r in cursor.execute(f"SELECT * FROM pg_class where relname = '{tbl}'", prepare=False)][0]

            ff = 1.0
            if pgc_record["reloptions"] is not None:
                for record in pgc_record["reloptions"]:
                    for key, value in re.findall(r'(\w+)=(\w*)', record):
                        if key == "fillfactor":
                            ff = float(value) / 100.0
                            break

            ougc.table_feature_state[tbl]["target_ff"] = ff


def compute_table_exec_features(ougc, tbl_summaries, window, output_df=False):
    """
    Computes the table execution features.
    """

    for tbl_summary in tbl_summaries:
        # update the table metadata with what happens in this window.
        ougc.table_feature_state[tbl_summary["target"]]["num_select_tuples"] = tbl_summary["num_select_tuples"]
        ougc.table_feature_state[tbl_summary["target"]]["num_insert_tuples"] = tbl_summary["num_insert_tuples"]
        ougc.table_feature_state[tbl_summary["target"]]["num_update_tuples"] = tbl_summary["num_update_tuples"]
        ougc.table_feature_state[tbl_summary["target"]]["num_delete_tuples"] = tbl_summary["num_delete_tuples"]
        ougc.table_feature_state[tbl_summary["target"]]["num_modify_tuples"] = tbl_summary["num_modify_tuples"]

        ougc.table_feature_state[tbl_summary["target"]]["num_select_queries"] = tbl_summary["num_select_queries"]
        ougc.table_feature_state[tbl_summary["target"]]["num_insert_queries"] = tbl_summary["num_insert_queries"]
        ougc.table_feature_state[tbl_summary["target"]]["num_update_queries"] = tbl_summary["num_update_queries"]
        ougc.table_feature_state[tbl_summary["target"]]["num_delete_queries"] = tbl_summary["num_delete_queries"]

    if ougc.table_feature_model is None:
        for t in ougc.table_feature_state:
            # By default, assume "default" behavior.
            for target in MODEL_WORKLOAD_TARGETS:
                ougc.table_feature_state[t][target] = 0.0

        return None
    else:
        with torch.no_grad():
            outputs, tbls, ret_df = ougc.table_feature_model.inference(ougc.table_feature_state, ougc.table_attr_map, ougc.table_keyspace_features, window, output_df=output_df)
            for i, t in enumerate(tbls):
                for out, target in enumerate(MODEL_WORKLOAD_TARGETS):
                    ougc.table_feature_state[t][target] = outputs[i][out]
            return ret_df


def compute_next_window_state(ougc, window, output_df=False):
    predicted_state = {t: {} for t in ougc.table_attr_map.keys()}
    if ougc.table_state_model is not None:
        with torch.no_grad():
            outputs, tbls, ret_df = ougc.table_state_model.inference(ougc.table_feature_state, ougc.table_attr_map, ougc.table_keyspace_features, window, output_df=output_df)
            for i, t in enumerate(tbls):
                for out, target in enumerate(STATE_WORKLOAD_TARGETS):
                    predicted_state[t][target] = outputs[i][out]
    else:
        ret_df = None

    # Make a copy of the dead tuples counts.
    prev_dead_tuples = {}
    for tbl, tbl_state in ougc.table_feature_state.items():
        prev_dead_tuples[tbl] = (tbl_state["dead_tuple_percent"] * tbl_state["num_pages"] * 8192.0) / tbl_state["tuple_len_avg"]

    # These values are all computed in expectation.
    for tbl, tbl_state in ougc.table_feature_state.items():
        # Analytical tuple count.
        est_tuple_count = tbl_state["tuple_count"]
        est_tuple_count += tbl_state["num_insert_tuples"]
        est_tuple_count -= tbl_state["num_delete_tuples"]
        est_tuple_count = max(0, tbl_state["tuple_count"])

        # Analytical new pages.
        ins_update = tbl_state["num_update_tuples"] - floor(tbl_state["num_update_tuples"] * tbl_state["hot_percent"])
        ins_insert = tbl_state["num_insert_tuples"]
        est_new_pages = tbl_state["num_pages"] + floor((ins_insert + ins_update) * tbl_state["extend_percent"])

        if ougc.table_state_model is not None:
            # Try to stabilize the result. Assume est_tuple_count is the upper bound.
            est_tuple_count = min(tbl_state["tuple_count"] + tbl_state["num_insert_tuples"], predicted_state[tbl]["next_table_num_tuples"])
            est_tuple_count = max(tbl_state["tuple_count"] - tbl_state["num_delete_tuples"], predicted_state[tbl]["next_table_num_tuples"])

        if ougc.table_state_model is not None:
            # We can't go negative in page delta.
            #next_new_pages = max(tbl_state["num_pages"], predicted_state[tbl]["next_table_num_pages"])
            ## Worse case we assume that all insert/updates extend under the current percentages.
            #deltas = tbl_state["num_insert_tuples"] + tbl_state["num_update_tuples"]
            #worst_new_pages = tbl_state["num_pages"] + floor(deltas * tbl_state["extend_percent"])
            #est_new_pages = min(worst_new_pages, next_new_pages)
            est_tuple_count = predicted_state[tbl]["next_table_num_pages"]

        tbl_state["num_pages"] = est_new_pages
        tbl_state["tuple_count"] = est_tuple_count

        #for indexoid, relname in ougc.indexoid_table_map.items():
        #    if relname == tbl:
        #        idx_state = ougc.index_feature_state[indexoid]
        #        for _ in range(ins_insert + ins_update):
        #            idx_state["tuple_count"] += 1
        #            idx_state["num_pages"] += mutate_index_state_will_extend(idx_state)

    # FIXME(VACUUM): Without a VACUUM model or some analytical setup, we can't wipe the dead tuple percents out.
    # This generally forces into a state where dead rises and free shrinks...does it matter though?
    # FIXME(STATS): There is no intuition of how tuple_len_avg will change over time nor any sense of defrags.
    for tbl, tbl_state in ougc.table_feature_state.items():
        new_dead_tuples = prev_dead_tuples[tbl] + tbl_state["num_delete_tuples"] + tbl_state["num_update_tuples"]
        est_dead_tuple_percent = max(0.0, min(1.0, new_dead_tuples * tbl_state["tuple_len_avg"] / (tbl_state["num_pages"] * 8192.0)))
        est_free_tuple_percent = max(0.0, min(1.0, 1 - (tbl_state["tuple_count"] + new_dead_tuples) * tbl_state["tuple_len_avg"] / (tbl_state["num_pages"] * 8192.0)))

        if ougc.table_state_model is not None:
            # Yoink from model if available.
            est_dead_tuple_percent = predicted_state[tbl]["next_table_dead_percent"]
            est_free_tuple_percent = predicted_state[tbl]["next_table_free_percent"]

        # Adjust these to be from 0-100%
        tbl_state["dead_tuple_percent"] = est_dead_tuple_percent * 100.0
        tbl_state["free_percent"] = est_free_tuple_percent * 100.0

        # Estimate the tuple len average from how much is live to how many tuples in table.
        # est_live_tuple_percent = max(0.0, min(1.0, 1 - est_dead_tuple_percent - est_free_tuple_percent))
        # tbl_state["tuple_len_avg"] = (tbl_state["num_pages"] * 8192.0 * est_live_tuple_percent) / (tbl_state["tuple_count"])

    return ret_df
