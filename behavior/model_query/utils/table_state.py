import torch
import re
from psycopg.rows import dict_row
from behavior import OperatingUnit
from behavior.model_workload.utils import OpType
from behavior.model_workload.models.table_feature_model import MODEL_WORKLOAD_TARGETS


def initial_trigger_metadata(target_conn, ougc):
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


def initial_table_feature_state(target_conn, ougc):
    table_feature_state = {}
    oid_table_map = {}

    # Here we assume that the "initial" state with which populate_data() was invoked with
    # and the state at which the query stream was captured is roughly consistent (along
    # with the state at which we are assessing pgstattuple_approx) to some degree.
    with target_conn.cursor(row_factory=dict_row) as cursor:
        for tbl in ougc.tables:
            result = [r for r in cursor.execute(f"SELECT * FROM pgstattuple_approx('{tbl}')", prepare=False)][0]
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
                "approx_free_percent": result["approx_free_percent"],
                "dead_tuple_percent": result["dead_tuple_percent"],
                "approx_tuple_count": result["approx_tuple_count"],
                "tuple_len_avg": result["approx_tuple_len"] / result["approx_tuple_count"],
                "target_ff": ff,
            }

    ougc.table_feature_state = table_feature_state
    ougc.oid_table_map = oid_table_map


def refresh_table_fillfactor(target_conn, ougc):
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


def initial_index_feature_state(target_conn, ougc):
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
                        "table_len": result["table_len"],
                        "approx_tuple_count": result["tuple_count"],
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


def compute_table_exec_features(ougc, query_plans, window):
    for t in ougc.table_feature_state:
        ougc.table_feature_state[t]["num_select_tuples"] = 0
        ougc.table_feature_state[t]["num_insert_tuples"] = 0
        ougc.table_feature_state[t]["num_update_tuples"] = 0
        ougc.table_feature_state[t]["num_delete_tuples"] = 0
        ougc.table_feature_state[t]["num_modify_tuples"] = 0

    for tbl, df in query_plans.groupby(by=["target"]):
        tbls = tbl.split(",")
        for tblc in tbls:
            ougc.table_feature_state[tblc]["num_insert_tuples"] += df[df.optype == OpType.INSERT.value].shape[0]
            if f"{tblc}_hits" in df:
                ougc.table_feature_state[tblc]["num_select_tuples"] += df[f"{tblc}_hits"].sum()
                ougc.table_feature_state[tblc]["num_update_tuples"] += df[df.optype == OpType.UPDATE.value][f"{tblc}_hits"].sum()
                ougc.table_feature_state[tblc]["num_delete_tuples"] += df[df.optype == OpType.DELETE.value][f"{tblc}_hits"].sum()

                ougc.table_feature_state[tblc]["num_modify_tuples"] += df[df.optype == OpType.UPDATE.value][f"{tblc}_hits"].sum()
                ougc.table_feature_state[tblc]["num_modify_tuples"] += df[df.optype == OpType.DELETE.value][f"{tblc}_hits"].sum()

    if ougc.table_feature_model is None:
        for t in ougc.table_feature_state:
            # By default, assume "default" behavior.
            for target in MODEL_WORKLOAD_TARGETS:
                ougc.table_feature_state[t][target] = 0.0
    else:
        with torch.no_grad():
            outputs, tbls = ougc.table_feature_model.inference(ougc.table_feature_state, ougc.table_attr_map, ougc.table_keyspace_features, window)
            for i, t in enumerate(tbls):
                for out, target in enumerate(MODEL_WORKLOAD_TARGETS):
                    ougc.table_feature_state[t][target] = outputs[i][out]


def compute_next_window_state(ougc, query_ous):
    prev_dead_tuples = {}
    for tbl, tbl_state in ougc.table_feature_state:
        prev_dead_tuples[tbl] = (tbl_state["dead_tuple_percent"] * tbl_state["num_pages"] * 8192.0) / tbl_state["tuple_len_avg"]

    delete_counters = {t: 0 for t in ougc.table_feature_state}
    update_counters = {t: 0 for t in ougc.table_feature_state}

    for query_ou in query_ous:
        if query_ou["node_type"] == OperatingUnit.ModifyTableInsert.name:
            # Indicate we've inserted a tuple.
            tbl = query_ou["ModifyTable_target_oid"]
            ougc.table_feature_state[tbl]["approx_tuple_count"] += 1
            ougc.table_feature_state[tbl]["num_pages"] += query_ou["ModifyTableInsert_num_extends"]
        elif query_ou["node_type"] == OperatingUnit.ModifyTableDelete.name:
            # Indicate we've deleted a tuple.
            tbl = query_ou["ModifyTable_target_oid"]
            ougc.table_feature_state[tbl]["approx_tuple_count"] -= query_ou["ModifyTableDelete_num_deletes"]
            delete_counters[tbl] += query_ou["ModifyTableDelete_num_deletes"]
        elif query_ou["node_type"] == OperatingUnit.ModifyTableUpdate.name:
            tbl = query_ou["ModifyTable_target_oid"]
            # Update doesn't incur a tuple count change because "delete 1, insert 1 mentality".
            ougc.table_feature_state[tbl]["num_pages"] += query_ou["ModifyTableUpdate_num_extends"]
            update_counters[tbl] += query_ou["ModifyTableUpdate_num_updates"]
        elif query_ou["node_type"] == OperatingUnit.ModifyTableIndexInsert.name:
            indexoid = query_ou["ModifyTableIndexInsert_indexid"]
            ougc.index_feature_state[indexoid]["approx_tuple_count"] += 1
            ougc.index_feature_state[indexoid]["num_pages"] += query_ou["ModifyTableIndexInsert_num_extends"]

    # Cap it at 0.
    for _, tbl_state in ougc.table_feature_state:
        tbl_state["approx_tuple_count"] = max(0, tbl_state["approx_tuple_count"])

    # FIXME(VACUUM): Without a VACUUM model or some analytical setup, we can't wipe the dead tuple percents out.
    # This generally forces into a state where dead rises and free shrinks...does it matter though?
    #
    # FIXME(STATS): There is no intuition of how tuple_len_avg will change over time nor any sense of defrags.
    for tbl, tbl_state in ougc.table_feature_state:
        new_dead_tuples = pre_dead_tuples[tbl] + dead_counters[tbl] + update_counters[tbl]
        est_dead_tuple_percent = new_dead_tuples * tbl_state["tuple_len_avg"] / (tbl_state["num_pages"] * 8192.0)
        est_free_tuple_percent = max(0.0, min(1.0, 1 - (tbl_state["approx_tuple_count"] + new_dead_tuples) * tbl_state["tuple_len_avg"] / (tbl_state["num_pages"] * 8192.0)))
        tbl_state["dead_tuple_percent"] = max(0.0, min(1.0, est_dead_tuple_percent))
        tbl_state["approx_free_percent"] = est_free_tuple_percent

