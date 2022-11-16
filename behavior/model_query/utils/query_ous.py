import random
import json
import copy
import numpy as np
from behavior import OperatingUnit


def prune_out_ous_for_trigger(ous):
    # Prune out OUs from the PLAN if we are invoking a trigger.
    return [o for o in ous if o["node_type"] != OperatingUnit.DestReceiverRemote.name]


##################################################################################
# General OU dictionary related helpers.
##################################################################################

# Finds a matching key in the dict other using "in".
def get_key(key, other):
    for subst_key in other:
        if key in subst_key:
            return other[subst_key]
    assert False, f"Could not find {key} in {other}"


# Returns the number of rows output by a plan. Uses the iterator_used if available.
# Otherwise defaults to plan_rows.
def get_plan_rows_matching_plan_id(ous, plan_id):
    for target_ou in ous:
        if target_ou["plan_node_id"] == plan_id:
            for key in target_ou:
                if "iterator_used" in key:
                    # This is a special means to get the iterator_used key if available.
                    return target_ou[key]

            return get_key("plan_plan_rows", target_ou)
    assert False, f"Could not find plan node with {plan_id}"


# Returns a matching OU that has child_plan_id as either left or right child.
def exist_ou_with_child_plan_id(ous, target_ou_type, child_plan_id):
    for target_ou in ous:
        ou_type = OperatingUnit[target_ou["node_type"]]
        if target_ou_type == ou_type and (target_ou["left_child_node_id"] == child_plan_id or target_ou["right_child_node_id"] == child_plan_id):
            return target_ou
    return None

##################################################################################
# Evaluate Query for OUs
##################################################################################

def evaluate_query(target_conn, query_text, args, qcache):
    if query_text in qcache:
        return copy.deepcopy(qcache[query_text])
    else:
        query_plan_ous = []
        target_conn.execute("DEALLOCATE ALL", prepare=False)
        target_conn.execute("PREPARE pp AS " + query_text, prepare=False)
        result = [r for r in target_conn.execute(f"EXPLAIN (format noisepage) EXECUTE pp (" + ",".join(args) + ")", prepare=False)][0][0]

        # Extract all the OUs from the plan.
        features = json.loads(result)[0]
        def extract_ou(plan):
            ou = {}
            accum_total_cost = 0.0
            accum_startup_cost = 0.0
            for key in plan:
                value = plan[key]
                if key == "Plans":
                    for p in plan[key]:
                        child_total_cost, child_startup_cost = extract_ou(p)
                        accum_total_cost += child_total_cost
                        accum_startup_cost += child_startup_cost
                    continue

                if isinstance(value, list):
                    # FIXME(LIST): For now, we simply featurize a list[] with a numeric length.
                    # This is likely insufficient if the content of the list matters significantly.
                    ou[key + "_len"] = len(value)
                ou[key] = value

            cur_total_cost = ou["total_cost"]
            cur_startup_cost = ou["startup_cost"]
            ou["total_cost"] = max(ou["total_cost"] - accum_total_cost, 0)
            ou["startup_cost"] = max(ou["startup_cost"] - accum_startup_cost, 0)
            query_plan_ous.append(ou)
            return cur_total_cost, cur_startup_cost

        extract_ou(features)
        qcache[query_text] = copy.deepcopy(query_plan_ous)
        return query_plan_ous

##################################################################################
# Generate Triggers for OUs.
##################################################################################

def generate_ou_triggers(target_conn, ou, ougc, qcache, use_plan_estimates):
    trigger_ous = []
    ou_type = OperatingUnit[ou["node_type"]]

    # Create the TupleAR[X]Triggers OU and AfterTriggerEndQuery OU.
    prefix = {
        OperatingUnit.ModifyTableInsert: "TupleARInsertTriggers",
        OperatingUnit.ModifyTableUpdate: "TupleARUpdateTriggers",
        OperatingUnit.ModifyTableDelete: "TupleARDeleteTriggers",
    }[ou_type]
    trigger_ous.append({
        "node_type": prefix,
        f"{prefix}_num_triggers": len(ou["ModifyTable_ar_triggers"]),
        "total_cost": 0,
        "startup_cost": 0
    })

    trigger_ous.append({
        "node_type": "AfterTriggerEndQuery", "AfterTriggerEndQuery_num_invoke": len(ou["ModifyTable_ar_triggers"]),
        "startup_cost": 0,
        "total_cost": 0,
    })

    for tgoid in ou["ModifyTable_ar_triggers"]:
        trigger_info = ougc.trigger_info_map[tgoid]
        if trigger_info["contype"] != "f":
            # UNIQUE constraints should be handled by indexes.
            continue

        if ou_type == OperatingUnit.ModifyTableInsert:
            # 1644 is the hardcoded code for RI_FKey_check_ins.
            assert trigger_info["tgfoid"] == 1644
            frelname = ougc.oid_table_map[trigger_info["confrelid"]]
            tgquery = f"SELECT 1 FROM {frelname} x WHERE "
            tgquery += " AND ".join([f"{attname} = ${i+1}" for i, attname in enumerate(trigger_info["attnames"])])
            tgquery += " FOR KEY SHARE OF x"

            # Get the OUs for the trigger query plan and compute the derived OUs.
            tgous = evaluate_query(target_conn, tgquery, ["0" for i in range(len(trigger_info["attnames"]))], qcache)
            tgous = prune_out_ous_for_trigger(tgous)
            tgous = augment_trigger_exec_features(target_conn, tgous, qcache, ougc, use_plan_estimates)
            # Add an InsertUpdateFKTrigger enforcement OU.
            trigger_ous.append({
                "node_type": OperatingUnit.InsertUpdateFKTriggerEnforce.name,
                "startup_cost": 0,
                "total_cost": 0
            })
            trigger_ous.extend(tgous)
        elif ou_type == OperatingUnit.ModifyTableUpdate:
            # Assert that the UPDATE/DELETE is basically a no-op
            # FIXME(TRIGGER): We assume that UPDATE/DELETE will not trigger FK enforcement.
            assert trigger_info['confupdtype'] == 'a'
        else:
            assert ou_type == OperatingUnit.ModifyTableDelete
            assert trigger_info['confdeltype'] == 'a'

    for tgou in trigger_ous:
        tgou["nonsynthetic"] = 0

    return trigger_ous


def augment_trigger_exec_features(target_conn, tgous, qcache, ougc, use_plan_estimates):
    tgous = augment_ous_exec_features(target_conn, None, tgous, qcache, ougc, use_plan_estimates)
    tgous.append({
        "node_type": OperatingUnit.DestReceiverSPI.name,
        "DestReceiverSPI_num_output": get_plan_rows_matching_plan_id(tgous, 0),
        "startup_cost": 0,
        "total_cost": 0,
    })
    tgous.append({
        "node_type": OperatingUnit.ExecutorStart.name,
        "ExecutorStart_num_marks": get_plan_rows_matching_plan_id(tgous, 0) if exist_ou_with_child_plan_id(tgous, OperatingUnit.LockRows, 1) else 0,
        "ExecutorStart_num_subplans": 0,
        "startup_cost": 0,
        "total_cost": 0,
    })

    for ou in tgous:
        ou_type = OperatingUnit[ou["node_type"]]
        if ou_type == OperatingUnit.LockRows:
            tgous.append({
                "startup_cost": 0,
                "total_cost": 0,
                "node_type": OperatingUnit.InitLockRows.name,
                "InitLockRows_num_marks": get_plan_rows_matching_plan_id(tgous, ou["plan_node_id"]),
            })
        elif ou_type == OperatingUnit.IndexScan:
            # FIXME(INDEX_QUAL): This feature is technically incorrect in the case
            # where the nmber of quals exceeds the number of index keys. But it's
            # also a good enough estimate probably.
            tgous.append({
                "startup_cost": 0,
                "total_cost": 0,
                "node_type": OperatingUnit.InitIndexScan.name,
                "InitIndexScan_num_scan_keys": get_key("indexqual_length", ou),
                "InitIndexScan_num_runtime_keys": get_key("indexqual_length", ou),
                "InitIndexScan_num_orderby_keys": get_key("indexorderby_length", ou),
            })
        elif ou_type == OperatingUnit.IndexOnlyScan:
            tgous.append({
                "startup_cost": 0,
                "total_cost": 0,
                "node_type": OperatingUnit.InitIndexOnlyScan.name,
                "InitIndexOnlyScan_num_scan_keys": get_key("indexqual_length", ou),
                "InitIndexOnlyScan_num_runtime_keys": get_key("indexqual_length", ou),
                "InitIndexOnlyScan_num_orderby_keys": get_key("indexorderby_length", ou),
            })
    return tgous

##################################################################################
# Generate Index OUs
##################################################################################

def mutate_index_state_will_extend(index_state):
    # FIXME(INDEX): We could theoretically use a learned model here to predict the
    # rate at which we accrue index splits and/or index extensions.
    index_state["num_inserts"] += 1
    return (index_state["num_inserts"] % int((8192.0 / index_state["tuple_len_avg"]) / 2) == 0)


def generate_index_inserts(ou, ougc, num_inserts):
    index_oids = ou["ModifyTable_indexupdates_oids"]
    idx_insert_ous = []
    for _ in range(num_inserts):
        for index_oid in index_oids:
            index_state = ougc.index_feature_state[index_oid]
            if mutate_index_state_will_extend(index_state):
                num_splits = 1
                idx_insert_ous.append({
                    'nonsynthetic': 0,
                    'node_type': OperatingUnit.ModifyTableIndexInsert.name,
                    'ModifyTableIndexInsert_indexid': index_oid,
                    'ModifyTableIndexInsert_num_splits': num_splits,
                    'ModifyTableIndexInsert_num_extends': 0,
                })
    return idx_insert_ous

##################################################################################
# Augment the OUs
##################################################################################

def augment_ous_exec_features(target_conn, query, ous, qcache, ougc, use_plan_estimates):
    new_ous = []

    # (1) First patch all of the num_iterator_used.
    for ou in ous:
        tbl = None
        it_label = None
        ou_type = OperatingUnit[ou["node_type"]]

        if ou_type == OperatingUnit.IndexOnlyScan or ou_type == OperatingUnit.IndexScan:
            prefix = "IndexOnlyScan" if ou_type == OperatingUnit.IndexOnlyScan else "IndexScan"
            # Set the number of outer loops to 1 by default. This will be fixed in the second pass for NestLoop.
            ou[f"{prefix}_num_outer_loops"] = 1.0

            # Set the num_iterator_used based on the {tbl}_hits or th plan rows if available.
            tbl = ougc.indexoid_table_map[ou[prefix + "_indexid"]]
            it_label = f"{prefix}_num_iterator_used"

        elif ou_type == OperatingUnit.SeqScan:
            # Set the num_iterator_used based on the {tbl}_hits or th plan rows if available.
            tbl = ougc.oid_table_map[ou["SeqScan_scanrelid_oid"]]
            it_label = "SeqScan_num_iterator_used"

        elif ou_type == OperatingUnit.BitmapIndexScan:
            tbl = ougc.oid_table_map[ou["BitmapIndexScan_scan_scanrelid_oid"]]
            it_label = "BitmapIndexScan_num_tids_found"

        if tbl is not None and it_label is not None:
            if use_plan_estimates or query is None or np.isnan(getattr(query, f"{tbl}_hits")):
                ou[it_label] = get_key("plan_rows", ou)
            else:
                ou[it_label] = getattr(query, f"{tbl}_hits")

            # In the case where there is a LIMIT directly above us, then we only fetch up to the Limit.
            # So take the min of whatever the num_iterator_used is set to and the plan_rows from the Limit.
            limit_ou = exist_ou_with_child_plan_id(ous, OperatingUnit.Limit, ou["plan_node_id"])
            if limit_ou is not None:
                ou[it_label] = min(ou[it_label], get_key("plan_rows", limit_ou))

    # (2) Now patch the rest of the OUs.
    for ou in ous:
        ou["nonsynthetic"] = 1
        ou_type = OperatingUnit[ou["node_type"]]

        if ou_type == OperatingUnit.DestReceiverRemote:
            # Number output is controlled by the output of plan node 0.
            ou["DestReceiverRemote_num_output"] = get_plan_rows_matching_plan_id(ous, 0)

        elif ou_type == OperatingUnit.IndexOnlyScan or ou_type == OperatingUnit.IndexScan or ou_type == OperatingUnit.SeqScan:
            prefix = {
                OperatingUnit.IndexOnlyScan: "IndexOnlyScan",
                OperatingUnit.IndexScan: "IndexScan",
                OperatingUnit.SeqScan: "SeqScan"
            }[ou_type]

            # Extract the table key.
            tbl = ougc.oid_table_map[ou["SeqScan_scanrelid_oid"]] if ou_type == OperatingUnit.SeqScan else ougc.indexoid_table_map[ou[prefix + "_indexid"]]

            # Set the num_defrag from defrag_percent which is normalized against number of tuples touched.
            ou[f"{prefix}_num_defrag"] = 0
            for _ in range(int(ou[f"{prefix}_num_iterator_used"])):
                ou[f"{prefix}_num_defrag"] += int(random.uniform(0, 1) <= (ougc.table_feature_state[tbl]["defrag_percent"]))

            if ou_type == OperatingUnit.IndexOnlyScan or ou_type == OperatingUnit.IndexScan:
                # See if there's a NestLoop OU that has us as the inner child (right).
                for other_ou in ous:
                    if other_ou["node_type"] == OperatingUnit.NestLoop.name and other_ou["right_child_node_id"] == ou["plan_node_id"]:
                        # Then set out num_outer_loops to the number of rows output by the left child.
                        ou[f"{prefix}_num_outer_loops"] = get_plan_rows_matching_plan_id(ous, other_ou["left_child_node_id"])
                        # Adjust num_iterator_used so it is a "per-iterator" estimate.
                        ou[f"{prefix}_num_iterator_used"] = max(1.0, ou[f"{prefix}_num_iterator_used"] / ou[f"{prefix}_num_outer_loops"])
                        break

        elif ou_type == OperatingUnit.SeqScan:
            tbl = ougc.oid_table_map[ou["SeqScan_scanrelid_oid"]]
            for _ in range(int(ou[f"{prefix}_num_iterator_used"])):
                ou[f"{prefix}_num_defrag"] += int(random.uniform(0, 1) <= (ougc.table_feature_state[tbl]["defrag_percent"]))

        elif ou_type == OperatingUnit.Agg:
            ou["Agg_num_input_rows"] = get_plan_rows_matching_plan_id(ous, ou["left_child_node_id"])

        elif ou_type == OperatingUnit.NestLoop:
            ou["NestLoop_num_outer_rows"] = get_plan_rows_matching_plan_id(ous, ou["left_child_node_id"])
            ou["NestLoop_num_inner_rows_cumulative"] = get_plan_rows_matching_plan_id(ous, ou["right_child_node_id"]) * ou["NestLoop_num_outer_rows"]

        elif ou_type == OperatingUnit.ModifyTableInsert:
            tbl = ougc.oid_table_map[ou["ModifyTable_target_oid"]]
            ou["ModifyTableInsert_num_extends"] = 0
            if random.uniform(0, 1) <= ougc.table_feature_state[tbl]["extend_percent"]:
                # FIXME(INSERT): We can only extend at-most 1 block. Maybe this needs to be a model input.
                # If it's an input from the model, we care a lot about model stability.
                ou["ModifyTableInsert_num_extends"] = 1

            # Check and generate OUs.
            new_ou = generate_ou_triggers(target_conn, ou, ougc, qcache, use_plan_estimates)
            new_ous.extend(new_ou)
            new_ous.extend(generate_index_inserts(ou, ougc, 1))

        elif ou_type == OperatingUnit.ModifyTableDelete:
            # Find the indexes that the ModifyTableDelete might effect.
            tbl = ougc.oid_table_map[ou["ModifyTable_target_oid"]]
            assert getattr(query, f"{tbl}_hits") > 0
            ou["ModifyTableDelete_num_deletes"] = getattr(query, f"{tbl}_hits")
            new_ou = generate_ou_triggers(target_conn, ou, ougc, qcache, use_plan_estimates)
            new_ous.extend(new_ou)

        elif ou_type == OperatingUnit.ModifyTableUpdate:
            tbl = ougc.oid_table_map[ou["ModifyTable_target_oid"]]
            num_modify = getattr(query, f"{tbl}_hits")
            assert num_modify > 0
            ou["ModifyTableUpdate_num_updates"] = num_modify
            ou["ModifyTableUpdate_num_extends"] = 0
            ou["ModifyTableUpdate_num_hot"] = 0

            num_index_inserts = 0
            for _ in range(int(num_modify)):
                # Flip a coin for each UPDATE tuple of whether it'll succeed or not..
                if random.uniform(0, 1) <= (ougc.table_feature_state[tbl]["hot_percent"]):
                    # We have performed a HOT update.
                    ou["ModifyTableUpdate_num_hot"] += 1
                else:
                    if random.uniform(0, 1) <= (ougc.table_feature_state[tbl]["extend_percent"]):
                        # We have performed a relation extend.
                        ou["ModifyTableUpdate_num_extends"] += 1
                    num_index_inserts += 1

            new_ou = generate_ou_triggers(target_conn, ou, ougc, qcache, use_plan_estimates)
            new_ous.extend(new_ou)
            new_ous.extend(generate_index_inserts(ou, ougc, num_index_inserts))

        elif ou_type == OperatingUnit.BitmapHeapScan:
            ou["BitmapHeapScan_scan_scanrelid_oid"] = 0
            for other_ou in ous:
                if other_ou["node_type"] == OperatingUnit.BitmapIndexScan.name and other_ou["plan_node_id"] == ou["left_child_node_id"]:
                    ou["BitmapHeapScan_scan_scanrelid_oid"] = other_ou["BitmapIndexScan_scan_scanrelid_oid"]
            ou["BitmapHeapScan_num_tuples_fetch"] = get_plan_rows_matching_plan_id(ous, ou["left_child_node_id"])

            tbl = ougc.oid_table_map[ou["BitmapHeapScan_scan_scanrelid_oid"]]
            for _ in range(int(ou[f"BitmapHeapScan_num_tuples_fetch"])):
                ou["BitmapHeapScan_num_defrag"] += int(random.uniform(0, 1) <= (ougc.table_feature_state[tbl]["defrag_percent"]))

    ous.extend(new_ous)
    return ous

##################################################################################
# Get the query OUs
##################################################################################

def generate_query_ous_window(target_conn, ougc, window_index, query_plans, use_plan_estimates, output_dir):
    qcache = {}
    args = [f"arg{i}" for i in range(0, query_plans.shape[1]) if f"arg{i}" in query_plans]
    col_idx_map = {k:i for i, k in enumerate(query_plans.columns)}
    total_window_ous = []
    with target_conn.transaction():
        for query in query_plans.itertuples(index=None):
            argvals = [f"'{query[col_idx_map[arg]]}'" for arg in args if query[col_idx_map[arg]] is not None]
            query_plan_ous = evaluate_query(target_conn, query.query_text, argvals, qcache)
            query_plan_ous = augment_ous_exec_features(target_conn, query, query_plan_ous, qcache, ougc, use_plan_estimates)
            for ou in query_plan_ous:
                ou["query_id"] = query.query_id
                ou["query_order"] = query.query_order
                ou["window_index"] = window_index
            total_window_ous.extend(query_plan_ous)
    return total_window_ous
