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
from pathlib import Path
from plumbum import cli

from behavior import OperatingUnit, Targets, BENCHDB_TO_TABLES
from behavior.datagen.pg_collector_utils import _parse_field, KNOBS
from behavior.utils.evaluate_ou import evaluate_ou_model
from behavior.utils.prepare_ou_data import prepare_index_input_data
from behavior.model_workload.utils import keyspace_metadata_read
from behavior.utils.process_pg_state_csvs import (
    process_time_pg_stats,
    process_time_pg_attribute,
    process_time_pg_index,
    process_time_pg_class,
    build_time_index_metadata
)
import behavior.model_workload.models as model_workload_models
import torch

logger = logging.getLogger(__name__)

##################################################################################
# Logic related to setting up the state.
##################################################################################

def compute_table_oids(conn):
    result = conn.execute("SELECT relname, oid from pg_class")
    return {r[1]: r[0] for r in result}


def compute_index_table_map(conn):
    result = conn.execute("""
            SELECT indexrelid,
                   t.relname
              FROM pg_index,
                   pg_class t
             WHERE pg_index.indrelid = t.oid
        """)
    return {tup[0]: tup[1] for tup in result}


def compute_index_keyspace_map(conn):
    # These ignore the primary key space which is the table keyspace.
    result = conn.execute("""
            SELECT indexrelid,
                   t.relname
              FROM pg_index,
                   pg_class t,
                   pg_namespace n
             WHERE pg_index.indexrelid = t.oid
               AND n.oid = t.relnamespace
               AND n.nspname = 'public'
        """)
    indexes = {tup[0]: tup[1] for tup in result}

    result = conn.execute("""
            select
                t.relname as table_name,
                i.relname as index_name,
                ix.indexrelid,
                ix.indisprimary,
                array_to_string(array_agg(a.attname), ',') as column_names
            from
                pg_class t,
                pg_class i,
                pg_index ix,
                pg_attribute a
            where
                t.oid = ix.indrelid
                and i.oid = ix.indexrelid
                and a.attrelid = t.oid
                and a.attnum = ANY(ix.indkey)
                and t.relkind = 'r'
            group by
                t.relname,
                i.relname,
                ix.indexrelid,
                ix.indisprimary
            order by
                t.relname,
                i.relname;
        """)

    res_indexes = {}
    for r in result:
        if r[2] in indexes:
            res_indexes[r[2]] = (r[1], r[4].split(","))
    return res_indexes


def compute_trigger_map(conn):
    with conn.cursor(row_factory=dict_row) as cursor:
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

    return triggers

##################################################################################
# Get query OUs from postgres and perform feature augmentation
##################################################################################

def evaluate_query(conn, query, qcache):
    if query in qcache:
        return copy.deepcopy(qcache[query])
    else:
        matches = re.findall(r'(\$[0-9])', query)
        if len(matches) == 0:
            result = [r for r in conn.execute(f"EXPLAIN (format noisepage) " + query, prepare=False)]
        else:
            conn.execute("DEALLOCATE ALL", prepare=False)
            conn.execute("PREPARE qq AS " + query, prepare=False)

            args = ",".join(["'0'" for i in range(len(matches))])
            result = [r for r in conn.execute(f"EXPLAIN (format noisepage) EXECUTE qq (" + args + ")", prepare=False)]

        result = result[0][0]

        # Extract all the OUs for a given query_text plan.
        features = json.loads(result)[0]
        template_ous = []

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
            ou["query_text"] = query
            template_ous.append(ou)
            return cur_total_cost, cur_startup_cost
        extract_ou(features)
        qcache[query] = copy.deepcopy(template_ous)
        return template_ous


def augment_ou_triggers(conn, ou, qcache, window_stats, index_table_map, table_oid_map, trigger_map):
    trigger_ous = []
    ou_type = OperatingUnit[ou["node_type"]]
    for tgoid in ou["ModifyTable_ar_triggers"]:
        trigger_info = trigger_map[tgoid]
        if trigger_info['contype'] != 'f':
            # UNIQUE constraints should be handled by indexes.
            continue

        if ou_type == OperatingUnit.ModifyTableInsert:
            # 1644 is the hardcoded code for RI_FKey_check_ins.
            assert trigger_info["tgfoid"] == 1644
            frelname = table_oid_map[trigger_info["confrelid"]]
            tgquery = f"SELECT 1 FROM {frelname} WHERE "

            for i, attname in enumerate(trigger_info["attnames"]):
                if i != 0:
                    tgquery = tgquery + " AND "
                # FIXME(TRIGGER): We currently use a placeholder value. We could try installing the true values in
                # since we are operating under the assumption that it succeeds.
                tgquery = tgquery + f"{attname} = '0'"

            # Get the OUs for the trigger query plan and compute the derived OUs.
            tgous = evaluate_query(conn, tgquery, qcache)
            ous, (_, idx_insert) = augment_single_query(conn, None, tgous, qcache, window_stats, index_table_map, table_oid_map, trigger_map)
            assert idx_insert == 0, "Trigger should not generate an index insert."
            trigger_ous.extend(ous)
        elif ou_type == OperatingUnit.ModifyTableUpdate:
            # Assert that the UPDATE/DELETE is basically a no-op
            # FIXME(TRIGGER): We assume that UPDATE/DELETE will not trigger FK enforcement.
            assert trigger_info['confupdtype'] == 'a'
        else:
            assert ou_type == OperatingUnit.ModifyTableDelete
            assert trigger_info['confdeltype'] == 'a'
    return trigger_ous


def augment_single_query(conn, query, ous, qcache, window_stats, index_table_map, table_oid_map, trigger_map):
    # Finds a matching key in the dict other using "in".
    def get_key(key, other):
        for subst_key in other:
            if key in subst_key:
                return other[subst_key]
        assert False, f"Could not find {key} in {other}"

    # Returns a matching OU that has child_plan_id as either left or right child.
    def exist_ou_with_child_plan_id(target_ou_type, child_plan_id):
        for target_ou in ous:
            ou_type = OperatingUnit[target_ou["node_type"]]
            if target_ou_type == ou_type and (target_ou["left_child_node_id"] == child_plan_id or target_ou["right_child_node_id"] == child_plan_id):
                return target_ou
        return None

    # Returns the number of rows output by a plan. Uses the iterator_used if available.
    # Otherwise defaults to plan_rows.
    def get_plan_rows_matching_plan_id(plan_id):
        for target_ou in ous:
            if target_ou["plan_node_id"] == plan_id:
                for key in target_ou:
                    if "iterator_used" in key:
                        # This is a special means to get the iterator_used key if available.
                        return target_ou[key]

                return get_key("plan_plan_rows", target_ou)
        assert False, f"Could not find plan node with {plan_id}"

    new_ous = []
    indexes_modify = ([], 0)

    # Augment the static OU features with the "dynamic" execution features.
    # These features are defined in behavior/__init__.py.
    for ou in ous:
        ou_type = OperatingUnit[ou["node_type"]]
        if ou_type == OperatingUnit.IndexOnlyScan or ou_type == OperatingUnit.IndexScan:
            prefix = "IndexOnlyScan" if ou_type == OperatingUnit.IndexOnlyScan else "IndexScan"
            tbl = index_table_map[ou[prefix + "_indexid"]]

            # Set the number of outer loops to 1 by default. This will be fixed in the second pass for NestLoop.
            ou[f"{prefix}_num_outer_loops"] = 1.0

            if query is None or np.isnan(getattr(query, f"{tbl}_pk_output")):
                # Case when we don't have a valid {tbl}_pk_output column that we can use.
                ou[f"{prefix}_num_iterator_used"] = get_key("plan_rows", ou)
            else:
                ou[f"{prefix}_num_iterator_used"] = getattr(query, f"{tbl}_pk_output")

            limit_ou = exist_ou_with_child_plan_id(OperatingUnit.Limit, ou["plan_node_id"])
            if limit_ou is not None:
                # In the case where there is a LIMIT directly above us, then we only fetch up to the Limit.
                # So take the min of whatever the num_iterator_used is set to and the plan_rows from the Limit.
                ou[f"{prefix}_num_iterator_used"] = min(ou[f"{prefix}_num_iterator_used"], get_key("plan_rows", limit_ou))

            # Set the num_defrag from defrag_percent which is normalized against number of tuples touched.
            ou[f"{prefix}_num_defrag"] = int(random.uniform(0, 1) <= (window_stats[tbl]["defrag_percent"] * ou[f"{prefix}_num_iterator_used"]))

        elif ou_type == OperatingUnit.Agg:
            # The number of input rows is the number of rows output by the child.
            ou["Agg_num_input_rows"] = get_plan_rows_matching_plan_id(ou["left_child_node_id"])

        elif ou_type == OperatingUnit.DestReceiverRemote:
            # Number output is controlled by the output of plan node 0.
            ou["DestReceiverRemote_num_output"] = get_plan_rows_matching_plan_id(0)

        elif ou_type == OperatingUnit.ModifyTableDelete:
            assert not np.isnan(query.num_modify)
            # Find the indexes that the ModifyTableDelete might effect.
            tbl = table_oid_map[ou["ModifyTable_target_oid"]]
            indexes = [idx for idx, t in index_table_map.items() if t == tbl]

            ou["ModifyTableDelete_num_deletes"] = query.num_modify
            new_ous.extend(augment_ou_triggers(conn, ou, qcache, window_stats, index_table_map, table_oid_map, trigger_map))
            indexes_modify = (indexes, query.num_modify)

        elif ou_type == OperatingUnit.ModifyTableUpdate:
            assert not np.isnan(query.num_modify)
            tbl = table_oid_map[ou["ModifyTable_target_oid"]]
            ou["ModifyTableUpdate_num_updates"] = query.num_modify
            ou["ModifyTableUpdate_num_extends"] = 0
            ou["ModifyTableUpdate_num_hot"] = 0

            num_index_inserts = 0
            for _ in range(int(query.num_modify)):
                if random.uniform(0, 1) <= (window_stats[tbl]["hot_percent"]):
                    # We have performed a HOT update.
                    ou["ModifyTableUpdate_num_hot"] = ou["ModifyTableUpdate_num_hot"] + 1
                else:
                    if random.uniform(0, 1) <= (window_stats[tbl]["extend_percent"]):
                        # We have performed a relation extend.
                        ou["ModifyTableUpdate_num_extends"] = ou["ModifyTableUpdate_num_extends"] + 1
                    num_index_inserts = num_index_inserts + 1
            new_ous.extend(augment_ou_triggers(conn, ou, qcache, window_stats, index_table_map, table_oid_map, trigger_map))

            if num_index_inserts > 0:
                indexes_modify = (ou["ModifyTable_indexupdates_oids"], num_index_inserts)

        elif ou_type == OperatingUnit.ModifyTableInsert:
            tbl = table_oid_map[ou["ModifyTable_target_oid"]]
            ou["ModifyTableInsert_num_extends"] = 0
            if random.uniform(0, 1) <= window_stats[tbl]["extend_percent"]:
                # We have performed a relation extend.
                ou["ModifyTableInsert_num_extends"] = 1

            # Check and generate OUs.
            new_ous.extend(augment_ou_triggers(conn, ou, qcache, window_stats, index_table_map, table_oid_map, trigger_map))
            indexes_modify = (ou["ModifyTable_indexupdates_oids"], 1)

    # We've now went through and set all the base table scans. We now handle all the other OUs
    # that might rely on that information (i.e., NestLoop joins).
    for ou in ous:
        ou_type = OperatingUnit[ou["node_type"]]
        if ou_type == OperatingUnit.IndexOnlyScan or ou_type == OperatingUnit.IndexScan:
            prefix = "IndexOnlyScan" if ou_type == OperatingUnit.IndexOnlyScan else "IndexScan"
            for other_ou in ous:
                # See if there's a NestLoop OU that has us as the inner child (right).
                if other_ou["node_type"] == OperatingUnit.NestLoop.name and other_ou["right_child_node_id"] == ou["plan_node_id"]:
                    # Then set out num_outer_loops to the number of rows output by the left child.
                    ou[f"{prefix}_num_outer_loops"] = get_plan_rows_matching_plan_id(other_ou["left_child_node_id"])
                    # Adjust num_iterator_used so it is a "per-iterator" estimate.
                    ou[f"{prefix}_num_iterator_used"] = max(1.0, ou[f"{prefix}_num_iterator_used"] / ou[f"{prefix}_num_outer_loops"])
                    break
        elif ou_type == OperatingUnit.NestLoop:
            ou["NestLoop_num_outer_rows"] = get_plan_rows_matching_plan_id(ou["left_child_node_id"])
            ou["NestLoop_num_inner_rows_cumulative"] = get_plan_rows_matching_plan_id(ou["right_child_node_id"]) * ou["NestLoop_num_outer_rows"]

    new_ous.extend(ous)
    return new_ous, indexes_modify

##################################################################################
# Generation of query OUs and state modification
##################################################################################

def implant_stats_to_postgres(conn, stats):
    for tbl, data in stats.items():
        if tbl == "chunk_num":
            # There is this dummy key.
            continue

        relpages = int(math.ceil(data["table_len"] / 8192))
        reltuples = data["approx_tuple_count"]

        # FIXME(INDEX): This is a back of the envelope estimation for the index height.
        height = 0
        if data["tuple_len_avg"] > 0:
            fanout = (8192 / data["tuple_len_avg"])
            height = math.ceil(np.log(relpages) / np.log(fanout))

        # FIXME(STATS): Should we try and fake the histogram?
        query = f"SELECT qss_install_stats('{tbl}', {relpages}, {reltuples}, {height})"
        conn.execute(query, prepare=False)


def mutate_table_window_state(tbl, pre_table_state, queries, ff_tbl_change_map, use_workload_table_estimates):
    if queries.shape[0] == 0:
        return pre_table_state

    num_insert = queries[queries.is_insert].num_modify.sum()
    num_delete = queries[queries.is_delete].num_modify.sum()
    num_update = queries[queries.OP == "UPDATE"].num_modify.sum()
    num_hot = num_update * min(1.0, max(0.0, pre_table_state["hot_percent"]))
    assert num_hot <= num_update

    if use_workload_table_estimates:
        delta_tuple_count = pre_table_state["norm_delta_tuple_count"] * (num_insert + num_delete) / NORM_RELATIVE_OPS
        delta_table_len = max(0, pre_table_state["norm_delta_table_len"] * (num_insert + num_update) / NORM_RELATIVE_OPS)
        delta_free_percent = pre_table_state["norm_delta_free_percent"] * (num_insert + num_update + num_delete) / NORM_RELATIVE_OPS
        delta_dead_percent = pre_table_state["norm_delta_dead_percent"] * (num_insert + num_update + num_delete) / NORM_RELATIVE_OPS

        est_tuple_count = max(0, pre_table_state["approx_tuple_count"] + delta_tuple_count)
        new_tuple_count = max(0, min(est_tuple_count, pre_table_state["approx_tuple_count"] + num_insert - num_delete))
    else:
        new_tuple_count = max(0, pre_table_state["approx_tuple_count"] + num_insert - num_delete)

    # Generate aggressively on the larger end so we can bound the model predictions.
    num_extend = (num_insert + num_update - num_hot) * min(1.0, max(0, pre_table_state["extend_percent"]))
    if use_workload_table_estimates:
        new_table_len = pre_table_state["table_len"] + min(num_extend * 8192, delta_table_len)
    else:
        new_table_len = pre_table_state["table_len"] + max(0, num_extend * 8192)

    new_dead_tuples = (pre_table_state["dead_tuple_percent"] * pre_table_state["table_len"] / pre_table_state["tuple_len_avg"]) + (num_delete + num_update)
    est_dead_tuple_percent = (new_dead_tuples * pre_table_state["tuple_len_avg"]) / new_table_len
    est_free_tuple_percent = max(0.0, min(1.0, 1 - (new_tuple_count + new_dead_tuples) / (new_table_len / pre_table_state["tuple_len_avg"])))

    pre_table_state["table_len"] = new_table_len
    pre_table_state["approx_tuple_count"] = new_tuple_count

    if use_workload_table_estimates:
        # Since our estimates are on the larger for "deadness" tuples, they provide a "higher bound" of sorts for deadness.
        pre_table_state["dead_tuple_percent"] = max(0, min(1, min(est_dead_tuple_percent, pre_table_state["dead_tuple_percent"] + delta_dead_percent)))
        # Since our estimates are on the larger for affected tuples, they provide a "lower bound" of sorts for the free percentages.
        pre_table_state["approx_free_percent"] = max(0, min(1, min(est_free_tuple_percent, pre_table_state["approx_free_percent"] + delta_free_percent)))
    else:
        pre_table_state["dead_tuple_percent"] = max(0, min(1, est_dead_tuple_percent))
        pre_table_state["approx_free_percent"] = max(0, min(1, est_free_tuple_percent))

    assert pre_table_state["dead_tuple_percent"] >= 0 and pre_table_state["dead_tuple_percent"] <= 1
    assert pre_table_state["approx_free_percent"] >= 0 and pre_table_state["approx_free_percent"] <= 1

    if tbl in ff_tbl_change_map and queries.shape[0] > 0:
        # Mutate the fill factor based on when we know the ALTER TABLE was executed.
        slots = ff_tbl_change_map[tbl]
        for (ts, ff) in reversed(slots):
            if queries.iloc[-1].statement_timestamp >= ts:
                # We have found a new fill-factor boundary. We want to traverse slots() in reverse since
                # we want to find the "last" ff setting that is valid (since slots is sorted in increasing
                # time order). If it's greater than slots[2], then it is greater than slots[1].
                pre_table_state["ff"] = ff
                break

    return pre_table_state


def generate_index_inserts(chunk_num, chunk, augment_chunk, keyspace_augs, index_keyspace_map, index_stats, scratch_ou):
    total = functools.reduce(lambda a,b: a+b, map(lambda x: len(x[1]), keyspace_augs.items()))
    pbar = tqdm(total=total)

    idx_insert_ous = []
    chunk.set_index(keys=["query_order"], inplace=True)
    augment_chunk.set_index(keys=["query_order"], inplace=True)
    for index_keyspace, v in keyspace_augs.items():
        index_name = index_keyspace_map[index_keyspace][0]
        if len(v) == 0:
            continue

        idx_ins = pd.DataFrame(v)
        assert idx_ins.target_index_name.nunique() == 1
        idx_ins.set_index(keys=["query_order"], inplace=True)
        augs = augment_chunk.join(idx_ins, how="inner")

        # FIXME(INDEX): Need to use the workload model in order to featurize [augs].
        # With the featurized [augs] we can then get a [% extend, % split].

        augs = chunk.join(idx_ins, how="inner")
        augs.reset_index(drop=False, inplace=True)
        augs = augs[~augs.target_index_modify_tuples.isna()]

        # These are the "index" got touched probes.
        pbar.update(idx_ins.shape[0] - augs.shape[0])

        # We remove the "DELETEs" since they don't trigger ModifyTableIndexInsert.
        total_mods = augs.shape[0]
        augs = augs[~augs.is_delete]
        num_deletes = total_mods - augs.shape[0]
        pbar.update(num_deletes)

        total_split = 0
        for aug in augs.itertuples():
            assert not aug.is_delete

            num_splits = 0
            index_stats[index_name]["num_inserts"] = index_stats[index_name]["num_inserts"] + 1
            if (index_stats[index_name]["num_inserts"] % int((8192 / index_stats[index_name]["tuple_len_avg"]) / 2) == 0):
                num_splits = 1
                total_split = total_split + 1

            idx_insert_ous.append({
                'query_id': aug.query_id,
                'query_order': aug.query_order,
                'window_slice': chunk_num,
                'node_type': OperatingUnit.ModifyTableIndexInsert.name,
                'ModifyTableIndexInsert_indexid': index_keyspace,
                'plan_node_id': -1,
                'ModifyTableIndexInsert_num_extends': num_splits,
                'ModifyTableIndexInsert_num_splits': num_splits,
            })
            pbar.update(1)

        # FIXME(INDEX): We need to adjust the state of the index_stats for the workload model.
        index_stats[index_name]["table_len"] = index_stats[index_name]["table_len"] + total_split * 8192
        # augs.shape[0] is the number of inserts caused by inserts/updates.
        index_stats[index_name]["approx_tuple_count"] = max(0, index_stats[index_name]["approx_tuple_count"] + augs.shape[0] - num_deletes)

    pbar.close()

    chunk.reset_index(drop=False, inplace=True)
    augment_chunk.reset_index(drop=False, inplace=True)
    pd.DataFrame(idx_insert_ous).to_feather(f"{scratch_ou}/ModifyTableIndexInsert.feather.{chunk_num}")
    del idx_insert_ous


def generate_ous_for_chunk(conn, workload_model, chunk_num, chunk,
                           augment_chunk, window_stats, index_stats,
                           index_table_map, index_keyspace_map, table_oid_map, trigger_map, scratch_ou):
    ou_features = {ou.name: [] for ou in OperatingUnit}
    keyspace_augs = {t: [] for t in index_keyspace_map.keys()}

    pbar = tqdm(total=chunk.shape[0])
    query_cache = {}
    for i, query in enumerate(chunk.itertuples()):
        # Get the OUs for evaluating the query.
        ous = evaluate_query(conn, query.query_text, query_cache)

        # Augment all the OUs.
        ous, (idxs, idx_insert) = augment_single_query(conn, query, ous, query_cache, window_stats, index_table_map, table_oid_map, trigger_map)

        # Add the OU features that we constructed.
        for ou in ous:
            ou["query_id"] = query.query_id
            ou["query_order"] = query.query_order
            ou["window_slice"] = chunk_num
            ou_features[ou["node_type"]].append(ou)

            ou_type = OperatingUnit[ou["node_type"]]
            if ou_type == OperatingUnit.IndexOnlyScan or ou_type == OperatingUnit.IndexScan:
                # We are using an index that is in index_keyspace_map for analysis.
                prefix = "IndexOnlyScan" if ou_type == OperatingUnit.IndexOnlyScan else "IndexScan"
                indexid = ou[prefix + "_indexid"]
                if indexid in index_keyspace_map:
                    keyspace_augs[indexid].append({
                        "query_order": query.query_order,
                        "target_index_name": index_keyspace_map[indexid][0],
                        "target_index_modify_tuples": np.nan,
                    })
            elif ou_type == OperatingUnit.ModifyTableInsert or ou_type == OperatingUnit.ModifyTableUpdate or ou_type == OperatingUnit.ModifyTableDelete:
                # We are using an index (for insert/update/delete) that is in index_keyspace_map for analysis.
                for idx in idxs:
                    keyspace_augs[idx].append({
                        "query_order": query.query_order,
                        "target_index_name": index_keyspace_map[idx][0],
                        "target_index_modify_tuples": idx_insert,
                    })
        pbar.update(1)
    pbar.close()
    del query_cache

    # Output all the OU features that have data.
    for key in ou_features:
        if len(ou_features[key]) > 0:
            filename = f"{scratch_ou}/{key}.feather.{chunk_num}"
            pd.DataFrame(ou_features[key]).to_feather(filename)

    del ou_features

    # Generate all the index related operations.
    generate_index_inserts(chunk_num, chunk, augment_chunk, keyspace_augs, index_keyspace_map, index_stats, scratch_ou)
    del keyspace_augs


def generate_query_ous(conn, compute_frames, use_workload_table_estimates, input_dir, tables, workload_model, ff_change_map, index_table_map, index_keyspace_map, table_oid_map, trigger_map, scratch_ous):
    if Path(f"{scratch_ous}/ou_done").exists():
        return

    # We assumed that we have already processed the window slices in a preceding step.
    # Here we just have to load them, compute relevant aspects and then voila!

    # We don't have a good way of ensuring that keyspace_metadata_read and the inputs are consistent.
    # Do your due diligence.
    table_attr_map, _, table_keyspace_map, _, _, _ = keyspace_metadata_read(f"{input_dir}/analysis")
    key_fn = lambda x: int(x.split("_")[-1])
    chunks = sorted(glob.glob(f"{input_dir}/snippets/chunk_*"), key=key_fn)

    data_map = {}
    window_stats = {"chunk_num": 0}
    index_stats = {"chunk_num": 0}

    # Find the last processed chunk and skip them.
    if Path(f"{scratch_ous}/state.pickle").exists():
        with open(f"{scratch_ous}/state.pickle", "rb") as f:
            window_stats = pickle.load(f)
            index_stats = pickle.load(f)

        if compute_frames:
            # Load the frame if it exists.
            for tbl in tables:
                if Path(f"{scratch_ous}/{tbl}.feather").exists():
                    data_map[tbl] = pd.read_feather(f"{scratch_ous}/{tbl}.feather")
        else:
            # Otherwise construct the correct visible table map.
            for tbl in tables:
                start = window_stats["chunk_num"]
                while start >= 0 and not Path(f"{input_dir}/snippets/chunk_{start}/{tbl}.feather").exists():
                    start = start - 1

                if Path(f"{input_dir}/snippets/chunk_{start}/{tbl}.feather").exists():
                    data_map[tbl] = pd.read_feather(f"{input_dir}/snippets/chunk_{start}/{tbl}.feather")

        chunks = chunks[window_stats["chunk_num"]:]
    else:
        # Here we assume that the "initial" state with which populate_data() was invoked with
        # and the state at which the query stream was captured is roughly consistent (along
        # with the state at which we are assessing pgstattuple_approx) to some degree.
        with conn.cursor(row_factory=dict_row) as cursor:
            for tbl in tables:
                result = [r for r in cursor.execute(f"SELECT * FROM pgstattuple_approx('{tbl}')", prepare=False)][0]
                pgc_record = [r for r in cursor.execute(f"SELECT * FROM pg_class where relname = '{tbl}'", prepare=False)][0]

                ff = 100
                if pgc_record["reloptions"] is not None:
                    for record in pgc_record["reloptions"]:
                        for key, value in re.findall(r'(\w+)=(\w*)', record):
                            if key == "fillfactor":
                                ff = float(value)
                                break

                window_stats[tbl] = {
                    "table_len": result["table_len"],
                    "approx_free_percent": result["approx_free_percent"] / 100.0,
                    "dead_tuple_percent": result["dead_tuple_percent"] / 100.0,
                    "approx_tuple_count": result["approx_tuple_count"],
                    "tuple_len_avg": result["approx_tuple_len"] / result["approx_tuple_count"],
                    "ff": ff,
                 }

            for _, (idx, _) in index_keyspace_map.items():
                result = [r for r in cursor.execute(f"SELECT * FROM pgstattuple('{idx}')", prepare=False)][0]
                index_stats[idx] = {
                    "table_len": result["table_len"],
                    "approx_tuple_count": result["tuple_count"],
                    "tuple_len_avg": 0.0 if result["tuple_count"] == 0 else result["tuple_len"] / result["tuple_count"],
                    "num_inserts": 0,
                }

            with open(f"{scratch_ous}/state.pickle.0", "wb") as f:
                pickle.dump(window_stats, f)
                pickle.dump(index_stats, f)

    for chunk in tqdm(chunks):
        chunk_num = key_fn(chunk)

        # If we aren't computing frames, look at if there are update frames to use.
        if not compute_frames:
            for tbl in tables:
                if Path(f"{chunk}/{tbl}.feather").exists():
                    data_map[tbl] = pd.read_feather(f"{chunk}/{tbl}.feather")

        # Perform a mind-trick on postgres.
        conn.execute("SELECT qss_clear_stats()", prepare=False)
        implant_stats_to_postgres(conn, window_stats)
        implant_stats_to_postgres(conn, index_stats)

        augment_chunk = pd.read_feather(f"{chunk}/augment_chunk.feather")
        assert np.sum(augment_chunk.target.isna()) == 0
        assert np.sum(augment_chunk.target_index_name.isna()) == augment_chunk.shape[0]
        assert np.sum(augment_chunk.num_modify.isna()) == 0.0
        augment_chunk.drop(columns=["target_index_name"], inplace=True)

        query_chunk = pd.read_feather(f"{chunk}/chunk.feather")
        assert np.sum(query_chunk.target_index_name.isna()) == query_chunk.shape[0]
        assert np.sum(query_chunk.num_modify.isna()) == 0.0
        query_chunk.drop(columns=["target_index_name"], inplace=True)

        for tbl, queries in augment_chunk.groupby(by=["target"]):
            # Assert that we have only 1 target table.
            assert "," not in tbl

            keyspace = []
            if tbl in table_keyspace_map and tbl in table_keyspace_map[tbl]:
                keyspace = table_keyspace_map[tbl][tbl]

            data = data_map[tbl].copy() if tbl in data_map else None
            inputs = WorkloadModel.featurize(queries, data, window_stats[tbl], keyspace, workload_model.get_hist_length(), train=False)
            inputs = workload_model.prepare_inputs(pd.DataFrame([inputs]), train=False)
            batch = next(iter(torch.utils.data.DataLoader(inputs, batch_size=1)))

            prediction = workload_model.predict(*batch)[0]
            for i, target in enumerate(workload_model.get_targets()):
                window_stats[tbl][target] = prediction[i].item()

                if target == "extend_percent" or target == "defrag_percent" or target == "hot_percent":
                    window_stats[tbl][target] = max(0, min(1, window_stats[tbl][target]))

        with open(f"{scratch_ous}/state.pickle.{chunk_num}", "wb") as f:
            # Redump the predictions.
            pickle.dump(window_stats, f)
            pickle.dump(index_stats, f)

        # Generate all the query operating units for this chunk.
        generate_ous_for_chunk(conn, workload_model, chunk_num, query_chunk, augment_chunk, window_stats, index_stats, index_table_map, index_keyspace_map, table_oid_map, trigger_map, scratch_ous)

        # Mutate all the table state.
        for tbl, queries in augment_chunk.groupby(by=["target"]):
            queries = queries[queries.OP != "SELECT"]
            window_stats[tbl] = mutate_table_window_state(tbl, window_stats[tbl], queries, ff_change_map, use_workload_table_estimates)

        # Compute the changes to the data map if needed.
        if compute_frames:
            join_map = {}
            touched_tbls = {}
            compute_frames_change(query_chunk, data_map, join_map, touched_tbls, table_attr_map, table_keyspace_map, logger)
            del join_map
            del touched_tbls

            for tbl, frame in data_map.items():
                # Write out a copy to facilitate restarts.
                frame.reset_index(drop=False).to_feather(f"{scratch_ous}/{tbl}.feather")

        next_chunk_num = chunk_num + 1
        window_stats["chunk_num"] = next_chunk_num
        index_stats["chunk_num"] = next_chunk_num
        with open(f"{scratch_ous}/state.pickle", "wb") as f:
            pickle.dump(window_stats, f)
            pickle.dump(index_stats, f)

        with open(f"{scratch_ous}/state.pickle.{next_chunk_num}", "wb") as f:
            pickle.dump(window_stats, f)
            pickle.dump(index_stats, f)

    open(f"{scratch_ous}/ou_done", "w").close()

##################################################################################
# Attach metadata to query operating units
##################################################################################

def prepare_metadata(conn, scratch_it):
    with conn.cursor(row_factory=dict_row) as cursor:
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

        # FIXME(KNOBS): We assume that knobs can't change over time. Otherwise, we need to capture that.
        pg_settings["time"] = 0.0
        pg_settings["unix_timestamp"] = 0.0
        pg_settings = pd.DataFrame([pg_settings])
        pg_settings.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
        pg_settings.sort_index(axis=0, inplace=True)

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

        # Now let's produce the pg_class entries.
        key_fn = lambda x: int(x.split(".")[-1])
        states = sorted(glob.glob(f"{scratch_it}/state.pickle.*"), key=key_fn)
        result_cls = [r for r in cursor.execute("SELECT c.* FROM pg_class c, pg_namespace n WHERE c.relnamespace = n.oid AND n.nspname = 'public'")]
        result_idx = [r for r in cursor.execute("SELECT i.*, c.relname FROM pg_index i, pg_class c WHERE i.indexrelid = c.oid")]
        pg_class_total = []
        pg_index_total = []
        for state in states:
            # Combine the stats together. We know that the names are unique.
            with open(state, "rb") as f:
                combined_stats = pickle.load(f)
                combined_stats.update(pickle.load(f))

            slice_num = key_fn(state)
            for record in result_cls:
                if "relname" in record and (record["relname"] in combined_stats):
                    relname = record["relname"]
                    record["reltuples"] = combined_stats[relname]["approx_tuple_count"]
                    record["relpages"] = combined_stats[relname]["table_len"] / 8192
                    record["time"] = (slice_num) * 1.0 * 1e6
                    pg_class_total.append(copy.deepcopy(record))

            for record in result_idx:
                if record["relname"] in combined_stats:
                    entry = copy.deepcopy(record)
                    entry.pop("relname")
                    entry["time"] = (slice_num) * 1.0 * 1e6
                    pg_index_total.append(entry)

    # Prepare all the augmented catalog data in timestamp order.
    process_tables, process_idxs = process_time_pg_class(pd.DataFrame(pg_class_total))
    process_pg_index = process_time_pg_index(pd.DataFrame(pg_index_total))
    time_pg_index = build_time_index_metadata(process_pg_index, process_tables.copy(deep=True), process_idxs, time_pg_attribute)
    return process_tables, time_pg_index, time_pg_stats, pg_settings


def attach_metadata_ous(conn, scratch_it):
    if ((scratch_it / "augment_query_ous")).exists():
        return

    # Get all the metadata in time order.
    process_tables, process_index, process_stats, process_settings = prepare_metadata(conn, scratch_it)

    # These are the INDEX OUs that require metadata augmentation.
    for index_ou in [OperatingUnit.IndexScan, OperatingUnit.IndexOnlyScan, OperatingUnit.ModifyTableIndexInsert]:
        column = {
            OperatingUnit.IndexOnlyScan: "IndexOnlyScan_indexid",
            OperatingUnit.IndexScan: "IndexScan_indexid",
            OperatingUnit.ModifyTableIndexInsert: "ModifyTableIndexInsert_indexid"
        }[index_ou]

        key_fn = lambda x: int(x.split(".")[-1])
        files = sorted(glob.glob(f"{scratch_it}/{index_ou.name}.feather.*"), key=key_fn)
        for target_file in files:
            if target_file.startswith("AUG"):
                # This file has already been augmented somehow.
                continue

            logger.info("[AUGMENT] Input %s", target_file)
            data = pd.read_feather(target_file)

            # This is super confusing but essentially "time" is the raw time. `unix_timestamp` is the time adjusted to
            # the correct unix_timestamp seconds. Essentially we want to join the unix_timestamp to window_slice
            # which is how the code is setup.
            #
            # TODO(TIME): Simplify and unify this.
            data["window_slice"] = data.window_slice.astype(np.float)
            data.set_index(keys=["window_slice"], drop=True, append=False, inplace=True)
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
            data.to_feather(scratch_it / f"AUG_{index_ou.name}.feather{Path(target_file).suffix}")

    process_tables.set_index(keys=["unix_timestamp"], drop=True, append=False, inplace=True)
    process_tables.sort_index(axis=0, inplace=True)
    for augment in [OperatingUnit.ModifyTableInsert, OperatingUnit.ModifyTableUpdate]:
        files = sorted(glob.glob(f"{scratch_it}/{augment.name}.feather.*"))
        for target_file in files:
            logger.info("[AUGMENT] Input %s", target_file)
            data = pd.read_feather(target_file)
            data["window_slice"] = data.window_slice.astype(np.float)
            data.set_index(keys=["window_slice"], drop=True, append=False, inplace=True)
            data.sort_index(axis=0, inplace=True)

            data = pd.merge_asof(data, process_tables, left_index=True, right_index=True, left_by=["ModifyTable_target_oid"], right_by=["oid"], allow_exact_matches=True)
            assert data.oid.isna().sum() == 0
            data.reset_index(drop=False, inplace=True)
            data.to_feather(scratch_it / f"AUG_{augment.name}.feather{Path(target_file).suffix}")

    open(scratch_it / "augment_query_ous", "w").close()

##################################################################################
# Evaluation of query plans
##################################################################################

def evaluate_query_plans(base_models, queries, scratch_it, output_path, eval_batch_size):
    if not Path(f"{scratch_it}/eval_done").exists():
        # Evaluate all the OUs.
        for ou in OperatingUnit:
            ou_type = ou.name
            key_fn = lambda x: int(x.split(".")[-1])
            files = sorted(glob.glob(f"{scratch_it}/AUG_{ou.name}.feather.*"), key=key_fn)
            if len(files) == 0:
                files = sorted(glob.glob(f"{scratch_it}/{ou.name}.feather.*"), key=key_fn)

            if len(files) == 0:
                # This case is no data for the OU.
                continue

            def logged_read_feather(target):
                logger.info("[^]: [LOAD] Input %s", target)
                return pd.read_feather(target)

            # Read each OU file in one by one and then evaluate it.
            (scratch_it / "evals" / ou_type).mkdir(parents=True, exist_ok=True)
            groupings = [files[i:i+eval_batch_size] for i in range(0,len(files),eval_batch_size)]
            for i, group in enumerate(groupings):
                df = pd.concat(map(logged_read_feather, group))
                df.reset_index(drop=True, inplace=True)
                if ou_type not in base_models:
                    # If we don't have the model for the particular OU, we just predict 0.
                    df["pred_elapsed_us"] = 0
                    # Set a bit in [error_missing_model]
                    df["error_missing_model"] = 1
                else:
                    df = evaluate_ou_model(base_models[ou_type], None, None, eval_df=df, return_df=True, output=False)
                    df["error_missing_model"] = 0

                    if OperatingUnit[ou_type] == OperatingUnit.IndexOnlyScan or OperatingUnit[ou_type] == OperatingUnit.IndexScan:
                        prefix = "IndexOnlyScan" if OperatingUnit[ou_type] == OperatingUnit.IndexOnlyScan else "IndexScan"
                        df["pred_elapsed_us"] = df.pred_elapsed_us * df[f"{prefix}_num_outer_loops"]
                df.to_feather(scratch_it / "evals" / ou_type / f"evals_{i}.feather")

        open(f"{scratch_it}/eval_done", "w").close()

    # Massage the frames together from disk to combat OOM.
    unified_df = None
    glob_files = []
    for ou in OperatingUnit:
        ou_type = ou.name
        glob_files.extend(glob.glob(f"{scratch_it}/evals/{ou_type}/evals_*.feather"))
    assert len(glob_files) > 0

    key_fn = lambda x: int(x.split(".feather")[0].split("_")[-1]) / eval_batch_size
    glob_files = sorted(glob_files, key=key_fn)
    df_files = []
    for _, group in itertools.groupby(glob_files, key=key_fn):
        def logged_read(input_file):
            logger.info("[^]: [MASSAGE] Input %s", input_file)
            keep_columns = ["query_id", "query_order", "pred_elapsed_us", "error_missing_model"]
            df = pd.read_feather(input_file, columns=keep_columns)
            return df

        df = pd.concat(map(logged_read, group))
        df.reset_index(drop=True).groupby(["query_id", "query_order"]).sum()
        df_files.append(df)

    combined_frame = pd.concat(df_files, ignore_index=True).groupby(by=["query_id", "query_order"]).sum()
    unified_df = combined_frame.reset_index(drop=False)
    del combined_frame
    gc.collect()

    assert unified_df is not None
    unified_df.sort_values(by=["query_id", "query_order"], inplace=True, ignore_index=True)
    unified_df.set_index(keys=["query_id", "query_order"], inplace=True)
    assert unified_df.index.is_unique

    queries.set_index(keys=["query_id", "query_order"], inplace=True)
    queries = queries.join(unified_df, how="inner")
    queries.reset_index(drop=False, inplace=True)
    assert np.sum(queries.pred_elapsed_us.isna()) == 0
    queries.to_feather(output_path / "query_results.feather")

##################################################################################
# Control
##################################################################################

def compute_ddl_changes(dir_data, tables):
    ddl = pd.read_csv(f"{dir_data}/pg_qss_ddl.csv")
    ddl = ddl[ddl.command == "AlterTableOptions"]
    ff_tbl_change_map = {t: [] for t in tables}
    for tbl in tables:
        query_str = ddl["query"].str
        slots = query_str.contains(tbl) & query_str.contains("fillfactor")
        tbl_changes = ddl[slots]
        for q in tbl_changes.itertuples():
            root = pglast.Node(pglast.parse_sql(q.query))
            for node in root.traverse():
                if isinstance(node, pglast.node.Node):
                    if isinstance(node.ast_node, pglast.ast.DefElem):
                        if node.ast_node.defname == "fillfactor":
                            ff = node.ast_node.arg.val
                            ff_tbl_change_map[tbl].append((q.statement_timestamp, ff))
    return ff_tbl_change_map


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
    model_dict = torch.load(path)

    args = model_dict["model_args"]
    num_outputs = model_dict["num_outputs"]
    model = model_cls(args, num_outputs)
    model.load_state_dict(model_dict["best_model"])
    return model


def main(workload_conn, target_conn, workload_analysis_prefix,
         input_dir, session_sql, ou_models_path,
         table_feature_model_path, buffer_page_model_path,
         buffer_access_model_path, concurrency_model_path,
         table_feature_granularity_queries,
         buffer_access_granularity_queries,
         concurrency_granularity_sec,
         histogram_width,
         output_dir):

    logger.info("Beginning eval_query_workload evaluation of %s", input_dir)
    assert False
    target_conn.execute("SET qss_capture_enabled = OFF")
    if session_sql is not None and session_sql.exists():
        with open(session_sql, "r") as f:
            for line in f:
                target_conn.execute(line)

    # Load the models
    ou_models = load_ou_models(ou_models_path)
    table_feature_model = load_model(table_feature_model_path, "TableFeatureModel")
    buffer_page_model = load_model(buffer_page_model_path, "BufferPageModel")
    buffer_access_model = load_model(buffer_access_model, "BufferAccessModel")
    concurrency_model = load_model(concurrency_model, "ConcurrencyModel")

    # Get the query order ranges.
    with workload_conn.transaction():
        r = [r for r in workload_conn.execute(f"SELECT min(query_order), max(query_order) FROM {workload_analysis_prefix}_mw_queries_args")][0]
        min_qo, max_qo = r[0], r[1]

    wa = keyspace_metadata_read(input_dir)[0]
    tables = wa.table_attr_map.keys()

    def save_bucket_keys_to_output(output):
        def save_df_return_none(tbl, df):
            Path(output).mkdir(parents=True, exist_ok=True)
            df["key_dist"] = [",".join(map(str, l)) for l in df.key_dist]
            df.to_feather(f"{output}/{tbl}.feather")
            return None

        return save_df_return_none

    # Populate the buffer access keyspace.
    if table_feature_granularity_queries != buffer_access_granularity_queries and not (Path(output_dir) / "scratch/buffer_access/done").exists():
        query_orders = range(min_qo, max_qo, buffer_access_granularity_queries)
        window_index_map = {t: [i for i in query_orders] for t in tables}
        callback = save_bucket_keys_to_output(Path(output_dir) / "scratch/buffer_access")
        construct_keyspaces(logger, workload_conn, workload_analysis_prefix, tables, wa.table_attr_map, window_index_map, histogram_width, gen_data=False, gen_op=True, callback_fn=callback)
        open(f"{output_dir}/scratch/buffer_access/done", "w").close()

    # Populate the table feature keyspace.
    if not (Path(output_dir) / "scratch/table_feature/done").exists():
        query_orders = range(min_qo, max_qo, table_feature_granularity_queries)
        window_index_map = {t: [i for i in query_orders] for t in tables}
        callback = save_bucket_keys_to_output(Path(output_dir) / "scratch/table_feature")
        construct_keyspaces(logger, connection, work_prefix, tables, wa.table_attr_map, window_index_map, buckets, gen_data=True, gen_op=True, callback_fn=callback)
        open(f"{output_dir}/scratch/table_feature/done", "w").close()

    shared_buffers = [r for r in target_conn.execute("SHOW shared_buffers")][0][0]
    ddl_changes = compute_ddl_changes(input_dir, tables)

    qos_windows = set(range(min_qo, max_qo, table_feature_granularity_queries) + range(min_qo, max_qo, buffer_access_granularity_queries))
    qos_windows = sorted(list(qos_windows))
    current_qo = 0
    while current_qo < qos_windows[-1]:
        pass



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

    table_feature_model_path = cli.SwitchAttr(
        "--table-feature-model-path",
        Path,
        default=None,
        help="Path to the table feature model that should be loaded.",
    )

    table_feature_granularity_queries = cli.SwitchAttr(
        "--table-feature-granularity-queries",
        int,
        default=1000,
        help="Granularity of window slices for table feature model in (# queries).",
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

    buffer_access_granularity_queries = cli.SwitchAttr(
        "--buffer-access-granularity-queries",
        int,
        default=1000,
        help="Granularity of window slices for buffer access model in (# queries).",
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
                     self.table_feature_model_path,
                     self.buffer_page_model_path,
                     self.buffer_access_model_path,
                     self.concurrency_model_path,
                     self.table_feature_granularity_queries,
                     self.buffer_access_granularity_queries,
                     self.concurrency_granularity_sec,
                     self.histogram_width,
                     self.output_dir)


if __name__ == "__main__":
    EvalQueryWorkloadCLI.run()
