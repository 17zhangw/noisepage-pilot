
def compute_buffer_page_features(ougc, query_ous):
    valid_ous = {
        OperatingUnit.IndexScan.name: ("IndexScan", "IndexScan_indexid"),
        OperatingUnit.IndexOnlyScan.name: ("IndexOnlyScan", "IndexOnlyScan_indexid"),
        OperatingUnit.SeqScan.name: ("SeqScan", "SeqScan_scanrelid_oid"),
        OperatingUnit.ModifyTableInsert.name: ("ModifyTableInsert", "ModifyTable_target_oid"),
        OperatingUnit.ModifyTableUpdate.name: ("ModifyTableUpdate", "ModifyTable_target_oid"),
        OperatingUnit.ModifyTableDelete.name: ("ModifyTableDelete", "ModifyTable_target_oid"),
    }

    if ougc.buffer_page_model is None:
        return

    frame = []
    for ou in query_ous:
        if ou["node_type"] not in valid_ous:
            continue

        comment, oid_code = valid_ous[ou["node_type"]]
        table_state = None
        if ou["node_type"] in [OperatingUnit.IndexScan.name, OperatingUnit.IndexOnlyScan.name]:
            indexoid = ou[oid_code]
            table_state = ougc.table_feature_state[ougc.indexoid_table_map[indexoid]]
        else:
            table_state = ougc.table_feature_state[ougc.oid_table_map[ou[oid_code]]]
        assert table_state is not None

        frame.append({
            "comment": comment,
            "relpages": table_state["num_pages"],
            "reltuples": table_state["approx_tuple_count"],
        })

    inference_pages = ougc.buffer_page_model.inference(frame)
    current_index = 0
    for ou in query_ous:
        if ou["node_type"] not in valid_ous:
            continue

        _, oid_code = valid_ous[ou["node_type"]]
        table_state = None
        max_pages = None
        if ou["node_type"] in [OperatingUnit.IndexScan.name, OperatingUnit.IndexOnlyScan.name]:
            indexoid = ou[oid_code]
            table_state = ougc.table_feature_state[ougc.indexoid_table_map[indexoid]]
        else:
            table_state = ougc.table_feature_state[ougc.oid_table_map[ou[oid_code]]]
        assert table_state is not None
        max_pages = table_state["num_pages"]

        # Clip the number of blocks requested to be at most the number of pages in the table.
        # We should probably not be fetching more even if the model says so...
        ou["total_blks_requested"] = min(inference_pages[current_index], max_pages)
        current_index += 1


def compute_buffer_access_features(ougc, query_ous, window, num_queries):
    if ougc.buffer_page_model is None or ougc.buffer_access_model is None:
        # We can't infer block hits or block misses.
        for ou in query_ous:
            if "total_blks_requested" in ou:
                ou["blk_hit"] = ou["total_blks_requested"]
            ou["blk_miss"] = 0
        return

    for t, tbl_state in ougc.table_feature_state:
        tbl_state["total_blks_requested"] = 0
        tbl_state["total_tuples_touched"] = tbl_state["num_select_tuples"] + tbl_state["num_modify_tuples"]

    valid_ous = {
        OperatingUnit.IndexScan.name: ("IndexScan", "IndexScan_indexid"),
        OperatingUnit.IndexOnlyScan.name: ("IndexOnlyScan", "IndexOnlyScan_indexid"),
        OperatingUnit.SeqScan.name: ("SeqScan", "SeqScan_scanrelid_oid"),
        OperatingUnit.ModifyTableInsert.name: ("ModifyTableInsert", "ModifyTable_target_oid"),
        OperatingUnit.ModifyTableUpdate.name: ("ModifyTableUpdate", "ModifyTable_target_oid"),
        OperatingUnit.ModifyTableDelete.name: ("ModifyTableDelete", "ModifyTable_target_oid"),
    }

    for ou in query_ous:
        if ou["node_type"] not in valid_ous:
            continue

        if ou["nonsynthetic"] == 0:
            continue

        if "total_blks_requested" not in ou:
            continue

        _, oid_code = valid_ous[ou["node_type"]]
        tbl_state = None
        if ou["node_type"] in [OperatingUnit.IndexScan.name, OperatingUnit.IndexOnlyScan.name]:
            indexoid = ou[oid_code]
            tbl_state = ougc.table_feature_state[ougc.indexoid_table_map[indexoid]]
        else:
            tbl_state = ougc.table_feature_state[ougc.oid_table_map[ou[oid_code]]]
        assert tbl_name is not None
        tbl_state["total_blks_requested"] += ou["total_blks_requested"]

    outputs, tbl_mapping = ougc.buffer_access_model.inference(window, num_queries, ougc.shared_buffers, ougc.table_state, ougc.table_attr_map, ougc.table_keyspace_features)

    for ou in query_ous:
        if ou["node_type"] not in valid_ous:
            continue

        if "total_blks_requested" not in ou:
            continue

        _, oid_code = valid_ous[ou["node_type"]]
        tbl_name = None
        if ou["node_type"] in [OperatingUnit.IndexScan.name, OperatingUnit.IndexOnlyScan.name]:
            indexoid = ou[oid_code]
            tbl_name = ougc.indexoid_table_map[indexoid]
        else:
            tbl_name = ougc.oid_table_map[ou[oid_code]]
        assert tbl_name is not None

        # Compute the number of pages that will probably HIT.
        ou["blk_hit"] = int(ou["total_blks_requested"] * outputs[0][tbl_mapping[tbl_name]])
        ou["blk_miss"] = ou["total_blks_requested"] - ou["blk_hit"]
