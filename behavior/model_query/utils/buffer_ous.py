import random
from behavior import OperatingUnit

def instantiate_buffer_page_reqs(ougc, window, output_df=False):
    for _, tblstate in ougc.table_feature_state.items():
        tblstate["buffer_usage"] = {}
    for _, idxstate in ougc.index_feature_state.items():
        idxstate["buffer_usage"] = {}

    results = ougc.buffer_page_model.inference(ougc, output_col="req_pages")
    for result in results.itertuples(index=False):
        tbl = result.table
        indexoid = result.indexoid
        ou_type = result.ou_type
        if indexname is None:
            ougc.table_feature_state[tbl]["buffer_usage"][ou_type] = result.req_pages
        else:
            ougc.index_feature_state[indexoid]["buffer_usage"][ou_type] = result.req_pages
    return results


def compute_buffer_page_features(ougc, query_ous):
    # FIXME(BITMAP): Consider the buffer page usage for bitmap.
    valid_ous = {
        OperatingUnit.IndexScan.name: ("IndexScan", "IndexScan_indexid", "IndexScan_scan_est_pages_needed"),
        OperatingUnit.IndexOnlyScan.name: ("IndexOnlyScan", "IndexOnlyScan_indexid", "IndexOnlyScan_scan_est_pages_needed"),
        OperatingUnit.SeqScan.name: ("SeqScan", "SeqScan_scanrelid_oid", "SeqScan_est_pages_needed"),
        OperatingUnit.ModifyTableInsert.name: ("ModifyTableInsert", "ModifyTable_target_oid", None),
        OperatingUnit.ModifyTableUpdate.name: ("ModifyTableUpdate", "ModifyTable_target_oid", None),
        OperatingUnit.ModifyTableDelete.name: ("ModifyTableDelete", "ModifyTable_target_oid", None),
    }

    current_index = 0
    for ou in query_ous:
        if ou["node_type"] not in valid_ous:
            continue

        _, oid_code, restrict = valid_ous[ou["node_type"]]
        state = None
        max_pages = None
        if ou["node_type"] in [OperatingUnit.IndexScan.name, OperatingUnit.IndexOnlyScan.name]:
            state = ougc.index_feature_state[ou[oid_code]]
        else:
            state = ougc.table_feature_state[ougc.oid_table_map[ou[oid_code]]]
        assert state is not None

        # Default restrict to the maximum limit in the table.
        max_pages = table_state["num_pages"]

        if "buffer_usage" in state and ou["node_type"] in state["buffer_usage"]:
            # We have a result that we can use!.
            inference = state["buffer_usage"][ou["node_type"]]
        elif ou["node_type"] in [OperatingUnit.IndexScan.name, OperatingUnit.IndexOnlyScan.name]:
            # Quote the estaimted number of pages by the query.
            assert "tree_level" in state
            inference = ou[restrict]
        elif ou["node_type"] in [OperatingUnit.ModifyTableInsert.name, OperatingUnit.ModifyTableUpdate.name, OperatingUnit.ModifyTableDelete.name]:
            # Quote 2 blocks needed.
            inference = 2
        else:
            # Otherwise, just use the number of pages in the table as an estimate.
            # This will be restricted again in the case of index scans.
            inference = table_state["num_pages"]

        # Clip the number of blocks requested to be at most the "clipped" amount of pages.
        # We should probably not be fetching more even if the model says so...
        # We should also only be able to request integral number of blocks so round.
        ou["total_blks_requested"] = round(min(inference, max_pages))
        current_index += 1


def compute_buffer_access_features(ougc, query_ous, window, num_queries):
    # We can't infer block hits or block misses.
    for ou in query_ous:
        if "total_blks_requested" in ou:
            ou["blk_hit"] = ou["total_blks_requested"]
        else:
            ou["blk_hit"] = 0
        ou["blk_miss"] = 0
