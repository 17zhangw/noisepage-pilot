import numpy as np
from behavior.model_workload.models import construct_stack, MAX_KEYS
from behavior.model_workload.utils import OpType

COMMENT_TO_OPTYPE = {
    "IndexScan": OpType.SELECT,
    "IndexOnlyScan": OpType.SELECT,
    "ModifyTableInsert": OpType.INSERT,
    "ModifyTableUpdate": OpType.UPDATE,
    "ModifyTableDelete": OpType.DELETE,
}

# Beware that this drops the "data" keyspace.
def extract_train_tables_keys_features(add_nonnorm, tbl_map, tbl_mapping, keys, hist_width, window, window_slot):
    key_bias = np.zeros((len(tbl_mapping), 4 * hist_width))
    key_dists = np.zeros((len(tbl_mapping), MAX_KEYS, 4 * hist_width))
    masks = np.zeros((len(tbl_mapping), MAX_KEYS, 1))
    addt_feats = np.zeros((len(tbl_mapping), (4 if add_nonnorm else 2)))
    all_bias = np.zeros((len(tbl_mapping), 1))
    all_requested = window.total_blks_requested.sum()

    for t, f in window.groupby(by=["target"]):
        all_bias[tbl_mapping[t], 0] = f.total_blks_requested.sum() / all_requested

        addt_feats[tbl_mapping[t], 0] = f.norm_relpages.mean()
        addt_feats[tbl_mapping[t], 1] = f.norm_reltuples.mean()
        if add_nonnorm:
            addt_feats[tbl_mapping[t], 2] = f.relpages.mean()
            addt_feats[tbl_mapping[t], 3] = f.reltuples.mean()

        if t in tbl_map:
            d = tbl_map[t]
            d = d[d.window_index == window_slot]
            for g in d.groupby(by=["optype"]):
                assert g[0] != "data"
                for ig in g[1].itertuples():
                    j = None
                    for idx_kt, kt in enumerate(keys[t]):
                        # Find the correct attribute index to use for the data.
                        if kt == ig.att_name:
                            j = idx_kt
                            break
                    assert j is not None, "There is a misalignment between what is considered a useful attribute by data pages and analysis."

                    # This is because OpType is 1-indexed.
                    key_dists[tbl_mapping[t], j, (int(g[0]) - 1) * hist_width:int(g[0]) * hist_width] = np.array([float(f) for f in ig.key_dist.split(",")])
                    masks[tbl_mapping[t], j] = 1

        if "total_tuples_touched" in f:
            total_access = f.total_tuples_touched.sum()
        else:
            assert "num_queries" in f
            total_access = f.num_queries.sum()

        if total_access > 0:
            for g in f.itertuples():
                g_access = g.total_tuples_touched if "total_tuples_touched" in f else g.num_queries
                g_index = COMMENT_TO_OPTYPE[g.comment].value if "comment" in f else g.optype

                sl = key_bias[tbl_mapping[t], (g_index-1)*hist_width:g_index*hist_width]
                sl += np.full(hist_width, g_access / total_access)
                key_bias[tbl_mapping[t], (g_index-1)*hist_width:g_index*hist_width] = sl

    return key_bias, key_dists, masks, all_bias, addt_feats


# Beware that this drops the "data" keyspace.
def extract_infer_tables_keys_features(model_args, window_slot, all_requested, tbl_mapping, table_attr_map, table_state, keyspace_feat_space):
    hist_width = model_args.hist_width
    add_nonnorm = model_args.add_nonnorm_features

    key_bias = np.zeros((len(tbl_mapping), 4 * hist_width))
    key_dists = np.zeros((len(tbl_mapping), MAX_KEYS, 4 * hist_width))
    masks = np.zeros((len(tbl_mapping), MAX_KEYS, 1))
    addt_feats = np.zeros((len(tbl_mapping), (4 if add_nonnorm else 2)))
    all_bias = np.zeros((len(tbl_mapping), 1))

    for tbl, state in table_state.items():
        all_bias[tbl_mapping[tbl], 0] = 1.0 if all_requested == 0.0 else state["total_blks_requested"] / all_requested

        addt_feats[tbl_mapping[tbl], 0] = state["norm_relpages"]
        addt_feats[tbl_mapping[tbl], 1] = state["norm_reltuples"]
        if add_nonnorm:
            addt_feats[tbl_mapping[tbl], 2] = state["num_pages"]
            addt_feats[tbl_mapping[tbl], 3] = state["tuple_count"]

        if tbl in keyspace_feat_space:
            window_df = keyspace_feat_space[tbl]
            window_df = window_df[window_df.window_index == window_slot]
            for g in window_df.groupby(by=["optype"]):
                if g[0] == "data":
                    # Skip the data featurization for buffer access.
                    continue

                for ig in g[1].itertuples():
                    j = None
                    for j, kt in enumerate(table_attr_map[tbl]):
                        # Find the correct attribute index to use for the data.
                        if kt == ig.att_name:
                            break
                    assert j is not None, "There is a misalignment between what is considered a useful attribute by data pages and analysis."

                    # This is because OpType is 1-indexed.
                    key_dists[tbl_mapping[tbl], j, (int(g[0]) - 1) * hist_width:int(g[0]) * hist_width] = np.array([float(f) for f in ig.key_dist.split(",")])
                    masks[tbl_mapping[tbl], j] = 1

        total_access = state["total_tuples_touched"]
        mods = [
            (OpType.SELECT.value, "num_select_tuples"),
            (OpType.INSERT.value, "num_insert_tuples"),
            (OpType.UPDATE.value, "num_update_tuples"),
            (OpType.DELETE.value, "num_delete_tuples"),
        ]

        if total_access > 0:
            for (g_index, num_queries) in mods:
                sl = key_bias[tbl_mapping[tbl], (g_index-1)*hist_width:g_index*hist_width]
                sl += np.full(hist_width, state[num_queries] / total_access)
                key_bias[tbl_mapping[tbl], (g_index-1)*hist_width:g_index*hist_width] = sl

    return key_bias, key_dists, masks, all_bias, addt_feats
