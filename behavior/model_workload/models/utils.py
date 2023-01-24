##################################################################################
# Table inputs for model training/inference
##################################################################################

def __generate_point_input_table(model_args, input_row, df, tbl, tbl_attr_keys, ff_value):
    hist_width = model_args.hist_width

    # Base inputs.
    row = {
        "free_percent": (input_row.free_percent if "free_percent" in input_row else input_row.approx_free_percent) / 100.0,
        "dead_tuple_percent": input_row.dead_tuple_percent / 100.0,
        "num_pages": input_row.num_pages,
        "tuple_count": input_row.tuple_count if "tuple_count" in input_row else input_row.approx_tuple_count,
        "tuple_len_avg": input_row.tuple_len_avg,
        "target_ff": ff_value,
        "vacuum": input_row.vaccum if "vacuum" in input_row else 0,
    }

    # Characterize the distribution of select, insert, update, and delete queries.
    num_queries = input_row.num_select_queries + input_row.num_insert_queries + input_row.num_update_queries + input_row.num_delete_queries
    num_touch = input_row.num_select_tuples + input_row.num_modify_tuples
    row.extend({
        "num_select_queries": input_row.num_select_queries,
        "num_insert_queries": input_row.num_insert_queries,
        "num_update_queries": input_row.num_update_queries,
        "num_delete_queries": input_row.num_delete_queries,

        "select_queries_dist": 0.0 if num_queries == 0 else input_row.num_select_queries / num_queries,
        "insert_queries_dist": 0.0 if num_queries == 0 else input_row.num_insert_queries / num_queries,
        "update_queries_dist": 0.0 if num_queries == 0 else input_row.num_update_queries / num_queries,
        "delete_queries_dist": 0.0 if num_queries == 0 else input_row.num_delete_queries / num_queries,

        "num_select_tuples": input_row.num_select_tuples,
        "num_insert_tuples": input_row.num_insert_tuples,
        "num_update_tuples": input_row.num_update_tuples,
        "num_delete_tuples": input_row.num_delete_tuples,

        "select_tuples_dist": 0.0 if num_touch == 0 else input_row.num_select_tuples / num_touch,
        "insert_tuples_dist": 0.0 if num_touch == 0 else input_row.num_insert_tuples / num_touch,
        "update_tuples_dist": 0.0 if num_touch == 0 else input_row.num_update_tuples / num_touch,
        "delete_tuples_dist": 0.0 if num_touch == 0 else input_row.num_delete_tuples / num_touch,
    })

    # Construct key dists.
    seen = []
    if df is not None and "att_name" in df:
        for ig in df.itertuples():
            j = None
            for idx_kt, kt in enumerate(tbl_attr_keys):
                if kt == ig.att_name:
                    j = idx_kt
                    break
            assert j is not None, "There is a misalignment between what is considered a useful attribute by data pages and analysis."
            assert (ig.optype, j) not in seen
            seen.append((ig.optype, j))

            col = tbl_attr_keys[j]
            name = ig.optype
            if not (ig.optype == "data" or ig.optype == "SELECT" or ig.optype == "INSERT" or ig.optype == "UPDATE" or ig.optype == "DELETE"):
                name = OpType(int(ig.optype)).name

            name = name.lower()
            key_dist = [float(f) for f in ig.key_dist.split(",")]
            for i in range(0, hist_width):
                if model_args.keep_identity:
                    row[f"{tbl}_{col}_{name}_{i}"] = key_dist[i]
                else:
                    row[f"key{j}_{name}_{i}"] = key_dist[i]
    return row


def generate_dataset_table(logger, model_args):
    tbl_attr_keys = {}
    # Construct the table attr mappings.
    for d in model_args.input_dirs:
        c = f"{d}/keyspaces.pickle"
        assert Path(c).exists()
        with open(c, "rb") as f:
            metadata = pickle.load(f)

        for t, k in metadata.table_attr_map.items():
            if t not in tbl_attr_keys:
                tbl_attr_keys[t] = k

    input_dataset = []
    correlated_tbls = []
    window_index = []
    for d in model_args.input_dirs:
        input_files = glob.glob(f"{d}/exec_features/data/*.feather")
        input_files = [f for f in input_files if "_index" not in f]
        pg_class = pd.read_csv(f"{d}/pg_class.csv")

        for input_file in input_files:
            root = Path(input_file).stem
            data = pd.read_feather(input_file)
            windows = pd.read_feather(f"{d}/exec_features/windows/{root}.feather")
            windows["num_pages"] = windows.table_len / 8192.0
            windows["tuple_len_avg"] = (windows.tuple_len / windows.tuple_count) if "tuple_count" in windows else (windows.approx_tuple_len / windows.approx_tuple_count)

            data.set_index(keys=["window_index"], inplace=True)
            windows.set_index(keys=["window_index"], inplace=True)
            data = data.join(windows, how="inner")

            if Path(f"{d}/exec_features/keys/{root}.feather").exists():
                keys = pd.read_feather(f"{d}/exec_features/keys/{root}.feather")
                keys = keys[keys.index_clause.isna()]
                keys.set_index(keys=["window_index"], inplace=True)
                data = data.join(keys, how="inner")

            relation = pg_class[pg_class.relname == root].iloc[0]
            ff_value = 1.0
            if relation.reloptions is not None and not isinstance(relation.reloptions, np.float):
                reloptions = ast.literal_eval(relation.reloptions)
                for opt in reloptions:
                    for key, value in re.findall(r'(\w+)=(\w*)', opt):
                        if key == "fillfactor":
                            # Fix fillfactor options.
                            ff_value = float(value) / 100.0

            data.reset_index(drop=False, inplace=True)
            for wi, df in data.groupby(by=["window_index"], sort=True):
                input_row = __generate_point_input_table(model_args, df.iloc[0], df, root, tbl_attr_keys[root], ff_value)

                # FIXME(TARGET): Assume all tuples have equal probability of triggering the event.
                actual_insert = df.iloc[0].num_insert_tuples + df.iloc[0].num_update_tuples - df.iloc[0].num_hot
                input_row["extend_percent"] = 0.0 if actual_insert == 0 else df.iloc[0].num_extend / actual_insert
                input_row["defrag_percent"] = 0.0 if df.iloc[0].num_select_tuples == 0 else df.iloc[0].num_defrag / df.iloc[0].num_select_tuples
                input_row["hot_percent"] = 0.0 if df.iloc[0].num_update_tuples == 0 else df.iloc[0].num_hot / df.iloc[0].num_update_tuples
                input_row = {k:v for k, v in input_row if k in TABLE_STATE_INPUTS}

                input_dataset.append(input_row)
                correlated_tbls.append(root)
                window_index.append(wi)

    return input_dataset, correlated_tbls, window_index


def generate_inference_table(model_args, predict_fn, table_state, table_attr_map, keyspace_feat_space, window, output_df=None):
    inputs = []
    tbl_keys = [t for t in table_state]
    for i, t in enumerate(tbl_keys):
        df = None
        if t in keyspace_feat_space:
            # Yoink the relevant keyspace features.
            df = keyspace_feat_space[t]
            df = df[df.window_index == window]
            df = df[df.index_clause.isna()]

        # Generate the inputs.
        input_row = __generate_point_input_table(model_args, Map(table_state[t]), df, t, table_attr_map[t], table_state[t]["target_ff"])
        input_row["vacuum"] = table_state[t]["vacuum"]
        input_row["table"] = t
        inputs.append(input_row)

    # Generate the predictions.
    inputs = pd.DataFrame(inputs)
    inputs.fillna(value=0, inplace=True)
    predictions = predict_fn(inputs)

    ret_df = None
    if output_df:
        inputs["window"] = window
        predictions.columns = "pred_" + predictions.columns
        rename_outputs = predictions.rename(columns={k: "pred_" + k for k in predictions.columns})
        ret_df = pd.concat([inputs, rename_outputs], axis=1)

    return tbl_keys, predictions, ret_df

##################################################################################
# Index inputs for model training/inference
##################################################################################

def __generate_point_input_index(model_args, input_row, df, idx, idx_attr_keys):
    hist_width = model_args.hist_width

    # Construct the base inputs.
    row = {
        "key_size": input_row.key_size,
        "key_natts": input_row.key_natts,
        "tree_level": input_row.tree_level,
        "num_pages": input_row.num_pages,
        "leaf_pages": input_row.leaf_pages,
        "empty_pages": input_row.empty_pages,
        "deleted_pages": input_row.deleted_pages,
        "avg_leaf_density": input_row.avg_leaf_density / 100.0,
        "rel_num_pages": input_row.rel_num_pages,
        "rel_num_tuples": input_row.rel_num_tuples,
        "num_index_inserts": input_row.num_inserts,
    }

    # Characterize the distribution of select, insert, update, and delete queries.
    num_queries = input_row.num_select_queries + input_row.num_insert_queries + input_row.num_update_queries + input_row.num_delete_queries
    num_touch = input_row.num_select_tuples + input_row.num_modify_tuples
    row.extend({
        "num_select_queries": input_row.num_select_queries,
        "num_insert_queries": input_row.num_insert_queries,
        "num_update_queries": input_row.num_update_queries,
        "num_delete_queries": input_row.num_delete_queries,

        "select_queries_dist": 0.0 if num_queries == 0 else input_row.num_select_queries / num_queries,
        "insert_queries_dist": 0.0 if num_queries == 0 else input_row.num_insert_queries / num_queries,
        "update_queries_dist": 0.0 if num_queries == 0 else input_row.num_update_queries / num_queries,
        "delete_queries_dist": 0.0 if num_queries == 0 else input_row.num_delete_queries / num_queries,

        "num_select_tuples": input_row.num_select_tuples,
        "num_insert_tuples": input_row.num_insert_tuples,
        "num_update_tuples": input_row.num_update_tuples,
        "num_delete_tuples": input_row.num_delete_tuples,

        "select_tuples_dist": 0.0 if num_touch == 0 else input_row.num_select_tuples / num_touch,
        "insert_tuples_dist": 0.0 if num_touch == 0 else input_row.num_insert_tuples / num_touch,
        "update_tuples_dist": 0.0 if num_touch == 0 else input_row.num_update_tuples / num_touch,
        "delete_tuples_dist": 0.0 if num_touch == 0 else input_row.num_delete_tuples / num_touch,
    })

    # Construct key dists.
    seen = []
    if df is not None and "att_name" in df:
        for ig in df.itertuples():
            j = None
            for idx_kt, kt in enumerate(tbl_attr_keys):
                if kt == ig.att_name:
                    j = idx_kt
                    break

            if j is None:
                continue

            assert (ig.optype, j) not in seen
            seen.append((ig.optype, j))

            col = tbl_attr_keys[j]
            name = ig.optype
            if not (ig.optype == "data" or ig.optype == "SELECT" or ig.optype == "INSERT" or ig.optype == "UPDATE" or ig.optype == "DELETE"):
                name = OpType(int(ig.optype)).name

            name = name.lower()
            key_dist = [float(f) for f in ig.key_dist.split(",")]
            for i in range(0, hist_width):
                if model_args.keep_identity:
                    row[f"{idx}_{col}_{name}_{i}"] = key_dist[i]
                else:
                    row[f"key{j}_{name}_{i}"] = key_dist[i]
    return row


def identify_key_size(input_dir, idx):
    # FIXME(DDL): Assume that there aren't DDL changes.
    pg_class = pd.read_csv(f"{input_dir}/pg_class.csv")
    pg_class = pg_class[pg_class.relname == idx]
    if pg_class.shape[0] == 0:
        return 0, 0

    idxoid = pg_class.iloc[0].oid
    pg_index = pd.read_csv(f"{input_dir}/pg_index.csv")
    pg_index = pg_idnex[pg_index.indexrelid == idxoid]
    if pg_index.shape[0] == 0:
        return 0, 0

    key_size = 0
    attnums = pg_index.iloc[0].indkey.str.split(" ")
    pg_att = pd.read_csv(f"{input_dir}/pg_attribute.csv")
    for attnum in attnums:
        atts = pg_att[(pg_att.attrelid == pg_index.indrelid) & (pg_att.attnum == int(attnum))]
        if atts.shape[0] == 0:
            return 0, 0

        if atts.iloc[0].attlen != -1:
            key_size += att.iloc[0].attlen
        elif atts.iloc[0].atttypmod != -1:
            key_size += att.iloc[0].atttypmod
    return key_size, len(attnums)


def generate_dataset_index(logger, model_args):
    idx_attr_keys = {}
    # Construct the table attr mappings.
    for d in model_args.input_dirs:
        c = f"{d}/keyspaces.pickle"
        assert Path(c).exists()
        with open(c, "rb") as f:
            metadata = pickle.load(f)

        for t, k in metadata.table_keyspace_map.items():
            for idx, idxk in k.items():
                if idx not in idx_attr_keys:
                    idx_attr_keys[idx] = idxk

    input_datasets = []
    correlated_idxs = []
    window_index = []
    for d in model_args.input_dirs:
        input_files = sorted(glob.glob(f"{d}/exec_features/data/*_index.feather"))
        for input_file in input_files:
            # Analyze each index in the input frame.
            all_idxs = pd.read_feather(input_file)
            all_idxs["unix_timestamp"] = postgres_julian_to_unix(all_idxs.start_timestamp).astype(float)
            unique_targets = [f for f in all_idxs.target.unique() if f not in metadata.table_attr_map]
            for idx in unique_targets:
                # Find the table this index belongs to.
                idxoid = None
                for idxo, name in metadata.indexoid_name_map.items():
                    if name == idx:
                        idxoid = idxo
                        break
                assert idxoid is not None
                tbl = metadata.indexoid_table_map[idxoid]

                data = all_idxs[(all_idxs.target == tbl) | (all_idxs.target == idx)].copy()
                data.set_index(keys=["unix_timestamp"], inplace=True)
                data.sort_index(inplace=True)

                # Read in the relevant parts of the relation.
                tbls_md = pd.read_csv(f"{d}/{tbl}.csv")
                tbls_md["time"] = tbls_md.time / 1e6
                tbls_md["rel_num_pages"] = tbls_md.table_len / 8192.0
                tbls_md["rel_num_tuples"] = tbls.tuple_count if "tuple_count" in tbls else tbls.approx_tuple_count
                tbls_md = tbls_md[["time", "rel_num_pages", "rel_num_tuples"]]
                tbls_md.set_index(keys=["time"], inplace=True)
                tbls_md.sort_index(inplace=True)
                data = pd.merge_asof(data, tbls_md, left_index=True, right_index=True, direction="forward", allow_exact_matches=True)
                data.reset_index(drop=False, inplace=True)
                assert np.sum(data.rel_num_pages.isna()) == 0
                data.set_index(keys=["unix_timestamp"], inplace=True)
                data.sort_index(inplace=True)

                # Read in the correct index metadata.
                windows = pd.read_csv(f"{d}/{idx}.csv")
                windows["time"] = windows.time / 1e6
                windows["num_pages"] = windows.index_size / 8192.0
                windows.set_index(keys=["time"], inplace=True)
                windows.sort_index(inplace=True)
                data = pd.merge_asof(data, windows, left_index=True, right_index=True, direction="forward", allow_exact_matches=True)
                data.reset_index(drop=False, inplace=True)
                assert np.sum(data.tree_level.isna()) == 0

                # FIXME(INDEX): There is some minor incongruity in how index execution features are summarized since they are
                # somewhat grouped against the index being used. In the cases of JOIN queries, they decompose into the index
                # entries and not the table.
                data = data.groupby(by=["window_index"]).max()

                # Now let's compute the keysize.
                key_size, numatts = identify_key_size(d, idx)
                data["key_size"] = key_size
                data["key_natts"] = key_natts

                if Path(f"{d}/exec_features/keys/{tbl}.feather").exists():
                    keys = pd.read_feather(f"{d}/exec_features/keys/{tbl}.feather")
                    # Only get relevant to this index.
                    keys = keys[(keys.index_clause == idx) | (keys.optype == "data") | (keys.optype != f"{OpType.SELECT.value}")]
                    keys.set_index(keys=["window_index"], inplace=True)
                    data = data.join(keys, on=["window_index"], how="inner")
                data.reset_index(drop=False, inplace=True)

                for wi, df in data.groupby(by=["window_index"]):
                    input_row = __generate_point_input_index(model_args, df.iloc[0], df, idx, idx_attr_keys[idx])

                    # FIXME(TARGET): Assume all tuples have equal probability of triggering the event.
                    input_row["extend_percent"] = 0.0 if df.iloc[0].num_inserts == 0 else df.iloc[0].num_extend / df.iloc[0].num_inserts
                    input_row["split_percent"] = 0.0 if df.iloc[0].num_inserts == 0 else df.iloc[0].num_split / df.iloc[0].num_inserts
                    input_datasets.append(input_row)

                    correlated_idxs.append(idx)
                    window_index.append(wi)

    return input_datasets, correlated_idxs, window_index


def generate_inference_index(model_args, predict_fn, ougc, window, output_df=None):
    inputs = []
    idx_keys = [idx for idx in ougc.index_feature_state]
    for i, idx in enumerate(idx_keys):
        tbl_state = ougc.table_feature_state[ougc.indexoid_table_map[idx]]
        extract = copy(index_feature_state[idx])
        idxname = extract["indexname"]

        # Yoink the keyspace features.
        df = None
        if idxname in keyspace_feat_space:
            # Yoink the relevant keyspace features.
            df = keyspace_feat_space[idxname]
            df = df[(df.index_clause == idxname) | (keys.optype == "data") | (keys.optype != f"{OpType.SELECT.value}")]

        tbl_copy = [
            ("rel_num_pages", "num_pages"),
            ("rel_num_tuples", "tuple_count"),
            ("num_select_queries", None),
            ("num_insert_queries", None),
            ("num_update_queries", None),
            ("num_delete_queries", None),
            ("num_select_tuples", None),
            ("num_insert_tuples", None),
            ("num_update_tuples", None),
            ("num_delete_tuples", None),
        ]
        for target, src in tbl_copy:
            extract[target] = tbl_state[src if src is not None else target]

        # Generate the inputs.
        input_row = __generate_point_input_index(model_args, Map(extract), df, idxname, ougc.table_keyspace_features[idxname])
        input_row["idx"] = idxname
        inputs.append(input_row)

    # Generate the predictions.
    inputs = pd.DataFrame(inputs)
    inputs.fillna(value=0, inplace=True)
    predictions = predict_fn(inputs)

    ret_df = None
    if output_df:
        inputs["window"] = window
        predictions.columns = "pred_" + predictions.columns
        rename_outputs = predictions.rename(columns={k: "pred_" + k for k in predictions.columns})
        ret_df = pd.concat([inputs, rename_outputs], axis=1)

    return idx_keys, predictions, ret_df
