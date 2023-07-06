import re
import ast
import glob
import pickle
import joblib
import pandas as pd
import numpy as np
import tempfile
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.utils.data import dataset
import torch.nn.functional as F
from behavior.model_workload.models import MAX_KEYS
from behavior.model_workload.utils import OpType, Map
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import shutil

try:
    from autogluon.tabular import TabularDataset, TabularPredictor
except:
    pass
from behavior.model_workload.models.multilabel_predictor import MultilabelPredictor
from behavior.model_workload.models.utils import identify_key_size

from behavior import OperatingUnit

SUPPORTED_OUS_MAPPING = {
    OperatingUnit.IndexScan.name: OpType.SELECT.value,
    OperatingUnit.IndexOnlyScan.name: OpType.SELECT.value,
    OperatingUnit.SeqScan.name: OpType.SELECT.value,
    OperatingUnit.ModifyTableInsert: OpType.INSERT.value,
    OperatingUnit.ModifyTableUpdate: OpType.UPDATE.value,
    OperatingUnit.ModifyTableDelete: OpType.DELETE.value,
}

BUFFER_PAGE_INPUTS = [
    "free_percent",
    "dead_tuple_percent",
    "num_pages",
    "tuple_count",
    "tuple_len_avg",
    "target_ff",

    "index_key_natts",
    "index_key_size",
    "index_tree_level",
    "index_num_pages",
    "index_leaf_pages",
]

BUFFER_PAGE_TARGETS = [
    "asked_pages_per_tuple",
]

BUFFER_PAGE_METRICS = [
    "mean_squared_error",
]

def generate_point_input(model_args, input_row, df, identity, tbl_attr_keys, ff_value):
    hist_width = model_args.hist_width
    row = {
        "free_percent": (input_row.free_percent if "free_percent" in input_row else input_row.approx_free_percent) / 100.0,
        "dead_tuple_percent": input_row.dead_tuple_percent / 100.0,
        "num_pages": input_row.num_page,
        "tuple_count": input_row.tuple_count,
        "tuple_len_avg": input_row.tuple_len_avg,
        "target_ff": ff_value,

        "index_key_natts": input_row.index_key_natts if "index_key_natts" in input_row else 0,
        "index_key_size": input_row.index_key_size if "index_key_size" in input_row else 0,
        "index_tree_level": input_row.index_tree_level if "index_tree_level" in input_row else 0,
        "index_num_pages": input_row.index_num_pages if "index_num_pages" in input_row else 0,
        "index_leaf_pages": input_row.index_leaf_pages if "index_leaf_pages" in input_row else 0,
    }

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
                    row[f"{identity}_{col}_{name}_{i}"] = key_dist[i]
                else:
                    row[f"key{j}_{name}_{i}"] = key_dist[i]
    return row


def generate_dataset(logger, model_args):
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
    for d in model_args.input_dirs:
        input_files = glob.glob(f"{d}/data_page_query/data_*.feather")
        pg_class = pd.read_csv(f"{d}/pg_class.csv")

        for input_file in input_files:
            tbl = Path(input_file).stem.split(".feather")[0].split("data_")[-1]

            stats = pd.read_csv(f"{d}/{tbl}.csv")
            stats["time"] = stats.time / 1e6
            stats.set_index(keys=["time"], inplace=True)

            data = pd.read_feather(input_file)
            # Eliminate everything else other than above.
            data = data[data.comment.isin([k for k in SUPPORTED_OUS_MAPPING.keys()])]
            data["optype"] = data.comment.apply(lambda x: str(int(SUPPORTED_OUS_MAPPING[x])))

            dfs = []
            for target, df in data.groupby(by=["target"]):
                df["num_pages"] = df.table_len / 8192.0
                df["tuple_count"] = df.tuple_count if "tuple_count" in df else df.approx_tuple_count
                df["tuple_len_avg"] = (df.tuple_len / df.tuple_count) if "tuple_len" in df else (df.approx_tuple_len / df.approx_tuple_count)
                if target == tbl:
                    df["index_natts"] = 0
                    df["index_key_size"] = 0
                    df["index_tree_level"] = 0
                    df["index_num_pages"] = 0
                    df["index_leaf_pages"] = 0
                else:
                    # If we're dealing with an index, perform a series of augmentations.
                    stats = pd.read_csv(f"{d}/{target}.csv")
                    stats["num_pages"] = stats.index_size / 8192.0
                    stats["time"] = stats.time / 1e6
                    stats = stats[["time", "tree_level", "index_size", "leaf_pages"]]

                    # Get the number of attributes and key size of the index.
                    key_size, natts = identify_key_size(d, target)

                    df.set_index(keys=["start_timestamp"], inplace=True)
                    df.sort_index(inplace=True)
                    df = pd.merge_asof(df, stats, left_index=True, right_index=True, allow_exact_matches=True)
                    df.reset_index(drop=False, inplace=True)
                    df.rename(columns={ "tree_level": "index_tree_level", "num_pages": "index_num_pages", "leaf_pages": "index_leaf_pages", }, inplace=True)
                    df["index_key_natts"] = natts
                    df["index_key_size"] = key_size

                dfs.append(df)
            data = pd.concat(dfs, ignore_index=True)

            if Path(f"{d}/data_page_query/keys/{tbl}.feather").exists():
                keys = pd.read_feather(f"{d}/data_page_query/keys/{tbl}.feather")
                keys = keys[keys.index_clause.isna()]
                data_ops = data.merge(keys, left_on=["window_bucket", "optype"], right_on=["window_index", "optype"], how="inner")
                data_base = data.drop(columns=["optype"]).merge(keys[keys.optype == "data"], left_on=["window_bucket"], right_on=["window_index"], how="inner")
                data = pd.concat([data_ops, data_base], ignore_index=True)
                assert not any([k.endswith("_x") or k.endswith("_y") for k in data.columns])

            relation = pg_class[pg_class.relname == tbl].iloc[0]
            ff_value = 1.0
            if relation.reloptions is not None and not isinstance(relation.reloptions, np.float):
                reloptions = ast.literal_eval(relation.reloptions)
                for opt in reloptions:
                    for key, value in re.findall(r'(\w+)=(\w*)', opt):
                        if key == "fillfactor":
                            # Fix fillfactor options.
                            ff_value = float(value) / 100.0

            for wi, df in data.groupby(by=["window_bucket", "comment"]):
                input_row = generate_point_input(model_args, df.iloc[0], df, tbl_attr_keys[tbl], ff_value)
                # FIXME(TARGET): Assume all tuples have equal probability of triggering the event.
                input_row["asked_pages_per_tuple"] = (df.total_blks_requested / df.total_tuples_touched).mean()
                input_row["ou_type"] = wi[1]

                input_dataset.append(input_row)
                correlated_tbls.append(tbl)

    return input_dataset, correlated_tbls


class AutoMLBufferPageModel():
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __init__(self, model_args):
        super(AutoMLBufferPageModel, self).__init__()
        self.model_args = model_args


    def load_model(model_file):
        with open(f"{model_file}/args.pickle", "rb") as f:
            model_args = pickle.load(f)

        model = AutoMLBufferPageModel(model_args)
        model.predictor = TabularPredictor.load(model_file)
        model.model_args = model_args
        return model


    def fit(self, dataset):
        predictor = TabularPredictor(label=BUFFER_PAGE_TARGETS[0],
                                     problem_type="regression",
                                     eval_metric=BUFFER_PAGE_METRICS[0],
                                     path=self.model_args.output_path)

        predictor.fit(dataset, time_limit=self.model_args.automl_timeout_secs, presets=self.model_args.automl_quality, num_cpus=self.model_args.num_threads)
        with open(f"{self.model_args.output_path}/args.pickle", "wb") as f:
            pickle.dump(self.model_args, f)


    def get_dataset(logger, model_args, inference=False):
        input_dataset, correlated_tbls, generate_dataset(logger, model_args)
        input_dataset = [{k:item[k] for k in BUFFER_PAGE_INPUTS + BUFFER_PAGE_TARGETS} for item in input_dataset]

        if inference:
            for i in range(len(input_dataset)):
                input_dataset[i]["table"] = correlated_tbls[i]

        inputs = pd.DataFrame(inputs)
        inputs.fillna(value=0, inplace=True)
        if not inference:
            inputs.to_feather(f"{model_args.output_path}/model_inputs.feather")
        return inputs


    def inference(self, ougc, window, output_col="req_pages"):
        tbls = [t for t in ougc.tables]
        keyspace_feat_map = ougc.table_keyspace_features
        inputs = []
        for tbl in tbls:
            # Get all indexes. 0 is the sentinel.
            idxoids = [0]
            if tbl in ougc.table_indexoid_map:
                idxoids.extend(table_indexoid_map[tbl])

            for idxoid in idxoids:
                for ou_type in SUPPORTED_OUS_MAPPING.keys():
                    if idxoid == 0:
                        # FIXME(BITMAP): Handle bitmap.
                        if ou_type not in [OperatingUnit.IndexScan.name, OperatingUnit.IndexOnlyScan.name]:
                            continue

                    info_tuple = copy(table_state[tbl])
                    target_ff = 1.0
                    indexoid = None
                    df = None
                    if idxoid == 0:
                        if tbl in keyspace_feat_space:
                            df = keyspace_feat_space[tbl]
                            df = df[df.window_index == window]
                            df = df[df.index_clause.isna()]
                            df = df[(df.optype == "data") | (df.optype == str(SUPPORTED_OUS_MAPPING[ou_type]))]
                        attkeys = ougc.table_attr_map[tbl]
                        target_ff = ougc.table_feature_state[tbl]["target_ff"]
                    else:
                        indexoid = idxoid
                        assert idxoid in ougc.index_feature_state
                        idxstate = ougc.index_feature_state[idxoid]
                        indexname = idxstate["indexname"]

                        if tbl in keyspace_feat_space:
                            df = keyspace_feat_space[tbl]
                            df = df[df.window_index == window]
                            df = df[(df.index_clause == indexname) | (keys.optype == "data") | (keys.optype != f"{OpType.SELECT.value}")]

                        info_tuple["index_tree_level"] = idxstate["tree_level"]
                        info_tuple["index_num_pages"] = idxstate["num_pages"]
                        info_tuple["index_leaf_pages"] = idxstate["leaf_pages"]
                        info_tuple["index_key_size"] = idxstate["key_size"]
                        info_tuple["index_key_natts"] = idxstate["key_natts"]
                        attkeys = ougc.table_attr_map[indexname]

                    input_row = generate_point_input(self.model_args, Map(info_tuple), df, table_attr_map[tbl], target_ff)
                    input_row["table"] = tbl
                    input_row["indexname"] = indexname
                    input_row["ou_type"] = ou_type
                    inputs.append(input_row)

        inputs = pd.DataFrame(inputs)
        inputs.fillna(value=0, inplace=True)
        predictions = np.clip(self.predictor.predict(inputs), 0, None)
        inputs["window"] = window
        inputs[output_col] = predictions
        return inputs
