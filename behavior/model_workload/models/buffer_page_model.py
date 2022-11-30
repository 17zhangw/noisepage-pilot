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
from behavior.model_workload.models import construct_stack, MAX_KEYS
from behavior.model_workload.utils import OpType, Map
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import shutil

try:
    from autogluon.tabular import TabularDataset, TabularPredictor
    from behavior.model_workload.models.multilabel_predictor import MultilabelPredictor
except:
    pass

try:
    from supervised.automl import AutoML
except:
    pass


MODEL_WORKLOAD_NORMAL_INPUTS = [
    "free_percent",
    "dead_tuple_percent",
    "norm_num_pages",
    "norm_tuple_count",
    "norm_tuple_len_avg",
    "target_ff",
]

MODEL_WORKLOAD_TARGETS = [
    "asked_pages_per_tuple",
]

MODEL_WORKLOAD_METRICS = [
    "root_mean_squared_error",
]

def generate_point_input(model_args, input_row, df, tbl_attr_keys, ff_value):
    hist_width = model_args.hist_width
    # Construct the augmented inputs.
    num_inputs = len(MODEL_WORKLOAD_NORMAL_INPUTS)
    input_args = np.zeros(num_inputs)
    input_args[0] = (input_row.free_percent if "free_percent" in input_row else input_row.approx_free_percent) / 100.0
    input_args[1] = input_row.dead_tuple_percent / 100.0
    input_args[2] = input_row.norm_num_pages
    input_args[3] = input_row.norm_tuple_count
    input_args[4] = input_row.norm_tuple_len_avg
    input_args[5] = ff_value

    # Construct key dists.
    key_dists = np.zeros((MAX_KEYS, 2 * hist_width))
    masks = np.zeros((MAX_KEYS, 1))
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

            if ig.optype == "data":
                key_dists[j][0:hist_width] = np.array([float(f) for f in ig.key_dist.split(",")])
            else:
                key_dists[j][hist_width:] = np.array([float(f) for f in ig.key_dist.split(",")])
            masks[j] = 1
    return input_args, key_dists, masks


def generate_automl_dataset(logger, model_args):
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

    global_args = []
    global_catargs = []
    global_dists = []
    global_masks = []
    global_targets = []
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
            mapping = {
                "IndexScan": OpType.SELECT.value,
                "IndexOnlyScan": OpType.SELECT.value,
                "ModifyTableInsert": OpType.INSERT.value,
                "ModifyTableUpdate": OpType.UPDATE.value,
                "ModifyTableDelete": OpType.DELETE.value
            }
            # Eliminate everything else other than above.
            data = data[data.comment.isin([k for k in mapping.keys()])]

            data["optype"] = data.comment.apply(lambda x: str(int(mapping[x])))
            data.set_index(keys=["start_timestamp"], inplace=True)
            data.sort_index(inplace=True)
            data = pd.merge_asof(data, stats, left_index=True, right_index=True, allow_exact_matches=True)
            data.reset_index(drop=False, inplace=True)

            data["norm_num_pages"] = data.table_len / 8192.0
            data["norm_tuple_count"] = data.tuple_count if "tuple_count" in data else data.approx_tuple_count
            data["norm_tuple_len_avg"] = (data.tuple_len / data.tuple_count) if "tuple_len" in data else (data.approx_tuple_len / data.approx_tuple_count)

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
                input_args, key_dists, masks = generate_point_input(model_args, df.iloc[0], df, tbl_attr_keys[tbl], ff_value)
                global_catargs.append([wi[1]])
                global_args.append(input_args)
                global_dists.append(key_dists)
                global_masks.append(masks)
                # FIXME(TARGET): Assume all tuples have equal probability of triggering the event.
                global_targets.append((df.total_blks_requested / df.total_tuples_touched).mean())
                correlated_tbls.append(tbl)

    return global_args, global_catargs, global_dists, global_masks, global_targets, tbl_attr_keys, correlated_tbls


class BufferPageModel():
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __init__(self, model_args):
        super(BufferPageModel, self).__init__()
        self.args = model_args

    def require_optimize():
        return False

    def load_model(model_file):
        with open(f"{model_file}/args.pickle", "rb") as f:
            args = pickle.load(f)

        model = BufferPageModel(args)
        model.automl = AutoML(results_path=f"{model_file}/perform")
        return model

    def fit(self, dataset):
        x, y = dataset[0], dataset[1]

        automl = AutoML(results_path=f"{self.args.output_path}/perform", ml_task="regression", eval_metric="mse", mode="Perform", total_time_limit=7200)
        automl.fit(x, y)

        with open(f"{self.args.output_path}/args.pickle", "wb") as f:
            pickle.dump(self.args, f)

    def get_dataset(logger, model_args):
        sub_dirs = []
        for d in model_args.input_dirs:
            for ws in model_args.window_slices:
                sub_dirs.append(Path(d) / f"data_page_{ws}/data.feather")

        data = pd.concat(map(pd.read_feather, sub_dirs), ignore_index=True)
        data["asked_pages"] = data.total_blks_requested / data.total_tuples_touched
        x = data[["comment", "reltuples", "relpages"]]
        y = data["asked_pages"]
        return (x, y)

    def inference(self, input_frame):
        df = pd.DataFrame(input_frame)
        return np.clip(self.automl.predict(df), 0, None)


class AutoMLBufferPageModel():
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __init__(self, model_args):
        super(AutoMLBufferPageModel, self).__init__()
        self.model_args = model_args

    def require_optimize():
        return False


    def load_model(model_file):
        with open(f"{model_file}/args.pickle", "rb") as f:
            model_args = pickle.load(f)

        model = AutoMLBufferPageModel(model_args)
        model.predictor = TabularPredictor.load(model_file)
        model.model_args = model_args
        return model


    def fit(self, dataset):
        predictor = TabularPredictor(label=MODEL_WORKLOAD_TARGETS[0], problem_type="regression", eval_metric=MODEL_WORKLOAD_METRICS[0], path=self.model_args.output_path)
        predictor.fit(dataset, time_limit=self.model_args.automl_timeout_secs, presets=self.model_args.automl_quality, num_cpus=self.model_args.num_threads)
        with open(f"{self.model_args.output_path}/args.pickle", "wb") as f:
            pickle.dump(self.model_args, f)


    def get_dataset(logger, model_args, inference=False):
        global_args, global_catargs, global_dists, global_masks, global_targets, tbl_attr_keys, correlated_tbls = generate_automl_dataset(logger, model_args)

        inputs = []
        hist = model_args.hist_width
        for i in range(len(global_args)):
            input_row = {
                "free_percent": global_args[i][0],
                "dead_tuple_percent": global_args[i][1],
                "norm_num_pages": global_args[i][2],
                "norm_tuple_count": global_args[i][3],
                "norm_tuple_len_avg": global_args[i][4],
                "target_ff": global_args[i][5],
                "ou_type": global_catargs[i][0],

                "asked_pages_per_tuple": global_targets[i],
            }

            if inference:
                input_row["table"] = correlated_tbls[i]

            hist_ranges = [
                ("data", 0, hist),
                ("op", hist, 2 * hist),
            ]

            keys = tbl_attr_keys[correlated_tbls[i]]
            for colidx, col in enumerate(keys):
                if global_masks[i][colidx] == 1:
                    for name, start, end in hist_ranges:
                        for j in range(start, end):
                            input_row[f"{correlated_tbls[i]}_{col}_{name}_{j % hist}"] = global_dists[i][colidx][j]
            inputs.append(input_row)

        inputs = pd.DataFrame(inputs)
        inputs.fillna(value=0, inplace=True)
        if not inference:
            inputs.to_feather(f"{model_args.output_path}/model_inputs.feather")
        return inputs


    def inference(self, table_state, table_attr_map, keyspace_feat_space, window, output_df=None):
        tbl_ous = {
            "warehouse": ["ModifyTableUpdate", "IndexScan"],
            "district": ["ModifyTableUpdate", "IndexScan"],
            "new_order": ["ModifyTableUpdate", "ModifyTableInsert", "IndexOnlyScan", "IndexScan"],
            "order_line": ["IndexScan", "ModifyTableInsert", "ModifyTableUpdate"],
            "oorder": ["IndexScan", "ModifyTableInsert", "ModifyTableUpdate"],
            "history": ["ModifyTableInsert"],
            "stock": ["IndexScan", "ModifyTableUpdate"],
            "customer": ["IndexScan", "ModifyTableUpdate"],
            "item": ["IndexScan"],
        }

        inputs = []
        hist = self.model_args.hist_width
        for tbl, state in table_state.items():
            for ou_type in tbl_ous[tbl]:
                mapping = {
                    "IndexScan": OpType.SELECT.value,
                    "IndexOnlyScan": OpType.SELECT.value,
                    "ModifyTableInsert": OpType.INSERT.value,
                    "ModifyTableUpdate": OpType.UPDATE.value,
                    "ModifyTableDelete": OpType.DELETE.value
                }

                df = None
                if tbl in keyspace_feat_space:
                    df = keyspace_feat_space[tbl]
                    df = df[df.window_index == window]
                    df = df[df.index_clause.isna()]
                    df = df[(df.optype == "data") | (df.optype == str(mapping[ou_type]))]

                input_args, key_dists, masks = generate_point_input(self.model_args, Map(table_state[tbl]), df, table_attr_map[tbl], 1.0)

                input_row = {
                    "free_percent": input_args[0],
                    "dead_tuple_percent": input_args[1],
                    "norm_num_pages": input_args[2],
                    "norm_tuple_count": input_args[3],
                    "norm_tuple_len_avg": input_args[4],
                    "target_ff": input_args[5],
                    "ou_type": ou_type,
                    "table": tbl,
                }

                hist_ranges = [
                    ("data", 0, hist),
                    ("op", hist, 2 * hist),
                ]

                keys = table_attr_map[tbl]
                for colidx, col in enumerate(keys):
                    if masks[colidx] == 1:
                        for name, start, end in hist_ranges:
                            for j in range(start, end):
                                input_row[f"{tbl}_{col}_{name}_{j % hist}"] = key_dists[colidx][j]
                inputs.append(input_row)

        inputs = pd.DataFrame(inputs)
        inputs.fillna(value=0, inplace=True)

        predictions = np.clip(self.predictor.predict(inputs), 0, None)

        inputs["window"] = window
        inputs["pred_asked_pages_per_tuple"] = predictions
        return inputs
