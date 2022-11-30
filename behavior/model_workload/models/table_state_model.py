import shutil
import tempfile
import ast
import re
import joblib
from pathlib import Path
import pickle
import time
import numpy as np
import argparse
import pandas as pd
import glob
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.data import dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from behavior.model_workload.models import construct_stack, MAX_KEYS
from behavior.model_workload.utils import OpType, Map
from sklearn.preprocessing import MinMaxScaler

try:
    from autogluon.tabular import TabularDataset, TabularPredictor
    from behavior.model_workload.models.multilabel_predictor import MultilabelPredictor
except:
    pass


STATE_WORKLOAD_TARGETS = [
    "next_table_free_percent",
    "next_table_dead_percent",
    "next_table_num_pages",
    "next_table_num_tuples",
]

STATE_WORKLOAD_METRICS = [
    "root_mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "mean_absolute_error",
]

MODEL_WORKLOAD_NORMAL_INPUTS = [
    "free_percent",
    "dead_tuple_percent",
    "norm_num_pages",
    "norm_tuple_count",
    "norm_tuple_len_avg",
    "target_ff",

    "num_select_queries",
    "num_insert_queries",
    "num_update_queries",
    "num_delete_queries",
    "num_select_tuples",
    "num_insert_tuples",
    "num_update_tuples",
    "num_delete_tuples",
]

MODEL_WORKLOAD_NONNORM_INPUTS = [
    "num_pages",
    "tuple_count",
    "tuple_len_avg",
]

MODEL_WORKLOAD_HIST_INPUTS = [
    "data",
    "select",
    "insert",
    "update",
    "delete",
]


def generate_point_input(model_args, input_row, df, tbl_attr_keys, ff_value):
    hist_width = model_args.hist_width
    # Construct the augmented inputs.
    num_inputs = len(MODEL_WORKLOAD_NORMAL_INPUTS)
    if model_args.add_nonnorm_features:
        num_inputs += len(MODEL_WORKLOAD_NONNORM_INPUTS)
    input_args = np.zeros(num_inputs)
    input_args[0] = (input_row.free_percent if "free_percent" in input_row else input_row.approx_free_percent) / 100.0
    input_args[1] = input_row.dead_tuple_percent / 100.0
    input_args[2] = input_row.norm_num_pages
    input_args[3] = input_row.norm_tuple_count
    input_args[4] = input_row.norm_tuple_len_avg
    input_args[5] = ff_value

    input_args[6] = input_row.num_select_queries
    input_args[7] = input_row.num_insert_queries
    input_args[8] = input_row.num_update_queries
    input_args[9] = input_row.num_delete_queries
    input_args[10] = input_row.num_select_tuples
    input_args[11] = input_row.num_insert_tuples
    input_args[12] = input_row.num_update_tuples
    input_args[13] = input_row.num_delete_tuples

    if model_args.add_nonnorm_features:
        input_args[14] = input_row.num_pages
        input_args[15] = input_row.tuple_count if "tuple_count" in input_row else input_row.approx_tuple_count
        input_args[16] = input_row.tuple_len_avg

    # Construct distribution scaler.
    dist_scalers = np.zeros(5 * hist_width)
    num_touch = input_row.num_select_tuples + input_row.num_modify_tuples
    dist_scalers[0*hist_width:1*hist_width] = 1.0
    dist_scalers[1*hist_width:2*hist_width] = 0.0 if num_touch == 0 else input_row.num_select_tuples / num_touch
    dist_scalers[2*hist_width:3*hist_width] = 0.0 if num_touch == 0 else input_row.num_insert_tuples / num_touch
    dist_scalers[3*hist_width:4*hist_width] = 0.0 if num_touch == 0 else input_row.num_update_tuples / num_touch
    dist_scalers[4*hist_width:5*hist_width] = 0.0 if num_touch == 0 else input_row.num_delete_tuples / num_touch

    # Construct key dists.
    key_dists = np.zeros((MAX_KEYS, 5 * hist_width))
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
                key_dists[j, 0:hist_width] = np.array([float(f) for f in ig.key_dist.split(",")])
            elif ig.optype == "SELECT" or ig.optype == "INSERT" or ig.optype == "UPDATE" or ig.optype == "DELETE":
                key_dists[j, (OpType[ig.optype].value) * hist_width:(OpType[ig.optype].value + 1) * hist_width] = np.array([float(f) for f in ig.key_dist.split(",")])
            else:
                key_dists[j, int(ig.optype) * hist_width:(int(ig.optype) + 1) * hist_width] = np.array([float(f) for f in ig.key_dist.split(",")])
            masks[j] = 1
    return input_args, dist_scalers, key_dists, masks


def generate_dataset(logger, model_args, automl=False):
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

    def produce_scaler():
        all_files = []
        globs = [all_files.extend([f for f in glob.glob(f"{d}/exec_features/windows/*.feather")]) for d in model_args.input_dirs]
        all_files = [f for f in all_files if "_index" not in f]

        data = pd.concat(map(pd.read_feather, all_files))
        data["num_pages"] = data.table_len / 8192.0
        data["tuple_len_avg"] = (data.tuple_len / data.tuple_count) if "tuple_count" in data else (data.approx_tuple_len / data.approx_tuple_count)

        num_page_scaler = MinMaxScaler().fit(data.num_pages.values.reshape(-1, 1))
        if "tuple_count" in data:
            tuple_count_scaler = MinMaxScaler().fit(data.tuple_count.values.reshape(-1, 1))
        else:
            tuple_count_scaler = MinMaxScaler().fit(data.approx_tuple_count.values.reshape(-1, 1))
        tuple_len_scaler = MinMaxScaler().fit(data.tuple_len_avg.values.reshape(-1, 1))
        return num_page_scaler, tuple_count_scaler, tuple_len_scaler

    if not automl:
        (Path(model_args.dataset_path)).mkdir(parents=True, exist_ok=True)
        num_pages_scaler, tuple_count_scaler, tuple_len_avg_scaler = produce_scaler()
        joblib.dump(num_pages_scaler, f"{model_args.dataset_path}/num_pages_scaler.gz")
        joblib.dump(tuple_count_scaler, f"{model_args.dataset_path}/tuple_count_scaler.gz")
        joblib.dump(tuple_len_avg_scaler, f"{model_args.dataset_path}/tuple_len_avg_scaler.gz")

    global_args = []
    global_dist_scalers = []
    global_dists = []
    global_masks = []
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
            if not automl:
                windows["norm_num_pages"] = num_pages_scaler.transform(windows.num_pages.values.reshape(-1, 1))
                tuple_count = windows.tuple_count if "tuple_count" in windows else windows.approx_tuple_count
                windows["norm_tuple_count"] = tuple_count_scaler.transform(tuple_count.values.reshape(-1, 1))
                windows["norm_tuple_len_avg"] = tuple_len_avg_scaler.transform(windows.tuple_len_avg.values.reshape(-1, 1))
            else:
                windows["norm_num_pages"] = windows.num_pages
                windows["norm_tuple_count"] = windows.tuple_count if "tuple_count" in windows else windows.approx_tuple_count
                windows["norm_tuple_len_avg"] = windows.tuple_len_avg

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
                input_args, dist_scalers, key_dists, masks = generate_point_input(model_args, df.iloc[0], df, tbl_attr_keys[root], ff_value)
                global_args.append(input_args)
                global_dist_scalers.append(dist_scalers)
                global_dists.append(key_dists)
                global_masks.append(masks)
                correlated_tbls.append(root)
                window_index.append(wi)

    if not automl:
        with tempfile.NamedTemporaryFile("wb") as f:
            pickle.dump(global_args, f)
            pickle.dump(global_dist_scalers, f)
            pickle.dump(global_dists, f)
            pickle.dump(global_masks, f)
            f.flush()
            shutil.copy(f.name, f"{model_args.dataset_path}/dataset.pickle")
    else:
        return global_args, global_dist_scalers, global_dists, global_masks, tbl_attr_keys, correlated_tbls, window_index


class AutoMLTableStateModel():
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def require_optimize():
        return False

    def __init__(self, model_args):
        super(AutoMLTableStateModel, self).__init__()
        self.model_args = model_args


    def get_dataset(logger, model_args, inference=False):
        global_args, global_dist_scalers, global_dists, global_masks, tbl_attr_keys, correlated_tbls, window_index = generate_dataset(logger, model_args, automl=True)

        inputs = []
        hist = model_args.hist_width
        for i in range(len(global_args) - 1):
            if correlated_tbls[i] != correlated_tbls[i+1]:
                # We've reached the end of the given table.
                continue

            if window_index[i] + 1 != window_index[i+1]:
                # There is a breakage so we omit the one right before the VACUUM since the "target" is garbage.
                continue

            input_row = {
                "free_percent": global_args[i][0],
                "dead_tuple_percent": global_args[i][1],
                "norm_num_pages": global_args[i][2],
                "norm_tuple_count": global_args[i][3],
                "norm_tuple_len_avg": global_args[i][4],
                "target_ff": global_args[i][5],

                "num_select_queries": global_args[i][6],
                "num_insert_queries": global_args[i][7],
                "num_update_queries": global_args[i][8],
                "num_delete_queries": global_args[i][9],
                "num_select_tuples": global_args[i][10],
                "num_insert_tuples": global_args[i][11],
                "num_update_tuples": global_args[i][12],
                "num_delete_tuples": global_args[i][13],

                "next_table_free_percent": global_args[i+1][0],
                "next_table_dead_percent": global_args[i+1][1],
                "next_table_num_pages": global_args[i+1][2],
                "next_table_num_tuples": global_args[i+1][3],
            }

            if inference:
                input_row["table"] = correlated_tbls[i]

            dist_range = [
                ("dist_data", range(0, hist)),
                ("dist_select", range(hist, 2 * hist)),
                ("dist_insert", range(2 * hist, 3 * hist)),
                ("dist_update", range(3 * hist, 4 * hist)),
                ("dist_delete", range(4 * hist, 5 * hist))
            ]
            for name, rg in dist_range:
                for j in rg:
                    input_row[f"{name}_{j}"] = global_dist_scalers[i][j]

            keys = tbl_attr_keys[correlated_tbls[i]]
            for colidx, col in enumerate(keys):
                if global_masks[i][colidx] == 1:
                    for name, rg in dist_range:
                        for j in rg:
                            # We have a valid array.
                            input_row[f"{correlated_tbls[i]}_{col}_{name}_{j % hist}"] = global_dists[i][colidx][j]
            inputs.append(input_row)

        inputs = pd.DataFrame(inputs)
        inputs.fillna(value=0, inplace=True)
        inputs.to_feather(f"{model_args.output_path}/model_inputs.feather")
        return inputs


    def load_model(model_file):
        with open(f"{model_file}/args.pickle", "rb") as f:
            model_args = pickle.load(f)

        model = AutoMLTableStateModel(model_args)
        model.predictor = MultilabelPredictor.load(model_file)
        return model


    def fit(self, dataset):
        model_file = self.model_args.output_path
        num = len(STATE_WORKLOAD_TARGETS)
        predictor = MultilabelPredictor(labels=STATE_WORKLOAD_TARGETS, problem_types=["regression"]*num, eval_metrics=STATE_WORKLOAD_METRICS, path=model_file, consider_labels_correlation=False)
        predictor.fit(dataset, time_limit=self.model_args.automl_timeout_secs, presets=self.model_args.automl_quality, num_cpus=self.model_args.num_threads)
        with open(f"{self.model_args.output_path}/args.pickle", "wb") as f:
            pickle.dump(self.model_args, f)


    def inference(self, table_state, table_attr_map, keyspace_feat_space, window, output_df=False):
        inputs = []
        tbl_keys = [t for t in table_state]
        for i, t in enumerate(tbl_keys):
            table_state[t]["norm_num_pages"] = table_state[t]["num_pages"]
            table_state[t]["norm_tuple_count"] = table_state[t]["tuple_count"]
            table_state[t]["norm_tuple_len_avg"] = table_state[t]["tuple_len_avg"]
            df = None
            if t in keyspace_feat_space:
                df = keyspace_feat_space[t]
                df = df[df.window_index == window]
                df = df[df.index_clause.isna()]

            input_args, dist_scalers, key_dists, masks = generate_point_input(self.model_args, Map(table_state[t]), df, table_attr_map[t], table_state[t]["target_ff"])
            input_row = {
                "free_percent": input_args[0],
                "dead_tuple_percent": input_args[1],
                "norm_num_pages": input_args[2],
                "norm_tuple_count": input_args[3],
                "norm_tuple_len_avg": input_args[4],
                "target_ff": input_args[5],

                "num_select_queries": input_args[6],
                "num_insert_queries": input_args[7],
                "num_update_queries": input_args[8],
                "num_delete_queries": input_args[9],
                "num_select_tuples": input_args[10],
                "num_insert_tuples": input_args[11],
                "num_update_tuples": input_args[12],
                "num_delete_tuples": input_args[13],

                "table": t,
            }

            hist = self.model_args.hist_width
            dist_range = [
                ("dist_data", range(0, hist)),
                ("dist_select", range(hist, 2 * hist)),
                ("dist_insert", range(2 * hist, 3 * hist)),
                ("dist_update", range(3 * hist, 4 * hist)),
                ("dist_delete", range(4 * hist, 5 * hist))
            ]
            for name, rg in dist_range:
                for j in rg:
                    input_row[f"{name}_{j}"] = dist_scalers[j]

            for colidx, col in enumerate(table_attr_map[t]):
                if masks[colidx] == 1:
                    for name, rg in dist_range:
                        for j in rg:
                            # We have a valid array.
                            input_row[f"{t}_{col}_{name}_{j%hist}"] = key_dists[colidx][j]
            inputs.append(input_row)

        inputs = pd.DataFrame(inputs)
        inputs.fillna(value=0, inplace=True)
        predictions = self.predictor.predict(inputs)
        predictions["next_table_free_percent"] = np.clip(predictions.next_table_free_percent, 0, 1)
        predictions["next_table_dead_percent"] = np.clip(predictions.next_table_dead_percent, 0, 1)
        outputs = np.zeros((len(tbl_keys), len(STATE_WORKLOAD_TARGETS)))
        for i, _ in enumerate(tbl_keys):
            for j, key in enumerate(STATE_WORKLOAD_TARGETS):
                outputs[i][j] = predictions[key].iloc[i]

        ret_df = None
        if output_df:
            inputs["window"] = window
            predictions.columns = "pred_" + predictions.columns
            ret_df = pd.concat([inputs, predictions], axis=1)

        return outputs, tbl_keys, ret_df
