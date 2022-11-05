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


MODEL_WORKLOAD_TARGETS = [
    "extend_percent",
    "defrag_percent",
    "hot_percent",
]

MODEL_WORKLOAD_NORMAL_INPUTS = [
    "free_percent",
    "dead_tuple_percent",
    "norm_num_pages",
    "norm_tuple_count",
    "norm_tuple_len_avg",
    "target_ff",
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
    input_args[0] = input_row.approx_free_percent / 100.0
    input_args[1] = input_row.dead_tuple_percent / 100.0
    input_args[2] = input_row.norm_num_pages
    input_args[3] = input_row.norm_tuple_count
    input_args[4] = input_row.norm_tuple_len_avg
    input_args[5] = ff_value
    if model_args.add_nonnorm_features:
        input_args[6] = input_row.num_pages
        input_args[7] = input_row.tuple_count
        input_args[8] = input_row.tuple_len_avg

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
            for j, kt in enumerate(tbl_attr_keys):
                if kt == ig.att_name:
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

    def produce_scaler():
        all_files = []
        globs = [all_files.extend([f for f in glob.glob(f"{d}/exec_features/windows/*.feather")]) for d in model_args.input_dirs]
        data = pd.concat(map(pd.read_feather, all_files))
        data["num_pages"] = data.table_len / 8192.0
        data["tuple_len_avg"] = data.table_len / data.approx_tuple_count
        return MinMaxScaler().fit(data.num_pages.values.reshape(-1, 1)), MinMaxScaler().fit(data.approx_tuple_count.values.reshape(-1, 1)), MinMaxScaler().fit(data.tuple_len_avg.values.reshape(-1, 1))

    (Path(model_args.dataset_path)).mkdir(parents=True, exist_ok=True)
    num_pages_scaler, tuple_count_scaler, tuple_len_avg_scaler = produce_scaler()
    joblib.dump(num_pages_scaler, f"{model_args.dataset_path}/num_pages_scaler.gz")
    joblib.dump(tuple_count_scaler, f"{model_args.dataset_path}/tuple_count_scaler.gz")
    joblib.dump(tuple_len_avg_scaler, f"{model_args.dataset_path}/tuple_len_avg_scaler.gz")

    global_args = []
    global_dist_scalers = []
    global_dists = []
    global_masks = []
    global_targets = []
    for d in model_args.input_dirs:
        input_files = glob.glob(f"{d}/exec_features/data/*.feather")
        pg_class = pd.read_csv(f"{d}/pg_class.csv")

        for input_file in input_files:
            root = Path(input_file).stem
            data = pd.read_feather(input_file)
            windows = pd.read_feather(f"{d}/exec_features/windows/{root}.feather")
            windows["num_pages"] = windows.table_len / 8192.0
            windows["tuple_len_avg"] = windows.table_len / windows.approx_tuple_count
            windows["norm_num_pages"] = num_pages_scaler.transform(windows.num_pages.values.reshape(-1, 1))
            windows["norm_tuple_count"] = tuple_count_scaler.transform(windows.approx_tuple_count.values.reshape(-1, 1))
            windows["norm_tuple_len_avg"] = tuple_len_avg_scaler.transform(windows.tuple_len_avg.values.reshape(-1, 1))

            data.set_index(keys=["window_index"], inplace=True)
            windows.set_index(keys=["window_index"], inplace=True)
            data = data.join(windows, how="inner")

            if Path(f"{d}/exec_features/keys/{root}.feather").exists():
                keys = pd.read_feather(f"{d}/exec_features/keys/{root}.feather")
                keys.set_index(keys=["window_index"], inplace=True)
                data = data.join(keys, how="inner")

            relation = pg_class[pg_class.relname == root].iloc[0]
            ff_value = 1.0
            if relation.reloptions is not None:
                reloptions = ast.literal_eval(relation.reloptions)
                for opt in reloptions:
                    for key, value in re.findall(r'(\w+)=(\w*)', opt):
                        if key == "fillfactor":
                            # Fix fillfactor options.
                            ff_value = float(value) / 100.0

            data.reset_index(drop=False, inplace=True)
            for wi, df in data.groupby(by=["window_index"]):
                input_args, dist_scalers, key_dists, masks = generate_point_input(model_args, df.iloc[0], df, tbl_attr_keys[root], ff_value)
                global_args.append(input_args)
                global_dist_scalers.append(dist_scalers)
                global_dists.append(key_dists)
                global_masks.append(masks)

                # Construct the targets.
                targets = np.zeros(len(MODEL_WORKLOAD_TARGETS))
                actual_insert = df.iloc[0].num_insert_tuples + df.iloc[0].num_update_tuples - df.iloc[0].num_hot
                actual_touch = df.iloc[0].num_select_tuples + df.iloc[0].num_update_tuples + df.iloc[0].num_delete_tuples
                targets[0] = 0.0 if actual_insert == 0 else df.iloc[0].num_extend / actual_insert
                targets[1] = 0.0 if actual_touch == 0 else df.iloc[0].num_defrag / actual_touch
                targets[2] = 0.0 if df.iloc[0].num_update == 0 else df.iloc[0].num_hot / df.iloc[0].num_update
                global_targets.append(targets)

    with tempfile.NamedTemporaryFile("wb") as f:
        pickle.dump(global_args, f)
        pickle.dump(global_dist_scalers, f)
        pickle.dump(global_dists, f)
        pickle.dump(global_masks, f)
        pickle.dump(global_targets, f)
        f.flush()
        shutil.copy(f.name, f"{model_args.dataset_path}/dataset.pickle")


class TableFeatureModel(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def require_optimize():
        return True

    def __init__(self, model_args, num_outputs):
        super(TableFeatureModel, self).__init__()

        hist_width = model_args.hist_width
        hid_units = model_args.hidden_size

        # This computes the distribution.
        self.dist = construct_stack(hist_width * len(MODEL_WORKLOAD_HIST_INPUTS), hid_units, hid_units, model_args.dropout, model_args.depth)
        # This computes the output targets.
        num_inputs = len(MODEL_WORKLOAD_NORMAL_INPUTS)
        if model_args.add_nonnorm_features:
            num_inputs += len(MODEL_WORKLOAD_NONNORM_INPUTS)
        self.outputs = construct_stack(hid_units + num_inputs, hid_units, len(MODEL_WORKLOAD_TARGETS), model_args.dropout, model_args.depth)

    def forward(self, **kwargs):
        global_args = kwargs["global_args"]
        global_dist_scalers = kwargs["global_dist_scalers"]
        global_dists = kwargs["global_dists"]
        global_masks = kwargs["global_masks"]

        bias_dists = global_dists * global_dist_scalers.unsqueeze(1)
        dists = self.dist(bias_dists)
        dists = dists * global_masks
        sum_dists = torch.sum(dists, dim=1, keepdim=False)
        adjust_masks = torch.sum(global_masks, dim=1, keepdim=False)
        adjust_masks[adjust_masks == 0] = 1
        adjust_sum_dists = sum_dists / adjust_masks

        concat_feats = torch.cat([global_args, adjust_sum_dists], dim=1)
        return self.outputs(concat_feats)

    def loss(self, target_outputs, model_outputs):
        outputs = target_outputs["global_targets"]
        loss = MSELoss()(outputs, model_outputs)
        return loss, loss.item()

    def get_dataset(logger, model_args):
        dataset_path = Path(model_args.dataset_path) / "dataset.pickle"
        if not dataset_path.exists():
            generate_dataset(logger, model_args)

        with open(dataset_path, "rb") as f:
            global_args = pickle.load(f)
            global_dist_scalers = pickle.load(f)
            global_dists = pickle.load(f)
            global_masks = pickle.load(f)
            global_targets = pickle.load(f)

        td = dataset.TensorDataset(
            torch.tensor(np.array(global_args, dtype=np.float32)),
            torch.tensor(np.array(global_dist_scalers, dtype=np.float32)),
            torch.tensor(np.array(global_dists, dtype=np.float32)),
            torch.tensor(np.array(global_masks, dtype=np.float32)),
            torch.tensor(np.array(global_targets, dtype=np.float32)))

        feat_names = [
            "global_args",
            "global_dist_scalers",
            "global_dists",
            "global_masks",
        ]

        target_names = [
            "global_targets"
        ]

        num_outputs = len(global_targets[0])
        return td, feat_names, target_names, num_outputs, None

    def load_model(model_file):
        model_obj = torch.load(f"{model_file}/best_model.pt")
        model = TableFeatureModel(model_obj["model_args"], model_obj["num_outputs"])
        model.model_args = model_obj["model_args"]
        model.load_state_dict(model_obj["best_model"])

        parent_dir = Path(model_file).parent
        model.num_pages_scaler = joblib.load(f"{parent_dir}/num_pages_scaler.gz")
        model.tuple_count_scaler = joblib.load(f"{parent_dir}/tuple_count_scaler.gz")
        model.tuple_len_avg_scaler = joblib.load(f"{parent_dir}/tuple_len_avg_scaler.gz")
        return model

    def inference(self, table_state, table_attr_map, keyspace_feat_space, window):
        global_args = []
        global_dist_scalers = []
        global_dists = []
        global_masks = []

        tbl_keys = [t for t in table_state]
        norm_num_pages = self.num_pages_scaler.transform(np.array([table_state[t]["num_pages"] for t in tbl_keys]).reshape(-1, 1))
        norm_tuple_count = self.tuple_count_scaler.transform(np.array([table_state[t]["approx_tuple_count"] for t in tbl_keys]).reshape(-1, 1))
        norm_tuple_len_avg = self.tuple_len_avg_scaler.transform(np.array([table_state[t]["tuple_len_avg"] for t in tbl_keys]).reshape(-1, 1))
        for i, t in enumerate(tbl_keys):
            table_state[t]["norm_num_pages"] = norm_num_pages[i][0]
            table_state[t]["norm_tuple_count"] = norm_tuple_count[i][0]
            table_state[t]["norm_tuple_len_avg"] = norm_tuple_len_avg[i][0]
            df = None
            if t in keyspace_feat_space:
                df = keyspace_feat_space[t]
                df = df[df.window_index == window]

            input_args, dist_scalers, key_dists, masks = generate_point_input(self.model_args, Map(table_state[t]), df, table_attr_map[t], table_state[t]["target_ff"])
            global_args.append(input_args)
            global_dist_scalers.append(dist_scalers)
            global_dists.append(key_dists)
            global_masks.append(masks)

        inputs = {
            "global_args": torch.tensor(np.array(global_args, dtype=np.float32)),
            "global_dist_scalers": torch.tensor(np.array(global_dist_scalers, dtype=np.float32)),
            "global_dists": torch.tensor(np.array(global_dists, dtype=np.float32)),
            "global_masks": torch.tensor(np.array(global_masks, dtype=np.float32)),
        }

        outputs = torch.clip(self(**inputs), 0, 1)
        return outputs, tbl_keys
