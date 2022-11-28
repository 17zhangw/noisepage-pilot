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
from behavior.utils.process_pg_state_csvs import postgres_julian_to_unix

try:
    from autogluon.tabular import TabularDataset, TabularPredictor
    from behavior.model_workload.models.multilabel_predictor import MultilabelPredictor
except:
    pass


MODEL_WORKLOAD_TARGETS = [
    "extend_pecent",
    "split_percent",
]

MODEL_WORKLOAD_NORMAL_INPUTS = [
    "tree_level",
    "num_pages",
    "leaf_pages",
    "deleted_pages",
    "avg_leaf_densty",
]

MODEL_WORKLOAD_HIST_INPUTS = [
    "data",
    "select",
    "insert",
    "update",
    "delete",
]


def generate_point_input(model_args, input_row, df, tbl_attr_keys):
    hist_width = model_args.hist_width
    # Construct the augmented inputs.
    num_inputs = len(MODEL_WORKLOAD_NORMAL_INPUTS)
    input_args = np.zeros(num_inputs)
    input_args[0] = input_row.tree_level
    input_args[1] = input_row.num_pages
    input_args[2] = input_row.leaf_pages
    input_args[3] = input_row.deleted_pages
    input_args[4] = input_row.avg_leaf_density / 100.0

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

            if j is None:
                # This is actually OK because full key space != index specific keyspace.
                continue

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

    global_args = []
    global_dist_scalers = []
    global_dists = []
    global_masks = []
    global_targets = []
    correlated_idxs = []
    for d in model_args.input_dirs:
        input_files = sorted(glob.glob(f"{d}/exec_features/data/*_index.feather"))
        for input_file in input_files:
            # Analyze each index in the input frame.
            all_idxs = pd.read_feather(input_file)
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

                data = all_idxs[(all_idxs.target == tbl) | (all_idxs.target == idx)]
                data["unix_timestamp"] = postgres_julian_to_unix(data.start_timestamp).astype(float)
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

                if Path(f"{d}/exec_features/keys/{tbl}.feather").exists():
                    keys = pd.read_feather(f"{d}/exec_features/keys/{tbl}.feather")
                    # Only get relevant to this index.
                    keys = keys[(keys.index_clause == idx) | (keys.optype == "data") | (keys.optype != f"{OpType.SELECT.value}")]
                    keys.set_index(keys=["window_index"], inplace=True)
                    data = data.join(keys, on=["window_index"], how="inner")
                data.reset_index(drop=False, inplace=True)

                for wi, df in data.groupby(by=["window_index"]):
                    input_args, dist_scalers, key_dists, masks = generate_point_input(model_args, df.iloc[0], df, idx_attr_keys[idx])
                    global_args.append(input_args)
                    global_dist_scalers.append(dist_scalers)
                    global_dists.append(key_dists)
                    global_masks.append(masks)

                    # FIXME(TARGET): Assume all tuples have equal probability of triggering the event.
                    targets = np.zeros(len(MODEL_WORKLOAD_TARGETS))
                    targets[0] = 0.0 if df.iloc[0].num_inserts == 0 else df.iloc[0].num_extend / df.iloc[0].num_inserts
                    targets[1] = 0.0 if df.iloc[0].num_inserts == 0 else df.iloc[0].num_split / df.iloc[0].num_inserts
                    global_targets.append(targets)
                    correlated_idxs.append(idx)

    return global_args, global_dist_scalers, global_dists, global_masks, global_targets, idx_attr_keys, correlated_idxs


class AutoMLIndexFeatureModel():
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def require_optimize():
        return False

    def __init__(self, model_args):
        super(AutoMLIndexFeatureModel, self).__init__()
        self.model_args = model_args


    def get_dataset(logger, model_args, inference=False):
        global_args, global_dist_scalers, global_dists, global_masks, global_targets, idx_attr_keys, correlated_idxs = generate_dataset(logger, model_args, automl=True)

        inputs = []
        hist = model_args.hist_width
        for i in range(len(global_args)):
            input_row = {
                "tree_level": global_args[i][0],
                "num_pages": global_args[i][1],
                "leaf_pages": global_args[i][2],
                "deleted_pages": global_args[i][3],
                "avg_leaf_density": global_args[i][4],

                "extend_percent": global_targets[i][0],
                "split_percent": global_targets[i][1],
            }

            if inference:
                input_row["idx"] = correlated_idxs[i]

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

            keys = idx_attr_keys[correlated_idxs[i]]
            for colidx, col in enumerate(keys):
                if global_masks[i][colidx] == 1:
                    for name, rg in dist_range:
                        for j in rg:
                            # We have a valid array.
                            input_row[f"{correlated_idxs[i]}_{col}_{name}_{j % hist}"] = global_dists[i][colidx][j]
            inputs.append(input_row)

        inputs = pd.DataFrame(inputs)
        inputs.fillna(value=0, inplace=True)
        if not inference:
            inputs.to_feather(f"{model_args.output_path}/model_inputs.feather")
        return inputs


    def load_model(model_file):
        with open(f"{model_file}/args.pickle", "rb") as f:
            model_args = pickle.load(f)

        model = AutoMLTableFeatureModel(model_args)
        model.predictor = MultilabelPredictor.load(model_file)
        return model


    def fit(self, dataset):
        model_file = self.model_args.output_path
        num = len(MODEL_WORKLOAD_TARGETS)
        predictor = MultilabelPredictor(labels=MODEL_WORKLOAD_TARGETS, problem_types=["regression"]*num, eval_metrics=["mean_absolute_error"]*num, path=model_file, consider_labels_correlation=False)
        predictor.fit(dataset, time_limit=self.model_args.automl_timeout_secs, presets=self.model_args.automl_quality, num_cpus=self.model_args.num_threads)
        with open(f"{self.model_args.output_path}/args.pickle", "wb") as f:
            pickle.dump(self.model_args, f)


    def inference(self, table_state, table_attr_map, keyspace_feat_space, window):
        assert False
