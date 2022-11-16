import shutil
import tempfile
from collections import namedtuple
from plumbum import cli
import argparse
from tqdm import tqdm
import copy
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib
import pickle

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.data import dataset
from torch.utils.data import DataLoader
import logging
import multiprocessing as mp
from behavior.model_workload.models import construct_stack, MAX_KEYS
from behavior.model_workload.models.utils import extract_train_tables_keys_features, extract_infer_tables_keys_features
from behavior.model_workload.utils import OpType

try:
    from autogluon.tabular import TabularDataset, TabularPredictor
    from behavior.model_workload.models.multilabel_predictor import MultilabelPredictor
except:
    pass


def bias_queries(histograms, queries):
    ranges = [(0, 64), (1, 1024), (2, None)]
    queries["pred_elapsed_us_bias"] = queries["pred_elapsed_us"]
    for i, (step, limit) in enumerate(ranges):
        if limit is not None:
            subqueries = queries[(queries.pred_elapsed_us < limit) & (queries.pred_elapsed_us >= (ranges[i-1][1] if i > 0 else 0))]
        else:
            subqueries = queries[queries.pred_elapsed_us > ranges[i-1][1]]

        if subqueries.shape[0] == 0:
            continue

        hist = histograms[step]
        hist = np.cumsum(hist)
        rands = np.random.uniform(0, 1, size=(subqueries.shape[0], 1))
        bins = np.argmax(rands < hist, axis=1)
        bias = (1 << bins)
        assert bias.shape[0] == subqueries.shape[0]
        queries.loc[subqueries.index, "pred_elapsed_us_bias"] = queries.loc[subqueries.index, "pred_elapsed_us_bias"] + bias
    return queries


def generate_dataset(logger, model_args, automl=False):
    hist_width = model_args.hist_width
    tbl_mapping = {}
    keys = {}

    global_feats = []
    global_bias = []
    global_targets = []
    global_addt_feats = []
    global_key_bias = []
    global_key_dists = []
    global_masks = []

    def produce_scaler(dirs):
        files = [f"{d}/concurrency/step{s}/data.feather" for d in dirs for s in model_args.steps]
        data = pd.concat(map(pd.read_feather, files))
        return MinMaxScaler().fit(data.relpages.values.reshape(-1, 1)), MinMaxScaler().fit(data.reltuples.values.reshape(-1, 1))

    def handle(in_dir, step, relpages_scaler, reltuples_scaler):
        logger.info("Processing input from: %s", in_dir)
        data = pd.read_feather(f"{in_dir}/concurrency/step{step}/data.feather")
        frame = pd.read_feather(f"{in_dir}/concurrency/step{step}/frame.feather")
        assert np.sum(data.step != step) == 0
        assert np.sum(frame.step != step) == 0

        tbl_map = {}
        for p in glob.glob(f"{in_dir}/concurrency/step{step}/keys/*.feather"):
            tbl_map[Path(p).stem] = pd.read_feather(p)

        data.set_index(keys=["mpi", "step", "window_bucket"], inplace=True)
        frame.set_index(keys=["mpi", "step", "window_index"], inplace=True)
        data = data.join(frame, on=["mpi", "step", "window_bucket"], how="inner")
        data.reset_index(drop=False, inplace=True)
        del frame

        if automl:
            data["norm_relpages"] = data.relpages
            data["norm_reltuples"] = data.reltuples
        else:
            data["norm_relpages"] = relpages_scaler.transform(data.relpages.values.reshape(-1, 1))
            data["norm_reltuples"] = reltuples_scaler.transform(data.reltuples.values.reshape(-1, 1))

        # Assume every query succeeds I guess...
        for window in data.groupby(by=["mpi", "step", "elapsed_slice", "window_bucket"]):
            mpi = window[0][0]

            all_queries = window[1].elapsed_slice_queries.iloc[0]
            target = window[1].targets.iloc[0]
            key_bias, key_dists, masks, all_bias, addt_feats = extract_train_tables_keys_features(model_args.add_nonnorm_features, tbl_map, tbl_mapping, keys, hist_width, window[1], window[0][3])

            # If you change the global_feats order, change the classes assignemnt.
            global_feats.append([all_queries, mpi, window[0][2]])
            global_bias.append(all_bias)
            global_targets.append(target)
            global_addt_feats.append(addt_feats)
            global_key_bias.append(key_bias)
            global_key_dists.append(key_dists)
            global_masks.append(masks)

    relpages_scaler, reltuples_scaler = produce_scaler(model_args.input_dirs)
    (Path(model_args.dataset_path)).mkdir(parents=True, exist_ok=True)
    joblib.dump(relpages_scaler, f"{model_args.dataset_path}/relpages_scaler.gz")
    joblib.dump(reltuples_scaler, f"{model_args.dataset_path}/reltuples_scaler.gz")

    for d in model_args.input_dirs:
        c = f"{d}/keyspaces.pickle"
        assert Path(c).exists()
        with open(c, "rb") as f:
            metadata = pickle.load(f)

        for t, k in metadata.table_attr_map.items():
            if t not in tbl_mapping:
                tbl_mapping[t] = len(tbl_mapping)
                keys[t] = k

    for d in model_args.input_dirs:
        for s in model_args.steps:
            handle(d, int(s), relpages_scaler, reltuples_scaler)

    if not automl:
        with tempfile.NamedTemporaryFile("wb") as f:
            pickle.dump(global_feats, f)
            pickle.dump(global_bias, f)
            pickle.dump(global_addt_feats, f)
            pickle.dump(global_key_bias, f)
            pickle.dump(global_key_dists, f)
            pickle.dump(global_masks, f)
            pickle.dump(global_targets, f)
            f.flush()
            shutil.copy(f.name, f"{model_args.dataset_path}/dataset.pickle")
    else:
        return global_feats, global_bias, global_addt_feats, global_key_bias, global_key_dists, global_masks, global_targets, tbl_mapping, keys


class ConcurrencyModel(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def require_optimize():
        return True

    def load_model(model_file):
        model_obj = torch.load(f"{model_file}/best_model.pt")
        model = ConcurrencyModel(model_obj["model_args"], model_obj["num_outputs"])
        model.load_state_dict(model_obj["best_model"])
        model.model_args = model_obj["model_args"]

        parent_dir = Path(model_file).parent
        model.relpages_scaler = joblib.load(f"{parent_dir}/relpages_scaler.gz")
        model.reltuples_scaler = joblib.load(f"{parent_dir}/reltuples_scaler.gz")
        return model

    def __init__(self, model_args, num_outputs):
        super(ConcurrencyModel, self).__init__()
        assert num_outputs > 0
        input_size = 4 * model_args.hist_width

        # Mapping from keyspace -> embedding.
        self.dist = construct_stack(input_size, model_args.hidden_size, model_args.hidden_size, model_args.dropout, model_args.depth, activation="Sigmoid")
        # Mapping from tbl + keyspace -> embedding
        num_inputs = 4 if model_args.add_nonnorm_features else 2
        self.comb_tbls = construct_stack(model_args.hidden_size + num_inputs, model_args.hidden_size, model_args.hidden_size, model_args.dropout, model_args.depth, activation="Sigmoid")
        # Final mapping.
        self.final = construct_stack(model_args.hidden_size + 3, model_args.hidden_size, num_outputs, model_args.dropout, model_args.depth, activation="Sigmoid")

    def forward(self, **kwargs):
        global_feats = kwargs["global_feats"]
        global_bias = kwargs["global_bias"]
        global_addt_feats = kwargs["global_addt_feats"]
        global_key_bias = kwargs["global_key_bias"]
        global_key_dists = kwargs["global_key_dists"]
        global_masks = kwargs["global_masks"]

        # First generate the key "embeding".
        # First scale based on tuple access distribution.
        global_key_dists = global_key_dists * global_key_bias.unsqueeze(2)

        key_dists = self.dist(global_key_dists)
        mask_dists = key_dists * global_masks
        sum_dists = torch.sum(mask_dists, dim=2, keepdim=False)
        adjust_masks = torch.sum(global_masks, dim=2, keepdim=False)
        adjust_masks[adjust_masks == 0] = 1
        adjust_sum_dists = sum_dists / adjust_masks

        # Attach the per-table features and "embedding".
        concat_feats = torch.cat([global_addt_feats, adjust_sum_dists], dim=2)
        tbl_feats = self.comb_tbls(concat_feats)

        # Bias with distribution and attach "global" state.
        bias_tbl_feats = tbl_feats * global_bias
        input_vec = torch.sum(bias_tbl_feats, dim=1, keepdim=False)
        input_vec = torch.cat([global_feats, input_vec], dim=1)
        outputs = self.final(input_vec)
        return F.softmax(outputs, dim=1)

    def loss(self, target_outputs, model_outputs):
        outputs = target_outputs["global_targets"]
        offset = outputs - model_outputs
        offset = torch.cumsum(offset, dim=1)
        offset = torch.abs(offset).sum(dim=1)
        loss = offset.mean()
        return loss, loss.item()

    def get_dataset(logger, model_args):
        dataset_path = Path(model_args.dataset_path) / "dataset.pickle"
        if not dataset_path.exists():
            generate_dataset(logger, model_args)

        with open(dataset_path, "rb") as f:
            global_feats = pickle.load(f)
            global_bias = pickle.load(f)
            global_addt_feats = pickle.load(f)
            global_key_bias = pickle.load(f)
            global_key_dists = pickle.load(f)
            global_masks = pickle.load(f)
            global_targets = pickle.load(f)

        classes = [(g[1], g[2]) for g in global_feats]

        td = [
            global_feats,
            global_bias,
            global_addt_feats,
            global_key_bias,
            global_key_dists,
            global_masks,
            global_targets,
        ]

        feat_names = [
            "global_feats",
            "global_bias",
            "global_addt_feats",
            "global_key_bias",
            "global_key_dists",
            "global_masks",
        ]

        target_names = [
            "global_targets"
        ]

        num_outputs = len(global_targets[0])
        return td, feat_names, target_names, num_outputs, classes

    def inference(self, window_slot, mpi, queries, table_state, table_attr_map, keyspace_feat_space):
        global_feats = []
        global_bias = []
        global_addt_feats = []
        global_key_bias = []
        global_key_dists = []
        global_masks = []

        tbl_mapping = {t:i for i, t in enumerate(table_attr_map)}
        norm_relpages = self.relpages_scaler.transform(np.array([table_state[t]["num_pages"] for t in tbl_mapping]).reshape(-1, 1))
        norm_reltuples = self.reltuples_scaler.transform(np.array([table_state[t]["approx_tuple_count"] for t in tbl_mapping]).reshape(-1, 1))

        for t, tbl_state in table_state.items():
            tbl_state["norm_relpages"] = norm_relpages[tbl_mapping[t]][0]
            tbl_state["norm_reltuples"] = norm_reltuples[tbl_mapping[t]][0]

        # FIXME(CONCURRENCY_SPACE): This is currently hardcoded.
        ranges = [(0, 64), (1, 1024), (2, None)]
        for i, (step, limit) in enumerate(ranges):
            if limit is not None:
                subqueries = queries[(queries.pred_elapsed_us < limit) & (queries.pred_elapsed_us >= (ranges[i-1][1] if i > 0 else 0))]
            else:
                subqueries = queries[queries.pred_elapsed_us > ranges[i-1][1]]

            global_blks_requested = 0
            for _, tbl_state in table_state.items():
                tbl_state["norm_relpages"] = norm_relpages[i][0]
                tbl_state["num_select_tuples"] = 0
                tbl_state["num_insert_tuples"] = 0
                tbl_state["num_update_tuples"] = 0
                tbl_state["num_delete_tuples"] = 0
                tbl_state["total_tuples_touched"] = 0
                tbl_state["total_blks_requested"] = 0

            for tbls, df in subqueries.groupby(by=["target"]):
                tbls = tbls.split(",")
                for tbl in tbls:
                    table_state[tbl]["num_select_tuples"] += np.sum(df.optype == OpType.SELECT.value)
                    table_state[tbl]["num_insert_tuples"] += np.sum(df.optype == OpType.INSERT.value)
                    table_state[tbl]["num_update_tuples"] += np.sum(df.optype == OpType.UPDATE.value)
                    table_state[tbl]["num_delete_tuples"] += np.sum(df.optype == OpType.DELETE.value)
                    table_state[tbl]["total_tuples_touched"] += df.shape[0]

                    blks = df.total_blks_requested.sum()
                    table_state[tbl]["total_blks_requested"] += blks
                    global_blks_requested += blks

            key_bias, key_dists, masks, all_bias, addt_feats = extract_infer_tables_keys_features(self.model_args,
                    window_slot,
                    global_blks_requested,
                    tbl_mapping,
                    table_attr_map,
                    table_state,
                    keyspace_feat_space)

            global_feats.append([subqueries.shape[0], mpi, step])
            global_bias.append(all_bias)
            global_addt_feats.append(addt_feats)
            global_key_bias.append(key_bias)
            global_key_dists.append(key_dists)
            global_masks.append(masks)

        inputs = {
            "global_feats": torch.tensor(np.array(global_feats, dtype=np.float32)),
            "global_bias": torch.tensor(np.array(global_bias, dtype=np.float32)),
            "global_addt_feats": torch.tensor(np.array(global_addt_feats, dtype=np.float32)),
            "global_key_bias": torch.tensor(np.array(global_key_bias, dtype=np.float32)),
            "global_key_dists": torch.tensor(np.array(global_key_dists, dtype=np.float32)),
            "global_masks": torch.tensor(np.array(global_masks, dtype=np.float32)),
        }

        with torch.no_grad():
            outputs = torch.clip(self(**inputs), 0, 1).numpy()
            pred_sums = outputs.sum(axis=1)
            outputs = outputs / pred_sums[:, np.newaxis]
        return outputs

    def bias(self, histograms, queries):
        return bias_queries(histograms, queries)


class AutoMLConcurrencyModel():
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def require_optimize():
        return False

    def load_model(model_file):
        with open(f"{model_file}/args.pickle", "rb") as f:
            model_args = pickle.load(f)

        model = AutoMLConcurrencyModel(model_args)
        model.predictor = MultilabelPredictor.load(model_file)
        return model

    def __init__(self, model_args):
        super(AutoMLConcurrencyModel, self).__init__()
        self.model_args = model_args


    def get_dataset(logger, model_args):
        global_feats, global_bias, global_addt_feats, global_key_bias, global_key_dists, global_masks, global_targets, tbl_mapping, tbl_attr_keys = generate_dataset(logger, model_args, automl=True)

        inputs = []
        hist = model_args.hist_width
        dist_range = [
            ("dist_select", range(0, hist)),
            ("dist_insert", range(hist, 2 * hist)),
            ("dist_update", range(2 * hist, 3 * hist)),
            ("dist_delete", range(3 * hist, 4 * hist))
        ]
        for i in range(len(global_feats)):
            input_row = {
                "num_queries": global_feats[i][0],
                "mpi": global_feats[i][1],
                "step": global_feats[i][2],
            }

            for tidx in range(len(global_targets[i])):
                input_row[f"target_window_{tidx}"] = global_targets[i][tidx]

            for t, tidx in tbl_mapping.items():
                input_row[f"{t}_bias"] = global_bias[i][tidx][0]
                input_row[f"{t}_norm_relpages"] = global_addt_feats[i][tidx][0]
                input_row[f"{t}_norm_reltuples"] = global_addt_feats[i][tidx][1]

                for name, rg in dist_range:
                    for j in rg:
                        input_row[f"{t}_{name}_{j%hist}"] = global_key_bias[i][tidx][j]

                keys = tbl_attr_keys[t]
                for colidx, col in enumerate(keys):
                    if global_masks[i][tidx][colidx] == 1:
                        for name, rg in dist_range:
                            for j in rg:
                                # We have a valid array.
                                input_row[f"{t}_{col}_{name}_{j%hist}"] = global_key_dists[i][tidx][colidx][j]
            inputs.append(input_row)

        inputs = pd.DataFrame(inputs)
        return inputs


    def fit(self, dataset):
        targets = [c for c in dataset if "target" in c]
        num = len(targets)
        model_file = self.model_args.output_path
        predictor = MultilabelPredictor(labels=targets, problem_types=["regression"]*num, eval_metrics=["mean_squared_error"]*num, path=model_file)
        predictor.fit(dataset, time_limit=self.model_args.automl_timeout_secs, presets="medium_quality")
        with open(f"{self.model_args.output_path}/args.pickle", "wb") as f:
            pickle.dump(self.model_args, f)


    def inference(self, window_slot, mpi, queries, table_state, table_attr_map, keyspace_feat_space):

        tbl_mapping = {t:i for i, t in enumerate(table_attr_map)}
        for _, tbl_state in table_state.items():
            tbl_state["norm_relpages"] = tbl_state["num_pages"]
            tbl_state["norm_reltuples"] = tbl_state["approx_tuple_count"]

        inputs = []
        hist = self.model_args.hist_width
        dist_range = [
            ("dist_select", range(0, hist)),
            ("dist_insert", range(hist, 2 * hist)),
            ("dist_update", range(2 * hist, 3 * hist)),
            ("dist_delete", range(3 * hist, 4 * hist))
        ]

        # FIXME(CONCURRENCY_SPACE): This is currently hardcoded.
        ranges = [(0, 64), (1, 1024), (2, None)]
        for i, (step, limit) in enumerate(ranges):
            if limit is not None:
                subqueries = queries[(queries.pred_elapsed_us < limit) & (queries.pred_elapsed_us >= (ranges[i-1][1] if i > 0 else 0))]
            else:
                subqueries = queries[queries.pred_elapsed_us > ranges[i-1][1]]

            global_blks_requested = 0
            for _, tbl_state in table_state.items():
                tbl_state["num_select_tuples"] = 0
                tbl_state["num_insert_tuples"] = 0
                tbl_state["num_update_tuples"] = 0
                tbl_state["num_delete_tuples"] = 0
                tbl_state["total_tuples_touched"] = 0
                tbl_state["total_blks_requested"] = 0

            for tbls, df in subqueries.groupby(by=["target"]):
                tbls = tbls.split(",")
                for tbl in tbls:
                    table_state[tbl]["num_select_tuples"] += np.sum(df.optype == OpType.SELECT.value)
                    table_state[tbl]["num_insert_tuples"] += np.sum(df.optype == OpType.INSERT.value)
                    table_state[tbl]["num_update_tuples"] += np.sum(df.optype == OpType.UPDATE.value)
                    table_state[tbl]["num_delete_tuples"] += np.sum(df.optype == OpType.DELETE.value)
                    table_state[tbl]["total_tuples_touched"] += df.shape[0]

                    blks = df.total_blks_requested.sum()
                    table_state[tbl]["total_blks_requested"] += blks
                    global_blks_requested += blks

            key_bias, key_dists, masks, all_bias, addt_feats = extract_infer_tables_keys_features(self.model_args,
                    window_slot,
                    global_blks_requested,
                    tbl_mapping,
                    table_attr_map,
                    table_state,
                    keyspace_feat_space)

            input_row = {
                "num_queries": subqueries.shape[0],
                "mpi": mpi,
                "step": step,
            }

            for t, tidx in tbl_mapping.items():
                input_row[f"{t}_bias"] = all_bias[tidx][0]
                input_row[f"{t}_norm_relpages"] = addt_feats[tidx][0]
                input_row[f"{t}_norm_reltuples"] = addt_feats[tidx][1]

                for name, rg in dist_range:
                    for j in rg:
                        input_row[f"{t}_{name}_{j%hist}"] = key_bias[tidx][j]

                for colidx, col in enumerate(table_attr_map[t]):
                    if masks[tidx][colidx] == 1:
                        for name, rg in dist_range:
                            for j in rg:
                                # We have a valid array.
                                input_row[f"{t}_{col}_{name}_{j%hist}"] = key_dists[tidx][colidx][j]
            inputs.append(input_row)

        inputs = pd.DataFrame(inputs)
        predictions = np.clip(self.predictor.predict(inputs), 0, 1)
        columns = sorted(predictions.columns, key=lambda k: int(k.split("_")[-1]))

        predictions = predictions[columns].values
        pred_sums = predictions.sum(axis=1)
        predictions = predictions / pred_sums[:, np.newaxis]
        return predictions


    def bias(self, histograms, queries):
        return bias_queries(histograms, queries)
