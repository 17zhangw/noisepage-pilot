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
from behavior.model_workload.models.utils import extract_train_tables_keys_features
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import shutil


def generate_dataset(logger, model_args):
    # Generate all the directories we want.
    sub_dirs = []
    for d in model_args.input_dirs:
        for ws in model_args.window_slices:
            sub_dirs.append(Path(d) / f"data_page_{ws}")
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
        data = pd.concat(map(pd.read_feather, [f"{d}/data.feather" for d in dirs]))
        return MinMaxScaler().fit(data.relpages.values.reshape(-1, 1)), MinMaxScaler().fit(data.reltuples.values.reshape(-1, 1))

    def handle(in_dir, relpages_scaler, reltuples_scaler):
        logger.info("Processing input from: %s", in_dir)
        pg_settings = pd.read_csv(f"{Path(in_dir).parent}/pg_settings.csv")
        sb_mb = pg_settings.iloc[0].shared_buffers / 1024 / 1024

        data = pd.read_feather(f"{in_dir}/data.feather")
        tbl_map = {}
        for p in glob.glob(f"{in_dir}/keys/*.feather"):
            tbl_map[Path(p).stem] = pd.read_feather(p)

        data["norm_relpages"] = relpages_scaler.transform(data.relpages.values.reshape(-1, 1))
        data["norm_reltuples"] = reltuples_scaler.transform(data.reltuples.values.reshape(-1, 1))

        # Assume every query succeeds I guess...
        for window in data.groupby(by=["window_bucket"]):
            all_queries = window[1].num_queries.sum()
            all_targets = np.zeros(len(tbl_mapping))

            # Compute the targets.
            for t, f in window[1].groupby(by=["target"]):
                all_targets[tbl_mapping[t]] = 1.0 if f.total_blks_requested.sum() == 0 else f.total_blks_hit.sum() / f.total_blks_requested.sum()

            key_bias, key_dists, masks, all_bias, addt_feats = extract_train_tables_keys_features(model_args.add_nonnorm_features, tbl_map, tbl_mapping, keys, hist_width, window[1], window[0])

            global_feats.append([all_queries, sb_mb])
            global_bias.append(all_bias)
            global_targets.append(all_targets)
            global_addt_feats.append(addt_feats)
            global_key_bias.append(key_bias)
            global_key_dists.append(key_dists)
            global_masks.append(masks)

    # Generate the scalers for relpages/reltuples.
    relpages_scaler, reltuples_scaler = produce_scaler(sub_dirs)
    joblib.dump(relpages_scaler, f"{model_args.dataset_path}/relpages_scaler.gz")
    joblib.dump(reltuples_scaler, f"{model_args.dataset_path}/reltuples_scaler.gz")

    # Construct the table attr mappings.
    for d in model_args.input_dirs:
        c = f"{d}/keyspaces.pickle"
        assert Path(c).exists()
        with open(c, "rb") as f:
            metadata = pickle.load(f)

        for t, k in metadata.table_attr_map.items():
            if t not in tbl_mapping:
                tbl_mapping[t] = len(tbl_mapping)
                keys[t] = k

    # Generate the feature data.
    for d in sub_dirs:
        handle(d, relpages_scaler, reltuples_scaler)

    (Path(model_args.dataset_path)).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb") as f:
        pickle.dump(global_feats, f)
        pickle.dump(global_bias, f)
        pickle.dump(global_targets, f)
        pickle.dump(global_addt_feats, f)
        pickle.dump(global_key_bias, f)
        pickle.dump(global_key_dists, f)
        pickle.dump(global_masks, f)
        f.flush()
        shutil.copy(f.name, f"{model_args.dataset_path}/dataset.pickle")


class BufferAccessModel(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def require_optimize():
        return True

    def __init__(self, model_args, num_outputs):
        super(BufferAccessModel, self).__init__()

        assert num_outputs > 0
        input_size = 4 * model_args.hist_width

        # Mapping from keyspace -> embedding.
        self.dist = construct_stack(input_size, model_args.hidden_size, model_args.hidden_size, model_args.dropout, model_args.depth)
        # Mapping from tbl + keyspace -> embedding
        num_inputs = 4 if model_args.add_nonnorm_features else 2
        self.comb_tbls = construct_stack(model_args.hidden_size + num_inputs, model_args.hidden_size, model_args.hidden_size, model_args.dropout, model_args.depth)
        # Final mapping.
        self.final = construct_stack(model_args.hidden_size + 2, model_args.hidden_size, num_outputs, model_args.dropout, model_args.depth)

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
        return self.final(input_vec)

    
    def loss(self, target_outputs, model_outputs):
        outputs = target_outputs["global_targets"]
        loss = MSELoss()(outputs, model_outputs)
        return loss, loss.item()


    def get_dataset(logger, model_args):
        dataset_path = Path(model_args.dataset_path) / "dataset.pickle"
        if not dataset_path.exists():
            generate_dataset(logger, model_args)

        with open(dataset_path, "rb") as f:
            global_feats = pickle.load(f)
            global_bias = pickle.load(f)
            global_targets = pickle.load(f)
            global_addt_feats = pickle.load(f)
            global_key_bias = pickle.load(f)
            global_key_dists = pickle.load(f)
            global_masks = pickle.load(f)

        td = dataset.TensorDataset(
            torch.tensor(np.array(global_feats, dtype=np.float32)),
            torch.tensor(np.array(global_bias, dtype=np.float32)),
            torch.tensor(np.array(global_addt_feats, dtype=np.float32)),
            torch.tensor(np.array(global_key_bias, dtype=np.float32)),
            torch.tensor(np.array(global_key_dists, dtype=np.float32)),
            torch.tensor(np.array(global_masks, dtype=np.float32)),
            torch.tensor(np.array(global_targets, dtype=np.float32)))

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
        return td, feat_names, target_names, num_outputs, None
