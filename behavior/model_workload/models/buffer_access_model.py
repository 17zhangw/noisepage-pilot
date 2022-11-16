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
from behavior.model_workload.models.utils import extract_train_tables_keys_features, extract_infer_tables_keys_features
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import shutil

try:
    from autogluon.tabular import TabularDataset, TabularPredictor
    from behavior.model_workload.models.multilabel_predictor import MultilabelPredictor
except:
    pass


def generate_dataset(logger, model_args, automl=False):
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

        if automl:
            data["norm_relpages"] = data.relpages
            data["norm_reltuples"] = data.reltuples
        else:
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

    if not automl:
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
    else:
        return global_feats, global_bias, global_targets, global_addt_feats, global_key_bias, global_key_dists, global_masks, tbl_mapping, keys


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

    def load_model(model_file):
        model_obj = torch.load(f"{model_file}/best_model.pt")
        model = BufferAccessModel(model_obj["model_args"], model_obj["num_outputs"])
        model.model_args = model_obj["model_args"]
        model.load_state_dict(model_obj["best_model"])

        parent_dir = Path(model_file).parent
        model.relpages_scaler = joblib.load(f"{parent_dir}/relpages_scaler.gz")
        model.reltuples_scaler = joblib.load(f"{parent_dir}/reltuples_scaler.gz")
        return model

    def inference(self, window_slot, num_queries, sb_bytes, table_state, table_attr_map, keyspace_feat_space):
        tbl_mapping = {t:i for i, t in enumerate(table_attr_map)}
        norm_relpages = self.relpages_scaler.transform(np.array([table_state[t]["num_pages"] for t in tbl_mapping]).reshape(-1, 1))
        norm_reltuples = self.reltuples_scaler.transform(np.array([table_state[t]["approx_tuple_count"] for t in tbl_mapping]).reshape(-1, 1))

        global_blks_requested = 0
        for tbl, i in tbl_mapping.items():
            table_state[tbl]["norm_relpages"] = norm_relpages[i][0]
            table_state[tbl]["norm_reltuples"] = norm_reltuples[i][0]
            global_blks_requested += table_state[tbl]["total_blks_requested"]

        key_bias, key_dists, masks, all_bias, addt_feats = extract_infer_tables_keys_features(self.model_args,
                window_slot,
                global_blks_requested,
                tbl_mapping,
                table_attr_map,
                table_state,
                keyspace_feat_space)

        global_feats = [[num_queries, sb_bytes / 1024 / 1024]]
        global_bias = [all_bias]
        global_addt_feats = [addt_feats]
        global_key_bias = [key_bias]
        global_key_dists = [key_dists]
        global_masks = [masks]

        with torch.no_grad():
            inputs = {
                "global_feats": torch.tensor(np.array(global_feats, dtype=np.float32)),
                "global_bias": torch.tensor(np.array(global_bias, dtype=np.float32)),
                "global_addt_feats": torch.tensor(np.array(global_addt_feats, dtype=np.float32)),
                "global_key_bias": torch.tensor(np.array(global_key_bias, dtype=np.float32)),
                "global_key_dists": torch.tensor(np.array(global_key_dists, dtype=np.float32)),
                "global_masks": torch.tensor(np.array(global_masks, dtype=np.float32)),
            }

            predictions = self(**inputs)[0]
            outputs = torch.clip(predictions, 0, 1)
        return outputs, tbl_mapping


class AutoMLBufferAccessModel():
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def require_optimize():
        return False

    def __init__(self, model_args):
        super(AutoMLBufferAccessModel, self).__init__()
        self.model_args = model_args

    def get_dataset(logger, model_args):
        global_feats, global_bias, global_targets, global_addt_feats, global_key_bias, global_key_dists, global_masks, tbl_mapping, tbl_attr_keys = generate_dataset(logger, model_args, automl=True)

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
                "sb_mb": global_feats[i][1],
            }

            for t, tidx in tbl_mapping.items():
                input_row[f"target_{t}"] = global_targets[i][tidx]
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


    def load_model(model_file):
        with open(f"{model_file}/args.pickle", "rb") as f:
            model_args = pickle.load(f)

        model = AutoMLBufferAccessModel(model_args)
        model.predictor = MultilabelPredictor.load(model_file)
        return model


    def fit(self, dataset):
        targets = [c for c in dataset if "target" in c]
        num = len(targets)
        model_file = self.model_args.output_path
        predictor = MultilabelPredictor(labels=targets, problem_types=["regression"]*num, eval_metrics=["mean_squared_error"]*num, path=model_file)
        predictor.fit(dataset, time_limit=self.model_args.automl_timeout_secs, presets="medium_quality")
        with open(f"{self.model_args.output_path}/args.pickle", "wb") as f:
            pickle.dump(self.model_args, f)


    def inference(self, window_slot, num_queries, sb_bytes, table_state, table_attr_map, keyspace_feat_space):
        tbl_mapping = {t:i for i, t in enumerate(table_attr_map)}
        global_blks_requested = 0
        for _, tbl_state in table_state.items():
            tbl_state["norm_relpages"] = tbl_state["num_pages"]
            tbl_state["norm_reltuples"] = tbl_state["approx_tuple_count"]
            global_blks_requested += tbl_state["total_blks_requested"]

        key_bias, key_dists, masks, all_bias, addt_feats = extract_infer_tables_keys_features(self.model_args,
                window_slot,
                global_blks_requested,
                tbl_mapping,
                table_attr_map,
                table_state,
                keyspace_feat_space)

        hist = self.model_args.hist_width
        dist_range = [
            ("dist_select", range(0, hist)),
            ("dist_insert", range(hist, 2 * hist)),
            ("dist_update", range(2 * hist, 3 * hist)),
            ("dist_delete", range(3 * hist, 4 * hist))
        ]

        input_row = {
            "num_queries": num_queries,
            "sb_mb": sb_bytes / 1024 / 1024,
        }
        for t, tidx in tbl_mapping.items():
            input_row[f"{t}_bias"] = key_bias[tidx][0]
            input_row[f"{t}_norm_relpages"] = addt_feats[tidx][0]
            input_row[f"{t}_norm_reltuples"] = addt_feats[tidx][1]

            for name, rg in dist_range:
                for j in rg:
                    input_row[f"{t}_{name}_{j%hist}"] = key_bias[tidx][j]

            keys = table_attr_map[t]
            for colidx, col in enumerate(keys):
                if masks[tidx][colidx] == 1:
                    for name, rg in dist_range:
                        for j in rg:
                            # We have a valid array.
                            input_row[f"{t}_{col}_{name}_{j%hist}"] = key_dists[tidx][colidx][j]

        inputs = pd.DataFrame([input_row])
        predictions = self.predictor.predict(inputs)
        outputs = np.zeros(len(tbl_mapping))
        for t, idx in tbl_mapping.items():
            outputs[idx] = predictions[f"target_{t}"].iloc[0]

        # Clip outputs to be between 0 and 1.
        outputs = np.clip(outputs, 0, 1)
        return outputs, tbl_mapping
