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
from behavior.model_workload.utils import OpType
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import shutil
from supervised.automl import AutoML


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

    def fit(self, dataset):
        x, y = dataset[0], dataset[1]

        automl = AutoML(results_path=f"{self.args.output_path}/perform", ml_task="regression", eval_metric="mse", mode="Perform")
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
