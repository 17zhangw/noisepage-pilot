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

try:
    from autogluon.tabular import TabularDataset, TabularPredictor
    from behavior.model_workload.models.multilabel_predictor import MultilabelPredictor
except:
    pass

try:
    from supervised.automl import AutoML
except:
    pass


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
        self.args = model_args

    def require_optimize():
        return False


    def load_model(model_file):
        with open(f"{model_file}/args.pickle", "rb") as f:
            model_args = pickle.load(f)

        model = AutoMLBufferPageModel(model_args)
        model.predictor = TabularPredictor.load(model_file)
        return model


    def fit(self, dataset):
        predictor = TabularPredictor(label="asked_pages", problem_type="regression", eval_metric="mean_squared_error", path=self.args.output_path)
        predictor.fit(dataset, time_limit=self.args.automl_timeout_secs, presets="medium_quality")
        with open(f"{self.args.output_path}/args.pickle", "wb") as f:
            pickle.dump(self.args, f)


    def get_dataset(logger, model_args):
        sub_dirs = []
        for d in model_args.input_dirs:
            for ws in model_args.window_slices:
                sub_dirs.append(Path(d) / f"data_page_{ws}/data.feather")

        data = pd.concat(map(pd.read_feather, sub_dirs), ignore_index=True)
        data["asked_pages"] = data.total_blks_requested / data.total_tuples_touched
        x = data[["comment", "reltuples", "relpages", "asked_pages"]]
        return x


    def inference(self, input_frame):
        df = pd.DataFrame(input_frame)
        return np.clip(self.predictor.predict(df), 0, None)
