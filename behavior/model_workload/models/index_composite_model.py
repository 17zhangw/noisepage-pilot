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
from behavior.model_workload.models import MAX_KEYS
from behavior.model_workload.utils import OpType, Map
from sklearn.preprocessing import MinMaxScaler
from behavior.utils.process_pg_state_csvs import postgres_julian_to_unix

from autogluon.tabular import TabularDataset, TabularPredictor
from behavior.model_workload.models.multilabel_predictor import MultilabelPredictor
from behavior.model_workload.models.utils import generate_dataset_index


INDEX_STATE_TARGETS = [
    ("extend_percent", None, 0),
    ("split_percent", None, 0),
    ("next_tree_level", "tree_level", 0),
    ("next_num_pages", "num_pages", 0),
    ("next_leaf_pages", "leaf_pages", 0),
    ("next_empty_pages", "empty_pages", 0),
    ("next_deleted_pages", "deleted_pages", 0),
    ("next_avg_leaf_density", "avg_leaf_density", 1),
]

INDEX_STATE_METRICS = [
    "mean_absolute_error",
    "mean_absolute_error",

    "mean_squared_error",
    "mean_squared_error",
    "mean_squared_error",
    "mean_squared_error",
    "mean_squared_error",
    "mean_absolute_error",
]

INDEX_STATE_INPUTS = [
    "key_size",
    "key_natts",
    "tree_level",
    "num_pages",
    "leaf_pages",
    "empty_pages",
    "deleted_pages",
    "avg_leaf_densty",
    "rel_num_pages",
    "rel_num_tuples",
    "num_index_inserts",

    "num_select_queries",
    "num_insert_queries",
    "num_update_queries",
    "num_delete_queries",

    "select_queries_dist",
    "insert_queries_dist",
    "update_queries_dist",
    "delete_queries_dist",

    "num_select_tuples",
    "num_insert_tuples",
    "num_update_tuples",
    "num_delete_tuples",

    "select_tuples_dist",
    "insert_tuples_dist",
    "update_tuples_dist",
    "delete_tuples_dist",
]


class AutoMLIndexCompositeModel():
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __init__(self, model_args):
        super(AutoMLIndexCompositeModel, self).__init__()
        self.model_args = model_args


    def get_dataset(logger, model_args, inference=False):
        input_dataset, correlated_idxs, window_index = generate_dataset(logger, model_args)
        input_dataset = [{k:item[k] for k in INDEX_STATE_INPUTS + [t[0] for t in INDEX_STATE_TARGETS]} for item in input_dataset]
        inputs = []

        for i in range(len(input_dataset) - 1):
            input_row = input_dataset[i]
            if correlated_idxs[i] != correlated_idxs[i+1]:
                # Reached the end of the given index.
                continue

            if window_index[i] + 1 != window_index[i+1]:
                # There is a breakage so we omit the one right before VACUUM since targets are garbage.
                continue

            for target, src, _ in INDEX_STATE_TARGETS:
                if src is None:
                    continue

                input_row[target] = input_dataset[i+1][src]

            if inference:
                input_row["idx"] = correlated_idxs[i]

            inputs.append(input_row)

        inputs = pd.DataFrame(inputs)
        inputs.fillna(value=0, inplace=True)
        if not inference:
            inputs.to_feather(f"{model_args.output_path}/model_inputs.feather")
        return inputs


    def load_model(model_file):
        with open(f"{model_file}/args.pickle", "rb") as f:
            model_args = pickle.load(f)

        model = AutoMLIndexStateModel(model_args)
        model.predictor = MultilabelPredictor.load(model_file)
        return model


    def fit(self, dataset):
        model_file = self.model_args.output_path
        predictor = MultilabelPredictor(labels=[t[0] for t in INDEX_STATE_TARGETS],
                                        problem_types=["regression"]*len(INDEX_STATE_TARGETS),
                                        eval_metrics=INDEX_STATE_METRICS,
                                        path=model_file,
                                        consider_labels_correlation=self.model_args.automl_consider_correlation)

        predictor.fit(dataset, time_limit=self.model_args.automl_timeout_secs, presets=self.model_args.automl_quality, num_cpus=self.model_args.num_threads)
        with open(f"{self.model_args.output_path}/args.pickle", "wb") as f:
            pickle.dump(self.model_args, f)


    def inference(self, ougc, window, output_df=False):
        def infer(inputs):
            predictions = self.predictor.predict(inputs)
            for target, _, clip in COMPOSITE_TARGETS:
                if clip == 1:
                    predictions[target] = np.clip(predictions[target], 0, 1)
            return predictions

        idx_keys, predictions, ret_df = generate_inference_index(self.model_args, infer, ougc, window, output_df=output_df)

        # Coerce the output predictions based on TABLE_FEATURE_TARGETS.
        outputs = np.zeros((len(idx_keys), len(COMPOSITE_TARGETS)))
        for i, _ in enumerate(idx_keys):
            for j, (key, _, _) in enumerate(COMPOSITE_TARGETS):
                outputs[i][j] = predictions[key].iloc[i]

        return idx_keys, outputs, ret_df
