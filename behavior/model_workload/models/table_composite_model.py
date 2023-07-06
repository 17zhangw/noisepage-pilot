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

try:
    from autogluon.tabular import TabularDataset, TabularPredictor
except:
    pass
from behavior.model_workload.models.multilabel_predictor import MultilabelPredictor
from behavior.model_workload.models.utils import generate_dataset_table, generate_inference_table


COMPOSITE_TARGETS = [
    ("extend_percent", None, 1),
    ("defrag_percent", None, 1),
    ("hot_percent", None, 1),
    ("next_table_free_percent", "free_percent", 1),
    ("next_table_dead_tuple_percent", "dead_tuple_percent", 1),
    ("next_table_num_pages", "num_pages", 0),
    ("next_table_tuple_count", "tuple_count", 0),
]

COMPOSITE_METRICS = [
    "mean_absolute_error",
    "mean_absolute_error",
    "mean_absolute_error",

    "mean_absolute_error",
    "mean_absolute_error",
    "mean_squared_error",
    "mean_squared_error",
]

COMPOSITE_INPUTS = [
    "free_percent",
    "dead_tuple_percent",
    "num_pages",
    "tuple_count",
    "tuple_len_avg",
    "target_ff",
    "vacuum",

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


class AutoMLTableCompositeModel():
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __init__(self, model_args):
        super(AutoMLTableCompositeModel, self).__init__()
        self.model_args = model_args


    def get_dataset(logger, model_args, inference=False):
        input_dataset, correlated_tbls, window_index = generate_dataset_table(logger, model_args)
        input_dataset = [{k:item[k] for k in COMPOSITE_INPUTS + [t[0] for t in COMPOSITE_TARGETS]} for item in input_dataset]
        inputs = []

        for i in range(len(input_dataset) - 1):
            input_row = input_dataset[i]
            if correlated_tbls[i] != correlated_tbls[i+1]:
                # We've reached the end of the given table.
                continue

            if window_index[i] + 1 != window_index[i+1]:
                # There is a breakage so we omit the one right before the VACUUM since the "target" is garbage.
                continue

            # Produce all the targets from the next state.
            for target, src, _ in COMPOSITE_TARGETS:
                if src is None:
                    continue

                input_row[target] = input_dataset[i+1][src]

            if inference:
                input_row["table"] = correlated_tbls[i]

            inputs.append(input_row)

        # Produce the dataset.
        inputs = pd.DataFrame(inputs)
        inputs.fillna(value=0, inplace=True)
        if not inference:
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
        predictor = MultilabelPredictor(labels=[t[0] for t in COMPOSITE_TARGETS],
                                        problem_types=["regression"]*len(TABLE_STATE_TARGETS),
                                        eval_metrics=TABLE_STATE_METRICS,
                                        path=model_file,
                                        consider_labels_correlation=self.model_args.automl_consider_correlation)

        predictor.fit(dataset, time_limit=self.model_args.automl_timeout_secs, presets=self.model_args.automl_quality, num_cpus=self.model_args.num_threads)
        with open(f"{self.model_args.output_path}/args.pickle", "wb") as f:
            pickle.dump(self.model_args, f)


    def inference(self, table_state, table_attr_map, keyspace_feat_space, window, output_df=False):
        def infer(inputs):
            predictions = self.predictor.predict(inputs)
            for target, _, clip in COMPOSITE_TARGETS:
                if clip == 1:
                    predictions[target] = np.clip(predictions[target], 0, 1)
            return predictions

        tbl_keys, predictions, ret_df = generate_inference_table(self.model_args, infer, table_state, table_attr_map, keyspace_feat_space, window, output_df=output_df)

        # Only clip these predictions from 0 and 1
        outputs = np.zeros((len(tbl_keys), len(COMPOSITE_TARGETS)))
        for i, _ in enumerate(tbl_keys):
            for j, (key, _, _) in enumerate(COMPOSITE_TARGETS):
                outputs[i][j] = predictions[key].iloc[i]

        return tbl_keys, outputs, ret_df
