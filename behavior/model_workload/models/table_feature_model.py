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

from autogluon.tabular import TabularDataset, TabularPredictor
from behavior.model_workload.models.multilabel_predictor import MultilabelPredictor
from behavior.model_workload.models.utils import generate_dataset_table, generate_inference_table


TABLE_FEATURE_TARGETS = [
    "extend_percent",
    "defrag_percent",
    "hot_percent",
]

TABLE_FEATURE_METRICS = [
    "mean_absolute_error",
    "mean_absolute_error",
    "mean_absolute_error",
]

TABLE_FEATURE_INPUTS = [
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


class AutoMLTableFeatureModel():
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __init__(self, model_args):
        super(AutoMLTableFeatureModel, self).__init__()
        self.model_args = model_args


    def get_dataset(logger, model_args, inference=False):
        input_dataset, correlated_tbls, _ = generate_dataset_table(logger, model_args)
        input_dataset = [{k:item[k] for k in TABLE_FEATURE_INPUTS + TABLE_FEATURE_TARGETS} for item in input_dataset]

        # Attach the table column if for inference.
        if inference:
            for i in range(len(input_dataset)):
                input_dataset[i]["table"] = correlated_tbls[i]

        inputs = pd.DataFrame(inputs)
        inputs.fillna(value=0, inplace=True)

        # Log the training data.
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
        predictor = MultilabelPredictor(labels=TABLE_FEATURE_TARGETS,
                                        problem_types=["regression"]*len(TABLE_FEATURE_TARGETS),
                                        eval_metrics=TABLE_FEATURE_METRICS,
                                        path=model_file,
                                        consider_labels_correlation=self.model_args.automl_consider_correlation)

        predictor.fit(dataset, time_limit=self.model_args.automl_timeout_secs, presets=self.model_args.automl_quality, num_cpus=self.model_args.num_threads)
        with open(f"{self.model_args.output_path}/args.pickle", "wb") as f:
            pickle.dump(self.model_args, f)


    def inference(self, table_state, table_attr_map, keyspace_feat_space, window, output_df=None):
        def infer(inputs):
            return np.clip(self.predictor.predict(inputs), 0, 1)

        tbl_keys, predictions, ret_df = generate_inference_table(self.model_args, infer, table_state, table_attr_map, keyspace_feat_space, window, output_df=output_df)

        # Coerce the output predictions based on TABLE_FEATURE_TARGETS.
        outputs = np.zeros((len(tbl_keys), len(TABLE_FEATURE_TARGETS)))
        for i, _ in enumerate(tbl_keys):
            for j, key in enumerate(TABLE_FEATURE_TARGETS):
                outputs[i][j] = predictions[key].iloc[i]

        return tbl_keys, outputs, ret_df
