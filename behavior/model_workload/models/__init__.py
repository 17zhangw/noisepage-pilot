from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_KEYS = 5

MODEL_ARGS_KEYS = [
    "model_name",
    "automl_timeout_secs",
    "automl_quality",
    "automl_forecast_horizon",
    "automl_splitter_offset",
    "lr",
    "num_epochs",
    "batch_size",
    "train_size",
    "cuda",
    "hidden_size",
    "output_path",
    "dataset_path",
    "depth",
    "input_dirs",
    "dropout",
    "num_threads",
    "ckpt_interval",
    "patience",
    "hist_width",
    "steps",
    "window_slices",
    "add_nonnorm_features",
]

ModelArgs = namedtuple("ModelArgs", MODEL_ARGS_KEYS)


def construct_stack(input_size, hidden_size, output_size, dropout, depth, activation="ReLU"):
    modules = [
        nn.Linear(input_size, hidden_size),
        getattr(nn, activation)(),
    ]

    if dropout:
        modules.append(nn.Dropout())

    for _ in range(depth - 1):
        modules.extend([
            nn.Linear(hidden_size, hidden_size),
            getattr(nn, activation)(),
        ])

        if dropout:
            modules.append(nn.Dropout())

    modules.append(nn.Linear(hidden_size, output_size))

    model = nn.Sequential(*modules)
    return model


from behavior.model_workload.models.buffer_access_model import BufferAccessModel
from behavior.model_workload.models.table_feature_model import TableFeatureModel
from behavior.model_workload.models.concurrency_model import ConcurrencyModel
from behavior.model_workload.models.buffer_page_model import BufferPageModel

from behavior.model_workload.models.buffer_access_model import AutoMLBufferAccessModel
from behavior.model_workload.models.table_feature_model import AutoMLTableFeatureModel
from behavior.model_workload.models.concurrency_model import AutoMLConcurrencyModel
from behavior.model_workload.models.buffer_page_model import AutoMLBufferPageModel

from behavior.model_workload.models.table_state_model import AutoMLTableStateModel
from behavior.model_workload.models.table_state_forecast import AutoMLTableStateForecastWide
from behavior.model_workload.models.table_state_forecast import AutoMLTableStateForecastNarrow
