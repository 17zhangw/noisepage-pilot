from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_KEYS = 5

MODEL_ARGS_KEYS = [
    "model_name",
    "input_dirs",
    "output_path",
    "automl_timeout_secs",
    "automl_quality",
    "automl_consider_correlation",
    "keep_identity",
    "hist_width",
    "window_slices",
    "num_threads",
]

ModelArgs = namedtuple("ModelArgs", MODEL_ARGS_KEYS)


from behavior.model_workload.models.table_feature_model import AutoMLTableFeatureModel
from behavior.model_workload.models.table_state_model import AutoMLTableStateModel
from behavior.model_workload.models.table_composite_model import AutoMLTableCompositeModel

from behavior.model_workload.models.index_feature_model import AutoMLIndexFeatureModel
from behavior.model_workload.models.index_state_model import AutoMLIndexStateModel
from behavior.model_workload.models.index_composite_model import AutoMLIndexCompositeModel

from behavior.model_workload.models.buffer_page_model import AutoMLBufferPageModel
