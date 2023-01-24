import math
from plumbum import cli
import argparse
from tqdm import tqdm
import copy
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import joblib
import pickle

import logging
import multiprocessing as mp

import torch
import torch.utils.data.dataset as torch_dataset
from torch.utils.data import DataLoader
import behavior.model_workload.models as models
from behavior.model_workload.models import ModelArgs, MODEL_ARGS_KEYS
from sklearn.model_selection import train_test_split

logger = logging.getLogger("build_exec_model")


def run_job(args):
    # Remove the stream handler.
    #logger.getLogger("build_exec_model").handlers.clear()
    (Path(args.output_path)).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(Path(args.output_path) / "output.log", mode="a")
    file_handler.setLevel(logging.INFO)

    try:
        from autogluon.common.utils import log_utils
        log_utils._logger_ag.addHandler(file_handler)
        log_utils._logger_ag.propagate = True
    except:
        pass

    logging.getLogger('').addHandler(file_handler)

    # Print the arguments.
    logger.info("%s", args)

    # Get the dataset.
    model_cls = getattr(models, args.model_name)
    dataset = model_cls.get_dataset(logger, args)
    model_cls(args).fit(dataset)
    open(f"{args.output_path}/outputs.pt", "w").close()

    # Remove the file handler.
    logging.getLogger('').removeHandler(file_handler)


def generate_job(model_name, input_dirs, output_dir,
                 automl_timeout_secs, automl_quality,
                 automl_consider_correlation,
                 keep_identity,
                 hist_width, num_cpus, max_threads,
                 window_slices):
    ws = window_slices.replace(",", "_")
    output = f"{output_dir}/{model_name}_step{s}_window{ws}"
    args = {
        "model_name": model_name,
        "input_dirs": input_dirs,
        "output_path": output,
        "automl_timeout_secs": automl_timeout_secs,
        "automl_quality": automl_quality,
        "automl_consider_correlation": automl_consider_correlation,
        "keep_identity": keep_identity,
        "hist_width": hist_width,
        "window_slices": window_slices.split(","),
        "num_threads": int(max(1, max_threads / num_cpus)),
    }

    v = [args[k] for k in MODEL_ARGS_KEYS]
    args = ModelArgs(*v)
    run_job(args)


class BuildExecModelCLI(cli.Application):
    model_name = cli.SwitchAttr(
        "--model-name",
        str,
        mandatory=True,
        help="Model Name that we should train.",
    )

    input_dirs = cli.SwitchAttr(
        "--input-dirs",
        str,
        mandatory=True,
        help="Path to multiple input directories.",
    )

    output_dir = cli.SwitchAttr(
        "--output-dir",
        Path,
        mandatory=True,
        help="Path to the containing output model directory.",
    )

    num_cpus = cli.SwitchAttr(
        "--num-cpus",
        int,
        default=mp.cpu_count(),
        help="Number of CPUs to use.",
    )

    max_threads = cli.SwitchAttr(
        "--max-threads",
        int,
        default=mp.cpu_count(),
        help="Maximum number of threads available to allocate.",
    )

    hist_width = cli.SwitchAttr(
        "--hist-width",
        int,
        default=10,
        help="Width of the histogram.",
    )

    window_slices = cli.SwitchAttr(
        "--window-slices",
        str,
        default="1000",
        help="Default slices to consider for buffer page model.",
    )

    automl_timeout_secs = cli.SwitchAttr(
        "--automl-timeout-secs",
        int,
        default=3600,
        help="Number of seconds for the AutoML timeout.",
    )

    automl_quality = cli.SwitchAttr(
        "--automl-quality",
        str,
        default=None,
        help="AutoML Quality to use.",
    )

    automl_consider_correlation = cli.Flag(
        "--automl-consider-correlation",
        default=False,
        help="Whether AutoML can consider multi-label correlation.",
    )

    keep_identity = cli.Flag(
        "--keep-identity",
        default=False,
        help="Whether to keep identifiers or not.",
    )

    def main(self):
        generate_jobs(self.model_name,
                      self.input_dirs.split(","),
                      self.output_dir,
                      self.automl_timeout_secs,
                      self.automl_quality,
                      self.automl_consider_correlation,
                      self.keep_identity,
                      self.hist_width,
                      self.num_cpus,
                      self.max_threads,
                      self.window_slices)


if __name__ == "__main__":
    BuildExecModelCLI.run()
