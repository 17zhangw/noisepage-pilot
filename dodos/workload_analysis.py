import os
from pathlib import Path

from doit.action import CmdAction
from plumbum import local, FG

import dodos.benchbase
import dodos.noisepage
from dodos import VERBOSITY_DEFAULT, default_artifacts_path, default_build_path
from dodos.benchbase import ARTIFACTS_PATH as BENCHBASE_ARTIFACTS_PATH
from dodos.noisepage import (
    ARTIFACTS_PATH as NOISEPAGE_ARTIFACTS_PATH,
    ARTIFACT_pgdata,
    ARTIFACT_psql,
)
from behavior import BENCHDB_TO_TABLES

ARTIFACTS_PATH = default_artifacts_path()
ARTIFACT_MODELS = ARTIFACTS_PATH / "workload_models"
BUILD_PATH = default_build_path()


def task_workload_analyze():
    """
    Workload Analysis: perform analysis of a workload and populate all data needed for further computation.
    """
    def workload_analyze(benchmark, input_workload, workload_only, psycopg2_conn, work_prefix, load_raw, load_initial_data, load_exec_stats):
        assert input_workload is not None
        assert work_prefix is not None
        assert len(benchmark.split(",")) == len(input_workload.split(","))

        for iw in input_workload.split(","):
            assert Path(iw).exists(), f"{iw} is not valid path."

        for bw in benchmark.split(","):
            assert bw in BENCHDB_TO_TABLES

        eval_args = (
            f"--benchmark {benchmark} "
            f"--dir-workload-input {input_workload} "
            f"--workload-only {workload_only} "
            f"--work-prefix {work_prefix} "
        )

        if load_raw is not None:
            eval_args += "--load-raw "

        if load_initial_data is not None:
            eval_args += "--load-initial-data "

        if load_exec_stats is not None:
            eval_args += "--load-exec-stats "

        if psycopg2_conn is not None:
            eval_args = eval_args + f"--psycopg2-conn \"{psycopg2_conn}\" "

        return f"python3 -m behavior workload_analyze {eval_args}"

    return {
        "actions": [CmdAction(workload_analyze, buffering=1),],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "benchmark",
                "long": "benchmark",
                "help": "Benchmark that is being analyzed.",
                "default": None,
            },
            {
                "name": "input_workload",
                "long": "input_workload",
                "help": "Path to the input workload that should be analyzed.",
                "default": None,
            },
            {
                "name": "workload_only",
                "long": "workload_only",
                "help": "Whether the input workload is only the workload or not.",
                "default": False,
            },
            {
                "name": "psycopg2_conn",
                "long": "psycopg2_conn",
                "help": "psycopg2 connection string to connect to the valid database instance.",
                "default": None,
            },
            {
                "name": "work_prefix",
                "long": "work_prefix",
                "help": "Prefix to use for working with the database.",
                "default": None,
            },
            {
                "name": "load_raw",
                "long": "load_raw",
                "help": "Whether to load the raw data or not.",
                "default": None,
            },
            {
                "name": "load_initial_data",
                "long": "load_initial_data",
                "help": "Load the initial data.",
                "default": None,
            },
            {
                "name": "load_exec_stats",
                "long": "load_exec_stats",
                "help": "Whether to load the execution statistics or not.",
                "default": None,
            },
        ],
    }


def task_workload_exec_feature_synthesis():
    """
    Workload Analysis: collect the input feature data for training exec feature model.
    """
    def workload_exec_feature_synthesis(input_workload, workload_only, psycopg2_conn, work_prefix, buckets, steps, slice_window, offcpu_logwidth, gen_exec_features, gen_data_page_features, gen_concurrency_features):
        assert input_workload is not None
        assert work_prefix is not None

        for iw in input_workload.split(","):
            assert Path(iw).exists(), f"{iw} is not valid path."

        eval_args = (
            f"--dir-workload-input {input_workload} "
            f"--workload-only {workload_only} "
            f"--work-prefix {work_prefix} "
            f"--buckets {buckets} "
            f"--steps {steps} "
            f"--slice-window {slice_window} "
            f"--offcpu-logwidth {offcpu_logwidth} "
        )

        if gen_exec_features is not None:
            eval_args += "--gen-exec-features "
        if gen_data_page_features is not None:
            eval_args += "--gen-data-page-features "
        if gen_concurrency_features is not None:
            eval_args += "--gen-concurrency-features "

        if psycopg2_conn is not None:
            eval_args = eval_args + f"--psycopg2-conn \"{psycopg2_conn}\" "

        return f"python3 -m behavior workload_exec_feature_synthesis {eval_args}"

    return {
        "actions": [CmdAction(workload_exec_feature_synthesis, buffering=1),],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "input_workload",
                "long": "input_workload",
                "help": "Path to the input workload that should be analyzed.",
                "default": None,
            },
            {
                "name": "workload_only",
                "long": "workload_only",
                "help": "Whether the input workload is only the workload or not.",
                "default": False,
            },
            {
                "name": "psycopg2_conn",
                "long": "psycopg2_conn",
                "help": "psycopg2 connection string to connect to the valid database instance.",
                "default": None,
            },
            {
                "name": "work_prefix",
                "long": "work_prefix",
                "help": "Prefix to use for working with the database.",
                "default": None,
            },
            {
                "name": "buckets",
                "long": "buckets",
                "help": "Number of buckets to use for bucketizing input data.",
                "default": 10,
            },
            {
                "name": "steps",
                "long": "steps",
                "help": "Summarization steps for concurrency histograms.",
                "default": "1",
            },
            {
                "name": "slice_window",
                "long": "slice_window",
                "help": "Slice window to use.",
                "default": "10000",
            },
            {
                "name": "offcpu_logwidth",
                "long": "offcpu_logwidth",
                "help": "Off CPU Log-width time (# buckets in histogram).",
                "default": 31,
            },
            {
                "name": "gen_exec_features",
                "long": "gen_exec_features",
                "help": "Whether to generate exec features data.",
                "default": None,
            },
            {
                "name": "gen_data_page_features",
                "long": "gen_data_page_features",
                "help": "Whether to generate data page features.",
                "default": None,
            },
            {
                "name": "gen_concurrency_features",
                "long": "gen_concurrency_features",
                "help": "Whether to generate concurrency features.",
                "default": None,
            },
        ],
    }


def task_workload_train():
    """
    Workload Analysis: train workload models.
    """

    def train_cmd(input_data, output_dir, separate, val_size, lr, epochs, batch_size, hidden_size, hist_length, cuda):
        if not Path(output_dir).is_absolute():
            # Make it a relative path to ARTIFACT_MODELS.
            output_dir = ARTIFACT_MODELS / output_dir

        train_args = (
            f"--dir-input \"{input_data}\" "
            f"--dir-output {output_dir} "
            f"--separate {separate} "
            f"--val-size {val_size} "
            f"--epochs {epochs} "
            f"--batch-size {batch_size} "
            f"--hidden-size {hidden_size} "
            f"--hist-length {hist_length} "
        )

        if cuda is not None:
            train_args += f"--cuda "

        return f"python3 -m behavior workload_train {train_args}"

    return {
        "actions": [f"mkdir -p {ARTIFACT_MODELS}", CmdAction(train_cmd, buffering=1)],
        "targets": [ARTIFACT_MODELS],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
        "params": [
            {
                "name": "input_data",
                "long": "input_data",
                "help": "Path to all the input data that should be used, comma separated if multiple globs.",
                "default": None,
            },
            {
                "name": "output_dir",
                "long": "output_dir",
                "help": "Path to the output folder where the workload models should be written to.",
                "default": None,
            },
            {
                "name": "separate",
                "long": "separate",
                "help": "Whether to train separate models for each workload execution feature.",
                "default": False,
            },
            {
                "name": "val_size",
                "long": "val_size",
                "help": "Percentage of the validation dataset.",
                "type": float,
                "default": 0.2,
            },
            {
                "name": "lr",
                "long": "lr",
                "help": "Learning rate for Adam optimizer.",
                "type": float,
                "default": 0.001,
            },
            {
                "name": "epochs",
                "long": "epochs",
                "help": "Number of epochs to train the model.",
                "type": int,
                "default": 1000,
            },
            {
                "name": "batch_size",
                "long": "batch_size",
                "help": "Batch size that should be used.",
                "type": int,
                "default": 1024,
            },
            {
                "name": "hidden_size",
                "long": "hidden_size",
                "help": "Number of hidden units.",
                "type": int,
                "default": 256,
            },
            {
                "name": "hist_length",
                "long": "hist_length",
                "help": "Length of histogram featurization.",
                "type": int,
                "default": 10,
            },
            {
                "name": "cuda",
                "long": "cuda",
                "help": "Whether to use CUDA or not.",
                "default": None,
            },
        ],
    }
