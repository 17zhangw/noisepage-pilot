import os
import multiprocessing as mp
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
    def workload_analyze(benchmark, input_workload, workload_only, psycopg2_conn, work_prefix, load_raw, load_initial_data, load_deltas, load_hits, load_exec_stats, load_windows):
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

        if load_deltas is not None:
            eval_args += "--load-deltas "

        if load_hits is not None:
            eval_args += "--load-hits "

        if load_exec_stats is not None:
            eval_args += "--load-exec-stats "

        if load_windows is not None:
            eval_args += "--load-windows "

        if psycopg2_conn is not None:
            eval_args = eval_args + f"--psycopg2-conn \"{psycopg2_conn}\" "

        return f"python3 -m behavior workload_analyze {eval_args}"

    return {
        "actions": [CmdAction(workload_analyze, buffering=1),],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            { "name": "benchmark", "long": "benchmark", "help": "Benchmark that is being analyzed.", "default": None, },
            { "name": "input_workload", "long": "input_workload", "help": "Path to the input workload that should be analyzed.", "default": None, },
            { "name": "workload_only", "long": "workload_only", "help": "Whether the input workload is only the workload or not.", "default": False, },
            { "name": "psycopg2_conn", "long": "psycopg2_conn", "help": "psycopg2 connection string to connect to the valid database instance.", "default": None, },
            { "name": "work_prefix", "long": "work_prefix", "help": "Prefix to use for working with the database.", "default": None, },
            { "name": "load_raw", "long": "load_raw", "help": "Whether to load the raw data or not.", "default": None, },
            { "name": "load_initial_data", "long": "load_initial_data", "help": "Load the initial data.", "default": None, },
            { "name": "load_deltas", "long": "load_deltas", "help": "Load the deltas.", "default": None, },
            { "name": "load_hits", "long": "load_hits", "help": "Load the hit.", "default": None, },
            { "name": "load_exec_stats", "long": "load_exec_stats", "help": "Whether to load the execution statistics or not.", "default": None, },
            { "name": "load_windows", "long": "load_windows", "help": "Whether to load the windows or not.", "default": None, },
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
                "default": "1000",
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


def task_workload_build_exec_model():
    """
    Workload Model: train execution feature models.
    """
    def workload_build_exec_model(
            model_name, input_dirs, output_dir,
            automl_timeout_secs, lr,
            epochs, batch_size, cuda, train_size, hidden,
            depth, sweep_dropout, add_nonnorm_features, num_iterations,
            num_cpus, max_threads, hist_width, patience, ckpt_interval,
            steps, window_slices):
        assert model_name is not None
        assert input_dirs is not None
        assert output_dir is not None

        for iw in input_dirs.split(","):
            assert Path(iw).exists(), f"{iw} is not valid path."

        eval_args = (
            f"--model-name {model_name} "
            f"--automl-timeout-secs {automl_timeout_secs} "
            f"--train-size {train_size} "
            f"--input-dirs {input_dirs} "
            f"--output-dir {output_dir} "
            f"--num-iterations {num_iterations} "
            f"--num-cpus {num_cpus} "
            f"--max-threads {max_threads} "
            f"--hist-width {hist_width} "
            f"--patience {patience} "
            f"--ckpt-interval {ckpt_interval} "
            f"--steps {steps} "
            f"--window-slices {window_slices} "
        )

        if lr:
            eval_args += f"--lr {lr} "
        if epochs:
            eval_args += f"--epochs {epochs} "
        if batch_size:
            eval_args += f"--batch-size {batch_size} "
        if hidden:
            eval_args += f"--hidden {hidden} "
        if depth:
            eval_args += f"--depth {depth} "
        if cuda:
            eval_args += "--cuda "
        if sweep_dropout:
            eval_args += "--sweep-dropout "
        if add_nonnorm_features:
            eval_args += "--add-nonnorm-features "

        return f"python3 -m behavior workload_build_exec_model {eval_args}"

    return {
        "actions": [CmdAction(workload_build_exec_model, buffering=1),],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            { "name": "model_name", "long": "model_name", "help": "Model Name from models to create.", "default": None, },
            { "name": "input_dirs", "long": "input_dirs", "help": "Path to multiple input directories.", "default": None, },
            { "name": "output_dir", "long": "output_dir", "help": "Path to the containing output model directory.", "default": None, },
            { "name": "automl_timeout_secs", "long": "automl_timeout_secs", "help": "AutoML timeout in seconds.", "default": 3600, },
            { "name": "lr", "long": "lr", "help": "Learning rate to use for training.", "default": None, },
            { "name": "epochs", "long": "epochs", "help": "Epochs to use for training.", "default": None, },
            { "name": "batch_size", "long": "batch_size", "help": "Batch size to use for training.", "default": None, },
            { "name": "cuda", "long": "cuda", "help": "Whether to use CUDA or not.", "default": False, },
            { "name": "train_size", "long": "train_size", "help": "Percentage of data to use for training.", "default": 0.8, },
            { "name": "hidden", "long": "hidden", "help": "Number of hidden to units to use across the model.", "default": None, },
            { "name": "depth", "long": "depth", "help": "Depth to use in the model.", "default": None, },
            { "name": "sweep_dropout", "long": "sweep_dropout", "help": "Whether to sweep dropout.", "default": False, },
            { "name": "add_nonnorm_features", "long": "add_nonnorm_features", "help": "Whether to add non normalization features>", "default": False, },
            { "name": "num_iterations", "long": "num_iterations", "help": "Number of iterations.", "default": 1, },
            { "name": "num_cpus", "long": "num_cpus", "help": "Number of CPUs.", "default": mp.cpu_count(), },
            { "name": "max_threads", "long": "max_threads", "help": "Maximum number of threads.", "default": mp.cpu_count(), },
            { "name": "hist_width", "long": "hist_width", "help": "Width of the histogram.", "default": 10, },
            { "name": "patience", "long": "patience", "help": "Patience for early stopping.", "default": 400, },
            { "name": "ckpt_interval", "long": "ckpt_interval", "help": "Interval for checkpointing.", "default": 100, },
            { "name": "steps", "long": "steps", "help": "Steps for concurrency.", "default": "1", },
            { "name": "window_slices", "long": "window_slices", "help": "Window slices for buffer pool model.", "default": "1000", },
        ],
    }
