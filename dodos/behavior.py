import os
from pathlib import Path

from datetime import datetime
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
BUILD_PATH = default_build_path()

# Input: various configuration files.
DATAGEN_CONFIG_FILE = Path("config/behavior/datagen.yaml").absolute()
MODELING_CONFIG_FILE = Path("config/behavior/modeling.yaml").absolute()
POSTGRESQL_CONF = Path("config/postgres/default_postgresql.conf").absolute()

# Output: model directory.
ARTIFACT_WORKLOADS = ARTIFACTS_PATH / "workloads"
ARTIFACT_DATA_RAW = ARTIFACTS_PATH / "data/raw"
ARTIFACT_DATA_OUS = ARTIFACTS_PATH / "data/ous"
ARTIFACT_MODELS = ARTIFACTS_PATH / "models"
ARTIFACT_EVALS_OU = ARTIFACTS_PATH / "evals_ou"
ARTIFACT_EVALS_QUERY = ARTIFACTS_PATH / "evals_query"
ARTIFACT_EVALS_QUERY_WORKLOAD = ARTIFACTS_PATH / "evals_query_workload"


def task_behavior_generate_workloads():
    """
    Behavior modeling: generate the workloads that we plan to execute for training data.
    """
    generate_workloads_args = (
        f"--config-file {DATAGEN_CONFIG_FILE} "
        f"--postgresql-config-file {POSTGRESQL_CONF} "
        f"--dir-benchbase-config {dodos.benchbase.CONFIG_FILES} "
        f"--dir-output {ARTIFACT_WORKLOADS} "
    )

    def conditional_clear(clear_existing):
        if clear_existing != "False":
            local["rm"]["-rf"][f"{ARTIFACT_WORKLOADS}"].run()

        return None

    return {
        "actions": [
            conditional_clear,
            f"python3 -m behavior generate_workloads {generate_workloads_args}",
        ],
        "file_dep": [
            dodos.benchbase.ARTIFACT_benchbase,
            dodos.noisepage.ARTIFACT_postgres,
        ],
        "targets": [ARTIFACT_WORKLOADS],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "clear_existing",
                "long": "clear_existing",
                "help": "Remove existing generated workloads.",
                "default": True,
            },
        ],
    }


def task_behavior_execute_workloads():
    """
    Behavior modeling: execute workloads to generate training data.
    """
    execute_args = (
        f"--workloads={ARTIFACT_WORKLOADS} "
        f"--output-dir={ARTIFACT_DATA_RAW} "
        f"--pgdata={ARTIFACT_pgdata} "
        f"--benchbase={BENCHBASE_ARTIFACTS_PATH} "
        f"--pg_binaries={NOISEPAGE_ARTIFACTS_PATH} "
    )

    return {
        "actions": [
            f"mkdir -p {ARTIFACT_DATA_RAW}",
            f"behavior/datagen/run_workloads.sh {execute_args}",
        ],
        "file_dep": [
            dodos.benchbase.ARTIFACT_benchbase,
            dodos.noisepage.ARTIFACT_postgres,
            dodos.noisepage.ARTIFACT_pg_ctl,
        ],
        "targets": [ARTIFACT_DATA_RAW],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_behavior_extract_ous():
    """
    Behavior modeling: extract OUs from the query state store.
    """

    def extract_ous(glob_pattern, work_prefix, host, port, db_name, user, preserve):
        assert work_prefix is not None
        assert db_name is not None
        assert user is not None

        args = (
            f"--dir-data {ARTIFACT_DATA_RAW} "
            f"--dir-output {ARTIFACT_DATA_OUS} "
            f"--work-prefix {work_prefix} "
            f"--db-name {db_name} "
            f"--user {user} "
        )

        if host is not None:
            args = args + f"--host {host} "

        if port is not None:
            args = args + f"--port {port} "

        if glob_pattern is not None:
            args = args + f"--glob-pattern '{glob_pattern}' "

        if preserve is not None:
            args = args + f"--preserve "

        return f"python3 -m behavior extract_ous {args}"

    return {
        "actions": [
            f"mkdir -p {ARTIFACT_DATA_OUS}",
            CmdAction(extract_ous, buffering=1),
        ],
        "targets": [ARTIFACT_DATA_OUS],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "glob_pattern",
                "long": "glob_pattern",
                "help": "Glob pattern for selecting which experiments to extract query state features for.",
                "default": None,
            },
            {
                "name": "work_prefix",
                "long": "work_prefix",
                "help": "Prefix to use for creating and operating tables.",
                "default": None,
            },
            {
                "name": "host",
                "long": "host",
                "help": "Host of the database instance to use.",
                "default": "localhost",
            },
            {
                "name": "port",
                "long": "port",
                "help": "Port of the database instance to connect to.",
                "default": "5432",
            },
            {
                "name": "db_name",
                "long": "db_name",
                "help": "Name of the database to use.",
                "default": None,
            },
            {
                "name": "user",
                "long": "user",
                "help": "User to connect to the database with.",
                "default": None,
            },
            {
                "name": "preserve",
                "long": "preserve",
                "help": "Whether to preserve state of the database.",
                "default": None,
            },
        ],
    }


def task_behavior_train():
    """
    Behavior modeling: train OU models.
    """

    def train_cmd(config_file, train_data, prefix_allow_derived_features, robust, log_transform, separate_indkey_features, trace_inputs, force_nonincremental, output_subpath):
        train_args = (
            f"--config-file {config_file} "
            f"--dir-data {train_data} "
        )

        if output_subpath is None:
            train_args += f"--dir-output {ARTIFACT_MODELS} "
        else:
            train_args += f"--dir-output {ARTIFACT_MODELS}/{output_subpath} "

        if force_nonincremental is not None:
            train_args = train_args + "--force-nonincremental "

        if robust is not None:
            train_args = train_args + "--robust "

        if log_transform is not None:
            train_args = train_args + "--log-transform "

        if prefix_allow_derived_features != "":
            train_args = train_args + f"--prefix-allow-derived-features {prefix_allow_derived_features} "

        if separate_indkey_features:
            train_args = train_args + f"--separate-indkey-features "

        if trace_inputs is not None:
            train_args = train_args + "--trace-inputs "

        return f"python3 -m behavior train {train_args}"

    return {
        "actions": [f"mkdir -p {ARTIFACT_MODELS}", CmdAction(train_cmd, buffering=1)],
        "targets": [ARTIFACT_MODELS],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
        "params": [
            { "name": "config_file", "long": "config_file", "help": "Path to configuration file to use.", "default": MODELING_CONFIG_FILE, },
            { "name": "train_data", "long": "train_data", "help": "Root of a recursive glob to gather all feather files for training data.", "default": ARTIFACT_DATA_OUS, },
            { "name": "prefix_allow_derived_features", "long": "prefix_allow_derived_features", "help": "List of prefixes to use for selecting derived features for training the model.", "default": "", },
            { "name": "robust", "long": "robust", "help": "Whether to use the robust scaler to normalize to the data.", "default": None, },
            { "name": "log_transform", "long": "log_transform", "help": "Whether to use the log_transform to the data.", "default": None, },
            { "name": "separate_indkey_features", "long": "separate_indkey_features", "help": "Whether to separate each indkey feature.", "default": None, },
            { "name": "trace_inputs", "long": "trace_inputs", "help": "Whether to trace transformed inputs to the model", "default": None, },
            { "name": "force_nonincremental", "long": "force_nonincremental", "help": "Whether to force nonincremental training.", "default": None, },
            { "name": "output_subpath", "long": "output_subpath", "help": "Subpath in the output directory.", "default": None, },
        ],
    }


def task_behavior_eval_ou():
    """
    Behavior modeling: eval OU models.
    """
    def eval_cmd(eval_data, skip_generate_plots, models, methods, output_name):
        if models is None:
            # Find the latest experiment by last modified timestamp.
            experiment_list = sorted((exp_path for exp_path in ARTIFACT_MODELS.glob("*")), key=os.path.getmtime)
            assert len(experiment_list) > 0, "No experiments found."
            models = experiment_list[-1]
        else:
            assert os.path.isdir(models), f"Specified path {models} is not a valid directory."

        assert Path(eval_data).exists(), f"Specified OU {eval_data} does not exist."

        if output_name is None:
            eval_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_name = f"eval_{eval_timestamp}"

        eval_args = (
            f"--dir-data {eval_data} "
            f"--dir-models {models} "
            f"--dir-evals-output {ARTIFACT_EVALS_OU}/{output_name} "
        )

        if methods is not None:
            eval_args = eval_args + f"--methods {methods}"

        if not skip_generate_plots:
            eval_args = eval_args + "--generate-plots"

        return f"python3 -m behavior eval_ou {eval_args}"

    return {
        "actions": [f"mkdir -p {ARTIFACT_EVALS_OU}", CmdAction(eval_cmd, buffering=1)],
        "targets": [ARTIFACT_EVALS_OU],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
        "params": [
            {
                "name": "eval_data",
                "long": "eval_data",
                "help": "Path to root folder containing feathers for evaluation purposes. (structure: [experiment]/[benchmark]/*.feather)",
                "default": ARTIFACT_DATA_OUS,
            },
            {
                "name": "skip_generate_plots",
                "long": "skip_generate_plots",
                "help": "Flag of whether to skip generate plots or not.",
                "type": bool,
                "default": False
            },
            {
                "name": "models",
                "long": "models",
                "help": "Path to folder containing models that should be used. Defaults to last trained models.",
                "default": None,
            },
            {
                "name": "methods",
                "long": "methods",
                "help": "Comma separated methods that should be evaluated. Defaults to None (all).",
                "default": None,
            },
            {
                "name": "output_name",
                "long": "output_name",
                "help": "Name of the output directory.",
                "default": None,
            },
        ],
    }


def task_behavior_eval_query_workload():
    """
    Behavior modeling: perform query-level model analysis using a workload model.
    """
    def eval_cmd(num_cpus, num_threads, session_sql, input_dir, output_dir, ou_models,
                 query_feature_granularity_queries,
                 table_feature_model_cls, table_feature_model_path,
                 buffer_page_model_cls, buffer_page_model_path,
                 buffer_access_model_cls, buffer_access_model_path,
                 concurrency_model_cls, concurrency_model_path,
                 concurrency_mpi, workload_analysis_conn,
                 target_db_conn, workload_analysis_prefix, histogram_width, forward_state,
                 use_plan_estimates):
        eval_args = (
            f"--num-cpus {num_cpus} "
            f"--num-threads {num_threads} "
            f"--session-sql {session_sql} "
            f"--input-dir {input_dir} "
            f"--output-dir {output_dir} "
            f"--ou-models {ou_models} "
            f"--query-feature-granularity-queries {query_feature_granularity_queries} "
            f"--table-feature-model-cls {table_feature_model_cls} "
            f"--table-feature-model-path {table_feature_model_path} "
            f"--buffer-page-model-cls {buffer_page_model_cls} "
            f"--buffer-page-model-path {buffer_page_model_path} "
            f"--buffer-access-model-cls {buffer_access_model_cls} "
            f"--buffer-access-model-path {buffer_access_model_path} "
            f"--concurrency-model-cls {concurrency_model_cls} "
            f"--concurrency-model-path {concurrency_model_path} "
            f"--concurrency-mpi {concurrency_mpi} "
            f"--workload-analysis-conn \"{workload_analysis_conn}\" "
            f"--target-db-conn \"{target_db_conn}\" "
            f"--workload-analysis-prefix {workload_analysis_prefix} "
            f"--histogram-width {histogram_width} "
        )

        if forward_state is not None:
            eval_args += f"--forward-state "

        if use_plan_estimates is not None:
            eval_args += f"--use-plan-estimates "

        return f"python3 -m behavior eval_query_workload {eval_args}"

    return {
        "actions": [f"mkdir -p {ARTIFACT_EVALS_QUERY_WORKLOAD}", CmdAction(eval_cmd, buffering=1)],
        "targets": [ARTIFACT_EVALS_QUERY_WORKLOAD],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
        "params": [
            { "name": "num_cpus", "long": "num_cpus", "help": "Number of CPUs to use for parallel OUs.", "default": 1, },
            { "name": "num_threads", "long": "num_threads", "help": "Number of threads/CPU to use for parallel OUs.", "default": 1, },
            { "name": "session_sql", "long": "session_sql", "help": "Path to a list of SQL statements that should be executed in the session prior to EXPLAIN.", "default": None, },
            { "name": "input_dir", "long": "input_dir", "help": "Input directory for data.", "default": None, },
            { "name": "output_dir", "long": "output_dir", "help": "Output directory for evals.", "default": None, },
            { "name": "ou_models", "long": "ou_models", "help": "Path to OU models.", "default": None, },
            { "name": "query_feature_granularity_queries", "help": "Granularity of window slices in queries.", "default": 1000, },
            { "name": "table_feature_model_cls", "long": "table_feature_model_cls", "help": "Table Feature model to instantiate", "default": "TableFeatureModel", },
            { "name": "table_feature_model_path", "long": "table_feature_model_path", "help": "Path to Table Feature models.", "default": None, },
            { "name": "buffer_page_model_cls", "long": "buffer_page_model_cls", "help": "Buffer Page model to instantiate.", "default": "BufferPageModel", },
            { "name": "buffer_page_model_path", "long": "buffer_page_model_path", "help": "Path to Buffer Page models.", "default": None, },
            { "name": "buffer_access_model_cls", "long": "buffer_acccess_model_cls", "help": "Buffer Access model to instantiate.", "default": "BufferAccessModel", },
            { "name": "buffer_access_model_path", "long": "buffer_acccess_model_path", "help": "Path to Buffer Access models.", "default": None, },
            { "name": "concurrency_model_cls", "long": "concurrency_model_cls", "help": "Concurrency Model to instantiate.", "default": "ConcurrencyModel", },
            { "name": "concurrency_model_path", "long": "concurrency_model_path", "help": "Path to Concurrency models.", "default": None, },
            { "name": "concurrency_mpi", "long": "concurrency_mpi", "help": "Concurrency MPI.", "default": None, },
            { "name": "workload_analysis_conn", "long": "workload_analysis_conn", "help": "psycopg2 connection string to connect to the workload database instance.", "default": None, },
            { "name": "target_db_conn", "long": "target_db_conn", "help": "psycopg2 connection string to connect to the target database instance.", "default": None, },
            { "name": "workload_analysis_prefix", "long": "workload_analysis_prefix", "help": "Workload analysis prefix to use.", "default": None, },
            { "name": "histogram_width", "long": "histogram_width", "help": "Number of buckets (histogram width).", "default": None, },
            { "name": "forward_state", "long": "forward_state", "help": "Whether to forward initial state or not.", "default": False, },
            { "name": "use_plan_estimates", "long": "use_plan_estimates", "help": "Whether to use postgres plan estimates as opposed to row execution features.", "default": False, },
        ],
    }


def task_behavior_eval_query_plots():
    """
    Behavior modeling: generate plots from eval_query[_workload] analysis.
    """
    def eval_cmd(input_dir, txn_analysis_file, generate_summary, generate_per_query, deviation_around_mean, generate_predict_abs_errors):
        eval_args = (
            f"--dir-input {input_dir} "
            f"--deviation-around-mean {deviation_around_mean} "
        )

        vals = [
            ("--generate-summary", generate_summary),
            ("--generate-per-query", generate_per_query),
            ("--generate-predict-abs-errors", generate_predict_abs_errors),
        ]

        for (k, v) in vals:
            if v is not None and v != "False":
                eval_args += f"{k} "

        if txn_analysis_file is not None:
            assert Path(txn_analysis_file).exists()
            eval_args += f"--txn-analysis-file {txn_analysis_file} "

        return f"python3 -m behavior eval_query_plots {eval_args}"

    return {
        "actions": [CmdAction(eval_cmd, buffering=1)],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
        "params": [
            { "name": "input_dir", "long": "input_dir", "help": "Path to root folder containing the input data to plot.", "default": None, },
            { "name": "txn_analysis_file", "long": "txn_analysis_file", "help": "Path to transaction analysis file.", "default": None, },
            { "name": "generate_summary", "long": "generate_summary", "help": "Whether to generate summary error information.", "default": None, },
            { "name": "generate_per_query", "long": "generate_per_query", "help": "Whether to generate per-query plots of the errors.", "default": None, },
            { "name": "deviation_around_mean", "long": "deviation_around_mean", "help": "Deviation to plot around mean for per-query plots.", "default": 0, },
            { "name": "generate_predict_abs_errors", "long": "generate_predict_abs_errors", "help": "Whether to generate abs errors against each table.", "default": None, },
        ],
    }


def task_behavior_eval_query_compare_plots():
    """
    Behavior modeling: generate plots from comparative workload analysis.
    """
    def eval_cmd(input_dir, input_names, dir_output, txn_analysis_file, generate_per_query, generate_predict_abs_errors):
        eval_args = (
            f"--dir-input {input_dir} "
            f"--input-names {input_names} "
            f"--dir-output {dir_output} "
        )

        vals = [
            ("--generate-per-query", generate_per_query),
            ("--generate-predict-abs-errors", generate_predict_abs_errors),
        ]

        for (k, v) in vals:
            if v is not None and v != "False":
                eval_args += f"{k} "

        if txn_analysis_file is not None:
            assert Path(txn_analysis_file).exists()
            eval_args += f"--txn-analysis-file {txn_analysis_file} "

        return f"python3 -m behavior eval_query_compare_plots {eval_args}"

    return {
        "actions": [CmdAction(eval_cmd, buffering=1)],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
        "params": [
            {
                "name": "input_dir",
                "long": "input_dir",
                "help": "Path to root folder containing the input data to plot.",
                "default": None,
            },
            {
                "name": "input_names",
                "long": "input_names",
                "help": "Names of the input models.",
                "default": None,
            },
            {
                "name": "dir_output",
                "long": "dir_output",
                "help": "Path to folder for output.",
                "default": None,
            },
            {
                "name": "txn_analysis_file",
                "long": "txn_analysis_file",
                "help": "Path to transaction analysis file.",
                "default": None,
            },
            {
                "name": "generate_per_query",
                "long": "generate_per_query",
                "help": "Whether to generate per-query plots of the errors.",
                "default": None,
            },
            {
                "name": "generate_predict_abs_errors",
                "long": "generate_predict_abs_errors",
                "help": "Whether to generate abs errors against each table.",
                "default": None,
            },
        ],
    }
