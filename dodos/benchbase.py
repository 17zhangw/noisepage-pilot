import os
from pathlib import Path
from plumbum import local

import doit
from doit.action import CmdAction
from plumbum import local

import dodos.noisepage
from dodos import VERBOSITY_DEFAULT, default_artifacts_path, default_build_path
from behavior import BENCHDB_TO_TABLES
from dodos.noisepage import ARTIFACT_psql

ARTIFACTS_PATH = default_artifacts_path()
BUILD_PATH = default_build_path()
CONFIG_FILES = Path("config/behavior/benchbase").absolute()

DEFAULT_DB = "benchbase"
DEFAULT_USER = "admin"
DEFAULT_PASS = "password"


# Output: BenchBase jar file.
ARTIFACT_benchbase = ARTIFACTS_PATH / "benchbase.jar"


def task_benchbase_clone():
    """
    BenchBase: clone.
    """

    def repo_clone(repo_url, branch_name):
        cmd = f"git clone {repo_url} --branch {branch_name} --single-branch --depth 1 {BUILD_PATH}"
        return cmd

    return {
        "actions": [
            # Set up directories.
            f"mkdir -p {BUILD_PATH}",
            # Clone BenchBase.
            CmdAction(repo_clone),
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "targets": [BUILD_PATH, BUILD_PATH / "mvnw"],
        "uptodate": [True],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "repo_url",
                "long": "repo_url",
                "help": "The repository to clone from.",
                "default": "https://github.com/cmu-db/benchbase.git",
            },
            {
                "name": "branch_name",
                "long": "branch_name",
                "help": "The name of the branch to checkout.",
                "default": "main",
            },
        ],
    }


def task_benchbase_build():
    """
    BenchBase: build.
    """

    return {
        "actions": [
            f"mkdir -p {ARTIFACTS_PATH}",
            lambda: os.chdir(BUILD_PATH),
            # Compile BenchBase.
            "./mvnw clean package -Dmaven.test.skip=true -P postgres",
            lambda: os.chdir("target"),
            "tar xvzf benchbase-postgres.tgz",
            # Move artifacts out.
            lambda: os.chdir(doit.get_initial_workdir()),
            f"mv {BUILD_PATH / f'target/benchbase-postgres/*'} {ARTIFACTS_PATH}",
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "file_dep": [BUILD_PATH / "mvnw"],
        "targets": [ARTIFACTS_PATH, ARTIFACT_benchbase],
        "uptodate": [True],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_benchbase_overwrite_config():
    """
    BenchBase: overwrite artifact config files with the ones in this repo.
    """

    return {
        "actions": [
            # Copy config files.
            f"cp {CONFIG_FILES / '*'} {ARTIFACTS_PATH / 'config/postgres/'}",
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "file_dep": [ARTIFACT_benchbase, *CONFIG_FILES.glob("*")],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_benchbase_bootstrap_dbms():
    """
    BenchBase: bootstrap the DBMS with any necessary databases and tables.
    """
    sql_list = [
        f"CREATE ROLE {DEFAULT_USER} WITH LOGIN SUPERUSER ENCRYPTED PASSWORD '{DEFAULT_PASS}'",
        f"ALTER DATABASE {DEFAULT_DB} SET compute_query_id = 'ON';",
    ]

    return {
        "actions": [
            f"{dodos.noisepage.ARTIFACT_dropdb} --if-exists {DEFAULT_DB}",
            f"{dodos.noisepage.ARTIFACT_dropuser} --if-exists {DEFAULT_USER}",
            f"{dodos.noisepage.ARTIFACT_createdb} {DEFAULT_DB}",
            *[f'{dodos.noisepage.ARTIFACT_psql} --dbname={DEFAULT_DB} --command="{sql}"' for sql in sql_list],
        ],
        "file_dep": [
            dodos.noisepage.ARTIFACT_dropdb,
            dodos.noisepage.ARTIFACT_dropuser,
            dodos.noisepage.ARTIFACT_createdb,
            dodos.noisepage.ARTIFACT_psql,
        ],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
    }


def task_benchbase_run():
    """
    BenchBase: run a specific benchmark.
    """

    def invoke_benchbase(benchmark, config, args, taskset_benchbase):
        if config is None:
            config = ARTIFACTS_PATH / f"config/postgres/{benchmark}_config.xml"
        elif not config.startswith("/"):
            # If config is not an absolute path,
            # because we must be in the BenchBase folder for the java invocation to work out,
            # we need to get the original relative path that the caller intended.
            config = (Path(doit.get_initial_workdir()) / config).absolute()

        cmd = f"java -jar benchbase.jar -b {benchmark} -c {config} {args}"
        if taskset_benchbase is not None and taskset_benchbase != 'None':
            cmd = f"taskset -c {taskset_benchbase} {cmd}"
        return cmd

    return {
        "actions": [
            lambda: os.chdir(ARTIFACTS_PATH),
            # Invoke BenchBase.
            CmdAction(invoke_benchbase),
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "file_dep": [ARTIFACT_benchbase],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "benchmark",
                "long": "benchmark",
                "help": "The benchmark to run.",
                "default": "tpcc",
            },
            {
                "name": "config",
                "long": "config",
                "help": (
                    "The config file to use for BenchBase."
                    "Defaults to the config in the artifacts folder for the selected benchmark."
                ),
                "default": None,
            },
            {
                "name": "args",
                "long": "args",
                "help": "Arguments to pass to BenchBase invocation.",
                "default": "--create=false --load=false --execute=false",
            },
            {
                "name": "taskset_benchbase",
                "long": "taskset_benchbase",
                "help": "CPU List to taskset benchbase to.",
                "default": None,
            },
        ],
    }


def task_benchbase_pg_analyze_benchmark():
    """
    Behavior modeling: Run ANALYZE on all the tables in the given benchmark.
    This updates internal statistics for estimating cardinalities and costs.

    Parameters
    ----------
    benchmark : str
        The benchmark whose tables should be analyzed.
    """

    def pg_analyze(benchmark):
        if benchmark is None or benchmark not in BENCHDB_TO_TABLES:
            print(f"Benchmark {benchmark} is not specified or does not exist.")
            return False

        for table in BENCHDB_TO_TABLES[benchmark]:
            query = f"ANALYZE VERBOSE {table};"
            local[str(ARTIFACT_psql)]["--dbname=benchbase"]["--command"][query]()

    return {
        "actions": [pg_analyze],
        "file_dep": [ARTIFACT_psql],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "benchmark",
                "long": "benchmark",
                "help": "Benchmark whose tables should be analyzed.",
                "default": None,
            },
        ],
    }


def task_benchbase_snapshot_benchmark():
    """
    BenchBase: snapshot all tables of a particular benchmark.
    """

    def snapshot(benchmark, output_dir):
        if benchmark is None or benchmark not in BENCHDB_TO_TABLES:
            print(f"Benchmark {benchmark} is not specified or does not exist.")
            return False

        for table in BENCHDB_TO_TABLES[benchmark]:
            query = f"\COPY (SELECT * FROM {table}) TO '{output_dir}/{table}_snapshot.csv' CSV HEADER;"
            local[str(ARTIFACT_psql)]["--dbname=benchbase"]["--command"][query]()

    return {
        "actions": [snapshot],
        "file_dep": [ARTIFACT_psql],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "benchmark",
                "long": "benchmark",
                "help": "Benchmark whose tables should be snapshot.",
                "default": None,
            },
            {
                "name": "output_dir",
                "long": "output_dir",
                "help": "Output directory that snapshots should be written to.",
                "default": None,
            },
        ],
    }


def task_benchbase_warm_benchmark():
    """
    Behavior modeling: Run warmer_prewarm() on all the tables in the given benchmark.
    This warms the buffer pool and OS page cache.

    Parameters
    ----------
    benchmark : str
        The benchmark whose tables should be prewarmed.
    """

    def pg_prewarm(benchmark):
        if benchmark is None or benchmark not in BENCHDB_TO_TABLES:
            print(f"Benchmark {benchmark} is not specified or does not exist.")
            return False

        for table in BENCHDB_TO_TABLES[benchmark]:
            query = f"SELECT * FROM warmer_prewarm('{table}');"
            local[str(ARTIFACT_psql)]["--dbname=benchbase"]["--command"][query]()

    return {
        "actions": [pg_prewarm],
        "file_dep": [ARTIFACT_psql],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "benchmark",
                "long": "benchmark",
                "help": "Benchmark whose tables should be analyzed.",
                "default": None,
            },
        ],
    }


def task_benchbase_prewarm_install():
    """
    BenchBase: install pg_prewarm for BenchBase benchmarks.
    """
    sql_list = ["CREATE EXTENSION IF NOT EXISTS pg_prewarm"]

    return {
        "actions": [
            *[f'{dodos.noisepage.ARTIFACT_psql} --dbname={DEFAULT_DB} --command="{sql}"' for sql in sql_list],
        ],
        "file_dep": [
            dodos.noisepage.ARTIFACT_psql,
        ],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
    }


def task_benchbase_pg_prewarm_benchmark():
    """
    Behavior modeling: Run warmer_prewarm() on all the tables in the given benchmark.
    This warms the buffer pool and OS page cache.

    Parameters
    ----------
    benchmark : str
        The benchmark whose tables should be prewarmed.
    """

    def pg_prewarm(benchmark):
        if benchmark is None or benchmark not in BENCHDB_TO_TABLES:
            print(f"Benchmark {benchmark} is not specified or does not exist.")
            return False

        for table in BENCHDB_TO_TABLES[benchmark]:
            query = f"SELECT * FROM pg_prewarm('{table}');"
            local[str(ARTIFACT_psql)]["--dbname=benchbase"]["--command"][query]()

    return {
        "actions": [pg_prewarm],
        "file_dep": [ARTIFACT_psql],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "benchmark",
                "long": "benchmark",
                "help": "Benchmark whose tables should be analyzed.",
                "default": None,
            },
        ],
    }
