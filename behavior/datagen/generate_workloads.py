from __future__ import annotations

import xml.etree.ElementTree as ET
import logging
import shutil
from pathlib import Path

import yaml
from plumbum import cli

from behavior import BENCHDB_TO_TABLES

logger = logging.getLogger(__name__)


def inject_param_xml(file_path, parameters):
    '''Inject and re-write XML file with given parameters.

    Parameters:
    -----------
    file_path : str
        XML file path to inject.
    parameters : List[Tuple(List[str], Any)]
        The list of parameter names and values to inject.
    '''
    conf_etree = ET.parse(file_path)
    root = conf_etree.getroot()
    parameters = [(key.split("."), parameters[key]) for key in parameters]

    for name_level, val in parameters:
        cursor = root
        # Traverse XML name levels.
        for key in name_level:
            cursor = cursor.find(key)
            if cursor is None:
                break

        if cursor is not None:
            cursor.text = str(val)

    conf_etree.write(file_path)


def generate_workload(config, mode_dir, benchbase_path, postgresql_config_file):
    output_name = config["output_name"]
    assert output_name is not None
    benchmark_dir = Path(mode_dir / output_name)
    benchmark_dir.mkdir(exist_ok=True)
    print(f"Creating workload configuration: {benchmark_dir}")

    benchbase_configs = []
    pg_configs = []
    post_execute = []
    pre_execute = []
    for idx, run in enumerate(config["runs"]):
        # Copy and inject the XML file of BenchBase.
        benchbase_config_file = Path(benchmark_dir / f"benchbase_config_{idx}.xml")
        shutil.copy(benchbase_path, benchbase_config_file)
        run["scalefactor"] = config["scalefactor"]
        inject_param_xml(benchbase_config_file.as_posix(), run)
        benchbase_configs.append(str(benchbase_config_file.resolve()))

        # Copy the default postgresql.conf file.
        benchbase_postgresql_config_file = Path(benchmark_dir / f"postgresql_{idx}.conf")
        shutil.copy(postgresql_config_file, benchbase_postgresql_config_file)
        if "options" in run and run["options"] is not None and len(run["options"]) > 0:
            with open(benchbase_postgresql_config_file, "a") as f:
                valid_option = []
                for option in run["options"]:
                    valid_option.append(option.split("=")[0])
                    f.write(f"{option}\n")

                if "default_options" in config and config["default_options"] is not None:
                    for option in config["default_options"]:
                        option_key = option.split("=")[0]
                        if option_key not in valid_option:
                            f.write(f"{option}\n")
        pg_configs.append(str(benchbase_postgresql_config_file.resolve()))
        if "post_execute_sql" in run and run["post_execute_sql"] is not None:
            post_execute.append(run["post_execute_sql"])
        else:
            post_execute.append(None)

        if "pre_execute_sql" in run and run["pre_execute_sql"] is not None:
            pre_execute.append(run["pre_execute_sql"])
        else:
            pre_execute.append(None)

    # Create the config.yaml file
    output = {
        "benchmark": config["benchmark"],
        "pg_analyze": config["pg_analyze"],
        "pg_prewarm": config["pg_prewarm"],
        "continuous": config["continuous"],
        "snapshot_data": config["snapshot_data"],
        "snapshot_metadata": config["snapshot_metadata"],
        "pg_configs": pg_configs,
        "benchbase_configs": benchbase_configs,
        "dump_db": config["dump_db"],
        "enable_collector": config["enable_collector"],
        "taskset_postgres": config["taskset_postgres"],
        "taskset_benchbase": config["taskset_benchbase"],
        "collector_interval": config["collector_interval"],
    }

    if "dump_db_path" in config and config["dump_db_path"] is not None:
        output["dump_db_path"] = config["dump_db_path"]

    output_post_execute = False
    for val in post_execute:
        output_post_execute |= val is not None
    if output_post_execute:
        output["post_execute"] = post_execute

    output_pre_execute = False
    for val in pre_execute:
        output_pre_execute |= val is not None
    if output_pre_execute:
        output["pre_execute"] = pre_execute

    if config["restore_db_path"] is not None:
        output["restore_db"] = True
        output["restore_db_path"] = config["restore_db_path"]

    if config["post_create_sql"] is not None:
        output["post_create"] = config["post_create_sql"]

    with (benchmark_dir / "config.yaml").open("w") as f:
        yaml.dump(output, f)


class GenerateWorkloadsCLI(cli.Application):
    config_file = cli.SwitchAttr(
        "--config-file",
        Path,
        mandatory=True,
        help="Path to configuration YAML containing datagen parameters.",
    )
    postgresql_config_file = cli.SwitchAttr(
        "--postgresql-config-file",
        Path,
        mandatory=True,
        help="Path to standard postgresql.conf that the workloads should execute with.",
    )
    dir_benchbase_config = cli.SwitchAttr(
        "--dir-benchbase-config",
        Path,
        mandatory=True,
        help="Path to BenchBase config files.",
    )
    dir_output = cli.SwitchAttr(
        "--dir-output",
        Path,
        mandatory=True,
        help="Directory to write generated data to.",
    )

    def main(self):
        config_path = Path(self.config_file)
        with config_path.open("r", encoding="utf-8") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)["datagen"]
        logger.setLevel(self.config["log_level"])

        self.dir_output.mkdir(parents=True, exist_ok=True)

        default_config = self.config["default_config"]
        configs = self.config[f"configs"]
        for augment in configs:
            config = default_config.copy()
            config.update(augment)

            benchmark = config["benchmark"]
            if benchmark not in BENCHDB_TO_TABLES:
                raise ValueError(f"Invalid benchmark: {benchmark}")
            benchbase_path = self.dir_benchbase_config / f"{benchmark}_config.xml"
            generate_workload(config, self.dir_output, benchbase_path, self.postgresql_config_file)


if __name__ == "__main__":
    GenerateWorkloadsCLI.run()
