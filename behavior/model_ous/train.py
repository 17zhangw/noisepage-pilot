from __future__ import annotations

import itertools
import logging
import os
import gc
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from pandas import DataFrame
from plumbum import cli
from sklearn import tree

from behavior import TARGET_COLUMNS, OperatingUnit, Targets, DERIVED_FEATURES_MAP
from behavior.utils.prepare_ou_data import OUDataLoader
from behavior.model_ous.model import BehaviorModel

logger = logging.getLogger(__name__)


def contains_data(train_files, ou):
    ou_results = [fp for fp in train_files if fp.name.startswith(ou.name)]
    for ou_result in ou_results:
        df = load_input_data(logger, ou_result, True)
        if df.shape[0] > 0:
            return df

    return None


def load_data(train_files, ou):
    """Load the training data.

    Parameters
    ----------
    train_files: List[Path]
        List of files to consider for training data.

    Returns
    -------
    dict[str, DataFrame]
        A map from operating unit names to their training data.

    Raises
    ------
    Exception
        If there is no valid training data.
    """
    # Load all the OU data from disk given the data directory.
    # We filter all files with zero results because it is common to only have data for
    # a few operating units.
    result_paths = [fp for fp in train_files if os.stat(fp).st_size > 0]
    ou_name_to_df: dict[str, DataFrame] = {}

    ou_results = [fp for fp in result_paths if fp.name.startswith(ou.name)]
    if len(ou_results) > 0:
        logger.debug("Found %s run(s) for %s", len(ou_results), ou.name)
        def invoke(path):
            # We are loading data for training purposes.
            return load_input_data(logger, path, True)

        return pd.concat(map(invoke, ou_results))

    return None


def main(
    config_file,
    dir_data,
    dir_output,
    prefix_allow_derived_features,
    separate_indkey_features,
    robust,
    log_transform,
    trace_inputs,
):
    # Load modeling configuration.
    if not config_file.exists():
        raise ValueError(f"Config file: {config_file} does not exist")

    logger.info("Loading config file: %s", config_file)
    with config_file.open("r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["modeling"]

    if robust:
        config["robust"] = True

    if log_transform:
        config["log_transform"] = True

    prefix_allows = prefix_allow_derived_features.split(",")

    # Mark this training-evaluation run with a timestamp for identification.
    training_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_files = list(dir_data.rglob("*.csv"))
    train_files = [fp for fp in train_files if os.stat(fp).st_size > 0 and pd.read_csv(fp, chunksize=4).get_chunk().shape[0] > 0]
    assert len(train_files) > 0, "No matching data files for training could be found."

    # Load the data and name the model.
    output_dir = dir_output / f"{config_file.stem}_{training_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "source.txt").open("w+") as f:
        f.write(f"Train: {train_files}\n")

    for ou in OperatingUnit:
        ou_name = ou.name
        ou_results = [fp for fp in train_files if fp.name == f"{ou_name}.csv"]

        # Arbitrarily set the chunk size to be 131072
        loader = OUDataLoader(logger, separate_indkey_features, ou_results, 131072, True)
        df_train = loader.get_next_data()

        # We have no data.
        if df_train is None or df_train.shape[0] == 0:
            continue

        # By default start with all non-plan features disabled.
        ignore_cols = [ "Node Type", "incomplete", "data_identifier", "identifier" ]
        ignore_cols.extend([f for f in df_train.columns if "indexid" in f])
        #ignore_cols.extend([f for f in df_train.columns if not f.startswith(ou_name + "_")])

        # Incrementally re-enable them.
        relevant_features = {k: v for k, v in DERIVED_FEATURES_MAP.items() if k.startswith(ou_name + "_") or k.endswith("_" + ou_name)}
        for n, v in relevant_features.items():
            allow = False
            for prefix_allow in prefix_allows:
                if n.startswith(prefix_allow):
                    allow = True

            if not allow:
                ignore_cols.append(v)
            elif v in ignore_cols:
                ignore_cols.remove(v)

        # Get a metadata representation for extracting all feature columns.
        features = sorted(list(set(df_train.columns) - set(TARGET_COLUMNS) - set(ignore_cols)))
        targets = [Targets.ELAPSED_US.value]
        logger.info("Begin Training OU: %s", ou_name)
        logger.info("Derived input features for OU: %s (%s)", ou_name, features)

        loader = OUDataLoader(logger, separate_indkey_features, ou_results, None, True)
        data = loader.get_next_data()
        for method in config["methods"]:
            model = BehaviorModel(
                method,
                ou_name,
                config,
                features,
                targets=targets,
            )

            logger.info("Fully training (%s, %s)", ou_name, model.method)
            df_train = data.copy()
            x_train = df_train[features]
            y_train = df_train[targets].values
            del df_train

            model.train(x_train, y_train)

            del x_train
            del y_train
            gc.collect()

            output = output_dir / model.method
            output.mkdir(parents=True, exist_ok=True)
            model.save(output)
            del model
        gc.collect()


class TrainCLI(cli.Application):
    config_file = cli.SwitchAttr(
        "--config-file",
        Path,
        mandatory=True,
        help="Path to configuration YAML containing modeling parameters.",
    )
    dir_data = cli.SwitchAttr(
        "--dir-data",
        Path,
        mandatory=True,
        help="Root of a recursive glob to gather all feather files for training data.",
    )
    dir_output = cli.SwitchAttr(
        "--dir-output",
        Path,
        mandatory=True,
        help="Folder to output models to.",
    )
    prefix_allow_derived_features = cli.SwitchAttr(
        "--prefix-allow-derived-features",
        help="List of prefixes to use for selecting derived features for training the model.",
        default=""
    )
    robust = cli.Flag(
        "--robust",
        default=False,
        help="Whether to force the robust scalar to the model.",
    )
    log_transform = cli.Flag(
        "--log-transform",
        default=False,
        help="Whether to force the log transform to the input data.",
    )
    separate_indkey_features = cli.Flag(
        "--separate-indkey-features",
        default=False,
        help="Whether to use separate indkey features.",
    )
    trace_inputs = cli.Flag(
        "--trace-inputs",
        default=False,
        help="Whether to trace the inputs to the models.",
    )

    def main(self):
        main(
            self.config_file,
            self.dir_data,
            self.dir_output,
            self.prefix_allow_derived_features,
            self.separate_indkey_features,
            self.robust,
            self.log_transform,
            self.trace_inputs,
        )


if __name__ == "__main__":
    TrainCLI.run()
