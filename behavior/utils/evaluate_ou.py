import re
import psycopg
from psycopg.rows import dict_row
from distutils import util
from pathlib import Path
from enum import Enum, auto, unique
import pandas as pd
import numpy as np

from behavior.utils.prepare_ou_data import OUDataLoader
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def compute_useful_measures(target_pred, target_true):
    true_mean = target_true.mean()
    pred_mean = target_pred.mean()
    mape = mean_absolute_percentage_error(target_true, target_pred)
    mse = mean_squared_error(target_true, target_pred)
    mae = mean_absolute_error(target_true, target_pred)
    rsquared = r2_score(target_true, target_pred)

    stats = {
        "Target Mean": round(true_mean, 2),
        "Predicted Mean": round(pred_mean, 2),
        "Mean Absolute Percentage Error": round(mape, 2),
        "Mean Squared Error": round(mse, 2),
        "Mean Absolute Error": round(mae, 2),
        "R-squared": round(rsquared, 2)
    }

    bins = [1, 5, 10, 30, 50, 100]
    for b in bins:
        stat = np.sum(np.abs(target_pred - target_true) <= b) / target_pred.shape[0]
        stats[f"Percentage within {b}"] = round(stat, 2)

    return stats


def evaluate_ou_model(model, output_dir, benchmark, eval_file=None, eval_df=None, return_df=False, output=True):
    assert eval_df is not None or eval_file is not None
    targets = model.targets
    pred_targets = [f"pred_{target}" for target in targets]

    if eval_file is not None:
        # Load the input OU file that we want to evaluate.
        # Load the entire input in and do evaluation.
        input_data = OUDataLoader(logger=None, separate_indkey_features=False, ou_files=[eval_file], chunksize=None, train=False).get_next_data()
    else:
        input_data = eval_df
        eval_file = Path(model.ou_name)

    # Run inference.
    y_pred = model.predict(input_data)

    # Write out the predictions.
    if output:
        y = input_data[model.targets].values

        # Add the source_file here for evaluation purposes.
        input_data["source_file"] = eval_file

        with open(output_dir / f"{model.method}_{benchmark}_{eval_file.stem}_preds.feather", "wb") as preds_file:
            columns = ["data_identifier", "source_file"]
            X = input_data[columns]

            temp: NDArray[Any] = np.concatenate((X, y, y_pred), axis=1)
            test_result_df = pd.DataFrame(
                temp,
                columns=columns + targets + pred_targets,
            )
            test_result_df["source_file"] = test_result_df.source_file.astype(str)
            test_result_df.to_feather(preds_file)
            del temp
            del test_result_df

        with open(output_dir / f"{model.method}_{benchmark}_{eval_file.stem}_stats.txt", "w+") as ou_eval_file:
            ou_eval_file.write(
                f"\n============= Evaluation {benchmark} {eval_file.stem}: Model Summary for {model.method} =============\n"
            )
            ou_eval_file.write(f"Features used: {model.features}\n")
            ou_eval_file.write(f"Num Features used: {len(model.features)}\n")

            # Evaluate performance for every resource consumption metric.
            for target_idx, target in enumerate(targets):
                ou_eval_file.write(f"===== Target: {target} =====\n")
                target_pred = y_pred[:, target_idx]
                target_true = y[:, target_idx]
                assert target_pred.shape[0] == target_true.shape[0]
                stats = compute_useful_measures(target_pred, target_true)
                for stat in stats:
                    ou_eval_file.write(f"{stat}: {stats[stat]}\n")
            ou_eval_file.write("======================== END SUMMARY ========================\n")

    if return_df:
        # Return the input data frame along with predicted targets.
        input_data[pred_targets] = y_pred
        return input_data

    return None
