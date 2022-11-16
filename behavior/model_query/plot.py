import gc
import glob
import yaml
import logging
import shutil
import copy
from tqdm import tqdm
from datetime import datetime
import numpy as np
import itertools
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from plumbum import cli

logger = logging.getLogger(__name__)


def generate_rolling(query_stream, dir_output):
    params = {
        'legend.fontsize': 'x-large',
        'axes.labelsize': 'x-large',
        'axes.titlesize':'x-large',
        'xtick.labelsize':'x-large',
        'ytick.labelsize':'x-large'
    }
    plt.rcParams.update(params)

    fig, ax = plt.subplots(1, 1, figsize=(25.6, 14.4))
    fig.suptitle(f"Rolling Workload Cost (sum, window=1000)", wrap=True)

    elapsed_us_windows = query_stream.elapsed_us.rolling(10000).sum()
    pred_elapsed_us_windows = query_stream.pred_elapsed_us.rolling(10000).sum()

    elapsed_us_windows.plot(ax=ax, label="True Workload Cost")
    pred_elapsed_us_windows.plot(ax=ax, label="Predicted Workload Cost")

    ax.set_ylabel("Total Microseconds")
    ax.set_xlabel("Window")
    ax.legend()

    plt.savefig(dir_output / f"rolling_sum.png")
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)


def generate_per_query_plots(query_stream, deviation_around_mean, dir_output):
    params = {
        'legend.fontsize': 'x-large',
        'axes.labelsize': 'x-large',
        'axes.titlesize':'x-large',
        'xtick.labelsize':'x-large',
        'ytick.labelsize':'x-large'
    }
    plt.rcParams.update(params)

    # Output error distribution plots based on query_id.
    qid_groups = query_stream.groupby(by=["query_id"])
    for group in qid_groups:
        logger.info("Processing query %s (%s)", group[0], group[1].iloc[0].query_text)
        fig, axes = plt.subplots(2, 1, figsize=(25.6, 14.4))
        fig.suptitle(f"{group[0]}")
        ax = axes[0]

        # Plot elapsed and predicted elapsed time on the same graph as a scatter.
        x_title = "order" if "order" in query_stream else "query_order"
        group[1].plot.scatter(x=x_title, y="elapsed_us", color='r', ax=ax, label="True Runtime")
        group[1].plot.scatter(x=x_title, y="pred_elapsed_us", color='b', ax=ax, label="Predicted Runtime")
        ax.set_xticks([])

        if deviation_around_mean > 0:
            avg = group[1].elapsed_us.mean()
            yrange = (max(0, avg - deviation_around_mean), min(group[1].elapsed_us.max() + 1, avg + deviation_around_mean))
            ax.set_ylim(yrange)

        if len(group[1]) > 1 and len(group[1].predicted_minus_elapsed.value_counts()) > 1:
            # Density of true and predicted.
            ax = axes[1]
            group[1].elapsed_us.plot.kde(color='r', ax=ax, label="True Runtime")
            if len(group[1].pred_elapsed_us.value_counts()) > 1:
                group[1].pred_elapsed_us.plot.kde(color='b', ax=ax, label="Predicted Runtime")
            else:
                ax.axvline(x=group[1].pred_elapsed_us.iloc[0], label="Predicted Runtime")
            ax.legend()

            # Only plot the second graph if there is more than 1 distinct value.
            #ax = axes[2]
            #group[1].predicted_minus_elapsed.plot.kde(color='r', ax=ax)

            #percentiles = [2.5, 5, 25, 50, 75, 95, 97.5]
            #percents = np.percentile(group[1].predicted_minus_elapsed, percentiles)
            #bounds = [group[1].predicted_minus_elapsed.min(), group[1].predicted_minus_elapsed.max()]
            #ax.scatter(bounds, [0, 0], color='b')
            #ax.scatter(percents, np.zeros(len(percentiles)), color='g')
            #ax.set_xlabel("predicted - elapsed")

        plt.savefig(dir_output / f"{group[0]}.png")
        plt.close()
    plt.rcParams.update(plt.rcParamsDefault)


def generate_predicted_query_error(query_stream, dir_output, min_error_threshold, max_error_threshold):
    x = 0
    width = 0.8
    ticks = []
    labels = []
    fig, axes = plt.subplots(3, 1, figsize=(25.6, 14.4))

    txtout = open(dir_output / "select_error.txt", "w")
    frame = query_stream[(query_stream.num_modify == 0)] if "num_modify" in query_stream else query_stream[(query_stream.OP == "SELECT")]
    modify_col = "target"
    for group in frame.groupby(by=[modify_col]):
        ticks.append(x)
        labels.append(group[0])

        txtout.write(f"{group[0]}\n")
        qgroups = group[1].groupby(by=["query_id"]).sum()
        colors = plt.cm.BuPu(np.linspace(0.25, 1.0, len(qgroups)))

        idx = 0
        for qgroup in qgroups.itertuples():
            error = qgroup.abs_diff / qgroup.cnt
            if error > min_error_threshold and (max_error_threshold is None or error < max_error_threshold):
                rect = axes[0].bar(x, qgroup.abs_diff / qgroup.cnt, width=width, color=colors[idx])[0]
                axes[0].text(rect.get_x() + rect.get_width() / 2., 1.05 * rect.get_height(), '%d' % error, ha='center', va='bottom')

                rect = axes[1].bar(x, qgroup.elapsed_us / qgroup.cnt, width=width, color=colors[idx])[0]
                axes[1].text(rect.get_x() + rect.get_width() / 2., 1.05 * rect.get_height(), '%d' % (qgroup.elapsed_us / qgroup.cnt), ha='center', va='bottom')

                rect = axes[2].bar(x, qgroup.pred_elapsed_us / qgroup.cnt, width=width, color=colors[idx])[0]
                axes[2].text(rect.get_x() + rect.get_width() / 2., 1.05 * rect.get_height(), '%d' % (qgroup.pred_elapsed_us / qgroup.cnt), ha='center', va='bottom')

                txtout.write(f"{qgroup.Index} avg. predicted/true: {error} / {qgroup.elapsed_us / qgroup.cnt}\n")

                x += width
                idx += 1

        x += 1

    axes[0].set_ylabel("Avg. Absolute Error (sum(|pred-elapsed|) / cnt)")
    axes[0].set_xticks(ticks, minor=False)
    axes[0].set_xticklabels(labels, fontdict=None, minor=False)

    axes[1].set_ylabel("Avg. Elapsed Runtime (sum(elapsed) / cnt)")
    axes[1].set_xticks(ticks, minor=False)
    axes[1].set_xticklabels(labels, fontdict=None, minor=False)

    axes[2].set_ylabel("Avg. Pred Elapsed Runtime (sum(pred) / cnt)")
    axes[2].set_xticks(ticks, minor=False)
    axes[2].set_xticklabels(labels, fontdict=None, minor=False)
    plt.savefig(dir_output / "summary_select.png")
    plt.close()
    txtout.close()


def generate_predicted_query_error_modify(query_stream, dir_output):
    x = 0
    width = 0.8
    ticks = []
    labels = []
    fig, axes = plt.subplots(2, 1, figsize=(25.6, 14.4))
    txtout = open(dir_output / "modify_error.txt", "w")

    frame = query_stream[(query_stream.num_modify != 0)] if "num_modify" in query_stream else query_stream[(query_stream.OP != "SELECT")]
    modify_col = "target"
    for group in frame.groupby(by=[modify_col]):
        ticks.append(x)
        labels.append(group[0])
        colors = plt.cm.BuPu(np.linspace(0.25, 1.0, 3))
        idx = 0

        if "OP" in query_stream:
            it = enumerate([group[1][group[1].OP == "INSERT"], group[1][group[1].OP == "UPDATE"], group[1][group[1].OP == "DELETE"]])
        else:
            it = enumerate([group[1][group[1].is_insert != 0], group[1][group[1].is_update != 0], group[1][group[1].is_delete != 0]])

        for t, subframe in it:
            if t == 0:
                txtout.write(f"{group[0]} INSERT\n")
            elif t == 1:
                txtout.write(f"{group[0]} UPDATE\n")
            else:
                txtout.write(f"{group[0]} DELETE\n")

            qgroups = subframe.groupby(by=["query_id"]).sum()
            for qgroup in qgroups.itertuples():
                error = qgroup.abs_diff / qgroup.cnt
                rect = axes[0].bar(x, qgroup.abs_diff / qgroup.cnt, width=width, color=colors[idx])[0]
                axes[0].text(rect.get_x() + rect.get_width() / 2., 1.05 * rect.get_height(), '%d' % error, ha='center', va='bottom')

                rect = axes[1].bar(x, qgroup.elapsed_us / qgroup.cnt, width=width, color=colors[idx])[0]
                axes[1].text(rect.get_x() + rect.get_width() / 2., 1.05 * rect.get_height(), '%d' % (qgroup.elapsed_us / qgroup.cnt), ha='center', va='bottom')
                txtout.write(f"{qgroup.Index} avg. predicted/true: {error} / {qgroup.elapsed_us / qgroup.cnt}\n")
                x += width

            idx += 1

        txtout.write("\n")
        x += 1

    axes[0].set_ylabel("Avg. Absolute Error (sum(|pred-elapsed|) / cnt)")
    axes[0].set_xticks(ticks, minor=False)
    axes[0].set_xticklabels(labels, fontdict=None, minor=False)

    axes[1].set_ylabel("Avg. Elapsed Runtime (sum(elapsed) / cnt)")
    axes[1].set_xticks(ticks, minor=False)
    axes[1].set_xticklabels(labels, fontdict=None, minor=False)
    plt.savefig(dir_output / "summary_modification.png")
    plt.close()
    txtout.close()


def generate_plots(query_stream, dir_output, generate_summary_flag, generate_per_query_flag, deviation_around_mean, generate_predict_abs_errors_flag):
    if generate_summary_flag:
        print("Generating summary plots.")
        with open(dir_output / "summary.txt", "w") as f:
            missing = query_stream[query_stream.error_missing_model > 0]
            f.write(f"Total Elapsed Us: {query_stream.elapsed_us.sum()}\n")
            f.write(f"Total Predicted Elapsed Us: {query_stream.pred_elapsed_us.sum()}\n")
            f.write(f"Total Missing Models: {missing.shape[0]}\n")
            f.write(f"Average # Missing Models: {missing.error_missing_model.sum() / max(1, missing.shape[0])}\n")

            def summary(df, prefix):
                f.write(f"Average Absolute Error ({prefix}): {(df.abs_diff.sum() / df.shape[0])}\n")

                box = [1, 5, 10, 100]
                for bound in box:
                    f.write(f"% within absolute error {bound} ({prefix}): {(len(df[df.abs_diff < bound]) / df.shape[0])}\n")
                f.write(f"% exceed absolute error {box[-1]} ({prefix}): {(len(df[df.abs_diff >= box[-1]]) / df.shape[0])}\n")

                for bound in box:
                    f.write(f"% underpredict error {bound} ({prefix}): {(len(df[(df.predicted_minus_elapsed < 0) & (df.abs_diff < bound)]) / df.shape[0])}\n")
                f.write(f"% exceed underpredict error {box[-1]} ({prefix}): {(len(df[(df.predicted_minus_elapsed < 0) & (df.abs_diff >= box[-1])]) / df.shape[0])}\n")

            if "num_modify" in query_stream:
                select = query_stream[query_stream.num_modify == 0]
                modify = query_stream[query_stream.num_modify != 0]
            else:
                select = query_stream[query_stream.OP == "SELECT"]
                modify = query_stream[query_stream.OP != "SELECT"]

            f.write(f"SELECT Elapsed Us: {select.elapsed_us.sum()}\n")
            f.write(f"SELECT Predicted Elapsed Us: {select.pred_elapsed_us.sum()}\n")
            f.write(f"MODIFY Elapsed Us: {modify.elapsed_us.sum()}\n")
            f.write(f"MODIFY Predicted Elapsed Us: {modify.pred_elapsed_us.sum()}\n")

            summary(query_stream, "All Queries")
            f.write("\n")

            summary(select, "SELECT")
            f.write("\n")

            summary(modify, "MODIFY")

        generate_rolling(query_stream, dir_output)

    if generate_per_query_flag:
        print("Generating per query plots.")
        generate_per_query_plots(query_stream, deviation_around_mean, dir_output)

    if generate_predict_abs_errors_flag:
        print("Generating predicted query error plots for SELECT.")
        generate_predicted_query_error(query_stream, dir_output, 0, None)

        print("Generating predicted query error plots for MODIFY.")
        generate_predicted_query_error_modify(query_stream, dir_output)


def generate_txn_plots(query_stream, output_dir, txn_analysis_file):
    if txn_analysis_file is None:
        return

    query_stream.sort_values(by=["statement_timestamp"], inplace=True)
    with open(txn_analysis_file, "r") as f:
        analysis = yaml.load(f, Loader=yaml.FullLoader)

    if not Path("/tmp/analysis/done").exists():
        query_stream["query_text"] = query_stream.query_text.str.lower()
        txn_groupings = {txn_name: [] for txn_name in analysis}
        for txn, txn_queries in tqdm(query_stream.groupby(by=["txn"])):
            valid_txn_name = None
            for txn_name, id_queries_set in analysis.items():
                for id_queries in id_queries_set:
                    valid = True
                    for i, query_text in enumerate(id_queries):
                        if i < txn_queries.shape[0] and query_text in txn_queries.iloc[i].query_text:
                            continue

                        valid = False
                        break

                    if valid:
                        valid_txn_name = txn_name
                        break

                if valid_txn_name is not None:
                    break

            assert valid_txn_name is not None, print(txn_queries.query_text)
            txn_groupings[valid_txn_name].append({"txn_elapsed_us": txn_queries.elapsed_us.sum(), "txn_pred_elapsed_us": txn_queries.pred_elapsed_us.sum()})

        txn_groupings = {txn: pd.DataFrame(v) for txn, v in txn_groupings.items()}

        Path(f"/tmp/analysis/").mkdir(parents=True, exist_ok=True)
        for txn_group, v in txn_groupings.items():
            v.to_feather(f"/tmp/analysis/{txn_group}.feather")
        open("/tmp/analysis/done", "w").close()
    else:
        files = glob.glob("/tmp/analysis/*.feather")
        txn_groupings = {}
        for f in files:
            name = Path(f).parts[-1].split(".feather")[0]
            txn_groupings[name] = pd.read_feather(f"/tmp/analysis/{name}.feather")

    for txn_name, points in txn_groupings.items():
        print(f"Processing the {txn_name} txn grouping")
        df = pd.DataFrame(points)
        df["order"] = df.index

        fig, axes = plt.subplots(4, 1, figsize=(25.6, 14.4))
        x_min = min(df.txn_elapsed_us.min(), df.txn_pred_elapsed_us.min())
        x_max = max(df.txn_elapsed_us.max(), df.txn_pred_elapsed_us.max())

        # Plot the true transaction elapsed us as a bar graph.
        # Try to plot the KDE too..
        df.txn_elapsed_us.plot.kde(color='b', ax=axes[0])
        axes[0].set_title(f"{txn_name} elapsed us")
        axes[0].set_xlim(x_min, x_max)

        df.txn_elapsed_us.hist(bins=1000, color='b', ax=axes[1])
        axes[1].set_xlim(x_min, x_max)

        # Plot the predicted transaction elapsed us as a bar graph.
        if df.txn_pred_elapsed_us.nunique() > 1:
            df.txn_pred_elapsed_us.plot.kde(color='r', ax=axes[2])
            axes[2].set_title(f"{txn_name} pred elapsed us")
            axes[2].set_xlim(x_min, x_max)

        df.txn_pred_elapsed_us.hist(bins=1000, color='r', ax=axes[3])
        axes[3].set_xlim(x_min, x_max)

        plt.savefig(output_dir / f"{txn_name}_dist.png")
        plt.close()

        fig, axes = plt.subplots(1, 1, figsize=(12.8, 7.2))
        df.txn_elapsed_us.hist(cumulative=True, density=True, histtype='step', bins=100, ax=axes, color='r', alpha=0.5, label="True Dist")
        df.txn_pred_elapsed_us.hist(cumulative=True, density=True, histtype='step', bins=100, ax=axes, color='b', alpha=0.5, label="Predicted Dist")
        axes.legend(["True Dist", "Predicted Dist"])
        plt.savefig(output_dir / f"{txn_name}_cdf.png")
        plt.close()

    shutil.rmtree("/tmp/analysis")


def main(dir_input, txn_analysis_file, generate_summary, generate_per_query, deviation_around_mean, generate_predict_abs_errors):
    inputs = [l for l in dir_input.rglob("query_results.feather")] + [l for l in dir_input.rglob("evals/")]
    inputs = sorted(inputs)
    for input_result in inputs:
        logger.info("Processing: %s", input_result)

        if Path(input_result).is_dir():
            if (Path(input_result) / "query_results.feather").exists():
                continue

            subframes = glob.glob(f"{input_result}/*.feather")
            query_stream = pd.concat(map(pd.read_feather, subframes), ignore_index=True)
            query_stream.loc[query_stream[query_stream.optype == 1].index, "OP"] = "SELECT"
            query_stream.loc[query_stream[query_stream.optype == 2].index, "OP"] = "INSERT"
            query_stream.loc[query_stream[query_stream.optype == 3].index, "OP"] = "UPDATE"
            query_stream.loc[query_stream[query_stream.optype == 4].index, "OP"] = "DELETE"
            query_stream.to_feather(f"{input_result}/query_results.feather")
            dir_output = Path(input_result) / "plots"
        else:
            query_stream = pd.read_feather(input_result)
            dir_output = input_result.parent / "plots"

        if "pred_elapsed_us_bias" in query_stream:
            query_stream["pred_elapsed_us"] = query_stream["pred_elapsed_us_bias"]

        dir_output.mkdir(parents=True, exist_ok=True)
        query_stream["predicted_minus_elapsed"] = query_stream["pred_elapsed_us"] - query_stream["elapsed_us"]
        query_stream["abs_diff"] = query_stream.predicted_minus_elapsed.apply(lambda c: abs(c))
        query_stream["cnt"] = 1

        if "query_order" not in query_stream:
            query_stream.sort_values(by=["statement_timestamp"], inplace=True, ignore_index=True)
            query_stream["query_order"] = query_stream.index

        consider_streams = [("original", query_stream),]
        if Path(input_result.parent / "ddl_changes.pickle").exists():
            with open(f"{input_result.parent}/ddl_changes.pickle", "rb") as f:
                ff_tbl_change_map = pickle.load(f)

            if len(ff_tbl_change_map) > 0:
                # We just use an arbitrary table's timestamps to segment.
                # The reason we do this: no queries run during a period of DDL changes.
                slots = ff_tbl_change_map[list(ff_tbl_change_map.keys())[0]]

                for i, (ts, _) in enumerate(slots):
                    consider_streams.append((f"{i}", query_stream[query_stream.statement_timestamp < ts].copy()))
                    query_stream = query_stream[query_stream.statement_timestamp >= ts]

                if query_stream.shape[0] > 0:
                    consider_streams.append((f"{len(slots)}", query_stream))

        for (flag, query_stream) in consider_streams:
            if flag == "original":
                generate_plots(query_stream, dir_output, generate_summary, generate_per_query, deviation_around_mean, generate_predict_abs_errors)
                generate_txn_plots(query_stream, dir_output, txn_analysis_file)
            elif flag != "original":
                sub_output = dir_output / f"{flag}"
                sub_output.mkdir(parents=True, exist_ok=True)
                generate_plots(query_stream, sub_output, generate_summary, generate_per_query, deviation_around_mean, generate_predict_abs_errors)
                generate_txn_plots(query_stream, sub_output, txn_analysis_file)

        del query_stream
        del consider_streams
        gc.collect()


class EvalQueryPlotsCLI(cli.Application):
    dir_input = cli.SwitchAttr(
        "--dir-input",
        Path,
        mandatory=True,
        help="Path to the input folder to recursively search.",
    )

    txn_analysis_file = cli.SwitchAttr(
        "--txn-analysis-file",
        Path,
        help="Path to transaction analysis file.",
    )

    generate_summary = cli.Flag(
        "--generate-summary",
        default=False,
        help="Whether to generate summary error information.",
    )

    generate_per_query = cli.Flag(
        "--generate-per-query",
        default=False,
        help="Whether to generate per-query plots of the errors.",
    )

    deviation_around_mean = cli.SwitchAttr(
        "--deviation-around-mean",
        int,
        help="Deviation to plot around mean for per-query plots.",
        default=0,
    )

    generate_predict_abs_errors = cli.Flag(
        "--generate-predict-abs-errors",
        default=False,
        help="Whether to generate abs errors against each table.",
    )

    def main(self):
        main(self.dir_input,
             self.txn_analysis_file,
             self.generate_summary,
             self.generate_per_query,
             self.deviation_around_mean,
             self.generate_predict_abs_errors)


if __name__ == "__main__":
    EvalQueryPlotsCLI.run()

