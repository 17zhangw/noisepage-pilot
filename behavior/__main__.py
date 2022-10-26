import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s:%(asctime)s %(message)s", level=logging.INFO)

from plumbum import cli

from behavior.datagen import generate_workloads
from behavior.datagen import pg_collector
from behavior.model_ous import train
from behavior.model_ous import extract_ous
from behavior.model_ous import eval_ou
from behavior.model_query import eval_query
from behavior.model_query import plot
from behavior.model_query import compare_plot

from behavior.model_workload import analyze
from behavior.model_workload import build_exec_model
from behavior.model_workload import exec_feature_synthesis
from behavior.model_query import eval_query_workload


class BehaviorCLI(cli.Application):
    def main(self) -> None:
        pass


if __name__ == "__main__":
    BehaviorCLI.subcommand("generate_workloads", generate_workloads.GenerateWorkloadsCLI)
    BehaviorCLI.subcommand("collector", pg_collector.CollectorCLI)

    BehaviorCLI.subcommand("extract_ous", extract_ous.ExtractOUsCLI)
    BehaviorCLI.subcommand("train", train.TrainCLI)
    BehaviorCLI.subcommand("eval_ou", eval_ou.EvalOUCLI)

    BehaviorCLI.subcommand("workload_analyze", analyze.AnalyzeWorkloadCLI)
    BehaviorCLI.subcommand("workload_build_exec_model", build_exec_model.BuildExecModelCLI)
    BehaviorCLI.subcommand("workload_exec_feature_synthesis", exec_feature_synthesis.ExecFeatureSynthesisCLI)
    BehaviorCLI.subcommand("eval_query_workload", eval_query_workload.EvalQueryWorkloadCLI)

    BehaviorCLI.subcommand("eval_query", eval_query.EvalQueryCLI)
    BehaviorCLI.subcommand("eval_query_plots", plot.EvalQueryPlotsCLI)
    BehaviorCLI.subcommand("eval_query_compare_plots", compare_plot.EvalQueryComparePlotsCLI)

    BehaviorCLI.run()
