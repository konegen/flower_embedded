import json
import os
from logging import INFO
from typing import Dict, List, Tuple

import yaml
from analysis_backend.deep_learning_backend.deep_learning_backend import (
    DeepLearningBackend,
)
from analysis_backend.machine_learning_backend.machine_learning_backend import (
    MachineLearningBackend,
)
from flwr.common import Context, Metrics, Scalar, ndarrays_to_parameters
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from strategy import AggregateCalculatedStatistics, AggregateCustomMetricStrategy
from task import evaluate_config, fit_config, get_model_parameters, load_config


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    """Compute weighted average.

    It is a generic implementation that averages only over floats and ints and drops the
    other data types of the Metrics.
    """
    # num_samples_list can represent the number of samples
    # or the number of batches depending on the client
    num_samples_list = [n_batches for n_batches, _ in metrics]
    num_samples_sum = sum(num_samples_list)
    metrics_lists: Dict[str, List[float]] = {}
    for num_samples, all_metrics_dict in metrics:
        #  Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            if isinstance(value, (float, int)):
                metrics_lists[single_metric] = []
        # Just one iteration needed to initialize the keywords
        break

    for num_samples, all_metrics_dict in metrics:
        # Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            # Add weighted metric
            if isinstance(value, (float, int)):
                metrics_lists[single_metric].append(float(num_samples * value))

    weighted_metrics: Dict[str, Scalar] = {}
    for metric_name, metric_values in metrics_lists.items():
        weighted_metrics[metric_name] = sum(metric_values) / num_samples_sum

    return weighted_metrics


def server_fn(context: Context) -> ServerAppComponents:

    if context.run_config["use-case"] == "statistic":

        # Define strategy
        strategy = AggregateCalculatedStatistics(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
            evaluate_metrics_aggregation_fn=weighted_average,
            on_evaluate_config_fn=evaluate_config,
        )
        server_config = ServerConfig(num_rounds=1)

        return ServerAppComponents(strategy=strategy, config=server_config)

    elif context.run_config["use-case"] == "train":

        analysis_config = load_config()

        if context.run_config["backend"] == "deep learning":
            analysis_backend = DeepLearningBackend()
        elif context.run_config["backend"] == "machine learning":
            analysis_backend = MachineLearningBackend()
        else:
            raise ValueError(
                f'Unknown analysis backend: {context.run_config["backend"]}'
            )

        model = analysis_backend.create_model(
            analysis_config["train"]["model"]["type"],
            analysis_config["train"]["model"]["parameters"],
        )

        parameters = get_model_parameters(model)

        # Define strategy
        strategy = strategy = AggregateCustomMetricStrategy(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
        )
        server_config = ServerConfig(num_rounds=context.run_config["num-server-rounds"])

        return ServerAppComponents(strategy=strategy, config=server_config)

    else:
        raise ValueError(f'Unknown fl_use_case: {server_config["use_case"]}')


# Create ServerApp
app = ServerApp(server_fn=server_fn)
