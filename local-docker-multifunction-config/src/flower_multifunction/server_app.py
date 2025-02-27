import json
import os
from logging import INFO
from typing import List, Tuple

import yaml
from analysis_backend.deep_learning_backend.deep_learning_backend import (
    DeepLearningBackend,
)
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from strategy import AggregateCalculatedStatistics, AggregateCustomMetricStrategy
from task import evaluate_config, fit_config, load_config


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):

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

        deep_learning_backend = DeepLearningBackend()

        model = deep_learning_backend.create_model(
            analysis_config["train"]["model"]["type"],
            analysis_config["train"]["model"]["parameters"]["layers"],
        )

        # Get parameters to initialize global model
        parameters = ndarrays_to_parameters(model.get_weights())

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


app = ServerApp(server_fn=server_fn)
