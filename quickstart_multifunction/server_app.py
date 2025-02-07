from typing import List, Tuple

from flwr.common import (
    Context,
    Metrics,
    ndarrays_to_parameters,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from task import load_model
from utils import evaluate_config, fit_config

from strategy import AggregateCalculatedStatistics, AggregateCustomMetricStrategy

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):

    # log(INFO, f"context: {context}")
    # log(INFO, f"context.run_config: {context.run_config}")

    if context.run_config["use-case"] == "statistic":

        # Define strategy
        strategy = AggregateCalculatedStatistics(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
            evaluate_metrics_aggregation_fn=weighted_average,
            on_evaluate_config_fn=evaluate_config,
        )
        config = ServerConfig(num_rounds=1)

        return ServerAppComponents(strategy=strategy, config=config)

    elif context.run_config["use-case"] == "train":

        # Get parameters to initialize global model
        parameters = ndarrays_to_parameters(load_model().get_weights())

        # Define strategy
        strategy = strategy = AggregateCustomMetricStrategy(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=fit_config,
        )
        config = ServerConfig(num_rounds=context.run_config["num-server-rounds"])

        return ServerAppComponents(strategy=strategy, config=config)

    else:
        raise ValueError(f'Unknown fl_use_case: {config["use_case"]}')


app = ServerApp(server_fn=server_fn)
