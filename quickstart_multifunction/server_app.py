import random
import time
from logging import INFO, WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    Context,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Metrics,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
)
from flwr.common.logger import log
from flwr.server import Driver, ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, Strategy

from quickstart_multifunction.task import load_model
from utils import evaluate_config, fit_config

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


class AggregateCalculatedStatistics:

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return None

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        metrics = {}

        for i, (_, res) in enumerate(results):
            log(INFO, f"i: {i}")
            log(INFO, f"res.metrics.items(): {res.metrics.items()}")
            log(INFO, f"res.num_examples: {res.num_examples}")

            if not f"Client_{i}" in metrics:
                metrics[f"Client_{i}"] = {}
            metrics[f"Client_{i}"]["num_examples"] = res.num_examples
            for key, value in res.metrics.items():
                metrics[f"Client_{i}"][key] = value

        log(INFO, f"metrics: {metrics}")

        aggregated_metrics = {}
        total_examples = 0
        # Aggregate values across all clients
        for _, client_metrics in metrics.items():
            num_examples = client_metrics["num_examples"]
            total_examples += num_examples

            for key, value in client_metrics.items():
                if key == "num_examples":
                    continue  # Skip counting num_examples in the metrics aggregation

                if key not in aggregated_metrics:
                    aggregated_metrics[key] = 0

                aggregated_metrics[key] += (
                    value * num_examples
                )  # Weighting by num_examples

        log(INFO, f"aggregated_metrics: {aggregated_metrics}")

        # Compute the averaged values
        averaged_metrics = {
            key: value / total_examples for key, value in aggregated_metrics.items()
        }

        log(INFO, f"averaged_metrics: {averaged_metrics}")

        return float(0), averaged_metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""

        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        return None

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)

        log(INFO, f"config: {config}")

        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]


class AggregateCustomMetricStrategy(FedAvg):
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(
            f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}"
        )

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return float(aggregated_loss), {"accuracy": float(aggregated_accuracy)}


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
