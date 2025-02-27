import json
from logging import INFO, WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

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

    def initialize_parameters(self, client_manager: ClientManager) -> None:
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
            log(INFO, "No results to aggregate.")
            return None, {}

        metrics = {}

        # log(INFO, f"results: {results}")
        # for i, (_, res) in enumerate(results):
        #     log(INFO, f"res.num_examples: {res.num_examples}")
        #     for key in res.metrics.keys():
        #         if not key in metrics:
        #             metrics[key] = {}
        #         metrics[key][f"Client_{i}"] = json.loads(res.metrics[key])
        # log(INFO, f"metrics: {metrics}")

        for i, (_, res) in enumerate(results):
            # log(INFO, f"i: {i}")
            # log(INFO, f"res.metrics.items(): {res.metrics.items()}")
            # log(INFO, f"res.num_examples: {res.num_examples}")

            if not f"Client_{i}" in metrics:
                metrics[f"Client_{i}"] = {}
            metrics[f"Client_{i}"]["num_examples"] = res.num_examples
            for key, value in res.metrics.items():
                metrics[f"Client_{i}"][key] = json.loads(value)

        log(INFO, f"metrics: {metrics}")
        log(INFO, f"metrics.items(): {metrics.items()}")

        aggregated_metrics = {}
        total_examples = 0
        # Aggregate values across all clients
        for _, client_metrics in metrics.items():
            num_examples = client_metrics["num_examples"]
            total_examples += num_examples
            log(INFO, f"client_metrics: {client_metrics}")

            if not isinstance(client_metrics, dict):
                continue

            self.recursive_aggregate(aggregated_metrics, client_metrics, num_examples)

        log(INFO, f"aggregated_metrics: {aggregated_metrics}")

        # Compute the averaged values
        averaged_metrics = {
            key: {
                sub_key: sub_value / total_examples
                for sub_key, sub_value in sub_dict.items()
            }
            for key, sub_dict in aggregated_metrics.items()
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

    def recursive_aggregate(self, target: dict, source: dict, weight: int) -> None:
        """
        Recursively aggregates values from source into target,
        weighting by num_examples. Ensures first-level keys remain dictionaries.
        """
        for key, value in source.items():
            if key == "num_examples" or key == "spearmanr":  # Skip non-metric keys
                continue

            if isinstance(value, dict):
                if key not in target:
                    target[key] = {}  # Ensure first-level keys are dictionaries
                self.recursive_aggregate(target[key], value, weight)
            else:
                if key not in target:
                    target[key] = 0  # Directly aggregate numerical values

                target[key] += value * weight  # Weighting by num_examples
