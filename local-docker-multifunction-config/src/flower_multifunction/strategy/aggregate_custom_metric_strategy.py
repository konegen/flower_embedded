from logging import INFO
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import EvaluateRes, FitRes, Scalar
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


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
        log(
            INFO,
            f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}",
        )

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return float(aggregated_loss), {"accuracy": float(aggregated_accuracy)}
