"""pandas_example: A Flower / Pandas app."""

import random
import time
from logging import INFO
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    Context,
    EvaluateRes,
    FitRes,
    Message,
    MessageType,
    Metrics,
    RecordSet,
    Scalar,
    ndarrays_to_parameters,
)
from flwr.common.logger import log
from flwr.server import Driver, ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from quickstart_multifunction.task import load_model
from utils import load_config

config = load_config()

if config["use_case"].lower() == "statistical":
    log(INFO, f"Statistical FL")
    app = ServerApp()

    @app.main()
    def main(driver: Driver, context: Context) -> None:
        """This `ServerApp` construct a histogram from partial-histograms reported by the
        `ClientApp`s."""

        num_rounds = context.run_config["num-server-rounds"]
        min_nodes = 2
        fraction_sample = context.run_config["fraction-sample"]

        for server_round in range(num_rounds):
            log(INFO, "")  # Add newline for log readability
            log(INFO, "Starting round %s/%s", server_round + 1, num_rounds)

            # Loop and wait until enough nodes are available.
            all_node_ids = []
            while len(all_node_ids) < min_nodes:
                all_node_ids = driver.get_node_ids()
                if len(all_node_ids) >= min_nodes:
                    # Sample nodes
                    num_to_sample = int(len(all_node_ids) * fraction_sample)
                    node_ids = random.sample(all_node_ids, num_to_sample)
                    break
                log(INFO, "Waiting for nodes to connect...")
                time.sleep(2)
            log(INFO, f"Context: {context}")

            log(INFO, "Sampled %s nodes (out of %s)", len(node_ids), len(all_node_ids))

            # Create messages
            recordset = RecordSet()
            messages = []
            for node_id in node_ids:  # one message for each node
                message = driver.create_message(
                    content=recordset,
                    message_type=MessageType.QUERY,  # target `query` method in ClientApp
                    dst_node_id=node_id,
                    group_id=str(server_round),
                )
                messages.append(message)

            # Send messages and wait for all results
            replies = driver.send_and_receive(messages)
            log(INFO, "Received %s/%s results", len(replies), len(messages))

            # Aggregate partial histograms
            aggregated_metric = aggregate_metric(replies)
            log(INFO, f"aggregated_metric: {aggregated_metric}")

    def aggregate_metric(messages: Message):
        """Aggregate metrics."""

        aggregated_metrics = {}
        total_count = 0
        for idx, rep in enumerate(messages):
            log(INFO, f"Node {idx+1}/{len(messages)}")
            aggregated_metrics[f"Node_{idx}"] = {}
            # log(INFO, f"rep: {rep}")
            if rep.has_error():
                continue
            query_results = rep.content.metrics_records["query_results"]
            # log(INFO, f"query_results: {query_results}")
            # Sum metrics
            for k, v in query_results.items():
                # log(INFO, f"query_results.items() k: {k}")
                # log(INFO, f"query_results.items() v: {v}")
                aggregated_metrics[f"Node_{idx}"][k] = v
                if "_count" in k:
                    total_count += v

        return aggregated_metrics

elif config["use_case"].lower() == "train":
    log(INFO, f"Train FL")

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
        # Read from config
        num_rounds = context.run_config["num-server-rounds"]

        # Get parameters to initialize global model
        parameters = ndarrays_to_parameters(load_model().get_weights())

        log(INFO, f"num_rounds: {num_rounds}")

        # Define strategy
        strategy = strategy = AggregateCustomMetricStrategy(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)

    app = ServerApp(server_fn=server_fn)
else:
    raise ValueError(f"Unknown fl_use_case: {config['use_case']}")
