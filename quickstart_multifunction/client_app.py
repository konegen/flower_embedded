"""pandas_example: A Flower / Pandas app."""

import warnings
from logging import INFO, WARNING

import numpy as np
from flwr.client import ClientApp
from flwr.common import Context, Message, MetricsRecord, RecordSet
from flwr.common.logger import log
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

from utils import load_config

warnings.filterwarnings("ignore", category=UserWarning)


from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from quickstart_multifunction.task import load_data, load_model

config = load_config()

if config["use_case"].lower() == "statistical":
    app = ClientApp()

    @app.query()
    def query(msg: Message, context: Context):
        """Construct histogram of local dataset and report to `ServerApp`."""

        log(INFO, f"ClientApp function QUERY was called")

        log(INFO, f"Context: {context}")

        config = load_config()
        log(INFO, f"config: {config}")

        # Read the node_config to fetch data partition associated to this node
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]

        x_train, _, _, _ = load_data(partition_id, num_partitions, "statistical")

        metrics = {}
        # Compute some statistics for each column in the dataframe
        for feature_name in x_train.columns:
            for metric in config["statistical_parameters"]:
                if hasattr(x_train[feature_name], metric):
                    # Falls die Metrik existiert, berechnen
                    metrics[f"{feature_name}_{metric}"] = getattr(
                        x_train[feature_name], metric
                    )()
                else:
                    # Falls die Metrik nicht existiert, Warnung ausgeben und None setzen
                    log(WARNING, f"Metric '{metric}' is not known")
                    metrics[f"{feature_name}_{metric}"] = None

        reply_content = RecordSet(
            metrics_records={"query_results": MetricsRecord(metrics)}
        )

        return msg.create_reply(reply_content)

elif config["use_case"].lower() == "train":
    # Define Flower Client and client_fn
    class FlowerClient(NumPyClient):
        def __init__(self, model, data, epochs, batch_size, verbose):
            self.model = model
            self.x_train, self.y_train, self.x_test, self.y_test = data
            self.epochs = epochs
            self.batch_size = batch_size
            self.verbose = verbose

        def fit(self, parameters, config):
            self.model.set_weights(parameters)
            self.model.fit(
                self.x_train,
                self.y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
            )
            return self.model.get_weights(), len(self.x_train), {}

        def evaluate(self, parameters, config):
            self.model.set_weights(parameters)
            loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
            return loss, len(self.x_test), {"accuracy": accuracy}

    def client_fn(context: Context):
        # Load model and data
        net = load_model()

        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        data = load_data(partition_id, num_partitions, "train")
        log(INFO, f"Num train data: {len(data[0])}")
        log(INFO, f"Num test data: {len(data[2])}")
        epochs = context.run_config["local-epochs"]
        batch_size = context.run_config["batch-size"]
        verbose = context.run_config.get("verbose")

        log(INFO, f"Start Flower Client")

        # Return Client instance
        return FlowerClient(net, data, epochs, batch_size, verbose).to_client()

    app = ClientApp(client_fn=client_fn)
else:
    raise ValueError(f"Unknown fl_use_case: {config['use_case']}")
