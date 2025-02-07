import json
import warnings
from logging import INFO, WARNING

import numpy as np
from flwr.client import ClientApp
from flwr.common import Context, Message, MetricsRecord, RecordSet
from flwr.common.logger import log

warnings.filterwarnings("ignore", category=UserWarning)


from flwr.client import NumPyClient
from flwr.common import Context

from quickstart_multifunction.task import load_data, load_model


class FlowerClientStatistic(NumPyClient):
    def __init__(self, data):
        self.x_train, self.y_train, self.x_test, self.y_test = data

    def evaluate(self, parameters, config):
        metrics = {}

        # Compute some statistics for each column in the dataframe
        for feature_name in self.x_train.columns:
            for metric in json.loads(config["metrics"]):
                if hasattr(self.x_train[feature_name], metric):
                    # Falls die Metrik existiert, berechnen
                    metrics[f"{feature_name}_{metric}"] = getattr(
                        self.x_train[feature_name], metric
                    )()
                else:
                    # Falls die Metrik nicht existiert, Warnung ausgeben und None setzen
                    log(WARNING, f"Metric '{metric}' is not known")
                    metrics[f"{feature_name}_{metric}"] = None
        return float(0.0), len(self.x_train), metrics


class FlowerClientTrain(NumPyClient):
    def __init__(self, model, data):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        log(INFO, f'config["local_epochs"]: {config["local_epochs"]}')
        log(INFO, f'config["batch_size"]: {config["batch_size"]}')
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=config["local_epochs"],
            batch_size=config["batch_size"],
            verbose=config["verbose"],
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}


def client_fn(context: Context):

    # log(INFO, f"context: {context}")

    if context.run_config["use-case"] == "statistic":
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        data = load_data(partition_id, num_partitions, "statistic")
        verbose = context.run_config.get("verbose")

        log(INFO, f"Num train data: {len(data[0])}")
        log(INFO, f"Num test data: {len(data[2])}")

        log(INFO, f"Start Flower Client Statistic")

        # Return Client instance
        return FlowerClientStatistic(data).to_client()

    elif context.run_config["use-case"] == "train":
        # Load model and data
        net = load_model()

        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        data = load_data(partition_id, num_partitions, "train")

        log(INFO, f"Num train data: {len(data[0])}")
        log(INFO, f"Num test data: {len(data[2])}")

        log(INFO, f"Start Flower Client Train")
        # Return Client instance
        return FlowerClientTrain(net, data).to_client()

    else:
        raise ValueError(f'Unknown Flower use case: {context.run_config["use-case"]}')


app = ClientApp(client_fn=client_fn)
