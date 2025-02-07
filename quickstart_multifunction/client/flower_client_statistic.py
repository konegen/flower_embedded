import json
from logging import WARNING
from flwr.common.logger import log

from flwr.client import NumPyClient


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
