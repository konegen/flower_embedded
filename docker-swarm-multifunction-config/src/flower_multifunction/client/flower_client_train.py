import json
import os
from logging import INFO

import pandas as pd
from analysis_backend.deep_learning_backend.deep_learning_backend import (
    DeepLearningBackend,
)
from flwr.client import NumPyClient
from flwr.common import Parameters
from flwr.common.logger import log


class FlowerClientTrain(NumPyClient):
    def __init__(
        self, deep_learning_backend: DeepLearningBackend, data: pd.DataFrame
    ) -> None:
        self.deep_learning_backend = deep_learning_backend
        self.data = data
        self.model = None

    def prepare_data(
        self, analysis_config: dict, data_type: str = "train"
    ) -> tuple[pd.DataFrame, pd.Series]:

        data = self.deep_learning_backend.label_encoding(
            self.data,
            analysis_config["data_info"]["target_column"],
            encoding=analysis_config["data_info"]["encoding"],
        )

        indexes = self.deep_learning_backend.get_split_indexes(
            data,
            analysis_config["data_info"]["target_column"],
            split_strategy=analysis_config["train"]["split_strategie"]["name"],
            split_parameter=analysis_config["train"]["split_strategie"]["parameters"],
        )

        data = data.iloc[indexes[0][data_type]]
        # Extract features and targets from train and test sets
        X = data.drop(columns=[analysis_config["data_info"]["target_column"]])
        y = data[analysis_config["data_info"]["target_column"]]

        return X, y

    def prepare_model(self, analysis_config: dict) -> None:

        self.model = self.deep_learning_backend.create_model(
            analysis_config["train"]["model"]["type"],
            analysis_config["train"]["model"]["parameters"]["layers"],
        )

        self.model = self.deep_learning_backend.compile_model(
            self.model,
            analysis_config["train"]["model"]["parameters"]["compiler"],
        )

    def fit(self, parameters: Parameters, config: dict) -> tuple[float, int, dict]:

        analysis_config = json.loads(config["config_json"])

        X_train, y_train = self.prepare_data(analysis_config, data_type="train")

        self.prepare_model(analysis_config)

        self.model.set_weights(parameters)

        log(
            INFO,
            f'{analysis_config["train"]["training"]["epochs"] = }',
        )
        log(
            INFO,
            f'{analysis_config["train"]["training"]["batch_size"] = }',
        )

        self.model = self.deep_learning_backend.train_model(
            self.model,
            X_train,
            y_train,
            analysis_config["train"]["training"],
        )

        return self.model.get_weights(), len(X_train), {}

    def evaluate(self, parameters: Parameters, config: dict) -> tuple[float, int, dict]:

        analysis_config = json.loads(config["config_json"])

        X_test, y_test = self.prepare_data(analysis_config, data_type="test")

        self.prepare_model(analysis_config)

        self.model.set_weights(parameters)

        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, len(X_test), {"accuracy": accuracy}
