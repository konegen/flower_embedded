import json
import os
import pickle
from logging import INFO
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from analysis_backend.deep_learning_backend.deep_learning_backend import (
    DeepLearningBackend,
)
from analysis_backend.machine_learning_backend.machine_learning_backend import (
    MachineLearningBackend,
)
from flwr.client import NumPyClient
from flwr.common import Parameters
from flwr.common.logger import log
from task import set_model_parameters


class FlowerClientTrain(NumPyClient):
    def __init__(
        self,
        analysis_backend: Union[DeepLearningBackend, MachineLearningBackend],
        data: pd.DataFrame,
    ) -> None:
        self.analysis_backend = analysis_backend
        self.data = data

        self.temp_dir = os.path.join(os.getcwd(), "temp")
        self.model_path = os.path.join(self.temp_dir, "model.pkl")
        self.data_path = os.path.join(self.temp_dir, "data.pkl")
        self.config_path = os.path.join(self.temp_dir, "config.json")

    def prepare_data(
        self, analysis_config: dict
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

        indexes = self.analysis_backend.get_split_indexes(
            self.data,
            analysis_config["data_info"]["target_column"],
            split_strategy=analysis_config["train"]["split_strategie"]["name"],
            split_parameter=analysis_config["train"]["split_strategie"]["parameters"],
        )

        data, label_col = self.analysis_backend.label_encoding(
            self.data,
            analysis_config["data_info"]["target_column"],
            encoding=analysis_config["data_info"]["encoding"],
        )

        data_train = data.iloc[indexes[0]["train"]]
        # Extract features and targets from train and test sets
        X_train = data_train.drop(columns=label_col)
        y_train = data_train[label_col]

        data_test = data.iloc[indexes[0]["test"]]
        # Extract features and targets from train and test sets
        X_test = data_test.drop(columns=label_col)
        y_test = data_test[label_col]

        return X_train, y_train, X_test, y_test

    def prepare_model(self, analysis_config: dict) -> object:

        model = self.analysis_backend.create_model(
            analysis_config["train"]["model"]["type"],
            analysis_config["train"]["model"]["parameters"],
        )
        if isinstance(self.analysis_backend, DeepLearningBackend):
            log(INFO, "Using DeepLearningBackend to create the model.")
            model = self.analysis_backend.compile_model(
                model,
                analysis_config["train"]["model"]["compiler"],
            )

        return model

    def save_temp(
        self,
        model: object,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        analysis_config: dict,
    ) -> None:
        """Save model, data and config temporarily and delete old files"""
        os.makedirs(self.temp_dir, exist_ok=True)

        for path in [self.model_path, self.data_path, self.config_path]:
            if os.path.exists(path):
                os.remove(path)

        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)

        with open(self.data_path, "wb") as f:
            pickle.dump((X_train, y_train, X_test, y_test), f)

        with open(self.config_path, "w") as f:
            json.dump(analysis_config, f)

    def load_temp(
        self,
    ) -> Tuple[
        Optional[object],
        Optional[pd.DataFrame],
        Optional[pd.Series],
        Optional[pd.DataFrame],
        Optional[pd.Series],
        Optional[dict],
    ]:
        """Load model, data and config"""
        if (
            os.path.exists(self.model_path)
            and os.path.exists(self.data_path)
            and os.path.exists(self.config_path)
        ):
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)

            with open(self.data_path, "rb") as f:
                X_train, y_train, X_test, y_test = pickle.load(f)

            with open(self.config_path, "r") as f:
                analysis_config = json.load(f)

            return model, X_train, y_train, X_test, y_test, analysis_config
        return None, None, None, None, None, None

    def fit(self, parameters: Parameters, config: dict) -> tuple[float, int, dict]:

        if config["current_round"] == 1:
            analysis_config = json.loads(config["config_json"])
            X_train, y_train, X_test, y_test = self.prepare_data(analysis_config)
            model = self.prepare_model(analysis_config)

            self.save_temp(model, X_train, y_train, X_test, y_test, analysis_config)

        else:

            model, X_train, y_train, X_test, y_test, analysis_config = self.load_temp()

        model = set_model_parameters(model, parameters)

        log(
            INFO,
            f'{analysis_config["train"]["training"]["epochs"] = }',
        )
        log(
            INFO,
            f'{analysis_config["train"]["training"]["batch_size"] = }',
        )
        log(INFO, f"{self.analysis_backend = }")

        model = self.analysis_backend.train_model(
            model,
            X_train,
            y_train,
            analysis_config["train"]["training"],
        )

        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters: Parameters, config: dict) -> tuple[float, int, dict]:

        log(INFO, "Evaluating model...")
        log(INFO, f"{config['current_round'] = }")

        model, _, _, X_test, y_test, _ = self.load_temp()

        model = set_model_parameters(model, parameters)

        log(INFO, f"{self.analysis_backend = }")

        test_prediction = self.analysis_backend.predict(model, X_test)

        test_validation = self.analysis_backend.validate(y_test, test_prediction)

        return 0.0, len(X_test), test_validation
