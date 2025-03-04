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
from task import set_model_parameters, set_initial_params, get_num_classes, get_model_parameters


class FlowerClientTrain(NumPyClient):
    def __init__(
        self,
        data: pd.DataFrame,
    ) -> None:
        self.data = data

        self.temp_dir = os.path.join(os.getcwd(), "temp")
        self.analysis_backend_path = os.path.join(self.temp_dir, "analysis_backend.pkl")
        self.model_path = os.path.join(self.temp_dir, "model.pkl")
        self.data_path = os.path.join(self.temp_dir, "data.pkl")
        self.config_path = os.path.join(self.temp_dir, "config.json")

        self.analysis_backend: Optional[Union[DeepLearningBackend, MachineLearningBackend]] = None
        self.model: Optional[object] = None
        self.analysis_config: Optional[dict] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None

    def prepare_data(self) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

        indexes = self.analysis_backend.get_split_indexes(
            self.data,
            self.analysis_config["data_info"]["target_column"],
            split_strategy=self.analysis_config["train"]["split_strategie"]["name"],
            split_parameter=self.analysis_config["train"]["split_strategie"]["parameters"],
        )

        data, label_col = self.analysis_backend.label_encoding(
            self.data,
            self.analysis_config["data_info"]["target_column"],
            encoding=self.analysis_config["data_info"]["encoding"],
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

    def prepare_model(self) -> object:

        model = self.analysis_backend.create_model(
            self.analysis_config["train"]["model"]["type"],
            self.analysis_config["train"]["model"]["parameters"],
        )
        if isinstance(self.analysis_backend, DeepLearningBackend):
            log(INFO, "Using DeepLearningBackend to create the model.")
            model = self.analysis_backend.compile_model(
                model,
                self.analysis_config["train"]["model"]["compiler"],
            )
        elif isinstance(self.analysis_backend, MachineLearningBackend):
            set_initial_params(model, get_num_classes(self.y_train), self.X_train.shape[1])

        return model

    def save_temp(
        self
    ) -> None:
        """Save model, data and config temporarily and delete old files"""
        os.makedirs(self.temp_dir, exist_ok=True)

        for path in [self.analysis_backend_path, self.model_path, self.data_path, self.config_path]:
            if os.path.exists(path):
                os.remove(path)

        with open(self.analysis_backend_path, "wb") as f:
            pickle.dump(self.analysis_backend, f)

        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

        with open(self.data_path, "wb") as f:
            pickle.dump((self.X_train, self.y_train, self.X_test, self.y_test), f)

        with open(self.config_path, "w") as f:
            json.dump(self.analysis_config, f)

    def load_temp(
        self,
    ) -> None:
        """Load model, data and config"""
        if (
            os.path.exists(self.analysis_backend_path)
            and os.path.exists(self.model_path)
            and os.path.exists(self.data_path)
            and os.path.exists(self.config_path)
        ):
            with open(self.analysis_backend_path, "rb") as f:
                self.analysis_backend = pickle.load(f)

            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

            with open(self.data_path, "rb") as f:
                self.X_train, self.y_train, self.X_test, self.y_test = pickle.load(f)

            with open(self.config_path, "r") as f:
                self.analysis_config = json.load(f)

    def fit(self, parameters: Parameters, config: dict) -> tuple[float, int, dict]:

        if config["current_round"] == 1:
            self.analysis_config = json.loads(config["config_json"])

            log(INFO, f'{self.analysis_config["backend"] = }')

            if self.analysis_config["backend"] == "deep learning":
                self.analysis_backend = DeepLearningBackend()
            elif self.analysis_config["backend"] == "machine learning":
                self.analysis_backend = MachineLearningBackend()
            else:
                raise ValueError(
                    f'Unknown analysis backend: {self.analysis_config["backend"]}'
                )
            self.X_train, self.y_train, self.X_test, self.y_test = self.prepare_data()
            self.model = self.prepare_model()
            parameters = get_model_parameters(self.model)

            self.save_temp()
            

        else:

            self.load_temp()

        self.model = set_model_parameters(self.model, parameters)

        self.model = self.analysis_backend.train_model(
            self.model,
            self.X_train,
            self.y_train,
            self.analysis_config["train"]["training"],
        )

        return get_model_parameters(self.model), len(self.X_train), {}

    def evaluate(self, parameters: Parameters, config: dict) -> tuple[float, int, dict]:

        log(INFO, "Evaluating model...")
        log(INFO, f"{config['current_round'] = }")

        self.load_temp()

        self.model = set_model_parameters(self.model, parameters)

        log(INFO, f"{self.analysis_backend = }")

        test_prediction = self.analysis_backend.predict(self.model, self.X_test)

        test_validation = self.analysis_backend.validate(self.y_test, test_prediction)

        return 0.0, len(self.X_test), test_validation
