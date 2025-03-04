import json
import os
from typing import List, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from flwr.common import Parameters
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.base import BaseEstimator


def load_config() -> dict:
    with open(os.path.join(os.getcwd(), "config", "config.yaml"), "r") as file:
        return yaml.safe_load(file)


def fit_config(server_round: int) -> dict:
    """Generate training configuration for each round."""
    # Create the configuration dictionary

    analysis_config = {"config_json": json.dumps(load_config())}

    analysis_config["current_round"] = server_round

    return analysis_config


def evaluate_config(server_round: int) -> dict:
    """Generate evaluation configuration for each round."""

    analysis_config = {"config_json": json.dumps(load_config())}

    analysis_config["current_round"] = server_round

    return analysis_config


def get_model_parameters(
    model: Union[tf.keras.Model, tf.Module, BaseEstimator]
) -> Union[np.ndarray, List[np.ndarray]]:

    if isinstance(model, (tf.keras.Model, tf.Module)):
        params = model.get_weights()
    elif isinstance(model, BaseEstimator):
        if model.fit_intercept:
            params = [
                model.coef_,
                model.intercept_,
            ]
        else:
            params = [
                model.coef_,
            ]
    else:
        return "unknown"

    return params


def set_model_parameters(
    model: Union[tf.keras.Model, tf.Module, BaseEstimator],
    parameters: Parameters,
) -> object:
    """Set the parameters of a sklean LogisticRegression model."""
    if isinstance(model, (tf.keras.Model, tf.Module)):
        model.set_weights(parameters)
    elif isinstance(model, BaseEstimator):
        model.coef_ = parameters[0]
        if model.fit_intercept:
            model.intercept_ = parameters[1]

    return model


def get_num_classes(y_train: Union[pd.Series, np.ndarray]) -> int:
    """Determines the number of classes from y_train (integer labels or one-hot encoded)."""
    if isinstance(y_train, (pd.DataFrame, pd.Series, np.ndarray)):
        if y_train.ndim == 1:  # Integer labels
            return len(np.unique(y_train))
        elif y_train.ndim == 2:  # One-hot encoded
            return y_train.shape[1]
    raise ValueError(
        "y_train must be either a 1D array (integer labels) or a 2D array (one-hot encoded)."
    )


def set_initial_params(model, n_classes: int, n_features: int):
    """Set initial parameters as zeros.

    Required since model params are uninitialized until model.fit is called but server
    asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def get_clientapp_dataset(path: str) -> pd.DataFrame:

    data = pd.read_csv(path)

    return data
