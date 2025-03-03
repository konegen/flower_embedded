import json
import os
from typing import List, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from flwr.common import Parameters, ndarrays_to_parameters
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

    params = ndarrays_to_parameters(params)

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


fds = None


def load_data(partition_id: int, num_partitions: int) -> pd.DataFrame:

    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="scikit-learn/iris",
            partitioners={"train": partitioner},
        )
    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]

    dataset = dataset.drop(columns=["Id"])

    return dataset
