import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner


def load_config() -> dict:
    with open(os.path.join(os.getcwd(), "config.yaml"), "r") as file:
        return yaml.safe_load(file)


def fit_config(server_round: int) -> dict:
    """Generate training configuration for each round."""
    # Create the configuration dictionary

    analysis_config = {"config_json": json.dumps(load_config())}

    # analysis_config["current_round"] = server_round

    return analysis_config


def evaluate_config(server_round: int) -> dict:
    """Generate evaluation configuration for each round."""

    analysis_config = {"config_json": json.dumps(load_config())}

    return analysis_config


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
