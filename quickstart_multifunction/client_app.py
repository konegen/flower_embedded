import warnings
from logging import INFO

import pandas as pd
from flwr.client import ClientApp
from flwr.common import Context
from flwr.common.logger import log

warnings.filterwarnings("ignore", category=UserWarning)

from client import FlowerClientStatistic, FlowerClientTrain
from task import load_data, load_model


def get_clientapp_dataset(path: str):

    data = pd.read_csv(path)
    return data


def client_fn(context: Context):

    # log(INFO, f"context: {context}")

    if context.run_config["use-case"] == "statistic":
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]

        data = get_clientapp_dataset(context.node_config["dataset-path"])

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
