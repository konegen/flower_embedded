import warnings
from logging import INFO

from flwr.client import ClientApp
from flwr.common.logger import log
from flwr.common import Context

warnings.filterwarnings("ignore", category=UserWarning)

from task import load_data, load_model

from client import FlowerClientStatistic, FlowerClientTrain


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
