import warnings
from logging import INFO

from flwr.client import ClientApp
from flwr.common import Context
from flwr.common.logger import log

warnings.filterwarnings("ignore", category=UserWarning)

from analysis_backend.deep_learning_backend.deep_learning_backend import (
    DeepLearningBackend,
)
from analysis_backend.statistic_backend.statistic_backend import StatisticBackend
from client import FlowerClientStatistic, FlowerClientTrain
from task import load_config, load_data


def client_fn(context: Context):

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)

    if context.run_config["use-case"] == "statistic":

        statistic_backend = StatisticBackend(log_logger_to_console=True)

        log(INFO, f"Start Flower Client Statistic")

        # Return Client instance
        return FlowerClientStatistic(statistic_backend, data).to_client()

    elif context.run_config["use-case"] == "train":

        deep_learning_backend = DeepLearningBackend()

        log(INFO, f"Start Flower Client Train")
        # Return Client instance
        return FlowerClientTrain(deep_learning_backend, data).to_client()

    else:
        raise ValueError(f'Unknown Flower use case: {context.run_config["use-case"]}')


app = ClientApp(client_fn=client_fn)
