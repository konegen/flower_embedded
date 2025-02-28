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
from task import get_clientapp_dataset


def client_fn(context: Context):

    data = get_clientapp_dataset(
        context.node_config["dataset-path"], context.run_config["use-case"]
    )

    if context.run_config["use-case"] == "statistic":

        statistic_backend = StatisticBackend(log_logger_to_console=True)

        print("\n\n")
        log(INFO, f"Start Flower Client Statistic")

        # Return Client instance
        return FlowerClientStatistic(statistic_backend, data).to_client()

    elif context.run_config["use-case"] == "train":

        deep_learning_backend = DeepLearningBackend()

        print("\n\n")
        log(INFO, f"Start Flower Client Train")
        # Return Client instance
        return FlowerClientTrain(deep_learning_backend, data).to_client()

    else:
        raise ValueError(f'Unknown Flower use case: {context.run_config["use-case"]}')


app = ClientApp(client_fn=client_fn)
