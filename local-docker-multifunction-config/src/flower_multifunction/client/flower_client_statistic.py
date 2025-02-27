import json
from logging import INFO, WARNING

import numpy as np
import pandas as pd
from analysis_backend.statistic_backend.statistic_backend import StatisticBackend
from flwr.client import NumPyClient
from flwr.common import Parameters
from flwr.common.logger import log


class FlowerClientStatistic(NumPyClient):
    def __init__(self, statistic_backend: StatisticBackend, data: pd.DataFrame) -> None:
        self.statistic_backend = statistic_backend
        self.data = data

    def evaluate(self, parameters: Parameters, config: dict) -> tuple[float, int, dict]:

        analysis_config = json.loads(config["config_json"])

        self.data = self.data.drop(
            columns=[analysis_config["data_info"]["target_column"]]
        )

        metrics = {}

        for metric, parameter in analysis_config["statistic"].items():

            calculated_statistic = self.statistic_backend.run_statistic(
                data=self.data,
                statistic_request=metric,
                columns=parameter["dataframe_parameters"],
                statistic_parameters=parameter["statistic_parameters"],
            )

            metrics[metric] = json.dumps(calculated_statistic)

            # if isinstance(calculated_statistic, pd.Series):
            #     metrics[metric] = json.dumps(calculated_statistic.to_dict())
            # elif isinstance(calculated_statistic, np.ndarray):
            #     metrics[metric] = json.dumps(
            #         dict(zip(["a", "b", "c", "d"], list(calculated_statistic)))
            #     )
            # else:
            #     log(WARNING, f"Metric '{metric}' is not known")
            #     metrics[metric] = 0

        # log(INFO, f"metrics: {metrics}")

        return float(0.0), len(self.data), metrics
