"""pandas_example: A Flower / Pandas app."""

import os
import warnings
from logging import INFO

import numpy as np
import pandas as pd
from flwr.client import ClientApp
from flwr.common import Context, Message, MetricsRecord, RecordSet
from flwr.common.logger import log

warnings.filterwarnings("ignore", category=UserWarning)


def get_clientapp_dataset(path: str):

    data = pd.read_csv(path)
    return data


# Flower ClientApp
app = ClientApp()


@app.query()
def query(msg: Message, context: Context):
    """Construct histogram of local dataset and report to `ServerApp`."""

    # Read the node_config to fetch data partition associated to this node
    dataset_path = context.node_config["dataset-path"]

    dataset = get_clientapp_dataset(dataset_path)

    metrics = {}
    # Compute some statistics for each column in the dataframe
    for feature_name in dataset.columns:
        if feature_name == "target":
            continue

        # Compute weighted average
        metrics[f"{feature_name}_avg"] = dataset[feature_name].mean()
        metrics[f"{feature_name}_std"] = dataset[feature_name].std()
        metrics[f"{feature_name}_median"] = dataset[feature_name].median()
        metrics[f"{feature_name}_count"] = len(dataset)

    reply_content = RecordSet(metrics_records={"query_results": MetricsRecord(metrics)})

    return msg.create_reply(reply_content)
