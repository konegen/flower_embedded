import json
import os

import numpy as np
import pandas as pd
import yaml


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


def get_clientapp_dataset(
    path: str, use_case: str
) -> pd.DataFrame:  # tuple[pd.DataFrame, pd.DataFrame]:

    data = pd.read_csv(path)

    return data

    # df_train = data.sample(frac=0.8, random_state=42)
    # df_test = data.drop(df_train.index)

    # return df_train, df_test
