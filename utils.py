import json


def fit_config(server_round: int):
    """Generate training configuration for each round."""
    # Create the configuration dictionary
    config = {
        "batch_size": 4,
        "current_round": server_round,
        "local_epochs": 3,
        "data_splits": json.dumps([0.8, 0.1, 0.1]),  # Example of serialized list
        "verbose": False,
    }
    return config


def evaluate_config(server_round: int):
    """Generate evaluation configuration for each round."""
    # Create the configuration dictionary
    config = {
        "metrics": json.dumps(["mean", "median", "std", "var", "min", "max"]),
    }
    return config
