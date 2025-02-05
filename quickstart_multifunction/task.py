"""quickstart-docker: A Flower / TensorFlow app."""

import os

import numpy as np
import tensorflow as tf
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model():

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(4,)),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(
        tf.optimizers.Adam(learning_rate=0.0005),
        "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions, use_case):

    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="scikit-learn/iris",
            partitioners={"train": partitioner},
        )
    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]

    df_train = dataset.sample(frac=0.8, random_state=42)
    df_test = dataset.drop(df_train.index)

    x_train = df_train.drop(columns=["Id", "Species"])
    x_test = df_test.drop(columns=["Id", "Species"])

    if use_case == "train":
        x_train, x_test = x_train.to_numpy(dtype=np.float32), x_test.to_numpy(
            dtype=np.float32
        )
        _, y_train = np.unique(df_train["Species"], return_inverse=True)
        _, y_test = np.unique(df_test["Species"], return_inverse=True)
    elif use_case == "statistical":
        y_train, y_test = df_train["Species"], df_test["Species"]

    return x_train, y_train, x_test, y_test
