import os

import numpy as np
import pandas as pd
import tensorflow as tf

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
        tf.optimizers.Adam(learning_rate=0.002),
        "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_clientapp_dataset(path: str, use_case: str):

    data = pd.read_csv(path)

    df_train = data.sample(frac=0.8, random_state=42)
    df_test = data.drop(df_train.index)

    x_train = df_train.drop(columns=["target"])
    x_test = df_test.drop(columns=["target"])

    if use_case == "statistic":
        y_train, y_test = df_train["target"], df_test["target"]

    elif use_case == "train":
        x_train, x_test = x_train.to_numpy(dtype=np.float32), x_test.to_numpy(
            dtype=np.float32
        )
        _, y_train = np.unique(df_train["target"], return_inverse=True)
        _, y_test = np.unique(df_test["target"], return_inverse=True)

    return x_train, y_train, x_test, y_test
