import os

import pandas as pd
from sklearn.model_selection import train_test_split


def create_iris_partitions(dataset, num_partitions, output_dir, test_size=0.2):
    # Erstellen des Hauptausgabeverzeichnisses
    os.makedirs(output_dir, exist_ok=True)

    # Aufteilen des Datensatzes in gleiche Teile
    partition_size = len(dataset) // num_partitions
    partitions = [
        dataset.iloc[i * partition_size : (i + 1) * partition_size]
        for i in range(num_partitions)
    ]

    # Erstellen der Partitionen
    for i, partition in enumerate(partitions):
        part_dir = os.path.join(output_dir, f"iris_part_{i+1}")
        os.makedirs(part_dir, exist_ok=True)

        # Aufteilen in Trainings- und Testdaten
        train_data, test_data = train_test_split(
            partition, test_size=test_size, random_state=42
        )

        # Pfade erstellen
        train_dir = os.path.join(part_dir, "train")
        test_dir = os.path.join(part_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Speichern als CSV
        train_data.to_csv(os.path.join(train_dir, "data.csv"), index=False)
        test_data.to_csv(os.path.join(test_dir, "data.csv"), index=False)

    print(f"{num_partitions} partitions created successfully in {output_dir}.")


# Laden des Iris-Datasets
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
iris_data = pd.concat([iris.data, iris.target], axis=1)

# Partitionierung erstellen
output_directory = "datasets"
num_partitions = 3  # Anzahl der Teile
create_iris_partitions(iris_data, num_partitions, output_directory)
