from logging import INFO
from flwr.common.logger import log

from flwr.client import NumPyClient


class FlowerClientTrain(NumPyClient):
    def __init__(self, model, data):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        log(INFO, f'config["local_epochs"]: {config["local_epochs"]}')
        log(INFO, f'config["batch_size"]: {config["batch_size"]}')
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=config["local_epochs"],
            batch_size=config["batch_size"],
            verbose=config["verbose"],
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}
