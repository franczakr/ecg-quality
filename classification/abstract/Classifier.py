import abc

import numpy as np

from classification.abstract.Autoencoder import Autoencoder


class Classifier(abc.ABC):
    def __init__(self):
        self.threshold = None

    @abc.abstractmethod
    def train(self, autoencoder: Autoencoder, slices: np.ndarray, classes: np.ndarray):
        pass

    @abc.abstractmethod
    def classify(self, autoencoder: Autoencoder, slices: np.ndarray) -> (np.ndarray, np.ndarray):
        pass
