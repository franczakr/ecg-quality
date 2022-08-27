import abc

import numpy as np
from torch import nn

from classification.abstract.Autoencoder import Autoencoder


class Trainer(abc.ABC):

    def __init__(self, epochs: int = 25, lr: float = 1e-3, batch_size: int = 128):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

    @abc.abstractmethod
    def train(self, autoencoder: Autoencoder, train_dataset: np.ndarray) -> Autoencoder:
        pass
