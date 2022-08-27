import abc

import numpy as np

from classification.abstract.Autoencoder import Autoencoder
from classification.abstract.Classifier import Classifier
from util.data_reader import GOOD_QUALITY, BAD_QUALITY


class AbstractSimpleClassifier(Classifier, abc.ABC):

    def train(self, autoencoder: Autoencoder, slices: np.ndarray, classes: np.ndarray):
        loss = self._get_loss(autoencoder, slices)

        gq_loss = [s[0] for s in list(zip(loss, classes)) if s[1] == GOOD_QUALITY]
        bq_loss = [s[0] for s in list(zip(loss, classes)) if s[1] == BAD_QUALITY]

        best_t = 0
        best_acc = 0
        for t in loss:
            TP = len(np.where(bq_loss > t)[0])
            TN = len(np.where(gq_loss <= t)[0])

            if TP <= 0:
                continue

            acc = (TP + TN) / len(loss)

            if acc > best_acc:
                best_acc = acc
                best_t = t

        self.threshold = best_t
        print(f"Classifier trained. Threshold: {self.threshold}")

    def classify(self, autoencoder: Autoencoder, slices: np.ndarray) -> np.ndarray:
        loss = self._get_loss(autoencoder, slices)
        return np.array([BAD_QUALITY if l > self.threshold else GOOD_QUALITY for l in loss])

    @abc.abstractmethod
    def _get_loss(self, autoencoder: Autoencoder, slices: np.ndarray) -> np.ndarray:
        pass
