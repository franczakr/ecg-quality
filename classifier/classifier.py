import numpy as np
import torch
import torch.nn as nn

from util.data_reader import GOOD_QUALITY, BAD_QUALITY


class Classifier:

    def __init__(self):
        self.treshold = None

    def train(self, autoencoder: nn.Module, slices: np.ndarray, classes: np.ndarray):
        loss = self._get_loss(autoencoder, slices)

        gq_loss = [s[0] for s in list(zip(loss, classes)) if s[1] == GOOD_QUALITY]
        bq_loss = [s[0] for s in list(zip(loss, classes)) if s[1] == BAD_QUALITY]

        bins = 200
        min = np.min(bq_loss)
        max = np.max(gq_loss)
        gq_hist = np.histogram(gq_loss, bins=bins, range=(min, max), density=True)
        bq_hist = np.histogram(bq_loss, bins=bins, range=(min, max), density=True)

        index = None
        for i in range(len(gq_hist[0])):
            if bq_hist[0][i] - gq_hist[0][i] > 0 and bq_hist[0][i + 1] - gq_hist[0][i + 1] > 0:
                index = i
                break

        if index is None:
            raise

        self.treshold = gq_hist[1][index]
        print(f"Classifier trained. Treshhold: {self.treshold}")

    def classify(self, autoencoder: nn.Module, slices: np.ndarray) -> (np.ndarray, np.ndarray):
        loss = self._get_loss(autoencoder, slices)
        return [BAD_QUALITY if l > self.treshold else GOOD_QUALITY for l in loss]

    def _get_loss(self, autoencoder, slices: np.ndarray):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        autoencoder = autoencoder.to(device)
        autoencoder.eval()
        with torch.no_grad():
            input = torch.tensor(slices).to(device)
            output = autoencoder(input).cpu()
            input = input.cpu()
            loss = np.array([autoencoder.loss_func(input[i], output[i]) for i in range(len(input))])
        return loss
