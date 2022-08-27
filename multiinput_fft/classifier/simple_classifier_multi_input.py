import numpy as np
import torch
import torch.nn as nn
from numpy.fft import fft

from util import config
from util.data_reader import GOOD_QUALITY, BAD_QUALITY


class SimpleClassifierMultiInput:

    def __init__(self):
        self.threshold = None

    def train(self, autoencoder: nn.Module, slices: np.ndarray, classes: np.ndarray):
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

    def classify(self, autoencoder: nn.Module, slices: np.ndarray) -> (np.ndarray, np.ndarray):
        loss = self._get_loss(autoencoder, slices)
        return loss, [BAD_QUALITY if l > self.threshold else GOOD_QUALITY for l in loss]

    def _get_loss(self, autoencoder, slices: np.ndarray):
        device = config.DEVICE
        autoencoder = autoencoder.to(device)
        autoencoder.eval()
        with torch.no_grad():
            input = torch.tensor(slices).to(device)
            input_signal = input
            input_fft = torch.tensor(np.asarray([np.abs(fft(i.numpy())) for i in input])).float().to(device)
            input = (input_signal, input_fft)
            output = autoencoder(input_signal, input_fft)
            loss = []
            for j in range(len(input[0])):
                i = [input[y][j] for y in range(len(input))]
                o = [output[y][j] for y in range(len(output))]
                loss.append(autoencoder.loss_func(i, o))
            loss = np.array(loss)
        return loss
