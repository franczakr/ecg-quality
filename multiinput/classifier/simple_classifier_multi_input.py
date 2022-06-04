import numpy as np
import torch
import torch.nn as nn

from util import config
from util.data_reader import GOOD_QUALITY, BAD_QUALITY


class SimpleClassifierMultiInput:

    def __init__(self):
        self.treshold = None

    def train(self, autoencoder: nn.Module, slices: np.ndarray, classes: np.ndarray):
        loss = self._get_loss(autoencoder, slices)

        gq_loss = [s[0] for s in list(zip(loss, classes)) if s[1] == GOOD_QUALITY]
        bq_loss = [s[0] for s in list(zip(loss, classes)) if s[1] == BAD_QUALITY]

        best_t = 0
        best_f1 = 0
        for t in loss:
            TP = len(np.where(bq_loss > t)[0])
            FP = len(np.where(gq_loss > t)[0])
            FN = len(np.where(bq_loss <= t)[0])

            if TP <= 0:
                continue

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

            f1 = (2 * precision * recall) / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        self.treshold = best_t
        print(f"Classifier trained. Treshhold: {self.treshold}")

    def classify(self, autoencoder: nn.Module, slices: np.ndarray) -> (np.ndarray, np.ndarray):
        loss = self._get_loss(autoencoder, slices)
        return [BAD_QUALITY if l > self.treshold else GOOD_QUALITY for l in loss]

    def _get_loss(self, autoencoder, slices: np.ndarray):
        device = config.DEVICE
        autoencoder = autoencoder.to(device)
        autoencoder.eval()
        with torch.no_grad():
            input = torch.tensor(slices).to(device)
            input_signal = input
            input_part_1 = torch.stack([i[:25] for i in input]).to(device)
            input_part_2 = torch.stack([i[25:50] for i in input]).to(device)
            input_part_3 = torch.stack([i[50:75] for i in input]).to(device)
            input_part_4 = torch.stack([i[75:] for i in input]).to(device)
            input = (input_signal, input_part_1, input_part_2, input_part_3, input_part_4)
            output = autoencoder(input_signal, input_part_1, input_part_2, input_part_3, input_part_4)
            loss = []
            for j in range(len(input[0])):
                i = [input[y][j] for y in range(len(input))]
                o = [output[y][j] for y in range(len(output))]
                loss.append(autoencoder.loss_func(i, o))
            loss = np.array(loss)
        return loss
