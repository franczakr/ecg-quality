import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from util import data_reader


class Classifier:

    def __init__(self):
        self.classifier = LogisticRegression()

    def train(self, autoencoder: nn.Module, slices: np.ndarray, classes: np.ndarray):
        slices, classes = self._get_random_data_for_train(slices, classes)

        loss = self._get_loss(autoencoder, slices)
        loss_matrix = loss.reshape(-1, 1)

        self.classifier.fit(loss_matrix, classes)

        predictions = self.classifier.predict(loss_matrix)

        print("CLASSIER TRAIN RESULTS:")
        TN, FP, FN, TP = confusion_matrix(classes, predictions).ravel()
        print('True Positive(TP)  = ', TP)
        print('False Positive(FP) = ', FP)
        print('True Negative(TN)  = ', TN)
        print('False Negative(FN) = ', FN)
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        print('Accuracy of the binary classification = {:0.3f}'.format(accuracy))

    def classify(self, autoencoder: nn.Module, slices: np.ndarray) -> (np.ndarray, np.ndarray):
        loss = self._get_loss(autoencoder, slices)
        loss_matrix = loss.reshape(-1, 1)
        predictions = self.classifier.predict(loss_matrix)
        return slices, predictions

    def _get_loss(self, autoencoder, slices: np.ndarray):
        autoencoder.eval()
        with torch.no_grad():
            input = torch.tensor(slices)
            output = autoencoder(input)
            loss = np.array([autoencoder.loss_func(input[i], output[i]) for i in range(len(input))])
        return loss

    def _get_random_data_for_train(self, slices: np.ndarray, classes: np.ndarray, count=1000) -> (
            np.ndarray, np.ndarray):
        slices_gq = [gq for gq in list(zip(slices, classes)) if gq[1] == data_reader.GOOD_QUALITY]
        slices_bq = [bq for bq in list(zip(slices, classes)) if bq[1] == data_reader.BAD_QUALITY]
        slices_gq_random = random.sample(slices_gq, count)
        slices_bq_random = random.sample(slices_bq, count)
        data = slices_gq_random + slices_bq_random
        random.shuffle(data)
        slices = np.array([x[0] for x in data])
        classes = np.array([x[1] for x in data])
        return slices, classes
