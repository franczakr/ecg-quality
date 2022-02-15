import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


class Classifier:

    def __init__(self):
        self.classifier = LogisticRegression()

    def train(self, autoencoder: nn.Module, slices: np.ndarray, classes: np.ndarray):
        loss = self._get_loss(autoencoder, slices)
        loss_matrix = loss.reshape(-1, 1)

        self.classifier.fit(loss_matrix, classes)

        predictions = self.classifier.predict(loss_matrix)

        print("CLASSIER TRAIN DATASET RESULTS:")
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
        device = 'cpu'
        autoencoder = autoencoder.to(device)
        autoencoder.eval()
        with torch.no_grad():
            input = torch.tensor(slices).to(device)
            output = autoencoder(input)
            loss = np.array([autoencoder.loss_func(input[i], output[i]) for i in range(len(input))])
        return loss
