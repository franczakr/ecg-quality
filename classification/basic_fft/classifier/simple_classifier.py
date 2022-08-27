import numpy as np
import torch
from scipy.fft import fft

from classification.abstract.AbstractSimpleClassifier import AbstractSimpleClassifier
from classification.abstract.Autoencoder import Autoencoder
from util import config


class SimpleClassifier(AbstractSimpleClassifier):

    def _get_loss(self, autoencoder: Autoencoder, slices: np.ndarray):
        device = config.DEVICE
        autoencoder = autoencoder.to(device)
        autoencoder.eval()

        input = np.abs(np.array([fft(i) for i in slices]))

        with torch.no_grad():
            input = torch.tensor(input).to(device)
            output = autoencoder(input).cpu()
            input = input.cpu()
            loss = np.array([autoencoder.loss_func(input[i], output[i]) for i in range(len(input))])
        return loss
