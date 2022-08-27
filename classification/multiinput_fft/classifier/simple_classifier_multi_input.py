import numpy as np
import torch
from numpy.fft import fft

from classification.abstract.AbstractSimpleClassifier import AbstractSimpleClassifier
from classification.abstract.Autoencoder import Autoencoder
from util import config


class SimpleClassifierMultiInput(AbstractSimpleClassifier):

    def _get_loss(self, autoencoder: Autoencoder, slices: np.ndarray):
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
