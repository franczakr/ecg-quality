import numpy as np
import torch
import torch.nn as nn

from classification.abstract.AbstractSimpleClassifier import AbstractSimpleClassifier
from classification.abstract.Autoencoder import Autoencoder
from util import config
from util.data_reader import GOOD_QUALITY, BAD_QUALITY


class SimpleClassifierMultiInput(AbstractSimpleClassifier):

    def _get_loss(self, autoencoder: Autoencoder, slices: np.ndarray):
        device = config.DEVICE
        autoencoder = autoencoder.to(device)
        autoencoder.eval()
        with torch.no_grad():
            input = torch.tensor(slices).to(device)
            input_part_1 = torch.stack([i[:25] for i in input]).to(device)
            input_part_2 = torch.stack([i[25:50] for i in input]).to(device)
            input_part_3 = torch.stack([i[50:75] for i in input]).to(device)
            input_part_4 = torch.stack([i[75:] for i in input]).to(device)
            input = (input_part_1, input_part_2, input_part_3, input_part_4)
            output = autoencoder(input_part_1, input_part_2, input_part_3, input_part_4)
            loss = []
            for j in range(len(input[0])):
                i = [input[y][j] for y in range(len(input))]
                o = [output[y][j] for y in range(len(output))]
                loss.append(autoencoder.loss_func(i, o))
            loss = np.array(loss)
        return loss
