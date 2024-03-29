import math

import numpy as np
import optuna.exceptions
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.fft import fft
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from classification.abstract.Autoencoder import Autoencoder
from classification.abstract.Trainer import Trainer
from util import config


class AETrainerMultiInput(Trainer):

    def train(self, autoencoder: Autoencoder, train_dataset: np.ndarray) -> Autoencoder:
        device = config.DEVICE

        autoencoder = autoencoder.to(device)

        optimizer = optim.Adam(autoencoder.parameters(), lr=self.lr)

        train_loader = DataLoader(
            TensorDataset(torch.tensor(train_dataset)), batch_size=self.batch_size, shuffle=True, pin_memory=True
        )

        # print("AE training started")

        for epoch in range(self.epochs):
            loss = 0

            for input_batch in train_loader:
                input_batch = input_batch[0].to(device)
                optimizer.zero_grad()
                input_batch_signal = input_batch
                input_batch_fft = torch.tensor(np.asarray([np.abs(fft(i.numpy())) for i in input_batch])).to(device)
                input_batch = (input_batch_signal, input_batch_fft)
                output_batch = autoencoder(input_batch_signal, input_batch_fft)
                train_loss = autoencoder.loss_func(output_batch, input_batch)
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()

            loss = loss / len(train_loader)

            if loss > 100000000 or math.isnan(loss):
                raise optuna.exceptions.TrialPruned()

            # print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

        # print("AE training finished")

        return autoencoder
