import math

import numpy as np
import optuna.exceptions
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.fft import fft

from util import config


class AETrainer:
    def __init__(self, epochs: int = 25, lr: float = 1e-3, batch_size: int = 128):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

    def train(self, autoencoder: nn.Module, train_dataset: np.ndarray) -> nn.Module:
        device = config.DEVICE

        autoencoder = autoencoder.to(device)

        optimizer = optim.Adam(autoencoder.parameters(), lr=self.lr)

        train_dataset = np.abs(np.array([fft(i) for i in train_dataset]))

        train_loader = DataLoader(
            TensorDataset(torch.tensor(train_dataset)), batch_size=self.batch_size, shuffle=True, pin_memory=True
        )

        # print("AE training started")

        for epoch in range(self.epochs):
            loss = 0

            for input_batch in train_loader:
                input_batch = input_batch[0].to(device)
                optimizer.zero_grad()
                output_batch = autoencoder(input_batch)
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
