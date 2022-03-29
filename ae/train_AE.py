import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_ae(autoencoder: nn.Module, train_dataset: np.ndarray, epochs: int) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = autoencoder.to(device)

    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_dataset)), batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )

    print("AE training started")

    for epoch in range(epochs):
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

        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

    print("ok")

    return autoencoder
