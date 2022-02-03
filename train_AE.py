import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from AE import AE


def train_ae(train_dataset: np.ndarray, epochs: int):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AE(input_shape=100).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.MSELoss()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )

    results = []

    for epoch in range(epochs):
        loss = 0
        for input_batch in train_loader:

            input_batch = input_batch.to(device)

            optimizer.zero_grad()

            output_batch = model(input_batch)

            train_loss = criterion(output_batch, input_batch)

            results.append([input_batch, output_batch, train_loss])

            train_loss.backward()

            optimizer.step()

            loss += train_loss.item()

        loss = loss / len(train_loader)

        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

    print("ok")