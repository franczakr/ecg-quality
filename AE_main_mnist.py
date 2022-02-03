import torch
from AE import AE
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(input_shape=784).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    epochs = 2

    results = []

    for epoch in range(epochs):
        loss = 0
        for input, _ in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            input = input.view(-1, 784).to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            output = model(input)


            # compute training reconstruction loss
            train_loss = criterion(output, input)


            results.append([input, output, train_loss])

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()


        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

    a = results[-1]
    a[0] = a[0].reshape(-1, 28, 28)
    plt.imshow(a[0][0].detach())
    plt.show()

    a[1] = a[1].reshape(-1, 28, 28)
    plt.imshow(a[1][0].detach())
    plt.show()


if __name__ == '__main__':
    main()