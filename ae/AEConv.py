import torch
import torch.nn as nn


class AEConv(nn.Module):

    # noinspection PyTypeChecker
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_func = nn.MSELoss()

        self.encoder_layer_1 = nn.Conv1d(in_channels=1,
                                         out_channels=32,
                                         kernel_size=11,
                                         stride=1,
                                         padding=5)

        self.encoder_layer_2 = nn.MaxPool1d(3,
                                            return_indices=True)

        self.encoder_layer_3 = nn.Conv1d(in_channels=32,
                                         out_channels=32,
                                         kernel_size=7,
                                         stride=1,
                                         padding=3)

        self.encoder_layer_4 = nn.Conv1d(in_channels=32,
                                         out_channels=64,
                                         kernel_size=7,
                                         stride=1,
                                         padding=3)

        self.encoder_layer_5 = nn.MaxPool1d(3,
                                            return_indices=True)

        self.encoder_layer_6 = nn.Conv1d(in_channels=64,
                                         out_channels=64,
                                         kernel_size=5,
                                         stride=1,
                                         padding=2)

        self.encoder_layer_7 = nn.Conv1d(in_channels=64,
                                         out_channels=64,
                                         kernel_size=5,
                                         stride=1,
                                         padding=2)

        self.decoder_layer_1 = nn.ConvTranspose1d(in_channels=32,
                                                  out_channels=1,
                                                  kernel_size=11,
                                                  stride=1,
                                                  padding=5)

        self.decoder_layer_2 = nn.MaxUnpool1d(3)

        self.decoder_layer_3 = nn.ConvTranspose1d(in_channels=32,
                                                  out_channels=32,
                                                  kernel_size=7,
                                                  stride=1,
                                                  padding=3)

        self.decoder_layer_4 = nn.ConvTranspose1d(in_channels=64,
                                                  out_channels=32,
                                                  kernel_size=7,
                                                  stride=1,
                                                  padding=3)

        self.decoder_layer_5 = nn.MaxUnpool1d(3)

        self.decoder_layer_6 = nn.ConvTranspose1d(in_channels=64,
                                                  out_channels=64,
                                                  kernel_size=5,
                                                  stride=1,
                                                  padding=2)

        self.decoder_layer_7 = nn.ConvTranspose1d(in_channels=64,
                                                  out_channels=64,
                                                  kernel_size=5,
                                                  stride=1,
                                                  padding=2)

        self.elu = torch.nn.ELU()

    def forward(self, features):
        features = torch.unsqueeze(features, 1)

        encoded = self.encoder_layer_1(features)

        encoded, indices_2 = self.encoder_layer_2(encoded)

        encoded = self.encoder_layer_3(encoded)
        encoded = self.elu(encoded)

        encoded = self.encoder_layer_4(encoded)
        encoded = self.elu(encoded)

        encoded, indices_5 = self.encoder_layer_5(encoded)

        encoded = self.encoder_layer_6(encoded)
        encoded = self.elu(encoded)

        encoded = self.encoder_layer_7(encoded)
        encoded = self.elu(encoded)

        reconstructed = self.decoder_layer_7(encoded)
        reconstructed = self.elu(reconstructed)

        reconstructed = self.decoder_layer_6(reconstructed)
        reconstructed = self.elu(reconstructed)

        reconstructed = self.decoder_layer_5(reconstructed, indices_5)

        reconstructed = self.decoder_layer_4(reconstructed)
        reconstructed = self.elu(reconstructed)

        reconstructed = self.decoder_layer_3(reconstructed)
        reconstructed = self.elu(reconstructed)

        reconstructed = self.decoder_layer_2(reconstructed, indices_2, features.size())

        reconstructed = self.decoder_layer_1(reconstructed)

        reconstructed = torch.squeeze(reconstructed, 1)

        return reconstructed
