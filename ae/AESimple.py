import torch
import torch.nn as nn


class AESimple(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.loss_func = nn.MSELoss()

        self.encoder_layer_1 = nn.Linear(in_features=100,
                                         out_features=125)

        self.decoder_layer_1 = nn.Linear(in_features=125,
                                         out_features=100)

        self.elu = torch.nn.ELU()

    def forward(self, features):
        encoded = self.encoder_layer_1(features)

        reconstructed = self.decoder_layer_1(encoded)

        return reconstructed