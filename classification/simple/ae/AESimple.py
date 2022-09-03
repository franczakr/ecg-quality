import torch.nn as nn

from classification.abstract.Autoencoder import Autoencoder


class AESimple(Autoencoder):

    def __init__(self, hidden_layer_width: int):
        super().__init__()

        self.loss_func = nn.L1Loss()

        self.encoder_layer_1 = nn.Linear(in_features=100,
                                         out_features=hidden_layer_width)

        self.decoder_layer_1 = nn.Linear(in_features=hidden_layer_width,
                                         out_features=100)

    def forward(self, features):
        encoded = self.encoder_layer_1(features)

        reconstructed = self.decoder_layer_1(encoded)

        return reconstructed
