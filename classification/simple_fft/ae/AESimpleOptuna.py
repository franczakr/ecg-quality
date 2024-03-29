import optuna
import torch.nn as nn

from classification.abstract.Autoencoder import Autoencoder


class AESimpleOptuna(Autoencoder):

    def __init__(self, trial: optuna.Trial):
        super().__init__()

        self.loss_func = nn.L1Loss()

        encoder_layers = []
        decoder_layers = []

        n_layers = 1

        in_features = 100
        for i in range(n_layers):
            out_features = trial.suggest_int(f'n_units_l{i}', 10, 100)

            encoder_layers.append(nn.Linear(in_features, out_features))

            decoder_layers.insert(0, nn.Linear(out_features, in_features))
            in_features = out_features

        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, features):
        encoded = features
        for encoder_layer in self.encoder_layers:
            encoded = encoder_layer(encoded)

        reconstructed = encoded
        for decoder_layer in self.decoder_layers:
            reconstructed = decoder_layer(reconstructed)

        return reconstructed
