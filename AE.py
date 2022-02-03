import torch.nn as nn
import torch

class AE(nn.Module):


    def __init__(self, **kwargs):
        super().__init__()
        self.current_epoch = 0

        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=125
        )
        # self.encoder_output_layer = nn.Linear(
        #     in_features=125, out_features=125
        # )
        # self.decoder_hidden_layer = nn.Linear(
        #     in_features=125, out_features=125
        # )
        self.decoder_output_layer = nn.Linear(
            in_features=125, out_features=kwargs["input_shape"]
        )

    def update_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, features):
        encoded = self.encoder_hidden_layer(features)
        encoded = torch.relu(encoded)
        # encoded = self.encoder_output_layer(encoded)
        # encoded = torch.relu(encoded)

        # reconstructed = self.decoder_hidden_layer(encoded)
        # reconstructed = torch.relu(reconstructed)
        # reconstructed = self.decoder_output_layer(reconstructed)
        reconstructed = self.decoder_output_layer(encoded)
        reconstructed = torch.relu(reconstructed)
        return reconstructed