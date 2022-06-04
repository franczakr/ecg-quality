import torch
import torch.nn as nn


class AEMultiInput(nn.Module):

    def __init__(self):
        super().__init__()

        def loss(input, output):
            loss = 0
            loss_func = nn.MSELoss()
            for i in range(len(input)):
                loss += loss_func(input[i], output[i])
            return loss

        self.loss_func = loss

        self.encoder_layer_signal = nn.Linear(in_features=100,
                                              out_features=120)

        self.encoder_layer_fft = nn.Linear(in_features=100,
                                             out_features=80)

        self.encoder_layer_final = nn.Linear(in_features=200,
                                             out_features=100)

        ##############################################

        self.decoder_layer_final = nn.Linear(in_features=100,
                                             out_features=200)

        self.decoder_layer_signal = nn.Linear(in_features=120,
                                              out_features=100)

        self.decoder_layer_fft = nn.Linear(in_features=80,
                                             out_features=100)

    def forward(self, signal, fft):

        # Encoder

        encoded_signal = self.encoder_layer_signal(signal)
        encoded_fft = self.encoder_layer_fft(fft)

        concatenated = torch.cat((encoded_signal, encoded_fft), dim=1)

        encoded = self.encoder_layer_final(concatenated)

        # Decoder

        reconstructed = self.decoder_layer_final(encoded)

        r_signal, r_fft = torch.split(reconstructed, [120, 80], dim=1)

        reconstructed_signal = self.decoder_layer_signal(r_signal)
        reconstructed_fft = self.decoder_layer_fft(r_fft)

        return reconstructed_signal, reconstructed_fft
