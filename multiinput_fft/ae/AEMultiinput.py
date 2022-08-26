import torch
import torch.nn as nn


class AEMultiInput(nn.Module):

    def __init__(self):
        super().__init__()

        loss_multiplier = [2, 6]

        def loss(input, output):
            loss = 0
            loss_func = nn.L1Loss()
            for i in range(len(input)):
                loss += loss_multiplier[i] * loss_func(input[i], output[i])
            return loss

        self.loss_func = loss

        self.encoder_layer_signal = nn.Linear(in_features=100,
                                              out_features=55)

        self.encoder_layer_fft = nn.Linear(in_features=100,
                                             out_features=12)

        self.encoder_layer_final = nn.Linear(in_features=67,
                                             out_features=139)

        ##############################################

        self.decoder_layer_final = nn.Linear(in_features=139,
                                             out_features=67)

        self.decoder_layer_signal = nn.Linear(in_features=55,
                                              out_features=100)

        self.decoder_layer_fft = nn.Linear(in_features=12,
                                             out_features=100)

    def forward(self, signal, fft):

        # Encoder

        encoded_signal = self.encoder_layer_signal(signal)
        encoded_fft = self.encoder_layer_fft(fft)

        concatenated = torch.cat((encoded_signal, encoded_fft), dim=1)

        encoded = self.encoder_layer_final(concatenated)

        # Decoder

        reconstructed = self.decoder_layer_final(encoded)

        r_signal, r_fft = torch.split(reconstructed, [55, 12], dim=1)

        reconstructed_signal = self.decoder_layer_signal(r_signal)
        reconstructed_fft = self.decoder_layer_fft(r_fft)

        return reconstructed_signal, reconstructed_fft
