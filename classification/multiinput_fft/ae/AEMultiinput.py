import torch
import torch.nn as nn

from classification.abstract.Autoencoder import Autoencoder


class AEMultiInput(Autoencoder):

    def __init__(self, loss_multiplier, signal_layer_width, fft_layer_width, final_layer_width):
        super().__init__()

        self.signal_layer_width = signal_layer_width
        self.fft_layer_width = fft_layer_width
        self.final_layer_width = final_layer_width

        def loss(input, output):
            loss = 0
            loss_func = nn.L1Loss()
            for i in range(len(input)):
                loss += loss_multiplier[i] * loss_func(input[i], output[i])
            return loss

        self.loss_func = loss

        self.encoder_layer_signal = nn.Linear(in_features=100,
                                              out_features=signal_layer_width)

        self.encoder_layer_fft = nn.Linear(in_features=100,
                                             out_features=fft_layer_width)

        self.encoder_layer_final = nn.Linear(in_features=signal_layer_width+fft_layer_width,
                                             out_features=final_layer_width)

        ##############################################

        self.decoder_layer_final = nn.Linear(in_features=final_layer_width,
                                             out_features=signal_layer_width+fft_layer_width)

        self.decoder_layer_signal = nn.Linear(in_features=signal_layer_width,
                                              out_features=100)

        self.decoder_layer_fft = nn.Linear(in_features=fft_layer_width,
                                             out_features=100)

    def forward(self, signal, fft):

        # Encoder

        encoded_signal = self.encoder_layer_signal(signal)
        encoded_fft = self.encoder_layer_fft(fft)

        concatenated = torch.cat((encoded_signal, encoded_fft), dim=1)

        encoded = self.encoder_layer_final(concatenated)

        # Decoder

        reconstructed = self.decoder_layer_final(encoded)

        r_signal, r_fft = torch.split(reconstructed, [self.signal_layer_width, self.fft_layer_width], dim=1)

        reconstructed_signal = self.decoder_layer_signal(r_signal)
        reconstructed_fft = self.decoder_layer_fft(r_fft)

        return reconstructed_signal, reconstructed_fft
