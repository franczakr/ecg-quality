import optuna
import torch
import torch.nn as nn


class AEMultiInputOptuna(nn.Module):

    def __init__(self, trial: optuna.Trial):
        super().__init__()

        loss_multiplier = [trial.suggest_int("loss_multiplier_0", 1, 10),
                           trial.suggest_int("loss_multiplier_1", 1, 10)]

        def loss(input, output):
            loss = 0
            loss_func = nn.MSELoss()
            for i in range(len(input)):
                loss += loss_multiplier[i] * loss_func(input[i], output[i])
            return loss

        self.signal_layer_out_features = trial.suggest_int("signal_layer_out_features", 10, 100)

        self.fft_layer_out_features = trial.suggest_int("fft_layer_out_features", 10, 100)

        self.final_layer_in_features = self.signal_layer_out_features \
                                       + self.fft_layer_out_features \

        self.final_layer_out_features = trial.suggest_int("final_layer_out_features", 10, 200)

        self.loss_func = loss

        self.encoder_layer_signal = nn.Linear(in_features=100,
                                              out_features=self.signal_layer_out_features)

        self.encoder_layer_fft = nn.Linear(in_features=100,
                                             out_features=self.fft_layer_out_features)

        self.encoder_layer_final = nn.Linear(in_features=self.final_layer_in_features,
                                             out_features=self.final_layer_out_features)

        ##############################################

        self.decoder_layer_final = nn.Linear(in_features=self.final_layer_out_features,
                                             out_features=self.final_layer_in_features)

        self.decoder_layer_signal = nn.Linear(in_features=self.signal_layer_out_features,
                                              out_features=100)

        self.decoder_layer_fft = nn.Linear(in_features=self.fft_layer_out_features,
                                             out_features=100)

    def forward(self, signal, fft):
        # Encoder

        encoded_signal = self.encoder_layer_signal(signal)
        encoded_fft = self.encoder_layer_fft(fft)

        concatenated = torch.cat((encoded_signal, encoded_fft), dim=1)

        encoded = self.encoder_layer_final(concatenated)

        # Decoder

        reconstructed = self.decoder_layer_final(encoded)

        r_signal, r_fft = torch.split(reconstructed, [self.signal_layer_out_features, self.fft_layer_out_features], dim=1)

        reconstructed_signal = self.decoder_layer_signal(r_signal)
        reconstructed_fft = self.decoder_layer_fft(r_fft)

        return reconstructed_signal, reconstructed_fft
