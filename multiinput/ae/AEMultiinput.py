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

        self.encoder_layer_part1 = nn.Linear(in_features=25,
                                             out_features=20)

        self.encoder_layer_part2 = nn.Linear(in_features=25,
                                             out_features=20)

        self.encoder_layer_part3 = nn.Linear(in_features=25,
                                             out_features=20)

        self.encoder_layer_part4 = nn.Linear(in_features=25,
                                             out_features=20)

        self.encoder_layer_final = nn.Linear(in_features=200,
                                             out_features=100)

        ##############################################

        self.decoder_layer_final = nn.Linear(in_features=100,
                                             out_features=200)

        self.decoder_layer_signal = nn.Linear(in_features=120,
                                              out_features=100)

        self.decoder_layer_part1 = nn.Linear(in_features=20,
                                             out_features=25)

        self.decoder_layer_part2 = nn.Linear(in_features=20,
                                             out_features=25)

        self.decoder_layer_part3 = nn.Linear(in_features=20,
                                             out_features=25)

        self.decoder_layer_part4 = nn.Linear(in_features=20,
                                             out_features=25)

    def forward(self, signal, part1, part2, part3, part4):

        # Encoder

        encoded_signal = self.encoder_layer_signal(signal)
        encoded_part1 = self.encoder_layer_part1(part1)
        encoded_part2 = self.encoder_layer_part2(part2)
        encoded_part3 = self.encoder_layer_part3(part3)
        encoded_part4 = self.encoder_layer_part4(part4)

        concatenated = torch.cat((encoded_signal, encoded_part1, encoded_part2, encoded_part3, encoded_part4), dim=1)

        encoded = self.encoder_layer_final(concatenated)

        # Decoder

        reconstructed = self.decoder_layer_final(encoded)

        r_signal, r_part1, r_part2, r_part3, r_part4 = torch.split(reconstructed, [120, 20, 20, 20, 20], dim=1)

        reconstructed_signal = self.decoder_layer_signal(r_signal)
        reconstructed_part1 = self.decoder_layer_part1(r_part1)
        reconstructed_part2 = self.decoder_layer_part2(r_part2)
        reconstructed_part3 = self.decoder_layer_part3(r_part3)
        reconstructed_part4 = self.decoder_layer_part4(r_part4)

        return reconstructed_signal, reconstructed_part1, reconstructed_part2, reconstructed_part3, reconstructed_part4
