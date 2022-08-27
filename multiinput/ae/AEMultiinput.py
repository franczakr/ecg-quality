import torch
import torch.nn as nn


class AEMultiInput(nn.Module):

    def __init__(self, loss_multipliers, partial_layers_width, hidden_layer_width):
        super().__init__()

        self.partial_layers_width = partial_layers_width

        if len(loss_multipliers) != 4:
            raise Exception('Wrong loss_multipliers size, should be 4')

        def loss(input, output):
            loss = 0
            loss_func = nn.L1Loss()
            for i in range(len(input)):
                loss += loss_multipliers[i] * loss_func(input[i], output[i])
            return loss

        self.loss_func = loss

        self.encoder_layer_part1 = nn.Linear(in_features=25,
                                             out_features=partial_layers_width)

        self.encoder_layer_part2 = nn.Linear(in_features=25,
                                             out_features=partial_layers_width)

        self.encoder_layer_part3 = nn.Linear(in_features=25,
                                             out_features=partial_layers_width)

        self.encoder_layer_part4 = nn.Linear(in_features=25,
                                             out_features=partial_layers_width)

        self.encoder_layer_final = nn.Linear(in_features=8,
                                             out_features=hidden_layer_width)

        ##############################################

        self.decoder_layer_final = nn.Linear(in_features=hidden_layer_width,
                                             out_features=8)

        self.decoder_layer_part1 = nn.Linear(in_features=partial_layers_width,
                                             out_features=25)

        self.decoder_layer_part2 = nn.Linear(in_features=partial_layers_width,
                                             out_features=25)

        self.decoder_layer_part3 = nn.Linear(in_features=partial_layers_width,
                                             out_features=25)

        self.decoder_layer_part4 = nn.Linear(in_features=partial_layers_width,
                                             out_features=25)

    def forward(self, part1, part2, part3, part4):

        # Encoder
        encoded_part1 = self.encoder_layer_part1(part1)
        encoded_part2 = self.encoder_layer_part2(part2)
        encoded_part3 = self.encoder_layer_part3(part3)
        encoded_part4 = self.encoder_layer_part4(part4)

        concatenated = torch.cat((encoded_part1, encoded_part2, encoded_part3, encoded_part4), dim=1)

        encoded = self.encoder_layer_final(concatenated)

        # Decoder

        reconstructed = self.decoder_layer_final(encoded)

        r_part1, r_part2, r_part3, r_part4 = torch.split(reconstructed, [self.partial_layers_width]*4, dim=1)

        reconstructed_part1 = self.decoder_layer_part1(r_part1)
        reconstructed_part2 = self.decoder_layer_part2(r_part2)
        reconstructed_part3 = self.decoder_layer_part3(r_part3)
        reconstructed_part4 = self.decoder_layer_part4(r_part4)

        return reconstructed_part1, reconstructed_part2, reconstructed_part3, reconstructed_part4
