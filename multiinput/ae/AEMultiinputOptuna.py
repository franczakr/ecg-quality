import optuna
import torch
import torch.nn as nn


class AEMultiInputOptuna(nn.Module):

    def __init__(self, trial: optuna.Trial):
        super().__init__()

        loss_multiplier = [trial.suggest_int("loss_multiplier_1", 1, 10),
                           trial.suggest_int("loss_multiplier_2", 1, 10),
                           trial.suggest_int("loss_multiplier_3", 1, 10),
                           trial.suggest_int("loss_multiplier_4", 1, 10)]

        def loss(input, output):
            loss = 0
            loss_func = nn.L1Loss()
            for i in range(len(input)):
                loss += loss_multiplier[i] * loss_func(input[i], output[i])
            return loss

        self.loss_func = loss

        # self.signal_layer_out_features = trial.suggest_int("signal_layer_out_features", 10, 100)

        self.part1_layer_out_features = trial.suggest_int("parts_layer_out_features", 2,  25)
        self.part2_layer_out_features = self.part1_layer_out_features
        self.part3_layer_out_features = self.part1_layer_out_features
        self.part4_layer_out_features = self.part1_layer_out_features

        self.final_layer_in_features = self.part1_layer_out_features \
                                       + self.part2_layer_out_features \
                                       + self.part3_layer_out_features \
                                       + self.part4_layer_out_features

        self.final_layer_out_features = trial.suggest_int("final_layer_out_features", 10, 200)

        self.encoder_layer_part1 = nn.Linear(in_features=25,
                                             out_features=self.part1_layer_out_features)

        self.encoder_layer_part2 = nn.Linear(in_features=25,
                                             out_features=self.part2_layer_out_features)

        self.encoder_layer_part3 = nn.Linear(in_features=25,
                                             out_features=self.part3_layer_out_features)

        self.encoder_layer_part4 = nn.Linear(in_features=25,
                                             out_features=self.part4_layer_out_features)

        self.encoder_layer_final = nn.Linear(in_features=self.final_layer_in_features,
                                             out_features=self.final_layer_out_features)

        ##############################################

        self.decoder_layer_final = nn.Linear(in_features=self.final_layer_out_features,
                                             out_features=self.final_layer_in_features)

        self.decoder_layer_part1 = nn.Linear(in_features=self.part1_layer_out_features,
                                             out_features=25)

        self.decoder_layer_part2 = nn.Linear(in_features=self.part2_layer_out_features,
                                             out_features=25)

        self.decoder_layer_part3 = nn.Linear(in_features=self.part3_layer_out_features,
                                             out_features=25)

        self.decoder_layer_part4 = nn.Linear(in_features=self.part4_layer_out_features,
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

        r_part1, r_part2, r_part3, r_part4 = torch.split(reconstructed, [self.part1_layer_out_features,
                                                                         self.part2_layer_out_features,
                                                                         self.part3_layer_out_features,
                                                                         self.part4_layer_out_features],
                                                         dim=1)

        reconstructed_part1 = self.decoder_layer_part1(r_part1)
        reconstructed_part2 = self.decoder_layer_part2(r_part2)
        reconstructed_part3 = self.decoder_layer_part3(r_part3)
        reconstructed_part4 = self.decoder_layer_part4(r_part4)

        return reconstructed_part1, reconstructed_part2, reconstructed_part3, reconstructed_part4
