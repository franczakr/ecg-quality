import argparse

import numpy as np
import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from basic_fft import AETrainer
from basic_fft.ae.AESimpleOptuna import AESimpleOptuna
from basic_fft.classifier.simple_classifier import SimpleClassifier
from basic_fft.AETrainer import AETrainer
from preprocess import PREPROCESSED_FILENAME
from util import data_reader, persistence, ecg_classifier
from util.ecg_classifier import EcgClassifier
from util.data_reader import BAD_QUALITY, GOOD_QUALITY


class OptunaTrainer:

    def __init__(self, gq_train_ae, slices_for_train_classifier, classes_for_train_classifier, slices_test,
                 classes_test):
        self.classes_test = classes_test
        self.slices_test = slices_test
        self.classes_for_train_classifier = classes_for_train_classifier
        self.slices_for_train_classifier = slices_for_train_classifier
        self.gq_train_ae = gq_train_ae

    def train(self, trial: optuna.trial.Trial):
        autoencoder = AESimpleOptuna(trial)

        epochs = trial.suggest_int('epochs', 5, 100)
        lr = trial.suggest_float("lr", 0, 1e-1)
        batch_size = trial.suggest_categorical("batch_size", [32, 128])
        autoencoder = AETrainer(epochs=epochs, lr=lr, batch_size=batch_size).train(autoencoder, self.gq_train_ae)

        classifier = SimpleClassifier()
        classifier.train(autoencoder, self.slices_for_train_classifier, self.classes_for_train_classifier)

        predictions = classifier.classify(autoencoder, self.slices_test)

        accuracy = accuracy_score(self.classes_test, predictions)

        return accuracy


def main(use_hearth_rate, n_trials):
    slices_train, slices_train_classifier, classes_train_classifier, slices_test, classes_test = EcgClassifier().load_fragments(use_hearth_rate)

    t = OptunaTrainer(slices_train, slices_train_classifier, classes_train_classifier, slices_test, classes_test)

    study = optuna.create_study(direction='maximize')
    study.optimize(t.train, n_trials=n_trials)
    print(optuna.importance.get_param_importances(study))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-hearth-rate', action="store_true")
    args = parser.parse_args()

    main(args.use_hearth_rate, n_trials=200)
