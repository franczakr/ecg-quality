import argparse

import numpy as np
import optuna
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

from multiinput_fft import AETrainerMultiInput
from multiinput_fft.ae.AEMultiinput import AEMultiInput
from multiinput_fft.ae.AEMultiinputOptuna import AEMultiInputOptuna
from multiinput_fft.classifier.simple_classifier_multi_input import SimpleClassifierMultiInput
from multiinput_fft.AETrainerMultiInput import AETrainerMultiInput
from util import data_reader
from util.data_reader import BAD_QUALITY, GOOD_QUALITY
from util.persistence import save_model, load_model


class OptunaTrainer:

    def __init__(self, gq_train_ae, slices_for_train_classifier, classes_for_train_classifier, slices_test,
                 classes_test):
        self.classes_test = classes_test
        self.slices_test = slices_test
        self.classes_for_train_classifier = classes_for_train_classifier
        self.slices_for_train_classifier = slices_for_train_classifier
        self.gq_train_ae = gq_train_ae

    def train(self, trial: optuna.trial.Trial):
        autoencoder = AEMultiInputOptuna(trial)

        epochs = trial.suggest_int('epochs', 5, 100)
        lr = trial.suggest_float("lr", 0, 1e-1)
        batch_size = trial.suggest_categorical("batch_size", [32, 128])
        autoencoder = AETrainerMultiInput.train(autoencoder, self.gq_train_ae, epochs=epochs, lr=lr, batch_size=batch_size)

        classifier = SimpleClassifierMultiInput()
        classifier.train(autoencoder, self.slices_for_train_classifier, self.classes_for_train_classifier)

        predictions = classifier.classify(autoencoder, self.slices_test)

        accuracy = accuracy_score(self.classes_test, predictions)

        return accuracy


def main(n_trials):
    slices, classes = data_reader.load_slices_from_csv()
    classes = np.array([BAD_QUALITY if x == '~' else GOOD_QUALITY for x in classes])

    slices_gq = [s for s in list(zip(slices, classes)) if s[1] == data_reader.GOOD_QUALITY]
    slices_bq = [s for s in list(zip(slices, classes)) if s[1] == data_reader.BAD_QUALITY]

    gq_train_ae, slices_gq = train_test_split(slices_gq, train_size=0.4)
    gq_train_ae = np.asarray([gq[0] for gq in gq_train_ae])
    gq_train_classifier, gq_test = train_test_split(slices_gq, train_size=0.3)
    bq_train_classifier, bq_test = train_test_split(slices_bq, train_size=0.3)
    slices_classes_train_classifier = gq_train_classifier + bq_train_classifier
    slices_for_train_classifier = np.asarray([s[0] for s in slices_classes_train_classifier])
    classes_for_train_classifier = np.asarray([s[1] for s in slices_classes_train_classifier])

    slices_classes_test = gq_test + bq_test
    slices_test = np.asarray([s[0] for s in slices_classes_test])
    classes_test = np.asarray([s[1] for s in slices_classes_test])

    t = OptunaTrainer(gq_train_ae, slices_for_train_classifier, classes_for_train_classifier, slices_test, classes_test)

    study = optuna.create_study(direction='maximize')
    study.optimize(t.train, n_trials=n_trials)
    print(optuna.importance.get_param_importances(study))


if __name__ == '__main__':
    main(n_trials=1000)
