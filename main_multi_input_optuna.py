import argparse

import optuna
from sklearn.metrics import accuracy_score

from classification.multiinput.MultiInputAETrainer import AETrainerMultiInput
from classification.multiinput.ae.AEMultiinputOptuna import AEMultiInputOptuna
from classification.multiinput.classifier.simple_classifier_multi_input import SimpleClassifierMultiInput
from util.ecg_classifier import EcgClassifier


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
        autoencoder = AETrainerMultiInput(epochs=epochs, lr=lr, batch_size=batch_size).train(autoencoder,
                                                                                             self.gq_train_ae)

        classifier = SimpleClassifierMultiInput()
        classifier.train(autoencoder, self.slices_for_train_classifier, self.classes_for_train_classifier)

        predictions = classifier.classify(autoencoder, self.slices_test)

        accuracy = accuracy_score(self.classes_test, predictions)

        return accuracy


def main(use_hearth_rate, n_trials):
    slices_train, slices_train_classifier, classes_train_classifier, slices_test, classes_test = EcgClassifier().load_fragments(
        use_hearth_rate)

    t = OptunaTrainer(slices_train, slices_train_classifier, classes_train_classifier, slices_test, classes_test)

    study = optuna.create_study(direction='maximize')
    study.optimize(t.train, n_trials=n_trials)
    print(optuna.importance.get_param_importances(study))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-hearth-rate', action="store_true")
    args = parser.parse_args()

    main(args.use_hearth_rate, n_trials=100)
