import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from classification.abstract.Autoencoder import Autoencoder
from classification.abstract.Classifier import Classifier
from classification.abstract.Trainer import Trainer
from util.config import PREPROCESSED_FILENAME_TRAIN, PREPROCESSED_FILENAME, PREPROCESSED_FILENAME_HR_TRAIN, \
    PREPROCESSED_FILENAME_HR
from util import persistence, data_reader
from util.data_reader import BAD_QUALITY, GOOD_QUALITY
from util.persistence import load_model, save_model


class EcgClassifier:

    def load_fragments(self, use_hearth_rate: bool):
        if use_hearth_rate:
            slices_train, _ = persistence.load_object(PREPROCESSED_FILENAME_HR_TRAIN)
            slices, classes = persistence.load_object(PREPROCESSED_FILENAME_HR)
        else:
            slices_train, _ = persistence.load_object(PREPROCESSED_FILENAME_TRAIN)
            slices, classes = persistence.load_object(PREPROCESSED_FILENAME)

        classes = np.array([BAD_QUALITY if x == '~' else GOOD_QUALITY for x in classes])

        slices_gq = [s for s in list(zip(slices, classes)) if s[1] == data_reader.GOOD_QUALITY]
        slices_bq = [s for s in list(zip(slices, classes)) if s[1] == data_reader.BAD_QUALITY]

        gq_train_classifier, gq_test = slices_gq[:1000], slices_gq[1000:3000]
        bq_train_classifier, bq_test = slices_bq[:1000], slices_bq[1000:3000]

        slices_for_train_classifier, classes_for_train_classifier = list(zip(*(gq_train_classifier + bq_train_classifier)))
        slices_test, classes_test = list(zip(*(gq_test + bq_test)))

        return slices_train, slices_for_train_classifier, classes_for_train_classifier, slices_test, classes_test

    def train_test(self, autoencoder: Autoencoder, classifier: Classifier, trainer: Trainer, use_hearth_rate: bool=False):
        slices_train, slices_train_classifier, classes_train_classifier, slices_test, classes_test = self.load_fragments(
            use_hearth_rate)

        if trainer is not None:
            autoencoder = trainer.train(autoencoder, slices_train)
            save_model(autoencoder)
        else:
            autoencoder = load_model(autoencoder)

        classifier.train(autoencoder, slices_train_classifier, classes_train_classifier)

        predictions = classifier.classify(autoencoder, slices_test)

        print("CLASSIER RESULTS:")
        TN, FP, FN, TP = confusion_matrix(classes_test, predictions).ravel()
        print(f'Sensitivity: {TP / (TP + FN)}')
        print(f'Specificity: {TN / (TN + FP)}')
        print(f'Accuracy: {accuracy_score(classes_test, predictions)}')
        print(f'F1: {f1_score(classes_test, predictions)}')


