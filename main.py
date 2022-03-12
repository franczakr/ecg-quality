import argparse
import random

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score

from ae import train_AE
from ae.AESimple import AESimple
from classifier.classifier import Classifier
from preprocess import PREPRECESSED_FILENAME
from util import persistence, data_reader
from util.data_reader import BAD_QUALITY, GOOD_QUALITY
from util.persistence import save_model, load_model


def main(train: bool):
    slices, classes = persistence.load_object(PREPRECESSED_FILENAME)

    classes = np.array([BAD_QUALITY if x == '~' else GOOD_QUALITY for x in classes])

    autoencoder = AESimple()

    if train:
        slices_gq = [gq[0] for gq in list(zip(slices, classes)) if gq[1] == data_reader.GOOD_QUALITY]
        slices_gq = np.array(slices_gq[:10000])  # TODO change to split data to train and test
        autoencoder = train_AE.train_ae(autoencoder, slices_gq, epochs=20)
        save_model(autoencoder)
    else:
        autoencoder = load_model(autoencoder)

    classifier = Classifier()

    slices_for_train_classifier, classes_for_train_classifier = _get_random_data(slices,
                                                                                 classes)  # TODO change to split data to train and test

    classifier.train(autoencoder, slices_for_train_classifier, classes_for_train_classifier)

    slices, classes = _get_random_data(slices, classes, count=3000)

    predictions = classifier.classify(autoencoder, slices)

    print("CLASSIER RESULTS:")
    TN, FP, FN, TP = confusion_matrix(classes, predictions).ravel()
    print('True Positive(TP)  = ', TP)
    print('False Positive(FP) = ', FP)
    print('True Negative(TN)  = ', TN)
    print('False Negative(FN) = ', FN)
    print(f'Accuracy: {accuracy_score(classes, predictions)}')
    print(f'Precision: {precision_score(classes, predictions)}')
    print(f'Recall: {recall_score(classes, predictions)}')
    print(f'F1: {f1_score(classes, predictions)}')


"""
    Chooses randomly count Good quality and Bad quality samples
"""


def _get_random_data(slices: np.ndarray, classes: np.ndarray, count=1000) -> (np.ndarray, np.ndarray):
    slices_gq = [gq for gq in list(zip(slices, classes)) if gq[1] == data_reader.GOOD_QUALITY]
    slices_bq = [bq for bq in list(zip(slices, classes)) if bq[1] == data_reader.BAD_QUALITY]
    slices_gq_random = random.sample(slices_gq, count)
    slices_bq_random = random.sample(slices_bq, count)
    data = slices_gq_random + slices_bq_random
    random.shuffle(data)
    slices = np.array([x[0] for x in data])
    classes = np.array([x[1] for x in data])
    return slices, classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-train', action="store_true")
    args = parser.parse_args()

    train = not args.no_train

    main(train)
