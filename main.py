import argparse
import random

import numpy
import numpy as np

from ae import train_AE
from ae.AESimple import AESimple
from classifier.classifier import Classifier
from preprocess import PREPRECESSED_FILENAME
from util import persistence, data_reader
from util.persistence import save_model, load_model


def main(train: bool):
    slices, classes = persistence.load_object(PREPRECESSED_FILENAME)
    autoencoder = AESimple()

    if train:
        slices_gq = [gq[0] for gq in list(zip(slices, classes)) if gq[1] == data_reader.GOOD_QUALITY]
        slices_gq = numpy.array(slices_gq[:10000])  # TODO change to split data to train and test
        autoencoder = train_AE.train_ae(autoencoder, slices_gq, epochs=20)
        save_model(autoencoder)
    else:
        autoencoder = load_model(autoencoder)

    classifier = Classifier()

    slices_for_train_classifier, classes_for_train_classifier = _get_random_data_for_train(slices,
                                                                                           classes)  # TODO change to split data to train and test

    classifier.train(autoencoder, slices_for_train_classifier, classes_for_train_classifier)

    # TODO add test data accuracy and f1 calculation


"""
    Chooses randomly count Good quality and Bad quality samples
"""
def _get_random_data_for_train(slices: np.ndarray, classes: np.ndarray, count=1000) -> (np.ndarray, np.ndarray):
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
