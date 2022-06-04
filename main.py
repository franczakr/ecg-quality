import argparse

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

from basic import AETrainer
from basic.ae.AESimple import AESimple
from basic.classifier.simple_classifier import SimpleClassifier
from basic.AETrainer import AETrainer
from preprocess import PREPRECESSED_FILENAME
from util import data_reader, persistence
from util.data_reader import BAD_QUALITY, GOOD_QUALITY
from util.persistence import save_model, load_model


def main(train: bool):
    slices, classes = persistence.load_object(PREPRECESSED_FILENAME)
    # slices, classes = data_reader.load_slices_from_csv()
    classes = np.array([BAD_QUALITY if x == '~' else GOOD_QUALITY for x in classes])

    slices_gq = [s for s in list(zip(slices, classes)) if s[1] == data_reader.GOOD_QUALITY]
    slices_bq = [s for s in list(zip(slices, classes)) if s[1] == data_reader.BAD_QUALITY]

    autoencoder = AESimple()

    gq_train_ae, slices_gq = train_test_split(slices_gq, train_size=0.4)
    gq_train_ae = np.asarray([gq[0] for gq in gq_train_ae])
    if train:
        autoencoder = AETrainer.train(autoencoder, gq_train_ae, epochs=30)
        save_model(autoencoder)
    else:
        autoencoder = load_model(autoencoder)

    classifier = SimpleClassifier()

    gq_train_classifier, gq_test = train_test_split(slices_gq, train_size=2000)
    bq_train_classifier, bq_test = train_test_split(slices_bq, train_size=2000)

    slices_classes_train_classifier = gq_train_classifier + bq_train_classifier
    slices_for_train_classifier = np.asarray([s[0] for s in slices_classes_train_classifier])
    classes_for_train_classifier = np.asarray([s[1] for s in slices_classes_train_classifier])
    classifier.train(autoencoder, slices_for_train_classifier, classes_for_train_classifier)

    slices_classes_test = gq_test + bq_test
    slices_test = np.asarray([s[0] for s in slices_classes_test])
    classes_test = np.asarray([s[1] for s in slices_classes_test])
    predictions = classifier.classify(autoencoder, slices_test)

    print("CLASSIER RESULTS:")
    TN, FP, FN, TP = confusion_matrix(classes_test, predictions).ravel()
    print('True Positive(TP)  = ', TP)
    print('False Positive(FP) = ', FP)
    print('True Negative(TN)  = ', TN)
    print('False Negative(FN) = ', FN)
    print(f'Accuracy: {accuracy_score(classes_test, predictions)}')
    print(f'Precision: {precision_score(classes_test, predictions)}')
    print(f'Recall: {recall_score(classes_test, predictions)}')
    print(f'F1: {f1_score(classes_test, predictions)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-train', action="store_true")
    args = parser.parse_args()

    train = not args.no_train

    main(train)
