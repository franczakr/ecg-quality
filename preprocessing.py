import argparse

import numpy as np

from util import persistence, data_reader
from util.config import PREPROCESSED_FILENAME_HR, PREPROCESSED_FILENAME, PREPROCESSED_FILENAME_TRAIN, PREPROCESSED_FILENAME_HR_TRAIN
from util.data_reader import load_train_data_and_labels
from util.SignalPreprocessor import SignalPreprocessor


def main(train_data_path: str, use_hearth_rate: bool):
    train_data_with_labels = load_train_data_and_labels(train_data_path)

    print("Loaded train dataset")

    train_data = [item[1] for item in train_data_with_labels]
    train_labels = [item[2] for item in train_data_with_labels]
    slices, classes = SignalPreprocessor().split_into_slices(train_data, train_labels, use_hearth_rate)
    print('Split dataset to fragments')

    slices_gq = [s for s in list(zip(slices, classes)) if s[1] == data_reader.GOOD_QUALITY]
    slices_bq = [s for s in list(zip(slices, classes)) if s[1] == data_reader.BAD_QUALITY]

    gq_test, gq_train = slices_gq[:1000], slices_gq[1000:]
    bq_test = slices_bq[:1000]

    slices_train, classes_train = list(zip(*gq_train))
    slices_test, classes_test = list(zip(*(gq_test + bq_test)))

    persistence.save_object(PREPROCESSED_FILENAME_HR_TRAIN if use_hearth_rate else PREPROCESSED_FILENAME_TRAIN, (slices_train, classes_train))
    persistence.save_object(PREPROCESSED_FILENAME_HR if use_hearth_rate else PREPROCESSED_FILENAME, (slices_test, classes_test))
    print("Saved preprocessed data")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path')
    parser.add_argument('--use_hearth_rate', action="store_true")
    args = parser.parse_args()

    main(args.train_data_path, args.use_hearth_rate)
