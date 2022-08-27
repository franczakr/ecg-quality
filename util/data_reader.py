import csv
from os import listdir
from os.path import isfile, join
from typing import List, Tuple

import numpy as np
import scipy.io

GOOD_QUALITY = 0
BAD_QUALITY = 1


def load_train_data_and_labels(train_data_path: str, limit: int = None) -> List[Tuple[str, np.ndarray, int]]:
    train_data = {}
    training_files = [f
                      for f in listdir(train_data_path)
                      if isfile(join(train_data_path, f)) and f.endswith('.mat')]

    if limit is not None:
        training_files = training_files[:limit]

    for filename in training_files:
        filename_without_extension = filename[:-4]
        train_data[filename_without_extension] = scipy.io.loadmat(join(train_data_path, filename))['val'][0]

    labels_file = join(train_data_path, 'REFERENCE.csv')

    with open(labels_file, newline='') as csvfile:
        train_labels = dict(csv.reader(csvfile))

    train = [(name, train_data[name], train_labels[name]) for name in train_data.keys()]

    return train


def load_slices_from_csv():
    slices_gq, classes_gq = _load_slices_from_csv('data/slices_gq/csv', 'N')
    slices_bq, classes_bq = _load_slices_from_csv('data/slices_bq/csv', '~')
    slices_af, classes_af = _load_slices_from_csv('data/slices_af/csv', 'A')
    slices_others, classes_others = _load_slices_from_csv('data/slices_others/csv', 'O')

    slices = np.concatenate([slices_gq, slices_bq, slices_af, slices_others])
    classes = np.concatenate([classes_gq, classes_bq, classes_af, classes_others])

    return slices, classes


def _load_slices_from_csv(path: str, classs: str):
    files_to_load = [f
                     for f in listdir(path)
                     if isfile(join(path, f)) and f.endswith('.csv')]

    slices = []

    for file in files_to_load:
        slices.append(np.loadtxt(path + '/' + file, delimiter="\n", dtype='float32'))

    slices = np.asarray(slices)
    classes = np.full(len(slices), classs)

    return slices, classes
