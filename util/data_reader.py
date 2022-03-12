import csv
from os import listdir
from os.path import isfile, join
from typing import List, Tuple

import numpy as np
import scipy.io

TRAIN_DATA_PATH = r"C:\Users\franc\Desktop\training2017"

GOOD_QUALITY = 0
BAD_QUALITY = 1


def load_train_data_and_labels(limit: int = None) -> List[Tuple[str, np.ndarray, int]]:
    train_data = {}
    training_files = [f
                      for f in listdir(TRAIN_DATA_PATH)
                      if isfile(join(TRAIN_DATA_PATH, f)) and f.endswith('.mat')]

    if limit is not None:
        training_files = training_files[:limit]

    for filename in training_files:
        filename_without_extension = filename[:-4]
        train_data[filename_without_extension] = scipy.io.loadmat(join(TRAIN_DATA_PATH, filename))['val'][0]

    labels_file = join(TRAIN_DATA_PATH, 'REFERENCE.csv')

    with open(labels_file, newline='') as csvfile:
        train_labels = dict(csv.reader(csvfile))

    train = [(name, train_data[name], train_labels[name]) for name in train_data.keys()]

    return train
