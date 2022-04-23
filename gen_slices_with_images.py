import os
import random

import matplotlib.pyplot as plt
import numpy as np

import preprocess


def save_slices_and_images(slices, path):
    path = 'data/' + path
    if not os.path.exists(path):
        os.makedirs(path)
    for i, slice in enumerate(slices):
        name = path + '/' + str(i + 500).zfill(5)
        np.savetxt(name + '.csv', slice, delimiter="\n")
        plt.plot(slice)
        plt.savefig(name + '.png')
        plt.clf()


def main():
    train_data_with_labels = preprocess.load_train_data_and_labels(1000)

    print("Loaded train dataset")

    train_data = [item[1] for item in train_data_with_labels]
    train_labels = [item[2] for item in train_data_with_labels]
    slices, classes = preprocess.split_into_slices(train_data, train_labels)
    print("Split dataset to fragments")

    slices_with_classes = list(zip(slices, classes))

    slices_gq = [x[0] for x in slices_with_classes if x[1] == 'N']  # GQ
    # slices_af = [x[0] for x in slices_with_classes if x[1] == 'A']  # AF
    # slices_o = [x[0] for x in slices_with_classes if x[1] == 'O']  # OTHERS
    # slices_bq = [x[0] for x in slices_with_classes if x[1] == '~']  # BQ

    random.shuffle(slices_gq)
    # random.shuffle(slices_af)
    # random.shuffle(slices_o)
    # random.shuffle(slices_bq)

    save_slices_and_images(slices_gq[:500], 'slices_gq')
    # save_slices_and_images(slices_af[:300], 'slices_af')
    # save_slices_and_images(slices_o[:200], 'slices_others')
    # save_slices_and_images(slices_bq[:1000], 'slices_bq')


if __name__ == '__main__':
    main()
