import persistence
import train_AE
from preprocess import PREPRECESSED_FILENAME

import torch


def main():
    # slices, classes, slices_with_context, context_length = persistence.load_object(PREPRECESSED_FILENAME)
    #
    # slices = slices.astype('float32')
    #
    # train_AE.train_ae(slices, epochs=3)

    print("a")


if __name__ == '__main__':
    main()
