import numpy

from ae import train_AE
from ae.AEConv import AEConv
from classifier.classifier import Classifier
from preprocess import PREPRECESSED_FILENAME
from util import persistence, data_reader
from util.persistence import save_model


def main():
    slices, classes = persistence.load_object(PREPRECESSED_FILENAME)
    autoencoder = AEConv()

    #  train
    slices_gq = [gq[0] for gq in list(zip(slices, classes)) if gq[1] == data_reader.GOOD_QUALITY]
    slices_gq = numpy.array(slices_gq[:10000])
    autoencoder = train_AE.train_ae(autoencoder, slices_gq, epochs=20)
    save_model(autoencoder)

    #  or load trained
    # autoencoder = load_model(autoencoder)

    classifier = Classifier()

    classifier.train(autoencoder, slices, classes)


if __name__ == '__main__':
    main()
