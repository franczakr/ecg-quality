import argparse

from basic_fft import AETrainer
from basic_fft.AETrainer import AETrainer
from basic_fft.ae.AESimple import AESimple
from basic_fft.classifier.simple_classifier import SimpleClassifier
from util.ecg_classifier import EcgClassifier


def main(train: bool, use_hearth_rate: bool):
    EcgClassifier().train_test(AESimple(hidden_layer_width=15),
                               SimpleClassifier(),
                               AETrainer(lr=0.042, batch_size=32, epochs=40) if train else None,
                               use_hearth_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-train', action="store_true")
    parser.add_argument('--use-hearth-rate', action="store_true")
    args = parser.parse_args()

    main(not args.no_train, args.use_hearth_rate)
