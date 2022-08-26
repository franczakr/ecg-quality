import argparse

from basic_fft import AETrainer
from basic_fft.AETrainer import AETrainer
from basic_fft.ae.AESimple import AESimple
from basic_fft.classifier.simple_classifier import SimpleClassifier
from util.classification import train_test


def main(train: bool, use_hearth_rate: bool):
    train_test(AESimple(), SimpleClassifier(), AETrainer() if train else None, use_hearth_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-train', action="store_true")
    parser.add_argument('--use-hearth-rate', action="store_true")
    args = parser.parse_args()

    main(not args.no_train, args.use_hearth_rate)
