import argparse

from multiinput import AETrainerMultiInput
from multiinput.AETrainerMultiInput import AETrainerMultiInput
from multiinput.ae.AEMultiinput import AEMultiInput
from multiinput.classifier.simple_classifier_multi_input import SimpleClassifierMultiInput
from util.classification import train_test


def main(train: bool, use_hearth_rate: bool):
    train_test(AEMultiInput(), SimpleClassifierMultiInput(), AETrainerMultiInput() if train else None, use_hearth_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-train', action="store_true")
    parser.add_argument('--use-hearth-rate', action="store_true")
    args = parser.parse_args()

    main(not args.no_train, args.use_hearth_rate)
