import argparse

from classification.simple.SimpleFftAETrainer import SimpleAETrainer
from classification.simple.ae.AESimple import AESimple
from classification.simple.classifier.simple_classifier import SimpleClassifier

from util.ecg_classifier import EcgClassifier


def main(train: bool, use_hearth_rate: bool):
    EcgClassifier().train_test(AESimple(hidden_layer_width=10),
                               SimpleClassifier(),
                               SimpleAETrainer(epochs=64, lr=0.25, batch_size=128) if train else None,
                               use_hearth_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-train', action="store_true")
    parser.add_argument('--use-hearth-rate', action="store_true")
    args = parser.parse_args()

    main(not args.no_train, args.use_hearth_rate)
