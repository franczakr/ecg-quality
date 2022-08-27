import argparse

from classification.multiinput_fft.AETrainerMultiInput import AETrainerMultiInput
from classification.multiinput_fft.ae.AEMultiinput import AEMultiInput
from classification.multiinput_fft.classifier.simple_classifier_multi_input import SimpleClassifierMultiInput
from util.ecg_classifier import EcgClassifier


def main(train: bool, use_hearth_rate: bool):
    EcgClassifier().train_test(
        AEMultiInput(signal_layer_width=55, fft_layer_width=12, final_layer_width=139, loss_multiplier=[2, 6]),
        SimpleClassifierMultiInput(),
        AETrainerMultiInput(lr=0.019, epochs=29, batch_size=32) if train else None,
        use_hearth_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-train', action="store_true")
    parser.add_argument('--use-hearth-rate', action="store_true")
    args = parser.parse_args()

    main(not args.no_train, args.use_hearth_rate)
