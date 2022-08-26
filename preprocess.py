import argparse
from typing import List

import numpy as np
from scipy.signal import medfilt, resample

from util import persistence
from util.data_reader import load_train_data_and_labels
from util.hearth_rate_calculator import HearthRateCalculator

PREPROCESSED_FILENAME = 'preprocessed.pkl'
PREPROCESSED_FILENAME_HR = 'preprocessed_hr.pkl'
PREPROCESSED_FILENAME_TRAIN = 'preprocessed_train.pkl'
PREPROCESSED_FILENAME_HR_TRAIN = 'preprocessed_hr_train.pkl'

ORIGINAL_SAMPLING_RATE = 300

BASELINE_FILTER_LEN_T = 600  # [ms]
BASELINE_FILTER_LEN_QRS_P = 200  # [ms]

IDX_OF_PEAK = 20
SLICE_LENGTH = 100
CONTEXT_LEN = int(round(2.5 * SLICE_LENGTH))


def split_into_slices(fragments: List[np.ndarray], classes: List[int], use_hearth_rate: bool) -> (np.ndarray, np.ndarray, np.ndarray, int):
    fragments_sliced = []
    fragments_sliced_with_context = []
    slice_counters = []

    for fragment in fragments:

        if use_hearth_rate:
            hr = HearthRateCalculator(fragment, ORIGINAL_SAMPLING_RATE)
            heart_rate = hr.calc_hearth_rate()
            sampling_rate = SLICE_LENGTH * heart_rate / 60
        else:
            sampling_rate = 125

        fragment = resample(fragment, int(sampling_rate * len(fragment) / ORIGINAL_SAMPLING_RATE))
        fragment = _cancel_baseline(fragment, sampling_rate)
        fragment = _subtract_mean(fragment)
        fragment = np.clip(fragment, -6000, 6000)
        slices, slices_with_context = _generate_slices(fragment)
        fragments_sliced.append(slices)
        fragments_sliced_with_context.append(slices_with_context)
        slice_counters.append(len(slices))

    sliced = np.concatenate(fragments_sliced, axis=0).reshape(-1, SLICE_LENGTH)

    classes_dupl = np.empty(0, int)

    for clazz, slice_cnt in zip(classes, slice_counters):
        classes_dupl = np.append(classes_dupl, np.repeat(clazz, slice_cnt, axis=0))

    return sliced.astype('float32'), classes_dupl

def _generate_slices(fragment):
    fragment_len = len(fragment)
    margin_left = IDX_OF_PEAK
    margin_right = SLICE_LENGTH - margin_left

    windows = [[start, start + CONTEXT_LEN] for start in range(0, fragment_len - CONTEXT_LEN + 1, CONTEXT_LEN)]

    slices_with_context = np.empty((len(windows), CONTEXT_LEN), fragment.dtype)

    for slice_cnt, (start, stop) in enumerate(windows):
        slices_with_context[slice_cnt] = fragment[start:stop].copy()

    windows[0][0] = margin_left
    windows[-1][1] = min(windows[-1][1] - margin_right, fragment_len)

    slices = np.empty((len(windows), SLICE_LENGTH), fragment.dtype)

    for slice_cnt, (start, stop) in enumerate(windows):
        slice_positive_peak = _cutout_slice(fragment, start, stop, margin_left, margin_right, np.argmax)
        slice_negative_peak = _cutout_slice(fragment, start, stop, margin_left, margin_right, np.argmin)

        slice = _select_slice(slice_positive_peak, slice_negative_peak)
        slice = normalize_slice(slice)
        slices[slice_cnt] = slice

    return slices, slices_with_context


def _cutout_slice(data, start, stop, margin_left, margin_right, method):
    peak_idx = start + method(data[start:stop])

    slice_start = peak_idx - margin_left
    slice_stop = peak_idx + margin_right

    slice = data[slice_start:slice_stop].copy()
    return _subtract_mean(slice)


def _select_slice(slice1, slice2):
    return slice1 if abs(slice1[IDX_OF_PEAK]) > abs(slice2[IDX_OF_PEAK]) else slice2


def normalize_slice(slice):
    if slice[IDX_OF_PEAK] < slice.mean():
        slice = -slice + slice.max()

    slice = (slice - slice.min()) / (slice.max() - slice.min())
    return slice


def _subtract_mean(data):
    return data - data.mean()


def _cancel_baseline(signal, sampling_rate):
    # T - wave
    filtered_stage1 = _apply_medfilt(signal, BASELINE_FILTER_LEN_T, sampling_rate)

    # QRS - complex and P - wave
    filtered_stage2 = _apply_medfilt(filtered_stage1, BASELINE_FILTER_LEN_QRS_P, sampling_rate)

    return filtered_stage2


def _apply_medfilt(signal, length_ms, sampling_rate):
    length = int(length_ms / 1000.0 * sampling_rate)
    if length % 2 == 0:
        length += 1
    filt_result = medfilt(signal, kernel_size=length)
    return signal - filt_result


def main(use_hearth_rate: bool):
    train_data_with_labels = load_train_data_and_labels()

    print("Loaded train dataset")

    train_data = [item[1] for item in train_data_with_labels]
    train_labels = [item[2] for item in train_data_with_labels]
    slices, classes = split_into_slices(train_data, train_labels, use_hearth_rate)
    print("Split dataset to fragments")

    persistence.save_object(PREPROCESSED_FILENAME_HR if use_hearth_rate else PREPROCESSED_FILENAME, (slices, classes))
    print("Saved preprocessed data")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_hearth_rate', action="store_true")
    args = parser.parse_args()

    main(args.use_hearth_rate)
