from typing import List

from scipy.signal import medfilt
import numpy as np

import persistence
from data_reader import load_train_data_and_labels

PREPRECESSED_FILENAME = 'preprocessed.pkl'

SAMPLING_RATE = 125

WND_LEN_RATIO = 2.5  # ratio of the slice length
POSITION_OF_MAX = 200  # position of the sample with the greatest abs amplitude [ms]

BASELINE_FILTER_LEN_T = 600  # [ms]
BASELINE_FILTER_LEN_QRS_P = 200  # [ms]

IDX_OF_PEAK = int(POSITION_OF_MAX / 1000.0 * SAMPLING_RATE)

SLICE_LENGTH = int(0.8 * SAMPLING_RATE)  # 0.8 second based on sampling rate
CONTEXT_LEN = int(round(WND_LEN_RATIO * SLICE_LENGTH))

def split_into_slices(fragments: List[np.ndarray], classes: List[int]) -> (np.ndarray, np.ndarray, np.ndarray, int):
    fragments_sliced = []
    fragments_sliced_with_context = []
    slice_counters = []

    for fragment in fragments:
        slices, slices_with_context = _generate_slices(fragment)
        fragments_sliced.append(slices)
        fragments_sliced_with_context.append(slices_with_context)
        slice_counters.append(len(slices))

    sliced = np.concatenate(fragments_sliced, axis=0).reshape(-1, SLICE_LENGTH)
    sliced_with_context = np.concatenate(fragments_sliced_with_context, axis=0).reshape(-1, CONTEXT_LEN)

    classes_dupl = np.empty(0, int)

    for clazz, slice_cnt in zip(classes, slice_counters):
        classes_dupl = np.append(classes_dupl, np.repeat(clazz, slice_cnt, axis=0))

    return sliced, classes_dupl, sliced_with_context, CONTEXT_LEN

def _generate_slices(fragment):
    fragment_len = len(fragment)
    margin_left = IDX_OF_PEAK
    margin_right = SLICE_LENGTH - margin_left

    if CONTEXT_LEN > fragment_len:
        raise Exception("Requested SLICE_LENGTH of slice ({} samples) requires at least {}-sample slice context. "
                        "Given signal is too short (only {} samples)."
                        .format(SLICE_LENGTH, CONTEXT_LEN, fragment_len))

    if margin_right > SLICE_LENGTH:
        raise Exception("Requested SLICE_LENGTH of slice ({} samples) is less than position of max value, which is {}ms"
                        .format(SLICE_LENGTH, POSITION_OF_MAX))

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
        slices[slice_cnt] = _select_slice(slice_positive_peak, slice_negative_peak, margin_left)

    return slices, slices_with_context


def _select_slice(slice1, slice2, peak_idx):
    return slice1 if abs(slice1[peak_idx]) > abs(slice2[peak_idx]) else slice2

def _cutout_slice(data, start, stop, margin_left, margin_right, method):
    peak_idx = start + method(data[start:stop])

    slice_start = peak_idx - margin_left
    slice_stop = peak_idx + margin_right

    slice = data[slice_start:slice_stop].copy()
    return _subtract_mean(slice)


def _subtract_mean(data):
    return data - np.mean(data)

def cancel_baseline(signal):
    # T - wave
    filtered_stage1 = _apply_medfilt(signal, BASELINE_FILTER_LEN_T)

    # QRS - complex and P - wave
    filtered_stage2 = _apply_medfilt(filtered_stage1,  BASELINE_FILTER_LEN_QRS_P)

    return filtered_stage2


def _apply_medfilt(signal, length_ms):
    length = int(length_ms / 1000.0 * SAMPLING_RATE)
    filt_result = medfilt(signal, kernel_size=(length, 1))
    return signal - filt_result


def main():
    train_data_with_labels = load_train_data_and_labels()

    print("Loaded train dataset")

    train_data = [item[1] for item in train_data_with_labels]
    train_labels = [item[2] for item in train_data_with_labels]
    slices, classes, slices_with_context, context_length = split_into_slices(train_data, train_labels)
    print("Split dataset to fragments")

    persistence.save_object(PREPRECESSED_FILENAME, (slices, classes, slices_with_context, context_length))
    print("Saved preprocessed data")


if __name__ == '__main__':
    main()