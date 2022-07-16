import numpy as np
from scipy import signal as sg
from scipy.signal import resample


class HearthRateCalculator:

    def __init__(self, signal, freq):
        self.RR1, self.RR2, self.probable_peaks, self.r_locs, self.peaks, self.result = ([] for _ in range(6))
        self.SPKI, self.NPKI, self.Threshold_I1, self.Threshold_I2, self.SPKF, self.NPKF, self.Threshold_F1, self.Threshold_F2 = (
            0 for _ in range(8))

        signal = resample(signal, int(125 * len(signal) / freq))

        hrc = HearthRateCalculator.PanTompkinsQRS()

        mwin, bpass = hrc.solve(signal, freq)

        self.T_wave = False
        self.m_win = mwin
        self.b_pass = bpass
        self.freq = 125
        self.signal = signal
        self.win_150ms = round(0.15 * self.freq)

        self.RR_Low_Limit = 0
        self.RR_High_Limit = 0
        self.RR_Missed_Limit = 0
        self.RR_Average1 = 0

    def approx_peak(self):
        # FFT convolution
        slopes = sg.fftconvolve(self.m_win, np.full((25,), 1) / 25, mode='same')

        # Finding approximate peak locations
        for i in range(round(0.5 * self.freq) + 1, len(slopes) - 1):
            if (slopes[i] > slopes[i - 1]) and (slopes[i + 1] < slopes[i]):
                self.peaks.append(i)

    def adjust_rr_interval(self, ind):
        self.RR1 = np.diff(self.peaks[max(0, ind - 8): ind + 1]) / self.freq

        self.RR_Average1 = np.mean(self.RR1)
        RR_Average2 = self.RR_Average1

        if ind >= 8:
            for i in range(0, 8):
                if self.RR_Low_Limit < self.RR1[i] < self.RR_High_Limit:
                    self.RR2.append(self.RR1[i])

                    if len(self.RR2) > 8:
                        self.RR2.remove(self.RR2[0])
                        RR_Average2 = np.mean(self.RR2)

        if len(self.RR2) > 7 or ind < 8:
            self.RR_Low_Limit = 0.92 * RR_Average2
            self.RR_High_Limit = 1.16 * RR_Average2
            self.RR_Missed_Limit = 1.66 * RR_Average2

    def searchback(self, peak_val, RRn, sb_win):
        global r_max
        x_max = None
        if RRn > self.RR_Missed_Limit:
            win_rr = self.m_win[peak_val - sb_win + 1: peak_val + 1]

            coord = np.asarray(win_rr > self.Threshold_I1).nonzero()[0]

            if len(coord) > 0:
                for pos in coord:
                    if win_rr[pos] == max(win_rr[coord]):
                        x_max = pos
                        break
            else:
                x_max = None

            if x_max is not None:
                self.SPKI = 0.25 * self.m_win[x_max] + 0.75 * self.SPKI
                self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
                self.Threshold_I2 = 0.5 * self.Threshold_I1

                win_rr = self.b_pass[x_max - self.win_150ms: min(len(self.b_pass) - 1, x_max)]

                coord = np.asarray(win_rr > self.Threshold_F1).nonzero()[0]

                if len(coord) > 0:
                    for pos in coord:
                        if win_rr[pos] == max(win_rr[coord]):
                            r_max = pos
                            break
                else:
                    r_max = None

                if r_max is not None:
                    if self.b_pass[r_max] > self.Threshold_F2:
                        self.SPKF = 0.25 * self.b_pass[r_max] + 0.75 * self.SPKF
                        self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
                        self.Threshold_F2 = 0.5 * self.Threshold_F1

                        self.r_locs.append(r_max)

    def find_t_wave(self, peak_val, RRn, ind, prev_ind):

        if self.m_win[peak_val] >= self.Threshold_I1:
            if ind > 0 and 0.20 < RRn < 0.36:
                curr_slope = max(np.diff(self.m_win[peak_val - round(self.win_150ms / 2): peak_val + 1]))
                last_slope = max(
                    np.diff(self.m_win[self.peaks[prev_ind] - round(self.win_150ms / 2): self.peaks[prev_ind] + 1]))

                if curr_slope < 0.5 * last_slope:
                    self.T_wave = True
                    self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI

            if not self.T_wave:
                if self.probable_peaks[ind] > self.Threshold_F1:
                    self.SPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.SPKI
                    self.SPKF = 0.125 * self.b_pass[ind] + 0.875 * self.SPKF

                    self.r_locs.append(self.probable_peaks[ind])

                else:
                    self.SPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.SPKI
                    self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

                    # Update noise thresholds
        elif (self.m_win[peak_val] < self.Threshold_I1) or (
                self.Threshold_I1 < self.m_win[peak_val] < self.Threshold_I2):
            self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI
            self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

    def adjust_thresholds(self, peak_val, ind):
        '''
        Adjust Noise and Signal Thresholds During Learning Phase
        :param peak_val: peak location in consideration
        :param ind: current index in peaks array
        '''

        if self.m_win[peak_val] >= self.Threshold_I1:
            self.SPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.SPKI

            if self.probable_peaks[ind] > self.Threshold_F1:
                self.SPKF = 0.125 * self.b_pass[ind] + 0.875 * self.SPKF

                self.r_locs.append(self.probable_peaks[ind])

            else:
                self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

        elif (self.m_win[peak_val] < self.Threshold_I2) or (
                self.Threshold_I2 < self.m_win[peak_val] < self.Threshold_I1):
            self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI
            self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

    def update_thresholds(self):
        '''
        Update Noise and Signal Thresholds for next iteration
        '''

        self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
        self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
        self.Threshold_I2 = 0.5 * self.Threshold_I1
        self.Threshold_F2 = 0.5 * self.Threshold_F1
        self.T_wave = False

    def ecg_searchback(self):
        '''
        Searchback in ECG signal to increase efficiency
        '''

        x_max = None
        self.r_locs = np.unique(np.array(self.r_locs).astype(int))

        win_200ms = round(0.3 * self.freq)

        for r_val in self.r_locs:
            coord = np.arange(r_val - win_200ms, min(len(self.signal), r_val + win_200ms + 1), 1)

            if len(coord) > 0:
                for pos in coord:
                    if self.signal[pos] == max(self.signal[coord]):
                        x_max = pos
                        break
            else:
                x_max = None

            if x_max is not None:
                self.result.append(x_max)

    def find_r_peaks(self):
        self.approx_peak()

        for ind in range(len(self.peaks)):

            peak_val = self.peaks[ind]
            win_300ms = np.arange(max(0, self.peaks[ind] - self.win_150ms),
                                  min(self.peaks[ind] + self.win_150ms, len(self.b_pass) - 1), 1)
            max_val = max(self.b_pass[win_300ms], default=0)

            if max_val != 0:
                x_coord = np.asarray(self.b_pass == max_val).nonzero()
                self.probable_peaks.append(x_coord[0][0])

            if ind < len(self.probable_peaks) and ind != 0:
                self.adjust_rr_interval(ind)

                if self.RR_Average1 < self.RR_Low_Limit or self.RR_Average1 > self.RR_Missed_Limit:
                    self.Threshold_I1 /= 2
                    self.Threshold_F1 /= 2

                RRn = self.RR1[-1]

                self.searchback(peak_val, RRn, round(RRn * self.freq))

                self.find_t_wave(peak_val, RRn, ind, ind - 1)

            else:
                self.adjust_thresholds(peak_val, ind)

            self.update_thresholds()

        self.ecg_searchback()

        return np.unique(np.array(self.result))

    def calc_hearth_rate(self):
        return (60 * self.freq) / np.average(np.diff(self.find_r_peaks()))

    class PanTompkinsQRS:

        def bandpass_filter(self, signal):
            sig = signal.copy()

            # Low pass filter
            for i in range(len(signal)):
                if i >= 1:
                    sig[i] += 2 * sig[i - 1]

                if i >= 2:
                    sig[i] -= sig[i - 2]

                if i >= 6:
                    sig[i] -= 2 * signal[i - 6]

                if i >= 12:
                    sig[i] += signal[i - 12]

            result = sig.copy()

            # High pass filter
            for i in range(len(signal)):
                result[i] = -1 * sig[i]

                if i >= 1:
                    result[i] -= result[i - 1]

                if i >= 16:
                    result[i] += 32 * sig[i - 16]

                if i >= 32:
                    result[i] += sig[i - 32]

            # Normalization
            max_val = max(abs(max(result)), abs(min(result)))
            result = result / max_val

            return result

        def derivative(self, signal, freq):
            result = signal.copy()

            for i in range(len(signal)):
                result[i] = 0

                if i >= 1:
                    result[i] -= 2 * signal[i - 1]

                if i >= 2:
                    result[i] -= signal[i - 2]

                if i <= len(signal) - 2:
                    result[i] += 2 * signal[i + 1]

                if i <= len(signal) - 3:
                    result[i] += signal[i + 2]

                result[i] = (result[i] * freq) / 8

            return result

        def squaring(self, signal):
            result = signal.copy()

            for index in range(len(signal)):
                result[index] = signal[index] ** 2

            return result

        def moving_window_integration(self, signal, freq):
            result = signal.copy()
            win_size = round(0.150 * freq)  # 150 ms
            sum = 0

            for i in range(win_size):
                sum += signal[i] / win_size
                result[i] = sum

            for i in range(win_size, len(signal)):
                sum += signal[i] / win_size
                sum -= signal[i - win_size] / win_size
                result[i] = sum

            return result

        def solve(self, signal, freq):
            bpass = self.bandpass_filter(signal)
            result = self.derivative(bpass, freq)
            result = self.squaring(result)
            mwin = self.moving_window_integration(result, freq)
            return bpass, mwin


