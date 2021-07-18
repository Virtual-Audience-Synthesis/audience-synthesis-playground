import numpy as np
import sys

# sys.path.append('../../utils')
from scipy.signal import lfilter

import pdb


def envelope(t, fs, attack_time=0.032, decay_time=0.052, c_attack=1, c_decay=0.1):
    c_attack = 44100 / fs
    c_decay = 44100 / fs * 0.1
    attack_time_n = int(attack_time * fs)
    decay_time_n = int(decay_time * fs)
    return np.min(
        np.stack(
            [
                np.power(0.99, (c_attack * (attack_time_n - t))),
                np.power(0.99, (c_decay * (t - attack_time_n))),
            ]
        ),
        axis=0,
    )


def clap(clap_n, env, fs):
    x = np.random.random(clap_n) * env
    y_bp1 = bandpass_(x, 10, 2000, fs)
    # plt.plot(y_bp1)
    # pdb.set_trace()
    y_hp1 = highpass_(y_bp1, 10, 500, fs)
    y_lp1 = lowpass_(y_hp1, 10, 2500, fs)

    y_bp2 = bandpass_(x, 10, 1000, fs)
    y_hp2 = highpass_(y_bp2, 10, 500, fs)
    y_lp2 = lowpass_(y_hp2, 10, 2500, fs)
    y = y_lp2 - y_lp1

    return y


## Functions

# lowpass(1+np.cos(w0)/2) *
def lowpass_(x, q_db, f0, Fs):
    # q_db: quality factor
    # f_0: center frequency
    # Fs: sampling frequency
    w0 = 2 * np.pi * f0 / Fs
    alpha_q_db = np.sin(w0) / (2 * np.power(10, (q_db / 20)))

    b = (1 - np.cos(w0) / 2) * np.array([1, 2, 1])
    a = np.array([1 + alpha_q_db, -2 * np.cos(w0), 1 - alpha_q_db])
    return lfilter(b, a, x)


# highpass
def highpass_(x, q_db, f0, Fs):
    # q_db: quality factor
    # f_0: center frequency
    # Fs: sampling frequency

    w0 = 2 * np.pi * f0 / Fs
    alpha_q_db = np.sin(w0) / (2 * np.power(10, (q_db / 20)))

    b = (1 + np.cos(w0) / 2) * np.array([1, -2, 1])
    a = np.array([1 + alpha_q_db, -2 * np.cos(w0), 1 - alpha_q_db])
    return lfilter(b, a, x)


# bandpass
def bandpass_(x, q_db, f0, Fs):
    # q_db: quality factor
    # f_0: center frequency
    # Fs: sampling frequency
    # pdb.set_trace()
    w0 = 2 * np.pi * f0 / Fs
    alpha_q_db = np.sin(w0) / (2 * np.power(10, (q_db / 20)))

    b = np.array([1, 0, -1])
    a = np.array([1 + alpha_q_db, -2 * np.cos(w0), 1 - alpha_q_db])
    return lfilter(b, a, x)


"""
Find 2^n that is equal to or greater than.
"""


def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n


#   This is internal function used by fft(), because the FFT routine
#   requires that the data size be a power of 2.


def fconv(x, h):
    # FCONV Fast Convolution
    # [y] = FCONV(x, h) convolves x and h, and normalizes the output
    # to +-1.
    # x = input vector
    # h = input vector
    #
    Ly = len(x) + len(h) - 1  #
    if nextpow2(Ly) == 0:
        print("nextpow2")
    Ly2 = np.power(2, nextpow2(Ly))  # Find smallest power of 2 that is > Ly
    X = np.fft.fft(x, n=Ly2)  # Fast Fourier transform
    H = np.fft.fft(h, n=Ly2)  # Fast Fourier transform
    Y = X * H  # DO CONVOLUTION
    y = np.fft.irfft(Y, n=Ly2)  # Inverse fast Fourier transform
    y = y[1:1:Ly]  # Take just the first N elements
    return y / max(abs(y))  # Normalize the output
