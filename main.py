#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:22:29 2024

@author: farismismar
"""

# This is re-write of the primer v0.8

import numpy as np
import pandas as pd
from scipy.constants import speed_of_light

import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
# import tikzplotlib

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

import os

from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.callbacks import ModelCheckpoint

from autoencoder import Autoencoder
from environment import radio_environment
from DQNLearningAgent import DQNLearningAgent as DQNAgent
from QLearningAgent import QLearningAgent as TabularAgent

import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# For Windows users
if os.name == 'nt':
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

print(tf.config.list_physical_devices('GPU'))

# The GPU ID to use, usually either "0" or "1" based on previous line.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

################################################
# Single OFDM symbol and single user simulator
################################################

###############################################################################
# Parameters
N_t = 2                                  # Number of transmit antennas
N_r = 2                                  # Number of receive antennas per user
N_sc = 64                                # Number of subcarriers
P_TX = 1                                 # Signal transmit power per antenna (bandwidth-normalized) [W]

max_transmissions = 500
precoder = 'SVD_Waterfilling'            # Also: identity, SVD, SVD_Waterfilling, dft_beamforming  # precoder is the problem
channel_type = 'CDL-E'                   # Channel type: awgn, rayleigh, ricean, CDL-A, CDL-C, CDL-E
quantization_b = np.inf                  # Quantization resolution

Df = 15e3                                # OFDM subcarrier bandwidth [Hz].
f_c = 1800e6                             # Center frequency [MHz] for large scale fading and DFT.
noise_figure = 5                         # Receiver noise figure [dB]
interference_power_dBm = -105            # dBm measured at the receiver
p_interference = 0.00                    # probability of interference

constellation = 'QAM'                    # QAM or PSK
M_constellation = 16                     # Square constellations only.
n_pilot = N_t                            # Number of pilots for channel estimation

MIMO_estimation = 'perfect'              # Also: perfect, LS, LMMSE (keep at perfect)
MIMO_equalization = 'MMSE'               # Also: MMSE, ZF
symbol_detection = 'ML'                  # Also: ML, kmeans, DNN, ensemble

crc_generator = 0b1000_0101              # CRC generator polynomial (n-bit)
channel_compression_ratio = 0            # Channel compression

K_factor = 4                             # For Ricean
shadowing_std_dev = 4                    # in dB

# Transmit SNR in dB
transmit_SNR_dB = [-10, -5, 0, 5, 10, 15, 20, 25, 30][::-1]
###############################################################################

plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = "14"

output_path = './'
prefer_gpu = True

seed = 42  # Reproduction
np_random = np.random.RandomState(seed=seed)

__ver__ = '0.8'
__data__ = '2025-03-19'


def create_bit_payload(payload_size):
    global np_random
    bits = np_random.binomial(1, 0.5, size=payload_size)
    bits = ''.join(map(str, bits))
    return bits


def create_constellation(constellation, M):
    if (constellation == 'PSK'):
        return _create_constellation_psk(M)
    elif (constellation == 'QAM'):
        return _create_constellation_qam(M)
    else:
        return None


def decimal_to_gray(n, k):
    gray = n ^ (n >> 1)
    gray = bin(gray)[2:]

    return '{}'.format(gray).zfill(k)


# Constellation based on Gray code
def _create_constellation_psk(M):
    k = np.log2(M)
    if k != int(k):  # only square constellations are allowed.
        print('Only square PSK constellations allowed.')
        return None

    k = int(k)
    constellation = pd.DataFrame(columns=['m', 'x_I', 'x_Q'])

    for m in np.arange(M):
        centroid_ = pd.DataFrame(data={'m': int(m),
                                       'x_I': np.sqrt(1 / 2) * np.cos(2*np.pi/M*m + np.pi/M),
                                       'x_Q': np.sqrt(1 / 2) * np.sin(2*np.pi/M*m + np.pi/M)}, index=[m])
        if constellation.shape[0] == 0:
            constellation = centroid_.copy()
        else:
            constellation = pd.concat([constellation, centroid_], ignore_index=True)

    gray = constellation['m'].apply(lambda x: decimal_to_gray(x, k))
    constellation['I'] = gray.str[:(k//2)]
    constellation['Q'] = gray.str[(k//2):]

    constellation.loc[:, 'x'] = constellation.loc[:, 'x_I'] + 1j * constellation.loc[:, 'x_Q']

    # Normalize the transmitted symbols
    # The average power is normalized to unity
    P_average = np.mean(np.abs(constellation.loc[:, 'x']) ** 2)
    constellation.loc[:, 'x'] /= np.sqrt(P_average)

    return constellation


# Constellation based on Gray code
def _create_constellation_qam(M):
    k = np.log2(M)
    if k != int(k):  # only square QAM is allowed.
        print('Only square QAM constellations allowed.')
        return None

    k = int(k)
    m = np.arange(M)
    Am_ = np.arange(-np.sqrt(M) + 1, np.sqrt(M), step=2, dtype=int)  # Proakis p105

    Am = np.zeros(M, dtype=np.complex64)
    idx = 0
    for Am_I in Am_:
        for Am_Q in Am_:
            Am[idx] = Am_I + 1j * Am_Q
            idx += 1

    # This will hold the transmitted symbols
    constellation = pd.DataFrame(data={'x_I': np.real(Am),
                                       'x_Q': np.imag(Am)})
    constellation.insert(0, 'm', m)
    constellation_ordered = pd.DataFrame()
    for idx, s in enumerate(np.array_split(constellation, int(np.sqrt(M)))):
        if idx % 2 == 1:
            s = s.iloc[::-1]  # Invert
        # print(s)
        constellation_ordered = pd.concat([constellation_ordered, s], axis=0)

    constellation = constellation_ordered.copy()
    constellation = constellation.reset_index(drop=True)
    constellation['m'] = constellation.index

    gray = constellation['m'].apply(lambda x: decimal_to_gray(x, k))
    constellation['I'] = gray.str[:(k//2)]
    constellation['Q'] = gray.str[(k//2):]

    constellation.loc[:, 'x'] = constellation.loc[:, 'x_I'] + \
        1j * constellation.loc[:, 'x_Q']

    # Normalize the transmitted symbols
    # The average power is normalized to unity
    P_average = _signal_power(constellation['x'])  # np.mean(np.abs(constellation.loc[:, 'x']) ** 2)
    constellation.loc[:, 'x'] /= np.sqrt(P_average)

    return constellation


def _signal_power(signal):
    return np.mean(np.abs(signal) ** 2, axis=0)


def generate_transmit_symbols(N_sc, N_t, alphabet, P_TX):
    global np_random
    global crc_generator, N_s

    k = int(np.log2(alphabet.shape[0]))

    payload_length = N_sc * N_t * k  # For the future, this depends on the MCS index.

    crc_length = len(bin(crc_generator)[2:])  # in bits.
    crc_pad_length = int(np.ceil(crc_length / k)) * k  # padding included.
    
    # The padding length in symbols is
    codeword_length = payload_length  # based on the MCS.
    padding_length_syms = N_sc * N_s - int(np.ceil((codeword_length + crc_length) / k))
    padding_length_bits = k * padding_length_syms

    bits = create_bit_payload(payload_length)

    bits = bits[:-crc_pad_length]
    crc_transmitter = compute_crc(bits, crc_generator)

    # Construct the payload frame.
    payload = bits + '0' * padding_length_bits + crc_transmitter
    ###

    assert(len(payload) == payload_length)

    x_b_i, x_b_q, x_information, x_symbols = bits_to_baseband(payload, alphabet)

    x_information = np.reshape(x_information, (N_sc, N_t))
    x_symbols = np.reshape(x_symbols, (N_sc, N_t))

    # Normalize and scale the transmit power of the symbols.
    x_symbols /= np.sqrt(_signal_power(x_symbols).mean() / P_TX)

    x_b_i = np.reshape(x_b_i, (-1, N_t))
    x_b_q = np.reshape(x_b_q, (-1, N_t))

    return x_information, x_symbols, [x_b_i, x_b_q], payload_length, crc_transmitter    


def generate_interference(Y, p_interference, interference_power_dBm):
    global np_random

    N_sc, N_r = Y.shape

    interference_power = _linear(interference_power_dBm)

    interf = np.sqrt(interference_power / 2) * \
        (np_random.normal(0, 1, size=(N_sc, N_r)) + \
         1j * np_random.normal(0, 1, size=(N_sc, N_r)))

    mask = np_random.binomial(n=1, p=p_interference, size=N_sc)

    # Apply some changes to interference
    for idx in range(N_sc):
        interf[idx, :] *= mask[idx]

    return interf


def generate_pilot_symbols(N_t, n_pilot, P_TX, kind='dft'):
    global np_random

    if kind == 'dft':
        n_generation = max(N_t, n_pilot)

        # Generate a DFT (Discrete Fourier Transform) matrix
        dft_matrix = np.fft.fft(np.eye(n_generation))

        # Select the first n_pilot rows (if n_pilot < N_t)
        pilot_matrix = dft_matrix[:n_pilot]

        # Normalize the DFT-based pilot matrix
        pilot_matrix /= np.sqrt(N_t)

    if kind == 'qr':
        # Generate a random complex Gaussian matrix
        random_matrix = np.sqrt(1 / 2) * (np_random.randn(n_pilot, N_t) + \
                                          1j * np_random.randn(n_pilot, N_t))

        # Perform QR decomposition on the random matrix to get a unitary matrix
        Q, R = np.linalg.qr(random_matrix)

        # Ensure Q is unitary (the first n_pilot rows are orthonormal)
        pilot_matrix = Q[:, :N_t]

    if kind == 'semi-unitary':
        # Compute a unitary matrix from a combinatoric of e
        I = np.eye(N_t)
        idx = np_random.choice(range(N_t), size=N_t, replace=False)
        Q = I[:, idx]

        assert(np.allclose(Q@Q.T, np.eye(N_t)))  # Q is indeed unitary, but square.

        # To make a semi-unitary, we need to post multiply with a rectangular matrix
        # Now we need a rectangular matrix (fat)
        A = np.zeros((N_t, n_pilot), int)
        np.fill_diagonal(A, 1)
        X_p = Q @ A

        assert(np.allclose(X_p@X_p.T, np.eye(N_t)))  # This is it

        # The training sequence is X_p.  It has n_pilot rows and N_t columns
        pilot_matrix = X_p.T

    # Normalize the pilot matrix such that its Frob norm sq is equal to P_TX.
    # Scale the pilot matrix
    pilot_matrix /= np.linalg.norm(pilot_matrix, ord='fro') / np.sqrt(P_TX)

    return pilot_matrix


def bits_to_baseband(x_bits, alphabet):
    k = int(np.log2(alphabet.shape[0]))

    df_ = alphabet[['m', 'I', 'Q']].copy()
    df_.loc[:, 'IQ'] = df_['I'].astype(str) + df_['Q'].astype(str)

    x_b_rev = x_bits[::-1]
    x_b_i = []
    x_b_q = []
    while len(x_b_rev) > 0:
        codeword = x_b_rev[:k][::-1].zfill(k)
        # print(codeword)
        x_b_i.append(codeword[:(k//2)])
        x_b_q.append(codeword[(k//2):])

        #print(x_b_rev, x_b_i)
        x_b_rev = x_b_rev[k:]

    x_b_i = x_b_i[::-1]
    x_b_q = x_b_q[::-1]

    x_sym = []
    x_info = []

    # Next is baseband which is the complex valued symbols
    for i, q in zip(x_b_i, x_b_q):
        sym = alphabet.loc[(alphabet['I'] == i) & (alphabet['Q'] == q), 'x'].values[0]
        info = int(alphabet.loc[(alphabet['I'] == i) & (alphabet['Q'] == q), 'm'].values[0])

        x_sym.append(sym)
        x_info.append(info)

    x_sym = np.array(x_sym)
    x_info = np.array(x_info)

    return x_b_i, x_b_q, x_info, x_sym


def channel_effect(H, X, snr_dB):
    global np_random
    global precoder

    N_sc = H.shape[0]

    # Set a flag to deal with beamforming.
    is_beamforming = True
    N_r = 1

    # Parameters
    if len(H.shape) == 3:  # MIMO case
        _, N_r, N_t = H.shape
        is_beamforming = False
        if N_t > 1 and N_r == 1 and precoder != 'dft_beamforming':
            raise ValueError('Only beamforming is supported for MISO.  Check the setting of precoder.')

    # Convert SNR from dB to linear scale
    snr_linear = _linear(snr_dB)

    # Compute the power of the transmit matrix X
    signal_power = np.mean(_signal_power(X))  # This must equal P_BS / N_t.

    # Calculate the noise power based on the input SNR
    noise_power = signal_power / snr_linear

    # Generate additive white Gaussian noise (AWGN)
    noise = np.sqrt(noise_power / 2) * (np_random.randn(N_sc, N_r) + 1j * np_random.randn(N_sc, N_r))

    received_signal = np.zeros((N_sc, N_r), dtype=np.complex128)

    if is_beamforming:
        received_signal = H*X + noise
        return received_signal, noise

    for sc in range(N_sc):
        received_signal[sc, :] = H[sc, :, :] @ X[sc, :] + noise[sc, :]

    return received_signal, noise


def equalize_channel(H, snr_dB, algorithm):
    if algorithm == 'ZF':
        return _equalize_channel_ZF(H)

    if algorithm == 'MMSE':
        return _equalize_channel_MMSE(H, snr_dB)

    return None


def _equalize_channel_ZF(H):
    N_sc, N_r, N_t = H.shape

    # Hermitian transpose of each subcarrier's H: (N_sc, N_t, N_r)
    H_hermitian = np.conjugate(np.transpose(H, (0, 2, 1)))

    # (H^H * H) for all subcarriers: (N_sc, N_t, N_t)
    H_herm_H = np.matmul(H_hermitian, H)

    # Compute the pseudo-inverse for each subcarrier (to handle singularity)
    H_pseudo_inv = np.linalg.pinv(H_herm_H)  # Shape (N_sc, N_t, N_t)

    # ZF equalization matrix: (H^H * H)^-1 * H^H for all subcarriers
    W_zf = np.matmul(H_pseudo_inv, H_hermitian)  # Shape (N_sc, N_t, N_r)

    return W_zf


def _equalize_channel_MMSE(H, snr_dB):
    N_sc, N_r, N_t = H.shape
    snr_linear = _linear(snr_dB)

    # Hermitian transpose of each subcarrier's H: (N_sc, N_t, N_r)
    H_hermitian = np.conjugate(np.transpose(H, (0, 2, 1)))

    # (H^H * H) for all subcarriers: (N_sc, N_t, N_t)
    H_herm_H = np.matmul(H_hermitian, H)

    # Add noise power to diagonal (1/SNR * I)
    identity = np.eye(N_t)[None, :, :]
    H_mmse_term = H_herm_H + (1 / snr_linear) * identity  # Shape (N_sc, N_t, N_t)

    # Compute the inverse of (H^H * H + (1/SNR) * I) for all subcarriers
    H_mmse_inv = np.linalg.inv(H_mmse_term)  # Shape: (N_sc, N_t, N_t)

    # MMSE equalization matrix: (H^H * H + (1/SNR) * I)^-1 * H^H
    W_mmse = np.matmul(H_mmse_inv, H_hermitian)  # Shape (N_sc, N_t, N_r)

    return W_mmse


def _estimate_channel_least_squares(X, Y):
    global N_sc

    # This is least square (LS) estimation
    H_estimated = Y@np.conjugate(np.transpose(X))

    # Repeat it across all N_sc
    H_estimated_full = np.repeat(H_estimated[np.newaxis, :, :], N_sc, axis=0)  # Repeat for all subcarriers

    return H_estimated_full


def _estimate_channel_linear_regression(X, Y):
    global N_t, N_r

    # This is only for real-valued data and one subcarrier.
    H_estimated = np.array([])

    for r in range(N_r):
        y_r = Y[:, r]
        reg = LinearRegression(n_jobs=-1, fit_intercept=True)
        reg.fit(X, y_r)
        rsq = reg.score(X, y_r)
        print(f'Fitting score on antenna {r}: {rsq:.2f}')
        coeff_r = reg.coef_

        if H_estimated.shape[0] == 0:
            H_estimated = coeff_r
        else:
            H_estimated = np.c_[H_estimated, coeff_r]

    # Repeat it across all N_sc
    H_estimated_full = np.repeat(H_estimated[np.newaxis, :, :], N_sc, axis=0)  # Repeat for all subcarriers

    return H_estimated_full


def _estimate_channel_LMMSE(X, Y, snr_dB):
    snr_linear = _linear(snr_dB)

    H_ls = _estimate_channel_least_squares(X, Y)

    N_sc = H_ls.shape[0]  # Number of subcarriers

    # Initialize the LMMSE channel estimate
    H_lmmse = np.zeros_like(H_ls, dtype=np.complex128)

    # Iterate over each subcarrier
    for sc in range(N_sc):
        # Compute the Frobenius norm squared for the current subcarrier's channel matrix (N_r * N_t)
        # which corresponds tr RHH corresponds to.
        frobenius_norm_squared = np.linalg.norm(H_ls[sc], 'fro') ** 2

        # Compute the LMMSE factor for this subcarrier
        lmmse_factor = frobenius_norm_squared / (frobenius_norm_squared + (1 / snr_linear))

        # Apply the LMMSE factor to the least squares estimate for this subcarrier
        H_lmmse[sc] = lmmse_factor * H_ls[sc]

    return H_lmmse, H_ls


def estimate_channel(X, Y, snr_dB, algorithm):
    H_lmmse, H_ls = _estimate_channel_LMMSE(X, Y, snr_dB)

    if algorithm == 'LS':
        return H_ls

    if algorithm == 'LMMSE':
        return H_lmmse

    return None


def quantize(X, b, max_iteration=200):
    if b == np.inf:
        return X

    X_re = np.real(X)
    X_im = np.imag(X)

    if b == 1:
        return np.sign(X_re) + 1j * np.sign(X_im)

    # Very slow
    Xb_re = np.apply_along_axis(_lloyd_max_quantization, 0, X_re, b, max_iteration)
    Xb_im = np.apply_along_axis(_lloyd_max_quantization, 0, X_im, b, max_iteration)

    return Xb_re + 1j * Xb_im


def _lloyd_max_quantization(x, b, max_iteration):
    # derives the quantized vector
    # https://gist.github.com/PrieureDeSion
    # https://github.com/stillame96/lloyd-max-quantizer
    from utils import normal_dist, expected_normal_dist, MSE_loss, LloydMaxQuantizer

    repre = LloydMaxQuantizer.start_repre(x, b)
    min_loss = 1.0
    min_repre = repre
    min_thre = LloydMaxQuantizer.threshold(min_repre)

    for i in np.arange(max_iteration):
        thre = LloydMaxQuantizer.threshold(repre)
        # In case wanting to use with another mean or variance,
        # need to change mean and variance in utils.py file
        repre = LloydMaxQuantizer.represent(thre, expected_normal_dist, normal_dist)
        x_hat_q = LloydMaxQuantizer.quant(x, thre, repre)
        loss = MSE_loss(x, x_hat_q)

        # Keep the threhold and representation that has the lowest MSE loss.
        if(min_loss > loss):
            min_loss = loss
            min_thre = thre
            min_repre = repre

    # x_hat_q with the lowest amount of loss.
    best_x_hat_q = LloydMaxQuantizer.quant(x, min_thre, min_repre)

    return best_x_hat_q


def create_channel(N_sc, N_r, N_t, channel='rayleigh'):
    global np_random
    global f_c
   
    if channel == 'awgn':
        return _create_awgn_channel(N_sc, N_r, N_t)
    
    if channel == 'ricean':
        return _create_ricean_channel(N_sc, N_r, N_t, K_factor=4)

    if channel == 'rayleigh':
        return _create_ricean_channel(N_sc, N_r, N_t, K_factor=0)

    if channel == 'CDL-A':
        return _generate_cdl_a_channel(N_sc, N_r, N_t, carrier_frequency=f_c)

    if channel == 'CDL-C':
        return _generate_cdl_c_channel(N_sc, N_r, N_t, carrier_frequency=f_c)

    if channel == 'CDL-E':
        return _generate_cdl_e_channel(N_sc, N_r, N_t, carrier_frequency=f_c)


def _create_awgn_channel(N_sc, N_r, N_t):
    H = np.eye(N_r, N_t)
    H = np.repeat(H[np.newaxis, :, :], N_sc, axis=0)  # Repeat for all subcarriers
    
    return H


def _create_ricean_channel(N_sc, N_r, N_t, K_factor):
    global np_random

    mu = np.sqrt(K_factor / (1 + K_factor))
    sigma = np.sqrt(1 / (1 + K_factor))

    # H = np_random.normal(loc=mu, scale=sigma, size=(N_sc, N_r, N_t)) + \
    #     1j * np_random.normal(loc=mu, scale=sigma, size=(N_sc, N_r, N_t))


    # # For each subcarrier and symbol, calculate the MIMO channel response
    # for sc in range(N_sc):
    #     # Normalize channel to unity gain
    #     H[sc, :, :] /= np.linalg.norm(H[sc, :, :], ord='fro')

    H = np_random.normal(loc=mu, scale=sigma, size=(N_r, N_t)) + \
        1j * np_random.normal(loc=mu, scale=sigma, size=(N_r, N_t))
    
    # Normalize channel to unity gain
    H /= np.linalg.norm(H, ord='fro')
    
    H = np.repeat(H[np.newaxis, :, :], N_sc, axis=0)  # Repeat for all subcarriers        
    
    return H


def _generate_cdl_a_channel(N_sc, N_r, N_t, carrier_frequency):
    global np_random, Df
    
    # Channel parameters from CDL-A model (delays, powers, AoD, AoA)
    delay_taps = np.array([0.0, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708, 0.5750, 0.7618, 1.5375, 1.8978, 2.2242, 2.1718, 2.4942, 2.5119, 3.0582, 4.0810, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586])
    powers_dB = np.array([-13.4, 0.0, -2.2, -4.0, -6.0, -8.2, -9.9, -10.5, -7.5, -15.9, -6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2, -18.3, -18.9, -16.6, -19.9, -29.7])
    aod = np.array([-178.1, -4.2, -4.2, -4.2, 90.2, 90.2, 90.2, 121.5, -81.7, 158.4, -83.0, 134.8, -153.0, -172.0, -129.9, -136.0, 165.4, 148.4, 132.7, -118.6, -154.1, 126.5, -56.2])
    aoa = np.array([51.3, -152.7, -152.7, -152.7, 76.6, 76.6, 76.6, -1.2, 10.5, -45.6, 88.1, 34.5, 45.0, -28.4, -90.2, 12.8, -9.7, 37.4, 28.2, 15.7, 3.0, 5.0, 16.0])

    num_taps = len(powers_dB)
    
    # Initialize MIMO OFDM channel matrix H with dimensions (N_sc, N_r, N_t)
    H = np.zeros((N_sc, N_r, N_t), dtype=np.complex128)

    # Frequency range for the subcarriers
    subcarrier_frequencies = carrier_frequency + (np.arange(N_sc) - N_sc // 2) * Df   # subcarrier indices
    
    # Generate channel response for each tap and apply delay phase shifts
    for tap in range(num_taps):
        # Delay in seconds
        delay = delay_taps[tap] * 1e-6  # convert from microseconds to seconds
        power = 10 ** (powers_dB[tap] / 10.)  # Linear scale of power
        aod_rad = np.radians(aod[tap])
        aoa_rad = np.radians(aoa[tap])
    
        # Apply the phase shift for each subcarrier based on the delay
        phase_shift = np.exp(-2j * np.pi * subcarrier_frequencies * delay)
        
        # For each subcarrier and symbol, calculate the MIMO channel response
        for sc in range(N_sc):
            # Generate the channel matrix for this subcarrier and symbol
            # For each antenna, the channel response is influenced by the AoD and AoA
            # Complex Gaussian fading for each tap, scaled by tap power
            H_tap = np.sqrt(power) * np.outer(np.exp(1j * aod_rad), np.exp(1j * aoa_rad)) * \
                                     (np_random.randn(N_sc, N_r, N_t) + 1j * np_random.randn(N_sc, N_r, N_t)) / np.sqrt(2)
    
            # Apply phase shift across subcarriers
            H += H_tap * phase_shift[sc]
        
    # Normalize channel gains
    for sc in range(N_sc):
        H[sc, :, :] /= np.linalg.norm(H[sc, :, :], ord='fro')
    
    return H


def _generate_cdl_c_channel(N_sc, N_r, N_t, carrier_frequency):
    global np_random, Df

    # Generates a 3GPP 38.900 CDL-C channel with dimensions (N_sc, N_r, N_t).
    delay_taps = [0, 0.209, 0.423, 0.658, 1.18, 1.44, 1.71]  # in microseconds
    powers_dB = [-0.2, -13.5, -15.4, -18.1, -20.0, -22.1, -25.2]  # tap power in dBm

    # Convert dB to linear scale for power
    powers_linear = 10 ** (np.array(powers_dB) / 10)
    num_taps = len(powers_dB)

    # Initialize the channel matrix (complex random Gaussian per subcarrier, tap, and antenna pair)
    H = np.zeros((N_sc, N_r, N_t), dtype=np.complex128)

    # Frequency range for the subcarriers
    subcarrier_frequencies = carrier_frequency + (np.arange(N_sc) - N_sc // 2) * Df   # subcarrier indices

    # Generate channel response for each tap and apply delay phase shifts
    for tap in range(num_taps):
        # Delay in seconds
        delay = delay_taps[tap] * 1e-6  # convert from microseconds to seconds

        # Apply the phase shift for each subcarrier based on the delay
        phase_shift = np.exp(-2j * np.pi * subcarrier_frequencies * delay)  # shape: (N_sc,)

        # Complex Gaussian fading for each tap, scaled by tap power
        H_tap = np.sqrt(powers_linear[tap]) * \
                (np_random.randn(N_r, N_t, N_sc) + 1j * np_random.randn(N_r, N_t, N_sc)) / np.sqrt(2)

        # Apply phase shift across subcarriers
        H += (H_tap * phase_shift).transpose(2, 0, 1)  # Adjust dimensions (N_sc, N_r, N_t)

    # Normalize channel gains
    for sc in range(N_sc):
        H[sc, :, :] /= np.linalg.norm(H[sc, :, :], ord='fro')

    return H


def _generate_cdl_e_channel(N_sc, N_r, N_t, carrier_frequency):
    global np_random, Df

    # Generates a 3GPP 38.900 CDL-E channel with dimensions (N_sc, N_r, N_t).
    delay_taps = [0, 0.264, 0.366, 0.714, 1.53, 1.91, 3.52, 4.20, 5.35]  # in microseconds
    powers_dB = [-0.03, -4.93, -8.03, -10.77, -15.86, -18.63, -21.11, -22.50, -25.63]  # tap power in dBm

    # Convert dB to linear scale for power
    powers_linear = 10 ** (np.array(powers_dB) / 10)
    num_taps = len(powers_dB)

    # Initialize the channel matrix (complex random Gaussian per subcarrier, tap, and antenna pair)
    H = np.zeros((N_sc, N_r, N_t), dtype=np.complex128)

    # Frequency range for the subcarriers
    subcarrier_frequencies = carrier_frequency + (np.arange(N_sc) - N_sc // 2) * Df   # subcarrier indices

    # Generate channel response for each tap and apply delay phase shifts
    for tap in range(num_taps):
        # Delay in seconds
        delay = delay_taps[tap] * 1e-6  # convert from microseconds to seconds

        # Apply the phase shift for each subcarrier based on the delay
        phase_shift = np.exp(-2j * np.pi * subcarrier_frequencies * delay)

        # Complex Gaussian fading for each tap, scaled by tap power
        H_tap = np.sqrt(powers_linear[tap]) * \
                (np_random.randn(N_r, N_t, N_sc) + 1j * np_random.randn(N_r, N_t, N_sc)) / np.sqrt(2)

        # Apply phase shift across subcarriers
        H += (H_tap * phase_shift).transpose(2, 0, 1)  # Adjust dimensions (N_sc, N_r, N_t)

    # Normalize channel gains
    for sc in range(N_sc):
        H[sc, :, :] /= np.linalg.norm(H[sc, :, :], ord='fro')

    return H


def compute_large_scale_fading(dist, f_c, D_t_dB=18, D_r_dB=-1, pl_exp=2):
    global N_sc, Df
    
    subcarrier_frequencies = f_c + (np.arange(N_sc) - N_sc // 2) * Df   # subcarrier indices

    wavelength = speed_of_light / subcarrier_frequencies
    G = _linear(D_t_dB + D_r_dB) * (wavelength / (4 * np.pi * dist)) ** pl_exp

    assert np.all(G <= 1)

    return G


def compute_shadow_fading(N_sc, large_scale_fading_dB, shadow_fading_std):
    global np_random
    global N_r, N_t
    
    chi_sigma = np_random.normal(loc=0, scale=shadow_fading_std, size=N_sc)
    G_dB = np.zeros_like(large_scale_fading_dB)
    
    G_dB = large_scale_fading_dB - chi_sigma
    
    return G_dB, chi_sigma


def compress_channel(H, compression_ratio, quantization_b, epochs=200, batch_size=16, learning_rate=1e-4, plotting=False):
    if compression_ratio >= 1 or compression_ratio < 0:
        raise ValueError("Compression choose compression ratio in [0,1).")

    if compression_ratio == 0:
        return H, H, np.nan

    global np_random, seed
    global output_path

    N_sc, N_r, N_t = H.shape

    # First step, convert the real and imag parts of H into a vector of the size 2N_t
    H = np.c_[np.real(H), np.imag(H)]
    H = H.reshape((1, N_sc, N_r, 2*N_t))

    # This dimension is the dimension of the compressed channel
    latent_dim = 2 * N_r * N_t * int(np.ceil((1 - compression_ratio) * N_sc))

    # For the encoder to learn to compress, pass X_train as an input and target
    # Decoder will learn how to reconstruct original
    X_train = H
    X_test = H

    autoencoder = Autoencoder(latent_dim,
                              shape=H.shape[1:], seed=seed)
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=losses.MeanSquaredError())

    autoencoder.train_on_batch(X_train, X_train)
    # callback = keras.callbacks.EarlyStopping(monitor='loss', patience=8)

    # Check if a keras object is stored, then use it.  Otherwise, train one.
    try:
        autoencoder.load_weights(f'autoencoder_compression_{compression_ratio}')
        # no training is done thus any score here is NaN.
    except Exception as e:
        print(f'Failed to load model due to {e}.  Training from scratch.')

        # Normalize train and test data
        min_val = tf.reduce_min(X_train)
        max_val = tf.reduce_max(X_train)

        X_train = (X_train - min_val) / (max_val - min_val)
        X_test = (X_test - min_val) / (max_val - min_val)

        # Build a foundational model
        history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
                        shuffle=True, # callbacks=[callback],
                        validation_data=(X_test, X_test))

        _plot_keras_learning(history, f'autoencoder_compression_{compression_ratio}')
        autoencoder.save_weights(f'autoencoder_compression_{compression_ratio}', save_format="tf")

    # Encoded samples are compressed and the latent vector has a lower
    # dimension as the result of compression.
    H_compressed = autoencoder.encoder.predict(X_test) # This is a vectorized channel.

    # Compress the channel at the transmit side
    # then quantize it and send it to the receiver, which will reconstruct it.
    H_compressed = quantize(H_compressed, quantization_b)

    # Decoder tries to reconstruct the true signal
    H_reconstructed = autoencoder.decoder.predict(H_compressed)

    H = H.reshape((-1, N_r, 2*N_t))
    H_compressed = H_compressed.reshape((-1, N_r, 2*N_t))
    H_reconstructed = H_reconstructed.reshape((-1, N_r, 2*N_t))

    # Now reassemble the channel, eliminating the dimension (this is correct)
    H = H[:, :, :N_t] + 1j * H[:, :, N_t:]
    H_compressed = H_compressed[:, :, :N_t] + 1j * H_compressed[:, :, N_t:]
    H_reconstructed = H_reconstructed[:, :, :N_t] + 1j * H_reconstructed[:, :, N_t:]

    error = _mse(H, H_reconstructed)

    if plotting:
        vmin = np.abs(H).min()
        vmax = np.abs(H).max()
        plot_channel(H_compressed, vmin=vmin, vmax=vmax, filename=f'compressed_{compression_ratio}')
        plot_channel(H_reconstructed, vmin=vmin, vmax=vmax, filename=f'reconstr_{compression_ratio}')

    return H_compressed, H_reconstructed, error


def denoise_signal(x_noisy, x_original, epochs=100, batch_size=16, learning_rate=1e-4, plotting=False):
    global seed

    # This is for real-valued vectors only.
    assert len(x_noisy.shape) == 1

    latent_dim = 128
    autoencoder = Autoencoder(latent_dim, shape=x_original.shape, seed=seed)
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=losses.MeanSquaredError())

    # First train on same clean data
    history = autoencoder.fit(x_original, x_original,
                             epochs=epochs, batch_size=batch_size,
                             shuffle=True)

    # Next, train on noisy data.
    history = autoencoder.fit(x_noisy, x_original,
                             epochs=epochs, batch_size=batch_size,
                             shuffle=True)

    # Now try to remove the noise.
    encoded_x = autoencoder.encoder(x_noisy).numpy()
    denoised_x = autoencoder.decoder(encoded_x).numpy()[:,0]

    if plotting:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(x_original, '--b', label='orig')
        ax.plot(x_noisy, 'r', alpha=0.5, label='noisy')
        ax.plot(denoised_x, 'k',  alpha=0.5, label='denoised')

        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        plt.savefig(f'{output_path}/denoising.pdf', format='pdf', dpi=fig.dpi)
        #tikzplotlib.save(f'{output_path}/denoising.tikz')

        plt.show()
        plt.close(fig)

    return denoised_x


def _vec(H):
    # H is numpy array
    return H.flatten(order='F')


def _mse(H_true, H_estimated):
    return np.mean(np.abs(H_true - H_estimated) ** 2)


def _dB(X):
    return 10 * np.log10(X)


def _linear(X):
    return 10 ** (X / 10.)


def detect_symbols(z, alphabet, algorithm):
    global np_random

    if algorithm == 'kmeans':
        return _detect_symbols_kmeans(z, alphabet)

    if algorithm == 'ML':
        return _detect_symbols_ML(z, alphabet)
    
    # Supervised learning detections
    y = alphabet['m'].values
    X = np.c_[np.real(alphabet['x']), np.imag(alphabet['x'])]

    X_infer = z.flatten()
    X_infer = np.c_[np.real(X_infer), np.imag(X_infer)]
    
    if algorithm == 'ensemble':    
        _, [training_accuracy_score, test_accuracy_score], y_infer =  \
            _detect_symbols_ensemble(X, y, X_infer)
                
        # print(f'Ensemble training accuracy is {training_accuracy_score:.2f}.')
        # print(f'Ensemble test accuracy is {test_accuracy_score:.2f}.')
    
    if algorithm == 'DNN':
        _, [train_acc_score, test_acc_score], y_infer = \
            _detect_symbols_DNN(X, y, X_infer)

        # print(f'DNN training accuracy is {train_acc_score:.2f}.')
        # print(f'DNN test accuracy is {test_acc_score:.2f}.')

    df = pd.merge(pd.DataFrame(data={'m': y_infer}), alphabet, how='left', on='m')

    # Reverse the flatten operation
    symbols = df['x'].values.reshape(z.shape)
    information = df['m'].values.reshape(z.shape)
    bits_i = df['I'].values.reshape(z.shape)
    bits_q = df['Q'].values.reshape(z.shape)
    
    return information, symbols, [bits_i, bits_q]


def _detect_symbols_kmeans(x_sym_hat, alphabet):
    global np_random

    x_sym_hat_flat = x_sym_hat.flatten()

    X = np.real(x_sym_hat_flat)
    X = np.c_[X, np.imag(x_sym_hat_flat)]
    X = X.astype('float32')

    centroids = np.c_[np.real(alphabet['x']), np.imag(alphabet['x'])]

    # Intialize k-means centroid location deterministcally as a constellation
    kmeans = KMeans(n_clusters=M_constellation, init=centroids, n_init=1,
                    random_state=np_random).fit(centroids)

    information = kmeans.predict(X)
    df_information = pd.DataFrame(data={'m': information})

    df = df_information.merge(alphabet, how='left', on='m')
    symbols = df['x'].values.reshape(x_sym_hat.shape)
    bits_i = df['I'].values.reshape(x_sym_hat.shape)
    bits_q = df['Q'].values.reshape(x_sym_hat.shape)

    return information, symbols, [bits_i, bits_q]


def _detect_symbols_ensemble(X_train, y_train, X_test):
    global np_random

    y_train = y_train.ravel()

    # The classifier hyperparameters need to be tuned.
    base_estimator = RandomForestClassifier(n_estimators=100, n_jobs=-1, 
                                            criterion='entropy',
                                            class_weight='balanced',
                                            random_state=np_random)

    # hyperparameters = {'criterion': ['entropy', 'gini'],
    #                     'min_impurity_decrease': [0.1, 0.2],
    #                     'min_weight_fraction_leaf': [0.1, 0.3],
    #                     'min_samples_split': [2, 10]
    #                     }

    # # kf = KFold(n_splits=3, shuffle=True, random_state=np_random)
    # loo = LeaveOneOut()
    # clf = GridSearchCV(base_estimator, param_grid=hyperparameters,
    #                    cv=loo, n_jobs=-1, scoring='roc_auc_ovr_weighted',
    #                    verbose=0)

    clf = base_estimator  # No cross validation is done.
    clf.fit(X_train, y_train)

    y_test_pred = clf.predict(X_test)
    training_accuracy_score = clf.score(X_train, y_train)
    
    return clf, [training_accuracy_score, np.nan], y_test_pred


def _detect_symbols_ML(symbols, alphabet):
    # Flatten symbols for easier processing
    symbols_flat = symbols.flatten()

    # Calculate distances to each constellation point
    distances = np.abs(symbols_flat[:, None] - alphabet['x'].values[None, :]) ** 2

    # Find the index of the closest constellation point
    m_star_indices = np.argmin(distances, axis=1)

    # Map indices to alphabet entries
    detected_alphabet = alphabet.iloc[m_star_indices]

    # Extract relevant information
    information = detected_alphabet['m'].values.reshape(symbols.shape)
    detected_symbols = detected_alphabet['x'].values.reshape(symbols.shape)
    bits_i = detected_alphabet['I'].values.reshape(symbols.shape)
    bits_q = detected_alphabet['Q'].values.reshape(symbols.shape)

    return information, detected_symbols, [bits_i, bits_q]


def _detect_symbols_DNN(X_train, y_train, X_test, depth=6, width=8,
                        epoch_count=512, batch_size=16):
    _, nX = X_test.shape

    # Make more data since the constellation size is small.
    # This improves the learning significantly.
    X_train_augmented = np.empty((0, nX))
    epsilons = [1e-2, 1e-3]
    
    for perturb in epsilons:
        X_train_i = X_train + np_random.normal(0, scale=perturb, size=X_train.shape)
        X_train_augmented = np.r_[X_train_augmented, X_train_i]
    
    X_train = np.r_[X_train, X_train_augmented]
    y_train = np.tile(y_train, len(epsilons) + 1)
    
    Y_train = keras.utils.to_categorical(y_train)
    
    # start_time = time.time()    
    model, [training_accuracy_score, test_accuracy_score], y_test_pred = _train_dnn(X_train, Y_train, X_test,
                                            depth=depth, width=width, epoch_count=epoch_count,
                                            batch_size=batch_size, 
                                            learning_rate=1e-2)

    # end_time = time.time()
    # print("Training took {:.2f} mins.".format((end_time - start_time) / 60.))

    return model, [training_accuracy_score, test_accuracy_score], y_test_pred


def _train_dnn(X_train, Y_train, X_test, depth, width, epoch_count,
               batch_size, learning_rate):
    global prefer_gpu
    global np_random
    
    use_cuda = len(tf.config.list_physical_devices('GPU')) > 0 and prefer_gpu
    device = "/gpu:0" if use_cuda else "/cpu:0"

    _, nX = X_train.shape
    _, nY = Y_train.shape

    # If a DNN model is stored, use it.  Otherwise, train a DNN.
    try:
        dnn_classifier = \
            keras.models.load_model('dnn_detection.keras',
                                    custom_objects={'__loss_fn_classifier': __loss_fn_classifier})
        training_accuracy_score = np.nan # no training is done.
    except Exception as e:
        print(f'Failed to load model due to {e}.  Training from scratch.')
        dnn_classifier = __create_dnn(input_dimension=nX, output_dimension=nY,
                                     depth=depth, width=width,
                                     learning_rate=learning_rate)

        with tf.device(device):
            history = dnn_classifier.fit(X_train, Y_train, epochs=epoch_count,
                               shuffle=False, 
                               batch_size=batch_size)
            dnn_classifier.save('dnn_detection.keras')
            Y_pred = dnn_classifier.predict(X_train)
            _, training_accuracy_score, _ = dnn_classifier.evaluate(X_train, Y_train)
            y_train_pred = np.argmax(Y_pred, axis=1)
            _plot_keras_learning(history, filename='dnn_detection')
            
    # Perform inference.
    with tf.device(device):
        Y_test_pred = dnn_classifier.predict(X_test, verbose=0)
        
    y_test_pred = np.argmax(Y_test_pred, axis=1)
    
    return dnn_classifier, [training_accuracy_score, np.nan], y_test_pred 


def __loss_fn_classifier(Y_true, Y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy()
    return cce(Y_true, Y_pred)


def _train_lstm(X_train, X_test, Y_train, Y_test, lookbacks, depth, width, epoch_count, batch_size, learning_rate):
    global prefer_gpu

    use_cuda = len(tf.config.list_physical_devices('GPU')) > 0 and prefer_gpu
    device = "/gpu:0" if use_cuda else "/cpu:0"

    # Store number of learning features
    mX, nX = X_train.shape
    _, nY = Y_train.shape

    # # Scale X features
    # sc = MinMaxScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    # Now, reshape input to be 3-D: [timesteps, batch, feature]
    X_train = np.reshape(X_train, (-1, lookbacks + 1, X_train.shape[1] // (lookbacks + 1)))
    X_test = np.reshape(X_test, (-1, lookbacks + 1, X_test.shape[1] // (lookbacks + 1)))

    Y_train = np.reshape(Y_train, (-1, nY))
    Y_test = np.reshape(Y_test, (-1, nY))

    # If a DNN model is stored, use it.  Otherwise, train a DNN.
    try:
        model = keras.models.load_model('lstm_classifier.keras')
        training_accuracy_score = np.nan # no training is done.
    except Exception as e:
        print(f'Failed to load model due to {e}.  Training from scratch.')

        model = __create_lstm(input_shape=(X_train.shape[1], X_train.shape[2]),
                                  output_shape=(X_train.shape[1], nY),
                                  depth=depth, width=width, learning_rate=learning_rate)

        with tf.device(device):
            # patience = 2 * lookbacks + 10
            # callback_list = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience)]

            # if callback is not None:
            #     callback_list = callback_list + [callback]

            history = model.fit(X_train, Y_train, epochs=epoch_count, batch_size=batch_size,
                                # callbacks=callback_list,
                                validation_data=(X_test, Y_test),
                                #validation_split=0.5,
                                # verbose=False,
                                shuffle=False)
            
            #  model.reset_states()  # For reproducibility, but does not work.
            model.save('lstm_classifier.keras')
            Y_pred = model.predict(X_train)
            _, training_accuracy_score = model.evaluate(X_train, Y_train)
            y_train_pred = np.argmax(Y_pred, axis=1)
            _plot_keras_learning(history, filename='lstm')

    # Perform inference.
    with tf.device(device):
        Y_test_pred = model.predict(X_test)
        loss, test_accuracy_score = model.evaluate(X_test, Y_test)
    y_test_pred = np.argmax(Y_test_pred, axis=1)

    return model, [training_accuracy_score, test_accuracy_score], y_test_pred


def __create_cnn(learning_rate):
    # Builds a CNN model that takes complex numbers and returns the real and imag parts.

    model = keras.Sequential()
    model.add(keras.Input(shape=(2,1)))   # Data must have dimensions (-1, 2, 1).  Kernel size is 2 (= 2 features) aligned with the input first dimension.
    model.add(layers.Conv1D(64, kernel_size=2, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(64, kernel_size=1, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dropout(0.3))
    model.add(layers.Dense(2, activation='linear'))  # output is 2D (real and imaginary parts)

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])

    # Reporting the number of parameters
    print(model.summary())

    num_params = model.count_params()
    print('Number of parameters: {}'.format(num_params))

    return model


def __create_dnn(input_dimension, output_dimension, depth, width, learning_rate):
    global seed    
    tf.random.set_seed(seed)  # Reproducibility
    
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dimension,)))
    
    for hidden in range(depth):
        model.add(layers.Dense(width, activation='relu'))
   
    model.add(layers.Dense(output_dimension, activation='softmax'))
    
    model.compile(loss=__loss_fn_classifier, optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy', 'categorical_crossentropy']) # Accuracy here is okay.  These metrics are what .evaluate() returns.
    
    # Reporting the number of parameters
    print(model.summary())
    
    num_params = model.count_params()
    print('Number of parameters: {}'.format(num_params))
    
    return model


def __create_lstm(input_shape, output_shape, depth, width, learning_rate):
    global seed
    tf.random.set_seed(seed)  # Reproducibility but it does not work on LSTMs.

    mX, nX = input_shape
    _, nY = output_shape # this becomes the dummy coded number of beams

    model = keras.Sequential()
    model.add(layers.LSTM(input_shape=(mX, nX), units=width, return_sequences=False))

    for hidden in np.arange(depth):
        model.add(layers.Dense(width, activation='relu'))

    model.add(layers.Dropout(0.3))
    # model.add(layers.LSTM(units=width, return_sequences=False))
    model.add(layers.Dense(nY, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    # Reporting the number of parameters
    print(model.summary())

    num_params = model.count_params()
    print('Number of parameters: {}'.format(num_params))

    return model


def bits_from_IQ(x_b_i, x_b_q):
    assert x_b_i.shape == x_b_q.shape
    shape = x_b_i.shape

    bits = np.array([f"{i}{q}" for i, q in zip(np.array(x_b_i).flatten(), np.array(x_b_q).flatten())])
    bits = np.reshape(bits, shape)

    flattened = ''.join(bits.flatten())

    return bits, flattened


def compute_bit_error_rate(a, b):
    assert(len(a) == len(b))

    length = len(a)
    bit_error = 0
    for idx in range(length):
        if a[idx] != b[idx]:
            bit_error += 1

    return bit_error / length


def compute_crc(x_bits_orig, crc_generator):
    # Introduce CRC to x
    length_crc = len(bin(crc_generator)[2:])

    x_bits = x_bits_orig.zfill(length_crc)

    crc = 0
    for position, value in enumerate(bin(crc_generator)[2:]):
        if value == '1':
            crc = crc ^ int(x_bits[position])

    crc = bin(crc)[2:]
    crc = crc.zfill(length_crc)

    return crc


def compute_precoder_combiner(H, P_BS, algorithm='SVD_Waterfilling'):
    N_sc, N_r, N_t = H.shape
    N_s = min(N_r, N_t)

    if N_s == 1 and algorithm != 'dft_beamforming':
        raise ValueError('Only beamforming is supported for MISO.  Check the setting of precoder.')

    if algorithm == 'identity':
        F = np.eye(N_t) #, N_s)
        Gcomb = np.eye(N_r)
        F = np.repeat(F[np.newaxis, :, :], N_sc, axis=0)  # Repeat for all subcarriers
        Gcomb = np.repeat(Gcomb[np.newaxis, :, :], N_sc, axis=0)  # Repeat for all subcarriers
        return F, Gcomb

    if algorithm == 'dft_beamforming':
        if N_r != 1:
            raise ValueError("Channel must have a single receive antenna.")
        F = _dft_codebook(N_t)

        # Search for the optimal beamformer
        max_sinr = -np.inf
        for idx in range(N_t):
            sinr = np.abs(np.vdot(H[0, :, :], F[idx, :])) ** 2
            if sinr > max_sinr:
                max_sinr = sinr
                f_opt = F[idx, :] # extract a vector

        Gcomb = np.ones((N_sc, 1))
        return f_opt, Gcomb

    if algorithm == 'SVD':
        U, S, Vh = _svd_precoder_combiner(H)

        F = np.conjugate(np.transpose(Vh, (0, 2, 1)))  # F is equal to V
        Gcomb = np.conjugate(np.transpose(U, (0, 2, 1)))  # G is equal to U*

        return F, Gcomb

    if algorithm == 'SVD_Waterfilling':
        U, S, Vh = _svd_precoder_combiner(H)

        try:
            D = _waterfilling(S[0, :], P_BS) # The power allocation matrix
            Dinv = np.linalg.inv(D)
        except:
            print('Waterfilling failed.  Returning identity power allocation.')
            D = np.eye(S.shape[1])
            Dinv = np.eye(S.shape[1])

        F = np.conjugate(np.transpose(Vh, (0, 2, 1)))@D
        Gcomb = Dinv@np.conjugate(np.transpose(U, (0, 2, 1)))

        return F, Gcomb


def _dft_codebook(N_t, k_oversample=1):
    global f_c

    wavelength = speed_of_light / f_c

    d = wavelength / 2.  # antenna spacing
    k = 2. * np.pi / wavelength  # wave number

    # Theta is [0, pi]
    theta = np.pi * np.arange(start=0., stop=1., step=1./(k_oversample*N_t))

    # The beamforming codebook
    F = np.zeros((N_t, theta.shape[0]), dtype=np.complex128)

    # Add DFT beamforming vectors to the codebook
    for i, theta_i in enumerate(theta):
        exponent = 1j * k * d * np.cos(theta_i) * np.arange(N_t)
        f_i = 1. / np.sqrt(N_t) * np.exp(exponent)
        F[i, :] = f_i

    return F


def _find_channel_eigenmodes(H, subcarrier_idx=0):
    _, S, _ = np.linalg.svd(H[subcarrier_idx,:,:], full_matrices=False)
    eigenmodes = S ** 2

    return eigenmodes


def _svd_precoder_combiner(H):
    U, S, Vh = np.linalg.svd(H, full_matrices=True)
    return U, S, Vh


def _waterfilling(S, power):
    n_modes = len(S)
    
    # Ensure eigenmodes are non-zero
    if np.any(S < 1):
        raise ValueError("Channel has very weak eigenmodes which prevents waterfilling.  Please use a different channel or a different precoder.")

    power_alloc = np.zeros(n_modes)  # Initialize power allocation array

    # Bisection method to find the optimal water level
    lower_bound = 0
    upper_bound = power + np.max(1 / S)  # Initial bounds for water level

    tolerance = 1e-5  # Precision for bisection convergence
    while upper_bound - lower_bound > tolerance:
        water_level = (upper_bound + lower_bound) / 2.0

        # Waterfilling formula: allocate power where water_level > 1/eigenmodes
        power_alloc = np.maximum(0, water_level - 1 / S)

        total_allocated_power = np.sum(power_alloc)  # Total power allocated

        if total_allocated_power > power:
            upper_bound = water_level  # Decrease water level
        else:
            lower_bound = water_level  # Increase water level

    P = np.diag(power_alloc)

    # Return the final power allocation across the eigenmodes
    # Note that np.trace(P) is equal to power.
    return P


def _matrix_vector_multiplication(A, B):
    try:
        N_sc = A.shape[0]

        ans = np.zeros((N_sc, A.shape[1]), dtype=np.complex128)
        for n in range(N_sc):
            ans[n, :] = A[n, :]@B[n]
        return ans
    except:
        # This is likely due to beamforming.
        return A@B


def plot_scatter(df, xlabel, ylabel, filename=None):
    global output_path
    
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.scatter(df[xlabel], df[ylabel], s=10, c='r', edgecolors='none', alpha=0.2)

    plt.grid(which='both')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    plt.show()
    plt.close(fig)
    

def plot_performance(df, xlabel, ylabel, semilogy=True, filename=None):
    global output_path

    cols = list(set([xlabel, ylabel, 'snr_dB']))
    df = df[cols]
    df_plot = df.groupby('snr_dB').mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))
    if semilogy:
        ax.set_yscale('log')
    ax.tick_params(axis=u'both', which=u'both')
    plt.plot(df_plot[xlabel].values, df_plot[ylabel].values, '--bo', alpha=0.7,
             markeredgecolor='k', markerfacecolor='r', markersize=6)

    plt.grid(which='both', axis='both')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(f'{output_path}/performance_{ylabel}_{filename}.pdf', format='pdf', dpi=fig.dpi)
        #tikzplotlib.save(f'{output_path}/performance_{ylabel}_{filename}.tikz')
    plt.show()
    plt.close(fig)


def plot_pdf(X, text=None, algorithm='empirical', num_bins=200, filename=None):
    global output_path

    X_re = np.real(X)
    X_im = np.imag(X)

    if text is None:
        text = ''
    else:
        text = f'-{text}'

    is_complex = True
    if np.sum(X_im) == 0:
        is_complex = False

    fig, ax = plt.subplots(figsize=(12, 6))

    if algorithm == 'empirical':
        for label, var in zip(['Re[X]', 'Im[X]'], [X_re, X_im]):
            # Is this a real quantity?
            if not is_complex and label == 'Im[X]':
                continue
            counts, bin_edges = np.histogram(var, bins=num_bins, density=True)
            pdf = counts / counts.sum()
            bin_edges = np.insert(bin_edges, 0, bin_edges[0] - (bin_edges[2] - bin_edges[1]))
            ax.plot(bin_edges[2:], pdf, '-', linewidth=1.5, label=f'{label}{text}')
        ax.legend()

    if algorithm == 'KDE':
        df_re = pd.DataFrame(X_re).add_suffix(f'-{text}-Re')
        df_im = pd.DataFrame(X_im).add_suffix(f'-{text}-Im')

        df = df_re.copy()
        if is_complex:
            df = pd.concat([df, df_im], axis=1, ignore_index=False)
        try:
            df.plot(kind='kde', bw_method=0.3, ax=ax)
        except Exception as e:
            print(f"Failed to generate plot due to {e}.")

    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('p(X)')
    plt.tight_layout()

    if filename is not None:
        plt.savefig(f'{output_path}/pdf_{algorithm}_{filename}.pdf', format='pdf', dpi=fig.dpi)
        #tikzplotlib.save(f'{output_path}/pdf_{algorithm}_{filename}.tikz')
    plt.show()
    plt.close(fig)


def _plot_keras_learning(history, filename=None):
    global output_path

    fig, ax = plt.subplots(figsize=(9,6))

    if 'val_loss' in history.history.keys():
        plt.plot(history.history['val_loss'], 'r', lw=2, label='Validation loss')
    plt.plot(history.history['loss'], 'b', lw=2, label='Loss')

    plt.grid(which='both', axis='both')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()

    if filename is not None:
        plt.savefig(f'{output_path}/history_keras_{filename}.pdf', format='pdf', dpi=fig.dpi)
        #tikzplotlib.save(f'{output_path}/history_keras_{filename}.tikz')
    plt.show()
    plt.close(fig)


def _plot_constellation(constellation, annotate=False, filename=None):
    global output_path

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plt.scatter(constellation['x_I'], constellation['x_Q'], c='k', marker='o', lw=2)

    if annotate:
        for idx, row in constellation.iterrows():
            x, y = row[['x_I', 'x_Q']]
            if y < 0:
                yshift = -0.1
            else:
                yshift = 0.05

            ax.text(row['x_I'], row['x_Q'] + yshift, s='{}{}'.format(row['I'], row['Q']))

    plt.grid(True)
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.tight_layout()

    if filename is not None:
        plt.savefig(f'{output_path}/constellation_{filename}.pdf', format='pdf', dpi=fig.dpi)
        #tikzplotlib.save(f'{output_path}/constellation_{filename}.tikz')
    plt.show()
    plt.close(fig)


def plot_channel(channel, vmin=None, vmax=None, filename=None):
    global output_path

    N_sc, N_r, N_t = channel.shape

    # Only plot first receive antenna
    H = channel[:,0,:]

    dB_gain = _dB(np.abs(H) ** 2 + 1e-5)

    # Create a normalization object
    norm = mcolors.Normalize(vmin=dB_gain.min(), vmax=dB_gain.max())

    # plt.rcParams['font.size'] = 36
    # plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(figsize=(12, 6))

    plt.imshow(dB_gain, aspect='auto', norm=norm)

    plt.xlabel('TX Antennas')
    plt.ylabel('Subcarriers')

    plt.xticks(range(N_t))
    plt.tight_layout()

    if filename is not None:
        plt.savefig(f'{output_path}/channel_{filename}.pdf', format='pdf', dpi=fig.dpi)
        #tikzplotlib.save(f'{output_path}/channel_{filename}.tikz')
    plt.show()
    plt.close(fig)


def plot_IQ(signal, filename=None):
    global output_path

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for idx in range(signal.shape[1]):
        X = signal[:, idx]
        plt.scatter(np.real(X), np.imag(X), marker='o', lw=2, label=f'TX ant. {idx}')

    plt.grid(True)
    plt.legend()
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.tight_layout()

    if filename is not None:
        plt.savefig(f'{output_path}/IQ_{filename}.pdf', format='pdf', dpi=fig.dpi)
        #tikzplotlib.save(f'{output_path}/IQ_{filename}.tikz')
    plt.show()
    plt.close(fig)


def _tabular_reinforcement_learning(max_episodes_to_run, max_timesteps_per_episode, plotting=False):
    global seed
    global prefer_gpu

    action_size = 3  # control command index up, nothing, index down

    env = radio_environment(action_size=action_size, min_reward=-1, max_reward=10, target=11,
                            max_step_count=max_timesteps_per_episode, seed=seed)

    agent = TabularAgent(state_size=env.observation_space.shape[0], action_size=action_size, seed=seed)

    successful = False
    episode_successful = [] # a list to save the optimal episodes

    optimal_episode = None
    optimal_reward = -np.inf

    print('Ep. | TS | Action | Power Control Index | Current SINR | Recv. SINR | Reward')
    _print_divider(rep=80)

    # These will keep track of the average Q values and losses experienced
    # per time step per episode.
    Q_values = []
    losses = []

    optimal_sinr_progress = []
    optimal_power_control_progress = []
    optimal_actions = []

    # Implement the Q-learning algorithm
    # Each episode z has a number of timesteps t
    for episode_index in 1 + np.arange(max_episodes_to_run):
        observation = env.reset()

        total_reward = 0
        done = False
        episode_loss = []
        episode_q = []

        # Read current envirnoment
        (current_sinr, _, pc_index) = observation
        action = agent.begin_episode(observation)

        actions = [action]
        sinr_progress = [current_sinr] # needed for the SINR based on the episode.
        power_control_progress = [pc_index]

        for timestep_index in 1 + np.arange(max_timesteps_per_episode):
            # Take a step
            next_observation, reward, done, abort = env.step(action)
            (current_sinr, received_sinr, pc_index) = next_observation

            loss, q = agent.get_performance()
            episode_loss.append(loss)
            episode_q.append(q)

            # make next_state the new current state for the next frame.
            observation = next_observation
            total_reward += reward

            successful = done and (abort == False)

            # Let us know how we did.
            print(f'{episode_index}/{max_episodes_to_run} | {timestep_index} | {action} | {pc_index} | {current_sinr:.2f} dB | {received_sinr:.2f} dB | {total_reward:.2f} | ', end='')

            sinr_progress.append(received_sinr)
            power_control_progress.append(pc_index)

            if abort == True:
                print('Episode aborted.')
                break

            if successful:
                print('Episode successful.')
                break

            # Update for the next time step
            action = agent.act(observation, reward)
            actions.append(action)
            print()
        # End For

        # Everytime an episode ends, compute its statistics.
        loss_z = np.mean(episode_loss)
        q_z = np.mean(episode_q)

        losses.append(loss_z)
        Q_values.append(q_z)

        if successful:
            episode_successful.append(episode_index)
            if (total_reward >= optimal_reward):
                optimal_reward, optimal_episode = total_reward, episode_index
                optimal_sinr_progress = sinr_progress
                optimal_power_control_progress = power_control_progress
                optimal_actions = actions
            optimal = 'Episode {}/{} has generated the highest reward {:.2f} yet.'.format(optimal_episode, max_episodes_to_run, optimal_reward)
            print(optimal)
        else:
            reward = 0
        _print_divider(rep=80)
    # End For (episodes)

    # Plot the current episode
    if plotting:
        _plot_Q_learning_performance(losses, max_episodes_to_run, is_loss=True, filename='tabular_loss')
        _plot_Q_learning_performance(Q_values, max_episodes_to_run, is_loss=False, filename='tabular')

        if len(episode_successful) > 0:
            _plot_environment_measurements(optimal_sinr_progress, max_timesteps_per_episode + 1, measurement='SINR_dB', filename='tabular')
            _plot_agent_actions(optimal_actions, max_timesteps_per_episode, filename='tabular')

    if (len(episode_successful) == 0):
        print("Goal cannot be reached after {} episodes.  Try to increase maximum episodes.".format(max_episodes_to_run))

    return Q_values, losses, optimal_episode, optimal_reward, optimal_sinr_progress, optimal_power_control_progress


def _deep_reinforcement_learning(max_episodes_to_run, max_timesteps_per_episode, batch_size=16, plotting=False):
    global seed
    global prefer_gpu

    action_size = 3   # control command index up, nothing, index down

    env = radio_environment(action_size=action_size, min_reward=-1, max_reward=10, target=11,
                            max_step_count=max_timesteps_per_episode, seed=seed)

    agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=action_size, prefer_gpu=prefer_gpu, seed=seed)

    successful = False
    episode_successful = []  # a list to save the optimal episodes

    optimal_episode = None
    optimal_reward = -np.inf

    print('Ep. | TS | Action | Power Control Index | Current SINR | Recv. SINR | Reward')
    _print_divider(rep=80)

    # These will keep track of the average Q values and losses experienced
    # per time step per episode.
    Q_values = []
    losses = []

    optimal_sinr_progress = []
    optimal_power_control_progress = []
    optimal_actions = []

    # Implement the Q-learning algorithm
    # Each episode z has a number of timesteps t
    for episode_index in 1 + np.arange(max_episodes_to_run):
        observation = env.reset()

        total_reward = 0
        done = False
        episode_loss = []
        episode_q = []

        # Read current envirnoment
        (current_sinr, _, pc_index) = observation
        action = agent.begin_episode(observation)

        actions = [action]
        sinr_progress = [current_sinr]  # needed for the SINR based on the episode.
        power_control_progress = [pc_index]

        for timestep_index in 1 + np.arange(max_timesteps_per_episode):
            # Take a step
            next_observation, reward, done, abort = env.step(action)
            (current_sinr, received_sinr, pc_index) = next_observation

            # Remember the previous state, action, reward, and done
            agent.remember(observation, action, reward, next_observation, done)

            # Sample replay batch from memory
            sample_size = min(len(agent.memory), batch_size)

            # Learn control policy
            loss, q = agent.replay(sample_size)

            episode_loss.append(loss)
            episode_q.append(q)

            # make next_state the new current state for the next frame.
            observation = next_observation
            total_reward += reward

            successful = done and (abort == False)

            # Let us know how we did.
            print(f'{episode_index}/{max_episodes_to_run} | {timestep_index} | {action} | {pc_index} | {current_sinr:.2f} dB | {received_sinr:.2f} dB | {total_reward:.2f} | ', end='')

            sinr_progress.append(received_sinr)
            power_control_progress.append(pc_index)

            if abort == True:
                print('Episode aborted.')
                break

            if successful:
                print('Episode successful.')
                break

            # Update for the next time step
            action = agent.act(observation)
            actions.append(action)
            print()
        # End For

        # Every time an episode ends, compute its statistics.
        loss_z = np.mean(episode_loss)
        q_z = np.mean(episode_q)

        losses.append(loss_z)
        Q_values.append(q_z)

        if successful:
            episode_successful.append(episode_index)
            if (total_reward >= optimal_reward):
                optimal_reward, optimal_episode = total_reward, episode_index
                optimal_sinr_progress = sinr_progress
                optimal_power_control_progress = power_control_progress
                optimal_actions = actions
            optimal = 'Episode {}/{} has generated the highest reward {:.2f} yet.'.format(optimal_episode, max_episodes_to_run, optimal_reward)
            print(optimal)
        else:
            reward = 0
        _print_divider(rep=80)
    # End For (episodes)

    # Plot the current episode
    if plotting:
        _plot_Q_learning_performance(losses, max_episodes_to_run, is_loss=True, filename='dqn_loss')
        _plot_Q_learning_performance(Q_values, max_episodes_to_run, is_loss=False, filename='dqn')

        if len(episode_successful) > 0:
            _plot_environment_measurements(optimal_sinr_progress, max_timesteps_per_episode + 1, measurement='SINR_dB', filename='dqn')
            _plot_agent_actions(optimal_actions, max_timesteps_per_episode, filename='dqn')

    if (len(episode_successful) == 0):
        print("Goal cannot be reached after {} episodes.  Try to increase maximum episodes.".format(max_episodes_to_run))

    tf.keras.backend.clear_session() # free up GPU memory

    return Q_values, losses, optimal_episode, optimal_reward, optimal_sinr_progress, optimal_power_control_progress


def _plot_Q_learning_performance(values, num_episodes, is_loss=False, filename=None):
    global output_path

    fig = plt.figure(figsize=(8, 5))

    y_label = 'Expected Action-Value Q' if not is_loss else r'Expected Loss'
    plt.xlabel('Episode')
    plt.ylabel(y_label)

    color = 'b' if not is_loss else 'r'
    plt.plot(1 + np.arange(num_episodes), values, linestyle='-', color=color)

    # These are integer actions.
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.grid(True)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(f'{output_path}/Qlearning_perf_{filename}.pdf', format='pdf', dpi=fig.dpi)
        #tikzplotlib.save(f'{output_path}/Qlearning_perf_{filename}.tikz')
    plt.show()
    plt.close(fig)


def _plot_environment_measurements(environment_measurements, time_steps, measurement=None, filename=None):
    global output_path

    fig = plt.figure(figsize=(8, 5))
    plt.plot(1 + np.arange(time_steps), environment_measurements, color='k')
    plt.xlabel('Time step t')

    if measurement is not None:
        plt.ylabel(measurement)

    plt.grid(True)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(f'{output_path}/environment_{measurement}_{filename}.pdf', format='pdf', dpi=fig.dpi)
        #tikzplotlib.save(f'{output_path}/environment_{measurement}_{filename}.tikz')
    plt.show()
    plt.close(fig)


def _plot_agent_actions(agent_actions, time_steps, filename=None):
    global output_path

    fig = plt.figure(figsize=(8, 5))
    plt.step(1 + np.arange(time_steps), agent_actions, color='k')
    plt.xlabel('Time step t')
    plt.ylabel('Action')

    # These are integer actions.
    fig.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.grid(True)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(f'{output_path}/actions_{filename}.pdf', format='pdf', dpi=fig.dpi)
        #tikzplotlib.save(f'{output_path}/actions_{filename}.tikz')
    plt.show()
    plt.close(fig)


def predict_trajectory_with_LSTM(df, target_variable, depth=1, width=2,
                                 lookahead_time=1, max_lookback=3,
                                 training_size=0.7, batch_size=16,
                                 epoch_count=10):
    if df is None:
        M = 50 # 50 records
        T = 10 # each record has 10 time steps
        df = pd.DataFrame(data={'Time': np.arange(M*T)})
        df['beam_index'] = np_random.randint(0,5, size=M*T)
        df['SINR'] = df['beam_index'] * 5 - np_random.uniform(0,10, size=M*T)
        df['RSRP'] = df['SINR'] - np_random.uniform(-94, -85, size=M*T)

        target_variable = 'beam_index'

    # lookahead_time = 1 # How far into the future are we predicting?
    # max_lookback = 3 # How many steps backwards do we need to look at?

    # X has a dimension of MT x {2 (max_lookback + 1)) * ncol}, with 2 being diff and shift.
    # y has a dimension of MT

    df_engf, target_var = timeseries_engineer_features(df, target_variable, lookahead_time, max_lookback, dropna=False)

    try:
        X_train, X_test, Y_train, Y_test, y_test_true = timeseries_train_test_split(df_engf,
                                                                 target_var, time_steps_per_block=T, train_size=training_size)
    except Exception as e:
        print(f'Critical: Failed to split data due to {e}.')
        raise e

    learning_rate = 1e-2
    start_time = time.time()
    # mc = ModelCheckpoint(model_filename, monitor='val_accuracy', mode='max', save_best_only=True)
    _, [training_accuracy_score, test_accuracy_score], y_test_pred = _train_lstm(X_train, X_test, Y_train, Y_test,
                                            lookbacks=max_lookback, depth=depth, width=width,
                                            epoch_count=epoch_count, batch_size=batch_size, learning_rate=learning_rate)

    print(f'LSTM training accuracy is {training_accuracy_score:.2f}.')
    print(f'LSTM test accuracy is {test_accuracy_score:.2f}.')

    end_time = time.time()
    print("Training took {:.2f} mins.".format((end_time - start_time) / 60.))

    return y_test_pred, test_accuracy_score


def timeseries_train_test_split(df, label, time_steps_per_block, train_size):
    # Avoid truncating training data or test data...
    y = df[label] # must be categorical
    X = df.drop(label, axis=1)

    # Split on border of time
    m = int(X.shape[0] / time_steps_per_block * train_size)
    train_rows = int(m * time_steps_per_block)

    test_offset = ((X.shape[0] - train_rows) // time_steps_per_block) * time_steps_per_block
    X_train = X.iloc[:train_rows, :]
    X_test = X.iloc[train_rows:(train_rows+test_offset), :]

    # This is useful only if the labels are seen in both sets.
    # from sklearn.preprocessing import LabelEncoder
    # le = LabelEncoder()
    # le.fit(y)
    # encoded_y = le.transform(y)
    # dummy_Y = keras.utils.to_categorical(encoded_y)

    dummy_Y = keras.utils.to_categorical(y)

    Y_train = dummy_Y[:train_rows]

    Y_test = dummy_Y[train_rows:(train_rows+test_offset)]
    y_test_true = y[train_rows:(train_rows+test_offset)]

    return X_train, X_test, Y_train, Y_test, y_test_true


def timeseries_engineer_features(df, target_variable, lookahead, max_lookback, dropna=False):
    df_ = df.set_index('Time')
    df_y = df_[target_variable].to_frame()

    # This is needed in case the idea is to predict y(t), otherwise
    # the data will contain y_t in the data and y_t+0, which are the same
    # and predictions will be trivial.
    if lookahead == 0:
        df_ = df_.drop(target_variable, axis=1)

    df_postamble = df_.add_suffix('_t')
    df_postamble = pd.concat([df_postamble, pd.DataFrame(np.zeros_like(df_), index=df_.index, columns=df_.columns).add_suffix('_d')], axis=1)

    df_shifted = pd.DataFrame()

    # Column order is important
    for i in range(max_lookback, 0, -1):
        df_shifted_i = df_.shift(i).add_suffix('_t-{}'.format(i))
        df_diff_i = df_.diff(i).add_suffix('_d-{}'.format(i)) # difference with previous time

        df_shifted = pd.concat([df_shifted, df_shifted_i, df_diff_i], axis=1)

    df_y_shifted = df_y.shift(-lookahead).add_suffix('_t+{}'.format(lookahead))
    df_output = pd.concat([df_shifted, df_postamble, df_y_shifted], axis=1)

    if dropna:
        df_output.dropna(inplace=True)
    else:
        # Do not drop data in a time series.  Instead, fill last value
        df_output.bfill(inplace=True)

        # Then fill the first value!
        df_output.ffill(inplace=True)

        # Drop whatever is left.
        df_output.dropna(how='any', axis=1, inplace=True)

    # Whatever it is, no more nulls shall pass!
    assert(df_output.isnull().sum().sum() == 0)

    engineered_target_variable = f'{target_variable}_t+{lookahead}'

    return df_output, engineered_target_variable


def _print_divider(rep=125):
    print('-' * rep)


def rotation_channel(X, theta=0, SNR_dB=30, noise='shot'):
    num_samples = X.shape[0]
    SNR = _linear(SNR_dB)

    # Rotate the symbols by an angle theta (counterclockwise)
    data = X * np.exp(1j * theta)

    real = np.real(data)
    imag = np.imag(data)

    # Now rotate real and imag
    # Noise power = 1/SNR because symbol power is 1.
    noise_power = 1. / SNR

    if noise == 'shot':
        lam = 1e9  # Electrons arrival rate = I Delta t / q
        noise_dc = np_random.poisson(lam=lam, size=num_samples)
        noise_ac = np.sqrt(noise_power / 2) * (np_random.normal(loc=0, scale=1, size=num_samples) + \
                                               1j * np_random.normal(loc=0, scale=1, size=num_samples))
        noise = noise_ac + noise_dc

        noise_real = np.real(noise)
        noise_imag = np.imag(noise)
    else:
        # Assume Gaussian
        noise_real = real + np_random.normal(0, np.sqrt(noise_power / 2.), num_samples)
        noise_imag = imag + np_random.normal(0, np.sqrt(noise_power / 2.), num_samples)

    return np.c_[noise_real, noise_imag], np.c_[np.real(X), np.imag(X)]


def equalize_rotation_channel_CNN(theta, SNR_dB, epochs=100, batch_size=64, training_ratio=0.95):
    global constellation, M_constellation
    global np_random

    if constellation != 'PSK' and M_constellation != 4:
        print("Error:  This is suitable only for QPSK.")
        return None, None, None

    alphabet = create_constellation(constellation=constellation, M=M_constellation)
    training_size = 10000  # in symbols

    # These are pilots.
    _, X_clean, _, _, _ = generate_transmit_symbols(training_size, 1, alphabet, 1)

    X_clean = X_clean.flatten() # ravel()
    X, y = rotation_channel(X_clean, theta=theta, SNR_dB=SNR_dB)  # SNR dB and a rotation of 7.5 deg.

    # Create train and test data
    train_idx = np_random.choice(np.arange(training_size), int(training_ratio * training_size), replace=False)
    test_idx = np.setdiff1d(np.arange(training_size), train_idx)

    X_train, y_train = X[train_idx, :], y[train_idx, :]
    X_test, y_test = X[test_idx, :], y[test_idx, :]

    # Plot the clean one
    constellation = pd.DataFrame({'x_I': np.real(X_clean),
                                  'x_Q': np.imag(X_clean)})
    _plot_constellation(constellation, filename='clean_symbols')

    # Then the noisy one
    constellation = pd.DataFrame({'x_I': X_test[:, 0],
                                  'x_Q': X_test[:, 1]})
    _plot_constellation(constellation, filename='noisy_symbols')

    # Reshape data for CNN
    X_train = X_train.reshape(-1, 2, 1)  # Reshaping to have 2 features (real and imag) per sample
    X_test = X_test.reshape(-1, 2, 1)

    # Build the CNN model
    model = __create_cnn(learning_rate=1e-2)

    # Find the best model
    checkpoint = ModelCheckpoint('CNN_equalization_weights.h5',
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min',
                             verbose=1)
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
    _plot_keras_learning(history, filename='CNN_equalization_learning')

    model.load_weights('CNN_equalization_weights.h5')

    # Evaluate the model
    loss, mae = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test MAE: {mae}')

    # See if this is an equalized instance
    y_pred = model.predict(X_test)

    constellation = pd.DataFrame({'x_I': y_pred[:, 0],
                                  'x_Q': y_pred[:, 1]})
    _plot_constellation(constellation, filename='equalized_symbols')

    return X_test, y_test, y_pred


# TODO:  Check why the bit error rate is off.
def run_simulation():
    global max_transmissions, constellation, M_constellation
    global transmit_SNR_dB, P_TX, N_t
    global N_sc, f_c, Df, noise_figure
    global channel_type, shadowing_std_dev
    global estimation, equalization, symbol_detection, N_s
    
    start_time = time.time()
    
    # This is the power of the signal (across all subcarriers for one OFDM symbol)
    P_BS = P_TX * N_t
    
    # Number of streams.
    N_s = min(N_r, N_t) if precoder != 'identity' else N_t
    
    if max_transmissions < 500:
        print('WARNING:  Low number of runs could cause statistically inaccurate results.')
    
    alphabet = create_constellation(constellation=constellation, M=M_constellation)
    _plot_constellation(alphabet, annotate=True, filename='constellation')

    code_rate = 1  # No FEC is introduced.
    k_constellation = int(np.log2(M_constellation))

    X_information, X, [x_b_i, x_b_q], payload_size, crc_transmitter = generate_transmit_symbols(N_sc, N_s, alphabet=alphabet, P_TX=P_TX)
    bits_transmitter, codeword_transmitter = bits_from_IQ(x_b_i, x_b_q)
    P_X = np.mean(_signal_power(X)) # * Df
    P_X_dBm = _dB(P_X * 1e3)
    
    # Number of streams.
    N_s = min(N_r, N_t) if precoder != 'identity' else N_t

    # Channel
    H = create_channel(N_sc, N_r, N_t, channel=channel_type)

    # Precoder and combiner
    F, Gcomb = compute_precoder_combiner(H, P_BS, algorithm=precoder)

    # Fading
    G = compute_large_scale_fading(dist=300, f_c=f_c, D_t_dB=18, D_r_dB=-1, pl_exp=2)
    G_dB = _dB(G)
    Gchi_dB, _ = compute_shadow_fading(N_sc, G_dB, shadowing_std_dev)  # Add shadowing \chi_\sigma
    Gchi = _linear(Gchi_dB)

    # Note to self, introducing G_fading to the channel creates detection problems.
    # H *= np.sqrt(Gchi)
    
    # Factor in the fading to the channel gain.
    GH = np.zeros_like(H)
    for sc in range(N_sc):
        GH[sc, :, :] = np.sqrt(Gchi[sc]) * H[sc, :, :]
        
    # Channel gain
    channel_gain = np.linalg.norm(np.mean(GH, axis=0), ord='fro') ** 2
    PL_dB = -_dB(channel_gain)
   
    # Pilot sequence 
    P = generate_pilot_symbols(N_t, n_pilot, P_TX, kind='dft')  # dimensions: N_t x n_pilot
    
    # The throughput can be computed by dividing the payload size by TTI (= 1 symbol duration)
    print(f'Payload to be transmitted: {payload_size} bits over one OFDM symbol duration (including CRC).')

    if precoder != 'dft_beamforming':
        print('Channel eigenmodes are: {}'.format(_find_channel_eigenmodes(H)))

    plot_channel(H, filename=channel_type)

    df = pd.DataFrame(columns=['n', 'snr_dB', 'Tx_EbN0_dB', 'Tx_Pwr_dBm',
                               'channel_estimation_error', 'compression_loss',
                               'PL_dB', 'Rx_Pwr_dBm', 'sinr_receiver_before_eq_dB',
                               'sinr_receiver_after_eq_dB', 'Rx_EbN0_dB',
                               'BER', 'BLER'])

    df_detailed = df.copy().rename({'BLER': 'total_block_errors'}, axis=1)
    df.drop(columns='n', inplace=True)
    
    print(' | '.join(df.columns))

    start_time = time.time()
    for item, snr_dB in enumerate(transmit_SNR_dB):
        block_error = 0
        BER = []

        if item % 2 == 0:
            _print_divider()

        EbN0_dB = snr_dB - _dB(k_constellation)

        for n_transmission in range(max_transmissions):
            # Precoding step
            FX = _matrix_vector_multiplication(F, X)  # Faris: Note this change.
            Y, noise = channel_effect(H, FX, snr_dB)
            T, _ = channel_effect(H[:P.shape[0], :], P, snr_dB)

            # Interference
            interference = generate_interference(Y, p_interference, interference_power_dBm)
            P_interference = np.mean(_signal_power(interference))  # * Df

            Y += interference
   
            # Left-multiply y and noise with Gcomb
            Y = _matrix_vector_multiplication(Gcomb, Y)
            noise = _matrix_vector_multiplication(Gcomb, noise)
               
            P_Y = np.mean(_signal_power(Y)) # * Df
            P_noise = np.mean(_signal_power(noise)) # * Df
               
            # Quantization is optional.
            Y = quantize(Y, b=quantization_b)
            P_Y = np.mean(_signal_power(Y)) # * Df
            
            # snr_transmitter_dB = _dB(P_X/P_noise) # This should be very close to snr_dB.
            
            # Note to self: PL is not P_Y / P_X.  The noise power is not subtracted.
            # Thus: P_noise + P_X / P_H ~ P_Y (where P_H is the path gain)
            P_Hx_dBm = P_X_dBm - PL_dB
            
            snr_rx_dB_pre_eq = P_Hx_dBm - _dB(P_noise + P_interference) - noise_figure
            
            # Estimate from pilots
            H_est = H if MIMO_estimation == 'perfect' else estimate_channel(P, T, snr_dB, algorithm=MIMO_estimation)
            estimation_error = _mse(H, H_est)
            
            # # Estimate from signal
            # H_est = H if MIMO_estimation == 'perfect' else estimate_channel(X, Y, snr_dB, algorithm=MIMO_estimation)

            # Compress channel before sending to the receiver
            # and only plot the first transmission (since all transmissions are assumed within channel coherence time).
            _, H_est, compression_loss = compress_channel(H_est, channel_compression_ratio, quantization_b, plotting=(n_transmission == 0))
  
            # Replace the channel H with Sigma as a result of the operations on
            # X and Y above.
            GH_est = Gcomb@H_est  # This is not Sigma
            Sigma = GH_est@F  # This is Sigma.  Is it diagonalized with elements equal the sqrt of eigenmodes when using SVD?  Yes.

            if ((precoder == 'SVD') or (precoder == 'SVD_Waterfilling')):
                if np.allclose(Sigma[0], np.diag(np.diagonal(Sigma[0]))) == False:
                    raise RuntimeError("SVD has failed.  Unable to achieve diagonalization.  Check channel estimation performance and try a different channel type.")

            # np.sqrt(_find_channel_eigenmodes(H)) == GH_estF[0].round(4)

            if (channel_compression_ratio == 0) and ((precoder == 'SVD') or (precoder == 'SVD_Waterfilling')):
                assert np.allclose(Sigma[0], np.diag(np.diagonal(Sigma[0])))

            if precoder != 'dft_beamforming':
                W = equalize_channel(Sigma, snr_dB, algorithm=MIMO_equalization)
            else:
                W = np.ones((N_t, N_sc)) # no equalization necessary for beamforming.

            # Derivation:
            # GY = G(HFx + n)
            # WGY = WGHFx + WGn
            # z = WSigma x + WGn
            # z = x + v
            # x_hat = argmax p_v(z | x)
            
            # import pdb; pdb.set_trace()
            # W@H_est is ones or I (W@H_est).round(2) or (W@Sigma).round(2) for precoding
            z = _matrix_vector_multiplication(W, Y)
            v = _matrix_vector_multiplication(W, noise)
            q = _matrix_vector_multiplication(W, interference)

            P_w_S_x_dBm = _dB(np.linalg.norm(W[0,:,:], ord='fro') ** 2) + _dB(np.linalg.norm(Sigma[0, :], ord='fro') ** 2) + P_X_dBm  # Ideally I should do across all subcarriers.
            P_v_dBm = _dB(np.linalg.norm(W[0, :, :]@Gcomb[0, :, :], ord='fro') ** 2) + _dB((P_noise + P_interference) * 1e3)

            snr_rx_dB = P_w_S_x_dBm - P_v_dBm - noise_figure
            EbN0_rx_dB = snr_rx_dB - _dB(k_constellation * code_rate)

            # Now conduct symbol detection to find x hat from z.
            X_hat_information, X_hat, [x_hat_b_i, x_hat_b_q] = detect_symbols(z, alphabet, algorithm=symbol_detection)

            bits_receiver, codeword_receiver = bits_from_IQ(x_hat_b_i, x_hat_b_q)

            # Remove the padding and CRC from here.
            crc_length = len(crc_transmitter)
            crc_pad_length = int(np.ceil(crc_length / k_constellation)) * \
                k_constellation  # padding included.

            codeword_receiver = codeword_receiver[:-crc_pad_length]

            # Performance measures are here.
            crc_receiver = compute_crc(codeword_receiver, crc_generator)

            # If CRC1 xor CRC2 is not zero, then error.
            if int(crc_transmitter, 2) ^ int(crc_receiver, 2) != 0:
                block_error += 1
            
            BER_i = compute_bit_error_rate(codeword_transmitter[:-crc_pad_length], codeword_receiver)
            BER.append(BER_i)
            
            to_append_i = [n_transmission, snr_dB, EbN0_dB, P_X_dBm,
                           estimation_error, compression_loss, PL_dB, P_Hx_dBm, snr_rx_dB_pre_eq, snr_rx_dB,  EbN0_rx_dB, BER_i, block_error]
            
            df_to_append_i = pd.DataFrame([to_append_i], columns=df_detailed.columns)

            # rounded = [x if ((i == 0) or (i == len(to_append_i))) else f'{x:.3f}' for i, x in enumerate(to_append_i)]
            # del to_append_i
            
            # print(' | '.join(map(str, rounded)))
            
            if df_detailed.shape[0] == 0:
                df_detailed = df_to_append_i.copy()
            else:
                df_detailed = pd.concat([df_detailed, df_to_append_i], ignore_index=True, axis=0)
            ###########################################################################

        BER = np.mean(BER)
        BLER = block_error / max_transmissions
        
        to_append = [snr_dB, EbN0_dB, P_X_dBm, estimation_error, compression_loss,
                       PL_dB, P_Hx_dBm, snr_rx_dB_pre_eq, snr_rx_dB, EbN0_rx_dB, BER, BLER]
        df_to_append = pd.DataFrame([to_append], columns=df.columns)
    
        rounded = [f'{x:.3f}' for x in to_append]
        del to_append
    
        if df.shape[0] == 0:
            df = df_to_append.copy()
        else:
            df = pd.concat([df, df_to_append], ignore_index=True, axis=0)
    
        print(' | '.join(map(str, rounded)))
        
    end_time = time.time()
    
    print(f'Time elapsed: {((end_time - start_time) / 60.):.2f} mins.')
    
    return df, df_detailed, v



# Which simulation scenario
# you would like to run?  Comment out the ones you are not
# interested in running.
###############################################################################

# Model-based and machine learning (supervised, unsupervised) simulations
###############################################################################

df_results, df_detailed_results, v = run_simulation()

df_detailed_results.to_csv('simulation_detailed_data.csv')
df_results.to_csv('simulation_data.csv')

plot_performance(df_detailed_results, xlabel='Tx_EbN0_dB', ylabel='BER', semilogy=True, filename='BER')
plot_performance(df_results, xlabel='Tx_EbN0_dB', ylabel='BLER', semilogy=True, filename='BLER')
plot_pdf(v, text='noise', algorithm='KDE', filename='enhanced_noise')
###############################################################################


# CNN-based equalization
###############################################################################
X_test, y_test, y_pred = equalize_rotation_channel_CNN(theta=np.pi/24, SNR_dB=30,
                                                           epochs=96, batch_size=32,
                                                           training_ratio=0.85)
###############################################################################

# Time series predictions
###############################################################################
y_test_pred, test_accuracy_score = predict_trajectory_with_LSTM(df=None,
                             target_variable='', depth=0, width=5,
                             lookahead_time=1, max_lookback=10,
                             training_size=0.7, batch_size=64,
                             epoch_count=128)
###############################################################################

# Tabular reinforcement learning simulation
###############################################################################
Q_values, losses, optimal_episode, optimal_reward, \
    optimal_environment_progress, optimal_action_progress = \
        _tabular_reinforcement_learning(max_episodes_to_run=100,
                                        max_timesteps_per_episode=15,
                                        plotting=True)
###############################################################################

# Deep reinforcement learning simulation
###############################################################################
Q_values, losses, optimal_episode, optimal_reward, \
    optimal_environment_progress, optimal_action_progress = \
    _deep_reinforcement_learning(max_episodes_to_run=200,
                                 max_timesteps_per_episode=15,
                                 plotting=True)
###############################################################################

# Using linear regresion for channel estimation.
###############################################################################
X = np_random.uniform(-1, 1, N_sc)
X /= np.sqrt(_signal_power(X))

X = np.repeat(X, N_t, axis=np.newaxis).reshape((N_sc, N_t))
H = np.tile(np.ones((N_r, N_t)), N_sc).reshape((N_sc, N_r, N_t))

Hx = np.zeros((N_sc, N_r))
for idx in range(N_sc):
    Hx[idx, :] = np.dot(H[idx, :, :], X[idx, :])

SNR_dB = 10
noise_power = 1 / _linear(SNR_dB)
y = Hx + np_random.normal(0, noise_power, (N_sc, N_r))

H_est = _estimate_channel_linear_regression(X, y)

estimation_error = _mse(H, H_est)
print(f'Using linear regression, estimation error is: {estimation_error:.4f}.')
