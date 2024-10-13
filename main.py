#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:22:29 2024

@author: farismismar
"""

import numpy as np
import pandas as pd
from scipy.constants import c

import time

import matplotlib.pyplot as plt
import tikzplotlib

from sklearn.cluster import KMeans

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# # For Windows users
# if os.name == 'nt':
#     os.add_dll_directory("/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# The GPU ID to use, usually either "0" or "1" based on previous line.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

################################################
# Single OFDM symbol and single user simulator
################################################

###############################################################################
# Parameters
N_t = 4                                  # Number of transmit antennas
N_r = 4                                  # Number of receive antennas per user
N_sc = 64                                # Number of subcarriers
P_BS = 4                                 # Base station transmit power [W] (across all transmitters)
    
max_transmissions = 200
precoder = 'identity'                    # Also: identity, SVD, SVD_Waterfilling, dft_beamforming
channel_type = 'CDL-E'                   # Channel type: rayleigh, ricean, CDL-C, CDL-E
quantization_b = np.inf                  # Quantization resolution

Df = 15e3                                # OFDM subcarrier bandwidth [Hz].
f_c = 1800e6                             # Center frequency [MHz] for large scale fading and DFT.
interference_power_dBm = -105            # dBm measured at the receiver
p_interference = 0.00                    # probability of interference

constellation = 'QAM'
M_constellation = 16
n_pilot = 4                              # Number of pilots for channel estimation

MIMO_estimation = 'perfect'              # Also: perfect, LS, LMMSE
MIMO_equalization = 'MMSE'               # Also: MMSE, ZF
symbol_detection = 'ML'                  # Also: ML, kmeans

crc_generator = 0b1100_1101              # CRC generator polynomial

# Transmit SNR in dB  
transmit_SNR_dB = [0, 5, 10, 15, 20, 25, 30][::-1]
###############################################################################

plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = "14"

prefer_gpu = True

seed = 42 # Reproduction
np_random = np.random.RandomState(seed=seed)

__ver__ = '0.6'
__data__ = '2024-10-11'


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
    if k != int(k): # only square constellations are allowed.
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
    if k != int(k): # only square QAM is allowed.
        print('Only square QAM constellations allowed.')
        return None

    k = int(k)
    m = np.arange(M)
    Am_ = np.arange(-np.sqrt(M) + 1, np.sqrt(M), step=2, dtype=int) # Proakis p105
    
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
            s = s.iloc[::-1] # Invert 
        # print(s)
        constellation_ordered = pd.concat([constellation_ordered, s], axis=0)
    
    constellation = constellation_ordered.copy()
    constellation = constellation.reset_index(drop=True)
    constellation['m'] = constellation.index
    
    gray = constellation['m'].apply(lambda x: decimal_to_gray(x, k))
    constellation['I'] = gray.str[:(k//2)]
    constellation['Q'] = gray.str[(k//2):]
    
    constellation.loc[:, 'x'] = constellation.loc[:, 'x_I'] + 1j * constellation.loc[:, 'x_Q']
    
    # Normalize the transmitted symbols
    # The average power is normalized to unity
    P_average = _signal_power(constellation['x']) #np.mean(np.abs(constellation.loc[:, 'x']) ** 2)
    constellation.loc[:, 'x'] /= np.sqrt(P_average)
    
    return constellation


def _signal_power(signal):
    return np.mean(np.abs(signal) ** 2, axis=0)


def generate_transmit_symbols(N_sc, N_t, alphabet, P_TX):
    global np_random
    
    k = int(np.log2(alphabet.shape[0]))
    
    payload_size = N_sc * N_t * k # For the future, this depends on the MCS index.
    bits = create_bit_payload(payload_size)
    
    x_b_i, x_b_q, x_information, x_symbols = bits_to_baseband(bits, alphabet)
    
    x_information = np.reshape(x_information, (N_sc, N_t))
    x_symbols = np.reshape(x_symbols, (N_sc, N_t))    
    
    # Normalize and scale the transmit power of the symbols.
    x_symbols /= np.sqrt(_signal_power(x_symbols).mean() / P_TX)
    
    x_b_i = np.reshape(x_b_i, (-1, N_t))
    x_b_q = np.reshape(x_b_q, (-1, N_t))
    
    return x_information, x_symbols, [x_b_i, x_b_q], payload_size


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
        # Generate a DFT (Discrete Fourier Transform) matrix of size N_t x N_t
        dft_matrix = np.fft.fft(np.eye(N_t))
        
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
        x_b_rev = x_b_rev[k:] #
    
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
        
    N_sc = H.shape[0]
    
    # Set a flag to deal with beamforming.
    is_beamforming = True
    N_r = 1
    
    # Parameters
    if len(H.shape) == 3:  # MIMO case
        _, N_r, N_t = H.shape
        is_beamforming = False
    
    # Convert SNR from dB to linear scale
    snr_linear = _linear(snr_dB)

    # Compute the power of the transmit matrix X
    signal_power = np.mean(_signal_power(X)) # This must equal P_BS / N_t.    

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
    H_pseudo_inv = np.linalg.pinv(H_herm_H) # Shape (N_sc, N_t, N_t)
    
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
    H_mmse_term = H_herm_H + (1 / snr_linear) * identity # Shape (N_sc, N_t, N_t)
    
    # Compute the inverse of (H^H * H + (1/SNR) * I) for all subcarriers
    H_mmse_inv = np.linalg.inv(H_mmse_term) # Shape: (N_sc, N_t, N_t)
    
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


def _estimate_channel_LMMSE(X, Y, snr_dB):
    # Assume that RHH is sigma_H^2 I for simplicity.
    snr_linear = _linear(snr_dB)
    
    H_ls = _estimate_channel_least_squares(X, Y)
    
    N_sc = H_ls.shape[0]  # Number of subcarriers
    
    # Initialize the LMMSE channel estimate
    H_lmmse = np.zeros_like(H_ls, dtype=np.complex128)
    
    # Iterate over each subcarrier
    for sc in range(N_sc):
        # Compute the Frobenius norm squared for the current subcarrier's channel matrix (N_r * N_t)
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

    
def create_channel(N_sc, N_r, N_t, shadow_fading_margin_dB=8, channel='rayleigh'):
    global np_random
    global f_c
    
    G = compute_large_scale_fading(d=1, f_c=f_c)

    if channel == 'ricean':
        return _create_ricean_channel(G, N_sc, N_r, N_t, K_factor=4, sigma_dB=shadow_fading_margin_dB)
    
    if channel == 'rayleigh':
        return _create_ricean_channel(G, N_sc, N_r, N_t, K_factor=0, sigma_dB=shadow_fading_margin_dB)

    if channel == 'CDL-C':
        return _generate_cdl_c_channel(G, N_sc, N_r, N_t, sigma_dB=shadow_fading_margin_dB)
    
    if channel == 'CDL-E':
        return _generate_cdl_e_channel(G, N_sc, N_r, N_t, sigma_dB=shadow_fading_margin_dB)
    

def _create_ricean_channel(G, N_sc, N_r, N_t, K_factor, sigma_dB):
    global np_random
  
    G_fading = _dB(G) - np_random.normal(loc=0, scale=np.sqrt(sigma_dB), size=(N_r, N_t))
    G_fading = np.array([_linear(g) for g in G_fading])
    
    mu = np.sqrt(K_factor / (1 + K_factor))
    sigma = np.sqrt(1 / (1 + K_factor))

    H = np_random.normal(loc=mu, scale=sigma, size=(N_r, N_t)) + \
        1j * np_random.normal(loc=mu, scale=sigma, size=(N_r, N_t))
    
    # Normalize channel to unity gain and add large scale gain
    # So the channel gain (tr(H)) is G.
    H /= np.trace(H)
    H *= np.sqrt(G_fading)  # element multiplication.

    H_full = np.repeat(H[np.newaxis, :, :], N_sc, axis=0)  # Repeat for all subcarriers

    return H_full


def _generate_cdl_c_channel(G, N_sc, N_r, N_t, sigma_dB):
    global np_random 
    
    # Generates a 3GPP 38.900 CDL-C channel with dimensions (N_sc, N_r, N_t).
    delay_taps = [0, 0.209, 0.423, 0.658, 1.18, 1.44, 1.71]  # in microseconds
    powers_dB = [-0.2, -13.5, -15.4, -18.1, -20.0, -22.1, -25.2]  # tap power in dBm

    # Convert dB to linear scale for power
    powers_linear = 10 ** (np.array(powers_dB) / 10)
    num_taps = len(powers_dB)
    
    # Initialize the channel matrix (complex random Gaussian per subcarrier, tap, and antenna pair)
    H = np.zeros((N_sc, N_r, N_t), dtype=np.complex128)
    
    # Apply shadow fading (log-normal) to the large-scale fading
    shadow_fading = 10 ** (np_random.normal(0, sigma_dB, size=(N_r, N_t)) / 10)
    
    # Frequency range for the subcarriers
    subcarrier_frequencies = np.arange(N_sc) / N_sc  # normalized subcarrier indices
    
    # Generate channel response for each tap and apply delay phase shifts
    for tap in range(num_taps):
        # Delay in seconds
        delay = delay_taps[tap] * 1e-6  # convert from microseconds to seconds
        
        # Apply the phase shift for each subcarrier based on the delay
        phase_shift = np.exp(-2j * np.pi * subcarrier_frequencies * delay)  # shape: (N_sc,)
        
        # Complex Gaussian fading for each tap, scaled by tap power, large-scale fading, and shadow fading
        H_tap = np.sqrt(powers_linear[tap] * G * shadow_fading[:, :, None]) * \
                (np_random.randn(N_r, N_t, N_sc) + 1j * np_random.randn(N_r, N_t, N_sc)) / np.sqrt(2)
        
        # Apply phase shift across subcarriers
        H += (H_tap * phase_shift).transpose(2, 0, 1)  # Adjust dimensions (N_sc, N_r, N_t)

    return H


def _generate_cdl_e_channel(G, N_sc, N_r, N_t, sigma_dB):
    global np_random
    
    # Generates a 3GPP 38.900 CDL-E channel with dimensions (N_sc, N_r, N_t).
    delay_taps = [0, 0.264, 0.366, 0.714, 1.53, 1.91, 3.52, 4.20, 5.35]  # in microseconds
    powers_dB = [-0.03, -4.93, -8.03, -10.77, -15.86, -18.63, -21.11, -22.50, -25.63]  # tap power in dBm

    # Convert dB to linear scale for power
    powers_linear = 10 ** (np.array(powers_dB) / 10)
    num_taps = len(powers_dB)
    
    # Initialize the channel matrix (complex random Gaussian per subcarrier, tap, and antenna pair)
    H = np.zeros((N_sc, N_r, N_t), dtype=np.complex128)
    
    # Apply shadow fading (log-normal) to the large-scale fading
    shadow_fading = 10 ** (np_random.normal(0, sigma_dB, size=(N_r, N_t)) / 10)
    
    # Frequency range for the subcarriers
    subcarrier_frequencies = np.arange(N_sc) / N_sc  # normalized subcarrier indices
    
    # Generate channel response for each tap and apply delay phase shifts
    for tap in range(num_taps):
        # Delay in seconds
        delay = delay_taps[tap] * 1e-6  # convert from microseconds to seconds
        
        # Apply the phase shift for each subcarrier based on the delay
        phase_shift = np.exp(-2j * np.pi * subcarrier_frequencies * delay)
        
        # Complex Gaussian fading for each tap, scaled by tap power, large-scale fading, and shadow fading
        H_tap = np.sqrt(powers_linear[tap] * G * shadow_fading[:, :, None]) * \
                (np_random.randn(N_r, N_t, N_sc) + 1j * np_random.randn(N_r, N_t, N_sc)) / np.sqrt(2)
        
        # Apply phase shift across subcarriers
        H += (H_tap * phase_shift).transpose(2, 0, 1)  # Adjust dimensions (N_sc, N_r, N_t)

    return H


def compute_large_scale_fading(d, f_c, D_t_dB=18, D_r_dB=2, pl_exp=1.07):
    l = c / f_c
    G = _linear(D_t_dB + D_r_dB) * (l / (4 * np.pi * d)) ** pl_exp

    assert (G <= 1)
    
    return G


def _vec(H):
    # H is numpy array
    return H.flatten(order='F')


def _mse(H_true, H_estimated):
    return np.mean(np.abs(H_true - H_estimated) ** 2)


def _dB(X):
    return 10 * np.log10(X)


def _linear(X):
    return 10 ** (X / 10.)


def detect_symbols(x_sym_hat, alphabet, algorithm):
    if algorithm == 'kmeans':
        return _detect_symbols_kmeans(x_sym_hat, alphabet)
    
    if algorithm == 'ML':
        return _detect_symbols_ML(x_sym_hat, alphabet)
    

def _detect_symbols_kmeans(x_sym_hat, alphabet):
    global np_random
    
    x_sym_hat_flat = x_sym_hat.flatten()

    X = np.real(x_sym_hat_flat)
    X = np.c_[X, np.imag(x_sym_hat_flat)]
    X = X.astype('float32')
    
    centroids = alphabet[['x_I', 'x_Q']].values
    centroids = centroids.astype('float32')
    
    # Intialize k-means centroid location deterministcally as a constellation
    kmeans = KMeans(n_clusters=M_constellation, init=centroids, n_init=1,
                    random_state=np_random).fit(centroids)
    
    information = kmeans.predict(X).reshape(x_sym_hat.shape)
    df_information = pd.DataFrame(data={'m': information})
    
    df = df_information.merge(alphabet, how='left', on='m')
    symbols = df['x'].values.reshape(x_sym_hat.shape)
    bits_i = df['I'].values.reshape(x_sym_hat.shape)
    bits_q = df['Q'].values.reshape(x_sym_hat.shape) 

    return information, symbols, [bits_i, bits_q]

    
def _detect_symbols_ML(symbols, alphabet):    
    df_information = pd.DataFrame()
    symbols_flat = symbols.flatten()
    
    for s in range(symbols_flat.shape[0]):
        x_hat = symbols_flat[s]
        # This function returns argmin |x - s_m| based on AWGN ML detection
        # for any arbitrary constellation denoted by the alphabet
        distances = alphabet['x'].apply(lambda x: np.abs(x - x_hat) ** 2)
        
        # Simple distances.idxmin is not cutting it.
        m_star = distances.idxmin(axis=0)
        
        df_i = pd.DataFrame(data={'m': m_star,
                                  'x': alphabet.loc[alphabet['m'] == m_star, 'x'],
                                  'I': alphabet.loc[alphabet['m'] == m_star, 'I'],
                                  'Q': alphabet.loc[alphabet['m'] == m_star, 'Q']})
        
        df_information = pd.concat([df_information, df_i], axis=0, ignore_index=True)
    
    information = df_information['m'].values.reshape(symbols.shape)
    
    # Now simply compute other elements.
    symbols = df_information['x'].values.reshape(symbols.shape)
    bits_i = df_information['I'].values
    bits_q = df_information['Q'].values
    
    bits_i = bits_i.reshape(symbols.shape)
    bits_q = bits_q.reshape(symbols.shape)
    
    return information, symbols, [bits_i, bits_q]


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


def compute_precoder_combiner(H, P_TX, algorithm='SVD_Waterfilling'):
    N_sc, N_r, N_t = H.shape
    N_s = min(N_r, N_t)    
    
    if algorithm == 'identity':
        F = np.eye(N_t, N_s)
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
    
        F = np.conjugate(np.transpose(Vh, (0, 2, 1)))
        Gcomb = np.conjugate(np.transpose(U, (0, 2, 1)))
        
        return F, Gcomb
    
    if algorithm == 'SVD_Waterfilling':
        U, S, Vh = _svd_precoder_combiner(H)
        
        try:
            D = _waterfilling(S[0, :], P_TX) # The power allocation matrix
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
    
    wavelength = c / f_c
    
    d = wavelength / 2. # antenna spacing 
    k = 2. * np.pi / wavelength # wave number

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
    _, S, _ = np.linalg.svd(H[subcarrier_idx,:,:], full_matrices=False) #, hermitian=True)
    eigenmodes = S ** 2
    
    return eigenmodes


def _svd_precoder_combiner(H):
    U, S, Vh = np.linalg.svd(H, full_matrices=True) #, hermitian=True)
    return U, S, Vh


def _waterfilling(S, power):
    n_modes = len(S)
    
    # Ensure eigenmodes are non-zero
    if 0 in S:
        raise ValueError("Zero eigenmode prevents waterfilling.  Please use a different precoder.")
        
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
    

def plot_performance(df, xlabel, ylabel, semilogy=True, filename=None):
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
        plt.savefig(f'performance_{ylabel}_{filename}.pdf', format='pdf', dpi=fig.dpi)
        tikzplotlib.save(f'performance_{ylabel}_{filename}.tikz')
    plt.show()
    plt.close(fig)
    

def plot_pdf(X, text=None, algorithm='empirical', num_bins=200, filename=None):
    X_re = np.real(X)
    X_im = np.imag(X)
    
    if text is None:
        text = ''
    else:
        text = f' - {text}'
        
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
        df_re = pd.DataFrame(X_re).add_suffix(f'_Re{text}')
        df_im = pd.DataFrame(X_im).add_suffix(f'_Im{text}')
        
        df = df_re.copy()
        if is_complex:
            df = pd.concat([df, df_im], axis=1, ignore_index=False)
            df.plot(kind='kde', bw_method=0.3, ax=ax)
    
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('p(X)')
    
    plt.tight_layout()
    if filename is not None:
        plt.savefig(f'pdf_{algorithm}_{filename}.pdf', format='pdf', dpi=fig.dpi)
        tikzplotlib.save(f'pdf_{algorithm}_{filename}.tikz')
    plt.show()
    plt.close(fig)
    

def _plot_constellation(constellation, filename=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plt.scatter(constellation['x_I'], constellation['x_Q'], c='k', marker='o', lw=2)
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
        plt.savefig(f'constellation_{filename}.pdf', format='pdf', dpi=fig.dpi)
        tikzplotlib.save(f'constellation_{filename}.tikz')
    plt.show()
    plt.close(fig)
    
    
def plot_channel(channel, filename=None):
    N_sc, N_r, N_t = channel.shape
    
    # Only plot first receive antenna
    H = channel[:,0,:]
    
    # plt.rcParams['font.size'] = 36
    # plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(figsize=(12, 6))

    plt.imshow(np.abs(H) ** 2, aspect='auto')
    
    plt.xlabel('TX Antennas')
    plt.ylabel('Subcarriers')
    
    plt.xticks(range(N_t))
    
    plt.tight_layout()
    if filename is not None:
        plt.savefig(f'channel_{filename}.pdf', format='pdf', dpi=fig.dpi)
        tikzplotlib.save(f'channel_{filename}.tikz')
    plt.show()
    plt.close(fig)


def plot_IQ(signal, filename=None):
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
        plt.savefig(f'IQ_{filename}.pdf', format='pdf', dpi=fig.dpi)
        tikzplotlib.save(f'IQ_{filename}.tikz')
    plt.show()
    plt.close(fig)

    
def _print_divider():
    print('-' * 125)
    

def run_simulation(transmit_SNR_dB, constellation, M_constellation, crc_generator, N_sc, N_r, N_t):
    global np_random
    global P_BS
    
    global precoder, channel_type, quantization_b, Df, max_transmissions
    global p_interference, interference_power_dBm
    
    start_time = time.time()
    
    P_TX = P_BS / N_t                # This is the power of one OFDM symbol (across all subcarriers)
    N_s = min(N_r, N_t)              # Number of streams.
    
    if max_transmissions < 300:
        print('WARNING:  Low number of runs could cause statistically inaccurate results.')
        
    alphabet = create_constellation(constellation=constellation, M=M_constellation)
    k_constellation = int(np.log2(M_constellation))
    
    X_information, X, [x_b_i, x_b_q], payload_size = generate_transmit_symbols(N_sc, N_s, alphabet=alphabet, P_TX=P_TX)
    bits_transmitter, codeword_transmitter = bits_from_IQ(x_b_i, x_b_q)
    P_X = np.mean(_signal_power(X)) * Df

    P = generate_pilot_symbols(N_t, n_pilot, P_TX, kind='dft')
    H = create_channel(N_sc, N_r, N_t, channel=channel_type, shadow_fading_margin_dB=8)

    # Precoder and combiner
    F, Gcomb = compute_precoder_combiner(H, P_BS, algorithm=precoder)
     
    # Precoding right-multiply H with F
    HF = H@F
    
    # The throughput can be computed by dividing the payload size by TTI.
    print(f'Payload to be transmitted: {payload_size} bits per TTI.')
    
    if precoder != 'dft_beamforming':
        print('Channel eigenmodes are: {}'.format(_find_channel_eigenmodes(H)))
    
    plot_channel(H, filename=channel_type)

    df = pd.DataFrame(columns=['snr_dB', 'n', 'EbN0_dB', 'snr_transmitter_dB', 
                               'channel_estimation_error', 
                               'PL_dB', 'sinr_receiver_after_eq_dB',
                               'BER', 'BLER'])
    
    df_detailed = df.copy().rename({'BLER': 'total_block_errors'}, axis=1)
    df.drop(columns='n', inplace=True)
    
    print(' | '.join(df.columns))
    
    for item, snr_dB in enumerate(transmit_SNR_dB):
        block_error = 0
        BER = []
        
        if item % 2 == 0:
            _print_divider()

        EbN0_dB = snr_dB - _dB(k_constellation)
        
        for n_transmission in range(max_transmissions):
            Y, noise = channel_effect(HF, X, snr_dB)
            T, _ = channel_effect(HF[:P.shape[0], :], P, snr_dB)
            
            # Interference
            interference = generate_interference(Y, p_interference, interference_power_dBm)
            Y += interference
            
            # Left-multiply y and noise with Gcomb            
            Y = _matrix_vector_multiplication(Gcomb, Y)
            noise = _matrix_vector_multiplication(Gcomb, noise)
            
            P_Y = np.mean(_signal_power(Y)) * Df
            P_noise = np.mean(_signal_power(noise)) * Df
        
            # Quantization is optional.
            Y = quantize(Y, b=quantization_b)
            
            P_Y = np.mean(_signal_power(Y)) * Df
            
            PL_dB = _dB(P_X) - _dB(P_Y)
        
            snr_transmitter_dB = _dB(P_X/P_noise) # This should be very close to snr_dB.
            # EbN0_transmitter_dB = snr_transmitter_dB - _dB(k_constellation)
            
            # Estimate from pilots
            H_est = H if MIMO_estimation == 'perfect' else estimate_channel(P, T, snr_dB, algorithm=MIMO_estimation)
            estimation_error = _mse(H, H_est)
            
            # Replace the channel H with Sigma as a result of the operations on
            # X and Y above.            
            GH_estF = Gcomb@H_est@F # This is Sigma.  Is it diagonalized with elements equal the sqrt of eigenmodes?  Yes.
            # np.sqrt(_find_channel_eigenmodes(H)) == GH_estF[0].round(4)
            
            if (precoder == 'SVD') or (precoder == 'SVD_Waterfilling'):
                assert np.allclose(GH_estF[0], np.diag(np.diagonal(GH_estF[0])))
            
            if precoder != 'dft_beamforming':
                W = equalize_channel(GH_estF, snr_dB, algorithm=MIMO_equalization)
            else:
                W = np.ones((N_t, N_sc)) # no equalization necessary for beamforming.
            
            # # Note:  Often, keep an eye on the product (W@GH_estF).round(1) and see how close it is to I.           
            # if not np.allclose((W@GH_estF)[0].round(1), np.eye(N_t)):
            #     print("WARNING")
            
            X_hat = _matrix_vector_multiplication(W, Y)
            v = _matrix_vector_multiplication(W, noise)
            q = _matrix_vector_multiplication(W, interference)
            
            P_X_hat = np.mean(_signal_power(X_hat)) * Df
            P_v = np.mean(_signal_power(v)) * Df
            P_q = np.mean(_signal_power(q)) * Df
            
            sinr_receiver_after_eq_dB = _dB(P_X_hat/(P_v + P_q))
            
            # Now conduct symbol detection
            X_hat_information, X_hat, [x_hat_b_i, x_hat_b_q] = detect_symbols(X_hat, alphabet, algorithm=symbol_detection)

            bits_receiver, codeword_receiver = bits_from_IQ(x_hat_b_i, x_hat_b_q)
            
            # Performance measures are here.
            crc_transmitter = compute_crc(codeword_transmitter, crc_generator)
            crc_receiver = compute_crc(codeword_receiver, crc_generator)
            
            # If CRC1 xor CRC2 is not zero, then error.
            if int(crc_transmitter, 2) ^ int(crc_receiver, 2) != 0:
                block_error += 1
            
            # For beamforming, the codeword is actually one symbol, and thus
            # bit error rate will be filled with NaN
            BER_i = np.nan
            if precoder != 'dft_beamforming':
                BER_i = compute_bit_error_rate(codeword_transmitter, codeword_receiver)
            BER.append(BER_i)
            
            to_append_i = [snr_dB, n_transmission, EbN0_dB, snr_transmitter_dB, estimation_error, PL_dB, sinr_receiver_after_eq_dB, BER_i, block_error]
            df_to_append_i = pd.DataFrame([to_append_i], columns=df_detailed.columns)
            
            if df_detailed.shape[0] == 0:
                df_detailed = df_to_append_i.copy()
            else:
                df_detailed = pd.concat([df_detailed, df_to_append_i], ignore_index=True, axis=0)
            ###########################################################################
            
        BER = np.mean(BER)
        BLER = block_error / max_transmissions

        to_append = [snr_dB, EbN0_dB, snr_transmitter_dB, estimation_error, PL_dB, sinr_receiver_after_eq_dB, BER, BLER]
        df_to_append = pd.DataFrame([to_append], columns=df.columns)
        
        rounded = [f'{x:.3f}' for x in to_append]
        del to_append
        
        if df.shape[0] == 0:
            df = df_to_append.copy()
        else:
            df = pd.concat([df, df_to_append], ignore_index=True, axis=0)
        
        print(' | '.join(map(str, rounded)))
    
    end_time = time.time()
    
    # Plots
    # noise after quantization for last run
    GHFX, _ = channel_effect(GH_estF, X, snr_dB)
    plot_pdf(Y - GHFX, text='noise', algorithm='KDE', filename='noise_alt')
    
    # Plot a quantized signal of the last run
    plot_IQ(Y, filename='IQ')
    
    # Plot the SNR distribution of the last run
    plot_pdf(df['sinr_receiver_after_eq_dB'], text='SINR receiver', num_bins=10, 
             filename='sinr', algorithm='empirical')
    ###########################################################################
    
    _print_divider()
    print(f'Time elapsed: {((end_time - start_time) / 60.):.2f} mins.')
    
    return df, df_detailed


df_results, df_detailed_results = run_simulation(transmit_SNR_dB,
                                                 constellation, 
                                                 M_constellation, crc_generator,
                                                 N_sc, N_r, N_t)                                                 

plot_performance(df_results, xlabel='EbN0_dB', ylabel='BER', semilogy=True, filename='BER')
plot_performance(df_results, xlabel='EbN0_dB', ylabel='BLER', semilogy=True, filename='BLER')
