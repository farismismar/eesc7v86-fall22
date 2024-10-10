#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:41:51 2024

@author: farismismar
"""

import numpy as np
import pandas as pd
from scipy.constants import c

import time

import matplotlib.pyplot as plt
import tikzplotlib

from sklearn.cluster import KMeans

import pdb

################################################
# Single OFDM symbol and single user simulator
################################################

###############################################################################
# Parameters
N_t = 4                                  # Number of transmit antennas
N_r = 4                                  # Number of receive antennas per user
N_sc = 64                                # Number of subcarriers
P_TX = 1                                 # in Watts
transmit_SNR_dB = [-5, 0, 5, 10, 15, 20, 25, 30][::-1]  # Transmit SNR in dB  # Transmit SNR in dB

constellation = 'QAM'
M_constellation = 16

n_pilot = 4                              # Number of pilots for channel estimation

MIMO_estimation = 'perfect' # perfect, LS, LMMSE
MIMO_equalization = 'ZF' # MMSE, ZF
symbol_detection = 'ML' # ML, kmeans

crc_generator = 0b1100_1101  # CRC generator polynomial
###############################################################################

plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = "14"

prefer_gpu = True

seed = 42 # Reproduction
np_random = np.random.RandomState(seed=seed)

__ver__ = '0.6'
__data__ = '2024-10-09'


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


def generate_transmit_symbols(N_sc, N_t, alphabet):
    global np_random
    
    k = int(np.log2(alphabet.shape[0]))
    bits = create_bit_payload(N_sc * N_t * k)
    x_b_i, x_b_q, x_information, x_symbols = bits_to_baseband(bits, alphabet)
    
    x_information = np.reshape(x_information, (N_sc, N_t))
    x_symbols = np.reshape(x_symbols, (N_sc, N_t))
    
    x_b_i = np.reshape(x_b_i, (-1, N_t))
    x_b_q = np.reshape(x_b_q, (-1, N_t))
    
    return x_information, x_symbols, [x_b_i, x_b_q]


def generate_pilot_symbols(N_t, n_pilot, kind='dft'):
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
        
        # # Scale the unitary matrix
        # Q /= np.linalg.norm(Q, ord='fro')
        
        # To make a semi-unitary, we need to post multiply with a rectangular matrix
        # Now we need a rectangular matrix (fat)
        A = np.zeros((N_t, n_pilot), int)
        np.fill_diagonal(A, 1)
        X_p = Q @ A
        
        assert(np.allclose(X_p@X_p.T, np.eye(N_t)))  # This is it
        
        # The training sequence is X_p.  It has n_pilot rows and N_t columns
        pilot_matrix = X_p.T
        
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
    
    # Parameters
    N_sc, N_r, N_t = H.shape

    # Convert SNR from dB to linear scale
    snr_linear = _linear(snr_dB)

    # Compute the power of the transmit matrix X
    signal_power = np.mean(_signal_power(X))

    # Calculate the noise power based on the input SNR
    noise_power = signal_power / snr_linear

    # Generate additive white Gaussian noise (AWGN)
    noise = np.sqrt(noise_power / 2) * (np_random.randn(N_sc, N_r) + 1j * np_random.randn(N_sc, N_r))
    
    received_signal = np.zeros((N_sc, N_r), dtype=np.complex128)
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
    
    # Compute the pseudo-inverse for each subcarrier (to handle singularity): (N_sc, N_t, N_t)
    H_pseudo_inv = np.linalg.pinv(H_herm_H)
    
    # ZF equalization matrix: (H^H * H)^-1 * H^H for all subcarriers
    W_zf = np.matmul(H_pseudo_inv, H_hermitian)  # Shape: (N_sc, N_t, N_r)
    
    return W_zf
    

def _equalize_channel_MMSE(H, snr_dB):
    N_sc, N_r, N_t = H.shape
    snr_linear = _linear(snr_dB)
    
    # Hermitian transpose of each subcarrier's H: (N_sc, N_t, N_r)
    H_hermitian = np.conjugate(np.transpose(H, (0, 2, 1)))
    
    # (H^H * H) for all subcarriers: (N_sc, N_t, N_t)
    H_herm_H = np.matmul(H_hermitian, H)
    
    # Add noise power to diagonal (1/SNR * I): (N_sc, N_t, N_t)
    identity = np.eye(N_t)[None, :, :]  # Shape (1, N_t, N_t) -> Broadcast to (N_sc, N_t, N_t)
    H_mmse_term = H_herm_H + (1 / snr_linear) * identity
    
    # Compute the inverse of (H^H * H + (1/SNR) * I) for all subcarriers: (N_sc, N_t, N_t)
    H_mmse_inv = np.linalg.inv(H_mmse_term)
    
    # MMSE equalization matrix: (H^H * H + (1/SNR) * I)^-1 * H^H
    W_mmse = np.matmul(H_mmse_inv, H_hermitian)  # Shape: (N_sc, N_t, N_r)
    
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
    H_lmmse = np.zeros_like(H_ls, dtype=np.complex64)
    
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


def quantize(X, b, tol=1e-6, max_iter=100):
    if np.isinf(b):
        return X
    
    # Implement Lloyd-Max
    # Number of quantization levels
    L = 2**b

    # Separate real and imaginary parts
    X_real = np.real(X)
    X_imag = np.imag(X)

    # Find min and max values in real and imaginary parts
    min_real, max_real = np.min(X_real), np.max(X_real)
    min_imag, max_imag = np.min(X_imag), np.max(X_imag)

    # Initialize the quantization levels (centroids) uniformly between min and max
    centroids_real = np.linspace(min_real, max_real, L)
    centroids_imag = np.linspace(min_imag, max_imag, L)

    for _ in range(max_iter):
        # Step 1: Partition boundaries (midpoints between centroids)
        boundaries_real = (centroids_real[:-1] + centroids_real[1:]) / 2
        boundaries_imag = (centroids_imag[:-1] + centroids_imag[1:]) / 2

        # Step 2: Assign each real/imaginary part of X to the nearest centroid
        X_real_b = np.digitize(X_real, boundaries_real)
        X_imag_b = np.digitize(X_imag, boundaries_imag)

        # Step 3: Recalculate the centroids
        new_centroids_real = np.array([X_real[X_real_b == i].mean() if len(X_real[X_real_b == i]) > 0 else centroids_real[i] for i in range(L)])
        new_centroids_imag = np.array([X_imag[X_imag_b == i].mean() if len(X_imag[X_imag_b == i]) > 0 else centroids_imag[i] for i in range(L)])

        # Step 4: Check for convergence
        if np.max(np.abs(new_centroids_real - centroids_real)) < tol and np.max(np.abs(new_centroids_imag - centroids_imag)) < tol:
            break

        centroids_real = new_centroids_real
        centroids_imag = new_centroids_imag

    # Quantize X to the nearest centroids
    X_real_b = centroids_real[np.digitize(X_real, boundaries_real)]
    X_imag_b = centroids_imag[np.digitize(X_imag, boundaries_imag)]

    # Combine the real and imaginary parts
    X_b = X_real_b + 1j * X_imag_b

    return X_b
    
    
def create_channel(N_sc, N_r, N_t, shadow_fading_margin_dB=8, channel='rayleigh'):
    global np_random
    
    G = 1 # compute_large_scale_fading(d=100, f_c=900e6)

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
    H /= np.trace(H)
    H *= np.sqrt(G_fading / 2) # element multiplication.

    H_full = np.repeat(H[np.newaxis, :, :], N_sc, axis=0)  # Repeat for all subcarriers

    return H_full


def _generate_cdl_c_channel(G, N_sc, N_r, N_t, sigma_dB):
    global np_random

    return H


def _generate_cdl_e_channel(G, N_sc, N_r, N_t, sigma_dB):
    global np_random


    return H


def compute_large_scale_fading(d, f_c, D_t_dB=12, D_r_dB=0, pl_exp=2):
    l = c / f_c
    G = _linear(D_t_dB * D_r_dB) * (l / (4 * np.pi * d)) ** pl_exp
    
    assert (G < 1)
    
    return G


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


def compute_precoder_combiner(H, snr_dB, algorithm='SVD_Waterfilling'):    
    U, S, Vh = _svd_precoder_combiner(H)
    
    try:
        D = waterfilling(S, snr_dB)
        Dinv = np.linalg.inv(D)
    except Exception as e:
        print(e)
        D = np.eye(S.shape[0])
        Dinv = np.eye(S.shape[0])

    F = np.conjugate(np.transpose(Vh, (0, 2, 1)))@D
    Gcomb = Dinv@np.conjugate(np.transpose(U, (0, 2, 1)))
   
    return F, Gcomb


def _svd_precoder_combiner(H):
    U, S, Vh = np.linalg.svd(H, full_matrices=False)
    return U, S, Vh


def waterfilling(S, rho):
    S = np.diag(S)
    
    n_channels = S.shape[0]
    mu = 0
    P = np.zeros(n_channels, dtype=np.complex128)
    
    # Implement bisection to find optimal water levels
    lower_bound = 0
    upper_bound = rho + np.max(S)
    
    tolerance = 1e-5
    while abs(upper_bound - lower_bound) > tolerance:
        mu = (upper_bound + lower_bound) / 2.
        
        # No negative powers allowed.
        P = np.maximum(0, mu - 1 / S)
        
        if np.sum(P) > rho:
            upper_bound = mu
        else:
            lower_bound = mu
    
    # Now final mu allows us to compute final water levels
    # P.sum() should be close to rho
    P = np.maximum(0, mu - 1 / S)
    
    return P


def _matrix_vector_multiplication(A, B):
    N_sc = A.shape[0]

    ans = np.zeros((N_sc, A.shape[1]), dtype=np.complex128)
    for n in range(N_sc):
        ans[n, :] = A[n, :] @ B[n]
        
    return ans


def plot_performance(df, xlabel, ylabel, semilogy=True):
    cols = list(set([xlabel, ylabel, 'snr_dB']))
    df = df[cols]
    df_plot = df.groupby('snr_dB').mean().reset_index()
    # xtick_labels = sorted(df[xlabel].unique())
    
    fig, ax = plt.subplots(figsize=(9,6))
    if semilogy:
        ax.set_yscale('log')
    ax.tick_params(axis=u'both', which=u'both')
    plt.plot(df_plot[xlabel].values, df_plot[ylabel].values, '--bo', alpha=0.7, 
             markeredgecolor='k', markerfacecolor='r', markersize=6)

    plt.grid(which='both', axis='both')
    # plt.xlim([min(xtick_labels), max(xtick_labels)*1.0001])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    tikzplotlib.save(f'output_{ylabel}_vs_{xlabel}.tikz')
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
        plt.savefig(f'channel_{filename}.pdf', format='pdf')
    plt.show()
    plt.close(fig)

    
def _print_divider():
    print('-' * 125)
    
    
def run_simulation(transmit_SNR_dB, constellation, M_constellation, crc_generator, N_sc, N_r, N_t, max_transmissions=500):
    global np_random
    
    start_time = time.time()
    
    if max_transmissions < 500:
        print('WARNING:  Low number of runs could cause statistically inaccurate results.')
        
    alphabet = create_constellation(constellation=constellation, M=M_constellation)
    k_constellation = int(np.log2(M_constellation))
    
    X_information, X, [x_b_i, x_b_q] = generate_transmit_symbols(N_sc, N_t, alphabet=alphabet)
    bits_transmitter, codeword_transmitter = bits_from_IQ(x_b_i, x_b_q)
    P_X = np.mean(_signal_power(X))

    P = generate_pilot_symbols(N_t, n_pilot, kind='dft')
    H = create_channel(N_sc, N_r, N_t, channel='rayleigh', shadow_fading_margin_dB=8)
    
    plot_channel(H)

    df = pd.DataFrame(columns = ['snr_dB', 'snr_transmitter_dB', 'EbN0_transmitter_dB', 
                                 'channel_estimation_error', 'PL_dB', 'snr_receiver_after_eq_dB', 
                                 'BER', 'BLER'])
    
    print(' | '.join(df.columns))
    
    for item, snr_dB in enumerate(transmit_SNR_dB):
        block_error = 0
        BER = []
        
        if item % 2 == 0:
            _print_divider()

        # Precoder and combiner
        # F, Gcomb = compute_precoder_combiner(H, snr_dB)

        for n_transmissions in range(max_transmissions):
            Y, noise = channel_effect(H, X, snr_dB)
            T, _ = channel_effect(H[:P.shape[0], :], P, snr_dB)
            P_Y = np.mean(_signal_power(Y))
            
            # Quantization is optional.
            Y = quantize(Y, b=np.inf)            
            
            PL_dB = _dB(P_X) - _dB(P_Y)
            
            P_noise = np.mean(_signal_power(noise))
    
            snr_transmitter_dB = _dB(P_X/P_noise) # This should be very close to snr_dB.
            EbN0_transmitter_dB = snr_transmitter_dB - _dB(k_constellation)
            
            # Estimate from pilots
            H_est = H if MIMO_estimation == 'perfect' else estimate_channel(P, T, snr_dB, algorithm=MIMO_estimation)
            estimation_error = _mse(H, H_est)
            
            W = equalize_channel(H_est, snr_dB, algorithm=MIMO_equalization)
            
            # Note:  Often, keep an eye on the product W@H and see how close it is to I.            
            X_hat = _matrix_vector_multiplication(W, Y)
            v =  _matrix_vector_multiplication(W, noise)
            
            P_X_hat = np.mean(_signal_power(X_hat))
            P_v = np.mean(_signal_power(v))
            
            snr_receiver_after_eq_dB = _dB(P_X_hat/P_v)
    
            # Now conduct symbol detection
            X_hat_information, X_hat, [x_hat_b_i, x_hat_b_q] = detect_symbols(X_hat, alphabet, algorithm=symbol_detection)

            bits_receiver, codeword_receiver = bits_from_IQ(x_hat_b_i, x_hat_b_q)
            
            # Performance
            crc_transmitter = compute_crc(codeword_transmitter, crc_generator)
            crc_receiver = compute_crc(codeword_receiver, crc_generator)
            
            # If CRC1 xor CRC2 is not zero, then error.
            if int(crc_transmitter, 2) ^ int(crc_receiver, 2) != 0:
                block_error += 1
            
            BER_i = compute_bit_error_rate(codeword_transmitter, codeword_receiver) 
            BER.append(BER_i)
    
        BER = np.mean(BER)
        BLER = block_error / max_transmissions

        to_append = [snr_dB, snr_transmitter_dB, EbN0_transmitter_dB, estimation_error, PL_dB, snr_receiver_after_eq_dB, BER, BLER]
        df_to_append = pd.DataFrame([to_append], columns=df.columns)
        
        rounded = [f'{x:.3f}' for x in to_append]
        del to_append
        
        if df.shape[0] == 0:
            df = df_to_append.copy()
        else:
            df = pd.concat([df, df_to_append], ignore_index=True, axis=0)
        
        print(' | '.join(map(str, rounded)))
    
    end_time = time.time()
    
    _print_divider()
    print(f'Time elapsed: {((end_time - start_time) / 60.):.2f} mins.')
    
    return df

df_results = run_simulation(transmit_SNR_dB, constellation, M_constellation,
                            crc_generator, N_sc, N_r, N_t,
                            max_transmissions=500)

plot_performance(df_results, xlabel='snr_transmitter_dB', ylabel='BER', semilogy=True)
plot_performance(df_results, xlabel='snr_transmitter_dB', ylabel='BLER', semilogy=True)
