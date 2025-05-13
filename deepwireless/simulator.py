#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:22:29 2024

@author: farismismar
"""

import os
import random
import numpy as np
import pandas as pd
from scipy.constants import speed_of_light
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

import DeepMIMOv3

################################################
# Single OFDM symbol and single user simulator
################################################
class simulator:
    __ver__ = '0.9.3'
    __date__ = '2025-05-15'

    def __init__(self, N_sc, N_r, N_t, transmit_power, f_c, precoder, channel_type,
                 pathloss_model, noise_figure, max_transmissions, constellation,
                 M_constellation, Df, K_factor, shadowing_std_dev, n_pilot=None, dnn_loss_function='mse',
                 quantization_b=None, crc_generator=0b1010_0101, payload_filename=None,
                 prefer_gpu=True, random_state=None):

        self.prefer_gpu = prefer_gpu

        if isinstance(random_state, np.random.mtrand.RandomState):
            self.random_state = random_state
            self.seed = 42  # Reproducibility
        else:
            self.random_state = np.random.RandomState(seed=random_state)
            self.seed = random_state

        # Necessary to ensure TensorFlow reproducibility.
        tf.random.set_seed(self.seed)
        random.seed(self.seed)

        self.payload_filename = payload_filename
        self.N_sc = N_sc
        self.N_r = N_r
        self.N_t = N_t
        self.transmit_power = transmit_power
        self.loss_function = dnn_loss_function
        self.f_c = f_c
        self.precoder = precoder
        self.channel_type = channel_type
        self.noise_figure = noise_figure
        self.max_transmissions = max_transmissions
        self.constellation = constellation
        self.M_constellation = M_constellation
        self.Df = Df
        self.crc_generator = crc_generator
        self.K_factor = K_factor
        self.shadowing_std_dev = shadowing_std_dev
        self.n_pilot = n_pilot if n_pilot is not None else N_t
        self.quantization_b = quantization_b

        # Number of streams.
        self.N_s = min(N_r, N_t) if precoder != 'identity' else N_t

        # Number of bits per symbol
        self.k_constellation = int(np.log2(M_constellation))


    def create_bit_payload(self, payload_size):
        bits = self.random_state.binomial(1, 0.5, size=payload_size)
        bits = ''.join(map(str, bits))
        return bits


    def create_constellation(self, constellation, M):
        if (constellation == 'PSK'):
            return self._create_constellation_psk(M)
        elif (constellation == 'QAM'):
            return self._create_constellation_qam(M)
        else:
            return None


    def decimal_to_gray(self, n, k):
        gray = n ^ (n >> 1)
        gray = bin(gray)[2:]

        return '{}'.format(gray).zfill(k)


    # Constellation based on Gray code
    def _create_constellation_psk(self, M):
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

        gray = constellation['m'].apply(lambda x: self.decimal_to_gray(x, k))
        constellation['I'] = gray.str[:(k//2)]
        constellation['Q'] = gray.str[(k//2):]

        constellation.loc[:, 'x'] = constellation.loc[:, 'x_I'] + 1j * constellation.loc[:, 'x_Q']

        # Normalize the transmitted symbols
        # The average power is normalized to unity
        P_average = np.mean(np.abs(constellation.loc[:, 'x']) ** 2)
        constellation.loc[:, 'x'] /= np.sqrt(P_average)

        return constellation


    # Constellation based on Gray code
    def _create_constellation_qam(self, M):
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

        gray = constellation['m'].apply(lambda x: self.decimal_to_gray(x, k))
        constellation['I'] = gray.str[:(k//2)]
        constellation['Q'] = gray.str[(k//2):]

        constellation.loc[:, 'x'] = constellation.loc[:, 'x_I'] + \
            1j * constellation.loc[:, 'x_Q']

        # Normalize the transmitted symbols
        # The average power is normalized to unity
        P_average = self.signal_power(constellation['x'])  # np.mean(np.abs(constellation.loc[:, 'x']) ** 2)
        constellation.loc[:, 'x'] /= np.sqrt(P_average)

        return constellation


    def signal_power(self, signal):
        return np.mean(np.abs(signal) ** 2, axis=0)


    def perform_framing(self, bits, max_codeword_length, crc_generator):
        N_sc = self.N_sc
        N_t = self.N_t
        k = self.k_constellation

        crc_length = len(bin(crc_generator)[2:])  # in bits.
        crc_pad_length = int(np.ceil(crc_length / k)) * k  # padding included.

        payload_length = min(len(bits), max_codeword_length - crc_pad_length)  # in bits
        # payload_length = max_codeword_length - crc_pad_length  # in bits
        padding_length_bits = max_codeword_length - payload_length  # in bits
        # padding_length_syms = int(np.ceil(padding_length_bits / k))

        bits = bits[:-crc_pad_length]
        crc_transmitter = self.compute_crc(bits, crc_generator)

        # Construct the codeword (frame containing payload and CRC)
        while True:
            codeword = bits + '0' * (padding_length_bits - crc_length) + crc_transmitter
            if (len(codeword) == max_codeword_length) and (len(codeword) % N_sc == 0) and (len(codeword) % N_t == 0):
                break

            else:
                padding_length_bits += 1
        ###
        assert(len(codeword) == max_codeword_length)  # in bits.

        return codeword, payload_length, crc_transmitter

    
    def generate_transmit_symbols(self, alphabet, mode='random', n_transmissions=None):
        transmit_power = self.transmit_power
        payload_filename = self.payload_filename

        N_sc = self.N_sc
        N_t = self.N_t
        k = self.k_constellation

        crc_generator = self.crc_generator
        max_codeword_length = N_sc * N_t * k  # [bits].  For the future, this depends on the MCS index.
        crc_length = len(bin(crc_generator)[2:])  # in bits.
        crc_pad_length = int(np.ceil(crc_length / k)) * k  # padding included.

        # The padding length in symbols is
        payload_length = max_codeword_length - crc_pad_length

        if mode == 'random':
            bits = self.create_bit_payload(payload_length)
            if n_transmissions is None:
                raise ValueError("Cannot use mode random without specifying n_transmissions.")
            codeword, payload_length, crc_transmitter = self.perform_framing(bits, max_codeword_length, crc_generator)
            x_b_i, x_b_q, x_information, x_symbols = self.bits_to_baseband(codeword, alphabet)

            x_information = np.reshape(x_information, (N_sc, N_t))
            x_symbols = np.reshape(x_symbols, (N_sc, N_t))

            # Normalize and scale the transmit power of the symbols.
            x_symbols /= np.sqrt(self.signal_power(x_symbols).mean() / transmit_power)
            x_b_i = np.reshape(x_b_i, (-1, N_t))
            x_b_q = np.reshape(x_b_q, (-1, N_t))

            n_transmissions = self.max_transmissions
            return x_information, x_symbols, [x_b_i, x_b_q], payload_length, crc_transmitter, n_transmissions

        elif mode == 'defined':
            bits = self.read_8_bit_bitmap(payload_filename)

            x_information_segmented = []
            x_symbols_segmented = []
            x_b_i_segmented = []
            x_b_q_segmented = []
            crc_transmitter_segmented = []
            payload_lengths = []

            print('Performing segmentation and framing of the payload (this may take long time)...')
            
            # The bitmap full size is len(bits)
            # Split bits to segments, the length of which is payload_length
            payload_length = min(len(bits), max_codeword_length - crc_pad_length)  # in bits

            # bits_reassembled = ''
            n_transmissions = int(np.ceil(len(bits) / payload_length))  # + 1
            for i in tqdm(range(n_transmissions)):
                start = i * payload_length
                end = start + payload_length
                bits_i = bits[start:end]
                codeword_i, _, crc_transmitter = self.perform_framing(bits_i, max_codeword_length, crc_generator)
                payload_lengths.append(len(bits_i))
                x_b_i, x_b_q, x_information, x_symbols = self.bits_to_baseband(codeword_i, alphabet)
                x_information_segmented.append(x_information)
                x_symbols_segmented.append(x_symbols)
                x_b_i_segmented.append(x_b_i)
                x_b_q_segmented.append(x_b_q)
                crc_transmitter_segmented.append(crc_transmitter)

            x_information_segmented = np.array(x_information_segmented)
            x_symbols_segmented = np.array(x_symbols_segmented)
            payload_lengths = np.array(payload_lengths)
            x_b_i_segmented = np.array(x_b_i_segmented)
            x_b_q_segmented = np.array(x_b_q_segmented)
            
            assert(payload_lengths.sum() == len(bits))

            crc_transmitter_segmented = np.array(crc_transmitter_segmented)

            return x_information_segmented, x_symbols_segmented, [x_b_i_segmented, x_b_q_segmented], payload_lengths, crc_transmitter_segmented, n_transmissions

        else:
            raise ValueError(f"The mode parameter must be set as 'random' or 'defined' and not {mode}.")


    def generate_interference(self, Y, p_interference, interference_power_dBm):
        N_sc, N_r = Y.shape

        interference_power = self.linear(interference_power_dBm)

        interf = np.sqrt(interference_power / 2) * \
            (self.random_state.normal(0, 1, size=(N_sc, N_r)) + \
             1j * self.random_state.normal(0, 1, size=(N_sc, N_r)))

        mask = self.random_state.binomial(n=1, p=p_interference, size=N_sc)

        # Apply some changes to interference
        for idx in range(N_sc):
            interf[idx, :] *= mask[idx]

        return interf


    def generate_pilot_symbols(self, pilot_power, kind='dft'):
        N_t = self.N_t
        n_pilot = self.n_pilot
        
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
            random_matrix = np.sqrt(1 / 2) * (self.random_state.randn(n_pilot, N_t) + \
                                              1j * self.random_state.randn(n_pilot, N_t))

            # Perform QR decomposition on the random matrix to get a unitary matrix
            Q, R = np.linalg.qr(random_matrix)

            # Ensure Q is unitary (the first n_pilot rows are orthonormal)
            pilot_matrix = Q[:, :N_t]

        if kind == 'semi-unitary':
            # Compute a unitary matrix from a combinatoric of e
            I = np.eye(N_t)
            idx = self.random_state.choice(range(N_t), size=N_t, replace=False)
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

        # Normalize the pilot matrix such that its Frob norm sq is equal to pilot_power.
        # Scale the pilot matrix
        pilot_matrix /= np.linalg.norm(pilot_matrix, ord='fro') / np.sqrt(pilot_power)

        return pilot_matrix


    def bits_to_baseband(self, x_bits, alphabet):
        # Determine k (bits per symbol)
        k = int(np.log2(alphabet.shape[0]))
        k_half = k // 2

        # Ensure x_bits is a string and pad if necessary
        x_bits = str(x_bits)
        num_symbols = len(x_bits) // k
        if len(x_bits) % k:
            x_bits = x_bits.zfill(num_symbols * k + k)  # Pad with zeros at the start
            num_symbols += 1

        # Convert bit string to array and reshape into symbols
        bit_array = np.array(list(x_bits), dtype=int)
        bit_array = bit_array[-num_symbols*k:]  # Trim to multiple of k from the end
        symbols_bits = bit_array.reshape(num_symbols, k)

        # Split into I and Q components
        x_b_i = symbols_bits[:, :k_half]
        x_b_q = symbols_bits[:, k_half:]

        # Convert binary arrays to strings for mapping
        x_b_i_str = np.array([''.join(map(str, row)) for row in x_b_i])
        x_b_q_str = np.array([''.join(map(str, row)) for row in x_b_q])

        # Create a mapping from I,Q pairs to m and x
        iq_to_m = {(row['I'], row['Q']): row['m'] for _, row in alphabet.iterrows()}
        iq_to_x = {(row['I'], row['Q']): row['x'] for _, row in alphabet.iterrows()}

        # Map I,Q pairs to m and x using vectorized operations
        iq_pairs = list(zip(x_b_i_str, x_b_q_str))
        x_info = np.array([iq_to_m.get(pair, 0) for pair in iq_pairs], dtype=int)
        x_sym = np.array([iq_to_x.get(pair, 0) for pair in iq_pairs], dtype=complex)

        return x_b_i_str, x_b_q_str, x_info, x_sym


    def channel_effect(self, H, X, snr_dB):
        N_sc = H.shape[0]

        # Set a flag to deal with beamforming.
        is_beamforming = True
        N_r = 1

        # Parameters
        if len(H.shape) == 3:  # MIMO case
            _, N_r, N_t = H.shape
            is_beamforming = False
            if N_t > 1 and N_r == 1 and self.precoder != 'dft_beamforming':
                raise ValueError('Only beamforming is supported for MISO.  Check the setting of precoder.')

        # Convert SNR from dB to linear scale
        snr_linear = self.linear(snr_dB)

        # Compute the power of the transmit matrix X
        signal_power = np.mean(self.signal_power(X))  # This is E[||X||^2] and must equal transmit_power / N_t.

        # Calculate the noise power based on the input SNR
        noise_power = signal_power / snr_linear

        # Generate additive white Gaussian noise (AWGN)
        noise = np.sqrt(noise_power / 2) * (self.random_state.randn(N_sc, N_r) + 1j * self.random_state.randn(N_sc, N_r))

        received_signal = np.zeros((N_sc, N_r), dtype=np.complex128)

        if is_beamforming:
            received_signal = H*X + noise
            return received_signal, noise

        for sc in range(N_sc):
            received_signal[sc, :] = H[sc, :, :] @ X[sc, :] + noise[sc, :]

        return received_signal, noise


    def equalize_channel(self, H, snr_dB, algorithm):
        N_t = self.N_t
        N_sc = self.N_sc

        if self.precoder == 'dft_beamforming':
            # Equalization for DFT beamforming is not necessary.
            return np.ones((N_t, N_sc))

        if algorithm == 'ZF':
            return self._equalize_channel_ZF(H)

        if algorithm == 'MMSE':
            return self._equalize_channel_MMSE(H, snr_dB)

        return None


    def _equalize_channel_ZF(self, H):
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


    def _equalize_channel_MMSE(self, H, snr_dB):
        N_sc, N_r, N_t = H.shape
        snr_linear = self.linear(snr_dB)

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


    def _estimate_channel_least_squares(self, X, Y):
        N_sc = self.N_sc

        # This is least square (LS) estimation
        H_estimated = Y@np.conjugate(np.transpose(X))

        # Repeat it across all N_sc
        H_estimated_full = np.repeat(H_estimated[np.newaxis, :, :], N_sc, axis=0)  # Repeat for all subcarriers

        return H_estimated_full


    def _estimate_channel_linear_regression(self, X, Y):
        # This is only for real-valued data and one subcarrier.
        N_sc, N_r = Y.shape
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


    def _estimate_channel_LMMSE(self, X, Y, snr_dB):
        snr_linear = self.linear(snr_dB)

        H_ls = self._estimate_channel_least_squares(X, Y)

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


    def estimate_channel(self, X, Y, snr_dB, algorithm):
        H_lmmse, H_ls = self._estimate_channel_LMMSE(X, Y, snr_dB)

        if algorithm == 'LS':
            return H_ls

        if algorithm == 'LMMSE':
            return H_lmmse

        return None


    def quantize(self, X):
        b = self.quantization_b

        if b == np.inf or b is None:
            return X

        X_re = np.real(X)
        X_im = np.imag(X)

        if b == 1:
            return np.sign(X_re) + 1j * np.sign(X_im)

        Xb_re = np.apply_along_axis(self._quantization, 0, X_re, b)
        Xb_im = np.apply_along_axis(self._quantization, 0, X_im, b)

        return Xb_re + 1j * Xb_im


    def _quantization(self, x, b):
        # Calculate number of quantization levels
        levels = 2 ** b
        
        # Find min and max of signal
        signal_min = np.min(x)
        signal_max = np.max(x)
        
        # Calculate quantization step size
        delta = (signal_max - signal_min) / (levels - 1)
        
        # Normalize signal to [0, levels-1]
        normalized = (x - signal_min) / delta
        
        # Round to nearest integer (quantization)
        quantized = np.round(normalized)
        
        # Scale back to original range
        quantized_signal = quantized * delta + signal_min
        
        # Clip to ensure values stay within bounds
        x_q = np.clip(quantized_signal, signal_min, signal_max)
        
        return x_q
    
    # very very slow
    # def _lloyd_max_quantization(self, x, b, max_iteration):
    #     # derives the quantized vector
    #     # https://gist.github.com/PrieureDeSion
    #     # https://github.com/stillame96/lloyd-max-quantizer
    #     from utils import normal_dist, expected_normal_dist, MSE_loss, LloydMaxQuantizer

    #     repre = LloydMaxQuantizer.start_repre(x, b)
    #     min_loss = 1.0
    #     min_repre = repre
    #     min_thre = LloydMaxQuantizer.threshold(min_repre)

    #     for i in np.arange(max_iteration):
    #         thre = LloydMaxQuantizer.threshold(repre)
    #         # In case wanting to use with another mean or variance,
    #         # need to change mean and variance in utils.py file
    #         repre = LloydMaxQuantizer.represent(thre, expected_normal_dist, normal_dist)
    #         x_hat_q = LloydMaxQuantizer.quant(x, thre, repre)
    #         loss = MSE_loss(x, x_hat_q)

    #         # Keep the threhold and representation that has the lowest MSE loss.
    #         if(min_loss > loss):
    #             min_loss = loss
    #             min_thre = thre
    #             min_repre = repre

    #     # x_hat_q with the lowest amount of loss.
    #     best_x_hat_q = LloydMaxQuantizer.quant(x, min_thre, min_repre)

    #     return best_x_hat_q


    def create_channel(self, N_sc, N_r, N_t, channel='rayleigh'):
        f_c = self.f_c
        K_factor = self.K_factor

        if channel == 'awgn':
            return self._create_awgn_channel(N_sc, N_r, N_t)
        elif channel == 'ricean':
            return self._create_ricean_channel(N_sc, N_r, N_t, K_factor=K_factor)
        elif channel == 'rayleigh':
            return self._create_ricean_channel(N_sc, N_r, N_t, K_factor=0)
        elif channel == 'CDL-A':
            return self._generate_cdl_a_channel(N_sc, N_r, N_t, carrier_frequency=f_c)
        elif channel == 'CDL-C':
            return self._generate_cdl_c_channel(N_sc, N_r, N_t, carrier_frequency=f_c)
        elif channel == 'CDL-E':
            return self._generate_cdl_e_channel(N_sc, N_r, N_t, carrier_frequency=f_c)
        elif channel == 'deep_mimo':
            return self._generate_deep_mimo_channel(N_sc, N_r, N_t, carrier_frequency=f_c)
        else:
            raise ValueError(f"create_channel does not take the value {channel}.")


    def _create_awgn_channel(self, N_sc, N_r, N_t):
        H = np.eye(N_r, N_t)
        H = np.repeat(H[np.newaxis, :, :], N_sc, axis=0)  # Repeat for all subcarriers

        return H


    def _create_ricean_channel(self, N_sc, N_r, N_t, K_factor):
        mu = np.sqrt(K_factor / (1 + K_factor))
        sigma = np.sqrt(1 / (1 + K_factor))

        # H = self.random_state.normal(loc=mu, scale=sigma, size=(N_sc, N_r, N_t)) + \
        #     1j * self.random_state.normal(loc=mu, scale=sigma, size=(N_sc, N_r, N_t))


        # # For each subcarrier and symbol, calculate the MIMO channel response
        # for sc in range(N_sc):
        #     # Normalize channel to unity gain
        #     H[sc, :, :] /= np.linalg.norm(H[sc, :, :], ord='fro')

        H = self.random_state.normal(loc=mu, scale=sigma, size=(N_r, N_t)) + \
            1j * self.random_state.normal(loc=mu, scale=sigma, size=(N_r, N_t))

        # Normalize channel to unity gain
        H /= np.linalg.norm(H, ord='fro')

        H = np.repeat(H[np.newaxis, :, :], N_sc, axis=0)  # Repeat for all subcarriers

        return H


    def _generate_cdl_a_channel(self, N_sc, N_r, N_t, carrier_frequency):
        Df = self.Df

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
                                         (self.random_state.randn(N_sc, N_r, N_t) + 1j * self.random_state.randn(N_sc, N_r, N_t)) / np.sqrt(2)

                # Apply phase shift across subcarriers
                H += H_tap * phase_shift[sc]

        # Normalize channel gains
        for sc in range(N_sc):
            H[sc, :, :] /= np.linalg.norm(H[sc, :, :], ord='fro')

        return H


    def _generate_cdl_c_channel(self, N_sc, N_r, N_t, carrier_frequency):
        Df = self.Df

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
                    (self.random_state.randn(N_r, N_t, N_sc) + 1j * self.random_state.randn(N_r, N_t, N_sc)) / np.sqrt(2)

            # Apply phase shift across subcarriers
            H += (H_tap * phase_shift).transpose(2, 0, 1)  # Adjust dimensions (N_sc, N_r, N_t)

        # Normalize channel gains
        for sc in range(N_sc):
            H[sc, :, :] /= np.linalg.norm(H[sc, :, :], ord='fro')

        return H


    def _generate_cdl_e_channel(self, N_sc, N_r, N_t, carrier_frequency):
        Df = self.Df

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
                    (self.random_state.randn(N_r, N_t, N_sc) + 1j * self.random_state.randn(N_r, N_t, N_sc)) / np.sqrt(2)

            # Apply phase shift across subcarriers
            H += (H_tap * phase_shift).transpose(2, 0, 1)  # Adjust dimensions (N_sc, N_r, N_t)

        # Normalize channel gains
        for sc in range(N_sc):
            H[sc, :, :] /= np.linalg.norm(H[sc, :, :], ord='fro')

        return H


    def _generate_deep_mimo_channel(self, N_sc, N_r, N_t, scenario=None):
        if scenario is None:
            scenario = 'O1_60'  # This is 60 GHz
        
        scenarios_folder = r'./deepmimo_scenarios/'

        # Note that the Deep MIMO model uses channels
        # that already have gain in them
        
        # Load the default parameters
        parameters = DeepMIMOv3.default_params()

        # Set scenario name
        parameters['scenario'] = scenario

        if not os.path.isdir(scenarios_folder):
            os.mkdir(scenarios_folder)

        # Set the main folder containing extracted scenarios
        parameters['dataset_folder'] = scenarios_folder

        # Single base station
        parameters['active_BS'] = np.array([1])

        # Single user
        parameters['user_rows'] = np.array([1])

        # OFDM MIMO
        parameters['OFDM']['bandwidth'] = self.Df * N_sc / 1e9 * 20/18 # in GHz
        parameters['OFDM']['subcarriers'] = N_sc

        # To adopt a ULA in x-z direction with spacing 0.5*wavelength, set
        parameters['ue_antenna']['shape'] = np.array([1, N_r, 1])
        parameters['ue_antenna']['spacing'] = 0.5

        parameters['bs_antenna']['shape'] = np.array([1, N_t, 1])
        parameters['bs_antenna']['spacing'] = 0.5

        # Generate data
        try:
            dataset = DeepMIMOv3.generate_data(parameters)
        except Exception as e:
            raise RuntimeError(f'{e}')

        H = dataset[0]['user']['channel']  # This is N_sc x N_r x Nt

        H = H[:N_sc, :, :, 0]  # First user

        # # Normalize channel gains
        # for sc in range(N_sc):
        #     H[sc, :, :] /= np.linalg.norm(H[sc, :, :], ord='fro')

        return H


    def compute_large_scale_fading(self, dist, f_c, model='FSPL', D_t_dB=18, D_r_dB=-1, pl_exp=2, h_BS=25, h_UT=1.5):
        N_sc = self.N_sc
        Df = self.Df

        subcarrier_frequencies = f_c + (np.arange(N_sc) - N_sc // 2) * Df   # subcarrier indices

        wavelength = speed_of_light / subcarrier_frequencies

        if model == 'FSPL':
            PL_FSPL = self.dB((4 * np.pi * dist / wavelength)) * pl_exp
            G = self.linear(D_t_dB + D_r_dB - PL_FSPL)
        elif model == 'UMa':
            dist_2D = dist  # Assuming 2D distance for simplicity
            f_c_GHz = f_c / 1e9  # Convert Hz to GHz

            # Breakpoint distance
            d_BP = 4 * h_BS * h_UT * wavelength

            PL_UMa = []  #  in dB
            for d in d_BP:
                if dist_2D <= d:
                    # Line-of-Sight (LOS) for distances less than breakpoint
                    PL_UMa.append(28.0 + 22 * np.log10(dist_2D) + 20 * np.log10(f_c_GHz))
                else:
                    # Non-Line-of-Sight (NLOS) approximated for simplicity
                    PL_UMa.append(13.54 + 39.08 * np.log10(dist_2D) + 20 * np.log10(f_c_GHz) - 0.6 * (h_UT - 1.5))
            PL_UMa = np.array(PL_UMa)

            # Convert pathloss to gain
            G = self.linear(D_t_dB + D_r_dB - PL_UMa)
        elif model == 'RMa':
            # Rural Macro (3GPP TR 38.901)
            dist_2D = dist
            f_c_GHz = subcarrier_frequencies / 1e9  # Convert Hz to GHz

            # Simplified LOS model
            PL_RMa = 20 * np.log10(40 * np.pi * dist_2D * f_c_GHz / 3) + \
                     min(0.03 * h_BS ** 1.72, 10) * np.log10(dist_2D) - \
                     min(0.044 * h_BS ** 1.72, 14.77) + 0.002 * np.log10(h_BS) * dist_2D

            # Convert pathloss to gain
            G = self.linear(D_t_dB + D_r_dB - PL_RMa)
        else:
            raise ValueError(f"Model must be 'FSPL', 'UMa', or 'RMa', but got {model}.")

        # Gain cannot exceed 1 or be negative.
        assert np.all(G <= 1)

        return G


    def compute_shadow_fading(self, N_sc, large_scale_fading_dB, shadow_fading_std):
        chi_sigma = self.random_state.normal(loc=0, scale=shadow_fading_std, size=N_sc)
        G_dB = np.zeros_like(large_scale_fading_dB)

        G_dB = large_scale_fading_dB - chi_sigma

        return G_dB, chi_sigma


    def _vec(self, H):
        # H is numpy array
        return H.flatten(order='F')


    def mse(self, H_true, H_estimated):
        return np.mean(np.abs(H_true - H_estimated) ** 2)


    def dB(self, X):
        return 10 * np.log10(X)


    def linear(self, X):
        return 10 ** (X / 10.)


    def detect_symbols(self, z, alphabet, algorithm):
        if algorithm == 'kmeans':
            return self._detect_symbols_kmeans(z, alphabet)

        if algorithm == 'ML':
            return self._detect_symbols_ML(z, alphabet)

        # Supervised learning detections
        y = alphabet['m'].values
        X = np.c_[np.real(alphabet['x']), np.imag(alphabet['x'])]

        X_infer = z.flatten()
        X_infer = np.c_[np.real(X_infer), np.imag(X_infer)]

        if algorithm == 'ensemble':
            _, [training_accuracy_score, test_accuracy_score], y_infer =  \
                self._detect_symbols_ensemble(X, y, X_infer)

            # print(f'Ensemble training accuracy is {training_accuracy_score:.2f}.')
            # print(f'Ensemble test accuracy is {test_accuracy_score:.2f}.')

        if algorithm == 'DNN':
            _, [train_acc_score, test_acc_score], y_infer = \
                self._detect_symbols_DNN(X, y, X_infer)

            # print(f'DNN training accuracy is {train_acc_score:.2f}.')
            # print(f'DNN test accuracy is {test_acc_score:.2f}.')

        df = pd.merge(pd.DataFrame(data={'m': y_infer}), alphabet, how='left', on='m')

        # Reverse the flatten operation
        symbols = df['x'].values.reshape(z.shape)
        information = df['m'].values.reshape(z.shape)
        bits_i = df['I'].values.reshape(z.shape)
        bits_q = df['Q'].values.reshape(z.shape)

        return information, symbols, [bits_i, bits_q]


    def _detect_symbols_kmeans(self, x_sym_hat, alphabet):
        x_sym_hat_flat = x_sym_hat.flatten()

        X = np.real(x_sym_hat_flat)
        X = np.c_[X, np.imag(x_sym_hat_flat)]
        X = X.astype('float32')

        centroids = np.c_[np.real(alphabet['x']), np.imag(alphabet['x'])]

        # Intialize k-means centroid location deterministcally as a constellation
        kmeans = KMeans(n_clusters=self.M_constellation, init=centroids, n_init=1,
                        random_state=self.random_state).fit(centroids)

        information = kmeans.predict(X)
        df_information = pd.DataFrame(data={'m': information})

        df = df_information.merge(alphabet, how='left', on='m')
        symbols = df['x'].values.reshape(x_sym_hat.shape)
        bits_i = df['I'].values.reshape(x_sym_hat.shape)
        bits_q = df['Q'].values.reshape(x_sym_hat.shape)

        return information, symbols, [bits_i, bits_q]


    def _detect_symbols_ensemble(self, X_train, y_train, X_test):
        y_train = y_train.ravel()

        # The classifier hyperparameters need to be tuned.
        base_estimator = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                                criterion='entropy',
                                                class_weight='balanced',
                                                random_state=self.random_state)

        # hyperparameters = {'criterion': ['entropy', 'gini'],
        #                     'min_impurity_decrease': [0.1, 0.2],
        #                     'min_weight_fraction_leaf': [0.1, 0.3],
        #                     'min_samples_split': [2, 10]
        #                     }

        # # kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        # loo = LeaveOneOut()
        # clf = GridSearchCV(base_estimator, param_grid=hyperparameters,
        #                    cv=loo, n_jobs=-1, scoring='roc_auc_ovr_weighted',
        #                    verbose=0)

        clf = base_estimator  # No grid search tuning is done.
        clf.fit(X_train, y_train)

        y_test_pred = clf.predict(X_test)
        training_accuracy_score = clf.score(X_train, y_train)

        return clf, [training_accuracy_score, np.nan], y_test_pred


    def _detect_symbols_ML(self, symbols, alphabet):
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


    def _detect_symbols_DNN(self, X_train, y_train, X_test, depth=6, width=8,
                            epoch_count=256, batch_size=16):
        _, nX = X_test.shape

        # Make more data since the constellation size is small.
        # This improves the learning significantly.
        X_train_augmented = np.empty((0, nX))
        epsilons = [1e-2, 1e-3]

        for perturb in epsilons:
            X_train_i = X_train + self.random_state.normal(0, scale=perturb, size=X_train.shape)
            X_train_augmented = np.r_[X_train_augmented, X_train_i]

        X_train = np.r_[X_train, X_train_augmented]
        y_train = np.tile(y_train, len(epsilons) + 1)

        Y_train = keras.utils.to_categorical(y_train)

        # start_time = time.time()
        model, [training_accuracy_score, test_accuracy_score], y_test_pred = self._train_dnn(X_train, Y_train, X_test,
                                                depth=depth, width=width, epoch_count=epoch_count,
                                                batch_size=batch_size, loss_function=self.loss_function,
                                                learning_rate=1e-2)

        # end_time = time.time()
        # print("Training took {:.2f} mins.".format((end_time - start_time) / 60.))

        return model, [training_accuracy_score, test_accuracy_score], y_test_pred


    def _train_dnn(self, X_train, Y_train, X_test, depth, width, epoch_count,
                   batch_size, loss_function, learning_rate):
        prefer_gpu = self.prefer_gpu

        use_cuda = len(tf.config.list_physical_devices('GPU')) > 0 and prefer_gpu
        device = "/gpu:0" if use_cuda else "/cpu:0"

        _, nX = X_train.shape
        _, nY = Y_train.shape

        # If a DNN model is stored, use it.  Otherwise, train a DNN.
        try:
            dnn_classifier = \
                keras.models.load_model('dnn_detection.keras',
                                        custom_objects={'__loss_fn_classifier': loss_function})
            training_accuracy_score = np.nan # no training is done.
        except Exception as e:
            print(f'Failed to load model due to {e}.  Training from scratch.')
            dnn_classifier = self.__create_dnn(input_dimension=nX, output_dimension=nY,
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
                self._plot_keras_learning(history, filename='dnn_detection')

        # Perform inference.
        with tf.device(device):
            Y_test_pred = dnn_classifier.predict(X_test, verbose=0)

        y_test_pred = np.argmax(Y_test_pred, axis=1)

        return dnn_classifier, [training_accuracy_score, np.nan], y_test_pred


    def _train_lstm(self, X_train, X_test, Y_train, Y_test, lookbacks, depth, width, epoch_count, batch_size, learning_rate):
        prefer_gpu = self.prefer_gpu

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

            model = self.__create_lstm(input_shape=(X_train.shape[1], X_train.shape[2]),
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
                self._plot_keras_learning(history, filename='lstm')

        # Perform inference.
        with tf.device(device):
            Y_test_pred = model.predict(X_test)
            loss, test_accuracy_score = model.evaluate(X_test, Y_test)
        y_test_pred = np.argmax(Y_test_pred, axis=1)

        return model, [training_accuracy_score, test_accuracy_score], y_test_pred


    def create_cnn(self, learning_rate):
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


    def __create_dnn(self, input_dimension, output_dimension, depth, width, learning_rate):
        model = keras.Sequential()
        model.add(keras.Input(shape=(input_dimension,)))

        for hidden in range(depth):
            model.add(layers.Dense(width, activation='relu'))

        model.add(layers.Dense(output_dimension, activation='softmax'))

        model.compile(loss=self.loss_function, optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=['accuracy', 'categorical_crossentropy']) # Accuracy here is okay.  These metrics are what .evaluate() returns.

        # Reporting the number of parameters
        print(model.summary())

        num_params = model.count_params()
        print('Number of parameters: {}'.format(num_params))

        return model


    def __create_lstm(self, input_shape, output_shape, depth, width, learning_rate):
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


    def bits_from_IQ(self, x_b_i, x_b_q):
        assert x_b_i.shape == x_b_q.shape
        shape = x_b_i.shape

        bits = np.array([f"{i}{q}" for i, q in zip(np.array(x_b_i).flatten(), np.array(x_b_q).flatten())])
        bits = np.reshape(bits, shape)

        flattened = ''.join(bits.flatten())

        return bits, flattened


    def compute_bit_error_rate(self, a, b):
        assert(len(a) == len(b))

        length = len(a)
        bit_error = 0
        for idx in range(length):
            if a[idx] != b[idx]:
                bit_error += 1

        return bit_error / length


    def compute_crc(self, x_bits_orig, crc_generator):
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


    def compute_precoder_combiner(self, H, P_total, algorithm='SVD_Waterfilling'):
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
            F = self._dft_codebook(N_t, f_c=self.f_c)

            # Search for the optimal beamformer
            max_sinr = -np.inf
            for idx in range(N_t):
                sinr = np.abs(np.vdot(H[0, :, :], F[idx, :])) ** 2
                if sinr > max_sinr:
                    max_sinr = sinr
                    f_opt = F[idx, :] # extract a vector

            # There is no combiner in single user beamforming (combine what?  A scalar!)
            Gcomb = np.ones((N_sc, 1))
            return f_opt, Gcomb

        if algorithm == 'SVD':
            U, S, Vh = self._svd_precoder_combiner(H)

            F = np.conjugate(np.transpose(Vh, (0, 2, 1)))  # F is equal to V
            Gcomb = np.conjugate(np.transpose(U, (0, 2, 1)))  # G is equal to U*

            return F, Gcomb

        if algorithm == 'SVD_Waterfilling':
            U, S, Vh = self._svd_precoder_combiner(H)

            try:
                D = self._waterfilling(S[0, :], P_total)  # The power allocation matrix using the first OFDM subcarrier
                Dinv = np.linalg.inv(D)
            except Exception as e:
                print(f'Waterfilling failed due to {e}.  Returning identity power allocation.')
                D = np.eye(S.shape[1])
                Dinv = np.eye(S.shape[1])

            F = np.conjugate(np.transpose(Vh, (0, 2, 1)))@D
            Gcomb = Dinv@np.conjugate(np.transpose(U, (0, 2, 1)))

            return F, Gcomb


    def _dft_codebook(self, N_t, f_c, k_oversample=1):
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
            F[:, i] = f_i.T

        return F


    def find_channel_eigenmodes(self, H, subcarrier_idx=0):
        _, S, _ = np.linalg.svd(H[subcarrier_idx,:,:], full_matrices=False)
        eigenmodes = S ** 2

        return eigenmodes


    def _svd_precoder_combiner(self, H):
        U, S, Vh = np.linalg.svd(H, full_matrices=True)
        return U, S, Vh


    def _waterfilling(self, S, power):
        n_modes = len(S)

        # Ensure eigenmodes are non-zero
        if np.any(S < 1):
            raise ValueError("Channel has very weak eigenmodes which prevents waterfilling.  Please use a different channel or a different precoder")

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


    def _matrix_vector_multiplication(self, A, B):
        try:
            N_sc = A.shape[0]

            ans = np.zeros((N_sc, A.shape[1]), dtype=np.complex128)
            for n in range(N_sc):
                ans[n, :] = A[n, :]@B[n]
            return ans
        except:
            # This is likely due to beamforming.
            return A@B


    def read_8_bit_bitmap(self, file):
        word_length = 8

        # This must be a 32x32 pixel bitmap image
        im_orig = plt.imread(file)
        im = im_orig.flatten()
        im = [bin(a)[2:].zfill(word_length) for a in im]

        im = ''.join(im)  # This is now a string of bits

        return im


    def plot_8_bit_rgb_payload(self, data1, data2=None, title=None):
        # binary_payload = ''.join(data1.flatten())
        word_length = 8  # Typically 8 bits per channel

        if data2 is not None:
            data1 = ''.join(np.array(data1).flatten())
            data2 = ''.join(np.array(data2).flatten())

            assert len(data1) == len(data2)

            # Calculate the number of complete pixels (each pixel needs 3 words for RGB)
            n = len(data1) // word_length  # Total number of 8-bit words
            num_pixels = n // 3  # Number of complete RGB pixels (each pixel needs 3 words)

            # Calculate the largest square image dimension possible
            dim = int(np.sqrt(num_pixels))  # Largest integer sqrt for a square image
            num_pixels_used = dim * dim  # Number of pixels to use for a square image

            # Process only the data for complete pixels
            data_vector_1 = []
            data_vector_2 = []
            for i in range(num_pixels_used * 3):  # Only process enough words for dim*dim pixels
                d = str(data1[i*word_length:(i+1)*word_length])
                d = int(d, 2)
                data_vector_1.append(d)

                d = str(data2[i*word_length:(i+1)*word_length])
                d = int(d, 2)
                data_vector_2.append(d)

            # Convert to numpy array and reshape into (dim, dim, 3) for RGB
            data_vector_1 = np.array(data_vector_1, dtype='uint8')
            data_vector_1 = data_vector_1.reshape(dim, dim, 3)
            data_vector_2 = np.array(data_vector_2, dtype='uint8')
            data_vector_2 = data_vector_2.reshape(dim, dim, 3)

            fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)
            ax1.imshow(data_vector_1)
            ax2.imshow(data_vector_2)
            ax1.set_aspect('equal', 'box')
            ax1.axis('off')
            ax2.set_aspect('equal', 'box')
            ax2.axis('off')
            if title is not None:
                plt.suptitle(title)

        else:
            data1 = ''.join(np.array(data1).flatten())

            n = len(data1) // word_length  # Total number of 8-bit words
            num_pixels = n // 3  # Number of complete RGB pixels (each pixel needs 3 words)

            # Calculate the largest square image dimension possible
            dim = int(np.sqrt(num_pixels))  # Largest integer sqrt for a square image
            num_pixels_used = dim * dim  # Number of pixels to use for a square image

            # Process only the data for complete pixels
            data_vector = []
            for i in range(num_pixels_used * 3):  # Only process enough words for dim*dim pixels
                d = str(data1[i*word_length:(i+1)*word_length])
                d = int(d, 2)
                data_vector.append(d)

            # Convert to numpy array and reshape into (dim, dim, 3) for RGB
            data_vector = np.array(data_vector, dtype='uint8')
            data_vector = data_vector.reshape(dim, dim, 3)

            fig, ax = plt.subplots()
            ax.imshow(data_vector)
            ax.set_aspect('equal', 'box')
            ax.axis('off')

            if title is not None:
                plt.title(title)

        plt.tight_layout()
        # plt.show()
        plt.close(fig)
        
    def _plot_keras_learning(self, history, filename=None):

        fig, ax = plt.subplots(figsize=(9, 6))

        if 'val_loss' in history.history.keys():
            plt.plot(history.history['val_loss'], 'r', lw=2, label='Validation loss')
        plt.plot(history.history['loss'], 'b', lw=2, label='Loss')

        plt.grid(which='both', axis='both')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()

        if filename is not None:
            plt.savefig(f'history_keras_{filename}.pdf', format='pdf', dpi=fig.dpi)

        # plt.show()
        plt.close(fig)
