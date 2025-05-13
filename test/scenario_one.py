#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 16:28:45 2025

@author: farismismar
"""

import os
import time

import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from deepwireless import simulator, plotter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# For Windows users
if os.name == 'nt':
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

print(tf.config.list_physical_devices('GPU'))

# The GPU ID to use, usually either "0" or "1" based on previous line.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###############################################################################
# Parameters
N_t = 2                                  # Number of transmit antennas
N_r = 2                                  # Number of receive antennas per user
N_sc = 64                                # Numer of subcarriers
transmit_power = 1                       # Total signal transmit power (bandwidth-normalized) [W].  Do not change it from 1 W.

max_transmissions = 700                  # Set to None if using a bitmap payload
payload_filename = 'faris.bmp'           # Must be an 8-bit 32x32 bitmap file
payload_mode = 'random'                  # random (for random bits) or defined (by a payload file)

constellation = 'QAM'                    # QAM or PSK
M_constellation = 16                     # Square constellations only.
code_rate = 1                            # Constant for now.

subcarrier_spacing = 15e3                # OFDM subcarrier bandwidth [Hz].
f_c = 1800e6                             # Center frequency [MHz] for large scale fading and DFT.

pathloss_model = 'FSPL'                  # Also: FSPL, UMa, and RMa.
channel_type = 'ricean'                  # Channel type: awgn, rayleigh, ricean, CDL-A, CDL-C, CDL-E, and deep_mimo
K_factor = 4                             # For Ricean
shadowing_std_dev = 4                    # in dB
channel_compression_ratio = 0            # Channel compression

estimation_algorithm = 'perfect'         # Also: perfect, LS, LMMSE (keep at perfect)
equalization_algorithm = 'MMSE'          # Also: MMSE, ZF
precoder = 'SVD_Waterfilling'            # Also: identity, SVD, SVD_Waterfilling, dft_beamforming
quantization_b = None                    # Quantization resolution

noise_figure = 2                         # Receiver noise figure [dB]
interference_power_dBm = -105            # dBm measured at the receiver
p_interference = 0.00                    # probability of interference

symbol_detection = 'ML'                  # Also: ML, kmeans, DNN, ensemble
crc_generator = 0b1001_0101              # CRC generator polynomial (n-bit)

# Transmit SNR in dB
transmit_SNR_dB = [-10, -5, 0, 5, 10, 15, 20, 25, 30][::-1]
###############################################################################

plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = "14"

output_path = './'
prefer_gpu = True

seed = 42  # Reproducibility
random_state = np.random.RandomState(seed=seed)

#############################################

# Pilot sequence consumes a small amount of the transmit power.
epsilon = 0.05

def __loss_fn_multi_class_classifier(Y_true, Y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy()
    return cce(Y_true, Y_pred)


# Objects for the simulation and plotting to work.
pl = plotter.plotter(output_path=output_path)
mlw = simulator.simulator(N_sc=N_sc, N_r=N_r, N_t=N_t, f_c=f_c,
                 transmit_power=transmit_power, precoder=precoder,
                 channel_type=channel_type, pathloss_model=pathloss_model,
                 noise_figure=noise_figure, max_transmissions=max_transmissions,
                 constellation=constellation, M_constellation=M_constellation,
                 quantization_b=quantization_b, Df=subcarrier_spacing,
                 payload_filename=payload_filename, crc_generator=crc_generator,
                 K_factor=K_factor, dnn_loss_function=__loss_fn_multi_class_classifier,
                 shadowing_std_dev=shadowing_std_dev, prefer_gpu=prefer_gpu,
                 random_state=random_state)

 
start_time = time.time()

#############################################
# 1) Initializations
#############################################
# This is the power of the signal (across all subcarriers for one OFDM symbol) or the trace of E XX*, which is P_X Nt
P_BS = transmit_power

alphabet = mlw.create_constellation(constellation=constellation, M=M_constellation)
pl.plot_constellation(alphabet, annotate=True, filename='constellation')
k_constellation = mlw.k_constellation

return_values = mlw.generate_transmit_symbols(alphabet, n_transmissions=max_transmissions, mode=payload_mode)

if payload_mode == 'random':
    X_information, X, [x_b_i, x_b_q], payload_size, crc_transmitter, n_transmissions = return_values

if payload_mode == 'defined':
    X_information_segmented, X_segmented, [x_b_i_segmented, x_b_q_segmented], payload_sizes, crc_transmitter_segmented, n_transmissions = return_values
    payload_size = payload_sizes[0]
    
if n_transmissions < 500:
    print('WARNING:  Low number of runs could cause statistically inaccurate results.')

H = mlw.create_channel(N_sc, N_r, N_t, channel=channel_type)

# Large scale fading and shadowing:
G = mlw.compute_large_scale_fading(dist=300, model=pathloss_model, f_c=f_c, D_t_dB=18, D_r_dB=-1, pl_exp=2)
G_dB = mlw.dB(G)
Gchi_dB, _ = mlw.compute_shadow_fading(N_sc, G_dB, shadowing_std_dev)  # Add shadowing \chi_\sigma
Gchi = mlw.linear(Gchi_dB)

# Note to self, introducing G_fading to the channel creates detection problems.
# H *= np.sqrt(Gchi)

# Factor in the fading to the channel gain.
GH = np.zeros_like(H)
for sc in range(N_sc):
    GH[sc, :, :] = np.sqrt(Gchi[sc]) * H[sc, :, :]

# Channel gain
channel_gain = np.linalg.norm(np.mean(GH, axis=0), ord='fro') ** 2
PL_dB = -mlw.dB(channel_gain)

pl.plot_channel(H, filename=channel_type)

F, Gcomb = mlw.compute_precoder_combiner(H, P_BS, algorithm=precoder)

if precoder != 'dft_beamforming':
    print('Channel eigenmodes are: {}'.format(mlw.find_channel_eigenmodes(H)))

P = mlw.generate_pilot_symbols(pilot_power=epsilon*mlw.transmit_power, kind='dft')  # dimensions: N_t x n_pilot

# The throughput can be computed by dividing the payload size by TTI (= 1 symbol duration)
print(f'Payload to be transmitted: {payload_size} bits over one OFDM symbol duration (including CRC).')

df_results = pd.DataFrame(columns=['n', 'snr_dB', 'Tx_EbN0_dB', 'Tx_Pwr_dBm',
                           'channel_estimation_error', 'compression_loss',
                           'PL_dB', 'Rx_Pwr_dBm', 'sinr_receiver_before_eq_dB',
                           'sinr_receiver_after_eq_dB', 'Rx_EbN0_dB',
                           'BER', 'BLER'])

df_detailed = df_results.copy().rename({'BLER': 'total_block_errors'}, axis=1)
df_results.drop(columns='n', inplace=True)

print(' | '.join(df_results.columns))

start_time = time.time()
for item, snr_dB in enumerate(transmit_SNR_dB):
    block_error = 0
    BER = []

    if item % 3 == 0:
        print('-' * 125)

    EbN0_dB = snr_dB - mlw.dB(mlw.k_constellation)

    full_payload_tx = []
    full_payload_rx = []

    for n_transmission in range(n_transmissions):
        if payload_mode == 'defined':
            x_b_i = x_b_i_segmented[n_transmission, :]
            x_b_q = x_b_q_segmented[n_transmission, :]
            X = X_segmented[n_transmission, :].reshape(N_sc, N_t)
            X_information = X_information_segmented[n_transmission, :]
            payload_size = payload_sizes[n_transmission]
            crc_transmitter = crc_transmitter_segmented[n_transmission]

        #############################################
        # 2) Transmit data
        #############################################
        bits_transmitter, codeword_transmitter = mlw.bits_from_IQ(x_b_i, x_b_q)

        P_X = np.mean(mlw.signal_power(X)) # * Df  # Power average per transmit antenna
        P_X_dBm = mlw.dB(P_X * 1e3)

        #############################################
        # 3) Precoding and Channel Effect
        #############################################
        FX = mlw._matrix_vector_multiplication(F, X)  # Faris: Note this change.
        Y, noise = mlw.channel_effect(H, FX, snr_dB)
        T, _ = mlw.channel_effect(H[:P.shape[0], :], P, snr_dB)

        #############################################
        # 4) Interference
        #############################################
        interference = mlw.generate_interference(Y, p_interference, interference_power_dBm)
        P_interference = np.mean(mlw.signal_power(interference))  # * Df

        Y += interference

        #############################################
        # 5) Combining
        #############################################
        # Left-multiply y and noise with Gcomb
        Y = mlw._matrix_vector_multiplication(Gcomb, Y)
        noise = mlw._matrix_vector_multiplication(Gcomb, noise)

        P_Y = np.mean(mlw.signal_power(Y)) # * Df
        P_noise = np.mean(mlw.signal_power(noise)) # * Df

        #############################################
        # 6) Quantization
        #############################################
        Y_unquant = Y
        Y = mlw.quantize(Y)
        P_Y = np.mean(mlw.signal_power(Y)) # * Df

        # snr_transmitter_dB = mlw.dB(P_X/P_noise) # This should be very close to snr_dB.

        # Note to self: PL is not P_Y / P_X.  The noise power is not subtracted.
        # Thus: P_noise + P_X / P_H ~ P_Y (where P_H is the path gain)
        P_Hx_dBm = P_X_dBm - PL_dB

        snr_rx_dB_pre_eq = P_Hx_dBm - mlw.dB(P_noise + P_interference) - mlw.noise_figure

        #############################################
        # 7) Channel Estimation
        #############################################
        if n_transmission == 0:
            # TODO: In the future, reassess pilot estimation.
            # Estimate from pilots
            H_est = H if estimation_algorithm == 'perfect' else mlw.estimate_channel(P, T, snr_dB, algorithm=estimation_algorithm)
            estimation_error = mlw.mse(H, H_est)

            # # Estimate from signal
            # H_est = H if MIMO_estimation == 'perfect' else estimate_channel(X, Y, snr_dB, algorithm=MIMO_estimation)

        #############################################
        # 8) Channel Compression
        #############################################
        if n_transmission == 0:
            # Compress the channel before sending to the receiver and use it for all subsequent transmissions
            # since all transmissions are assumed within channel coherence time.
            # _, H_est, compression_loss = compress_channel(mlw, pl, H_est, channel_compression_ratio, plotting=True)
            compression_loss = np.nan
            
        # Replace the channel H with Sigma as a result of the operations on
        # X and Y above.
        GH_est = Gcomb@H_est  # This is not Sigma
        Sigma = GH_est@F  # This is Sigma.  Is it diagonalized with elements equal the sqrt of eigenmodes when using SVD?  Yes.

        if (channel_compression_ratio == 0) and ((precoder == 'SVD') or (precoder == 'SVD_Waterfilling')):
            if np.allclose(Sigma[0], np.diag(np.diagonal(Sigma[0]))) == False:
                raise RuntimeError("SVD has failed.  Unable to achieve diagonalization.  Check channel estimation performance and try a different channel type.")

        # np.sqrt(_find_channel_eigenmodes(H)) == GH_estF[0].round(4)

        if (channel_compression_ratio == 0) and ((precoder == 'SVD') or (precoder == 'SVD_Waterfilling')):
            assert np.allclose(Sigma[0], np.diag(np.diagonal(Sigma[0])))

        #############################################
        # 9) Channel Equalization
        #############################################
        W = mlw.equalize_channel(Sigma, snr_dB, equalization_algorithm)
        z = mlw._matrix_vector_multiplication(W, Y)
        z_unquant = mlw._matrix_vector_multiplication(W, Y_unquant)
        v = mlw._matrix_vector_multiplication(W, noise)
        q = mlw._matrix_vector_multiplication(W, interference)

        #############################################
        # 10) Plots and Statistics
        #############################################        
        # Once plot the received symbols for the first antenna after equalization
        # Note that only the random mode is likely to obtain all constellation symbols in this plot
        if payload_mode == 'random' and item == 0 and n_transmission == 0:
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.set_aspect('equal', 'box')
            plt.scatter(np.real(z[:, 0]), np.real(z[:, 1]), s=50, c='b', edgecolors='none', alpha=0.4, label=f'Quantized (b={quantization_b})')
            plt.scatter(np.real(z_unquant[:, 0]), np.real(z_unquant[:, 1]), s=50, c='r', edgecolors='none', alpha=0.2, label='No quantization')
            plt.scatter(np.real(X[:, 0]), np.imag(X[:, 0]), s=50, c='k', edgecolors='none', label='Transmit')

            # Axes
            plt.axhline(0, color='black')
            plt.axvline(0, color='black')

            plt.grid(which='both')

            plt.xlabel('$I$')
            plt.ylabel('$Q$')
            plt.legend()
            plt.tight_layout()

            plt.show()
            plt.close(fig)

        P_w_S_x_dBm = mlw.dB(np.linalg.norm(W[0,:,:], ord='fro') ** 2) + mlw.dB(np.linalg.norm(Sigma[0, :], ord='fro') ** 2) + P_X_dBm  # Ideally I should do across all subcarriers.
        P_v_dBm = mlw.dB(np.linalg.norm(W[0, :, :]@Gcomb[0, :, :], ord='fro') ** 2) + mlw.dB((P_noise + P_interference) * 1e3)

        snr_rx_dB = P_w_S_x_dBm - P_v_dBm - noise_figure

        assert (code_rate == 1)  # No FEC is introduced.  This is like a hard-coded parameter for now.

        EbN0_rx_dB = snr_rx_dB - mlw.dB(k_constellation * code_rate)

        # Now conduct symbol detection to find x hat from z.
        X_hat_information, X_hat, [x_hat_b_i, x_hat_b_q] = mlw.detect_symbols(z, alphabet, algorithm=symbol_detection)
        bits_receiver, codeword_receiver = mlw.bits_from_IQ(x_hat_b_i, x_hat_b_q)

        # Remove the padding and CRC from here.
        crc_length = len(crc_transmitter)
        crc_pad_length = int(np.ceil(crc_length / k_constellation)) * \
            k_constellation  # padding included.

        # Reassemble
        full_payload_tx.append(''.join(codeword_transmitter[:payload_size]))
        full_payload_rx.append(''.join(codeword_receiver[:payload_size]))

        codeword_transmitter = codeword_transmitter[:-crc_pad_length]
        codeword_receiver = codeword_receiver[:-crc_pad_length]
        
        # Show a plot once per SNR
        if n_transmission == n_transmissions - 1:
            # Reassemble the segments
            mlw.plot_8_bit_rgb_payload(''.join(full_payload_tx), ''.join(full_payload_rx), title=f'Transmit SNR is {snr_dB} dB')

        # Performance measures are here.
        crc_receiver = mlw.compute_crc(codeword_receiver, crc_generator)

        # If CRC1 xor CRC2 is not zero, then error.
        if int(crc_transmitter, 2) ^ int(crc_receiver, 2) != 0:
            block_error += 1

        BER_i = mlw.compute_bit_error_rate(codeword_transmitter, codeword_receiver)
        BER.append(BER_i)

        to_append_i = [n_transmission, snr_dB, EbN0_dB, P_X_dBm,
                       estimation_error, compression_loss, PL_dB, P_Hx_dBm, snr_rx_dB_pre_eq, snr_rx_dB,  EbN0_rx_dB, BER_i, block_error]

        df_to_append_i = pd.DataFrame([to_append_i], columns=df_detailed.columns)

        if df_detailed.shape[0] == 0:
            df_detailed = df_to_append_i.copy()
        else:
            df_detailed = pd.concat([df_detailed, df_to_append_i], ignore_index=True, axis=0)
    ##############################################################################

    BER = np.mean(BER)
    BLER = block_error / n_transmissions

    to_append = [snr_dB, EbN0_dB, P_X_dBm, estimation_error, compression_loss,
                   PL_dB, P_Hx_dBm, snr_rx_dB_pre_eq, snr_rx_dB, EbN0_rx_dB, BER, BLER]
    df_to_append = pd.DataFrame([to_append], columns=df_results.columns)

    rounded = [f'{x:.3f}' for x in to_append]
    del to_append

    if df_results.shape[0] == 0:
        df_results = df_to_append.copy()
    else:
        df_results = pd.concat([df_results, df_to_append], ignore_index=True, axis=0)

    print(' | '.join(map(str, rounded)))

end_time = time.time()

print('-' * 125)
print(f'Time elapsed: {((end_time - start_time) / 60.):.2f} mins.')

df_detailed.to_csv('simulation_detailed_data.csv')
df_results.to_csv('simulation_data.csv')

pl.plot_performance(df_detailed, xlabel='Tx_EbN0_dB', ylabel='BER', semilogy=True, filename='BER')
pl.plot_performance(df_results, xlabel='Tx_EbN0_dB', ylabel='BLER', semilogy=True, filename='BLER')
pl.plot_pdf(v, text='noise', algorithm='KDE', filename='enhanced_noise')
###############################################################################
