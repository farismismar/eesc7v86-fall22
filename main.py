# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:19:40 2023

@author: farismismar
"""

import numpy as np
import pandas as pd
from scipy.constants import pi, c
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tikzplotlib
import copy

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# For Windows users
if os.name == 'nt':
    os.add_dll_directory("/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))

# The GPU ID to use, usually either "0" or "1" based on previous line.
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

from tensorflow import keras
from tensorflow.keras import layers

from keras import backend as K

import time

# System parameters
file_name = 'faris.bmp' # either a file name or a payload
payload_size = 0 # 30000 # bits
constellation = 'QAM'
M_constellation = 16
MIMO_equalizer = 'ZF'
seed = 42

n_pilot = 8
N_sc = 128
N_t = 4
N_r = 4
freq = 2100e6 # Hz
Df = 15e3 # Hz

quantization_r = 6 # np.inf # number of decimal places.  np.inf means no quantization.

constellation = 'QAM'
M_constellation = 16
MIMO_equalizer = 'MMSE' # 'ZF', 'MMSE'  

epoch_count = 128
batch_size = 16

interference_power = -105 # dBm measured at the receiver
p_interference = 0.00 # probability of interference

shadowing_std = 8  # dB
K_factor = 2

crc_polynomial = 0b1001_0011
crc_length = 8 # bits

Tx_SNRs = [0, 5, 10, 15, 20, 25, 30][::-1]

prefer_gpu = True
##################

__release_date__ = '2024-07-29'
__ver__ = '0.50'

##################
plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = "14"

np_random = np.random.RandomState(seed=seed)


# https://github.com/farismismar/eesc7v86-fall22/
def _cplex_mse(y_true, y_pred):
    shape = np.prod(y_true.shape[1:])
    
    y_true = K.cast(y_true, tf.complex64)
    y_pred = K.cast(y_pred, tf.complex64) 
    
    error_vector_sq = tf.math.abs(y_true - y_pred) ** 2
        
    return K.cast(tf.math.sqrt(tf.math.scalar_mul(100 / shape, tf.math.reduce_sum(error_vector_sq))), tf.float64)
    

def create_constellation(constellation, M):
    if (constellation == 'PSK'):
        return _create_constellation_psk(M)
    elif (constellation == 'QAM'):
        return _create_constellation_qam(M)
    else:
        return None


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
    P_average = np.mean(np.abs(constellation.loc[:, 'x']) ** 2)
    constellation.loc[:, 'x'] /= np.sqrt(P_average)
    
    return constellation


def quantize(x, s):
    if s == np.inf:
        return x
   
    return np.round(x, s)


def decimal_to_gray(n, k):
    gray = n ^ (n >> 1)
    gray = bin(gray)[2:]
    
    return '{}'.format(gray).zfill(k)


def _plot_constellation(constellation):
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
    plt.show()
    tikzplotlib.save('constellation.tikz')
    plt.close(fig)
    

def _ber(a, b):
    assert(len(a) == len(b))
    
    length = len(a)
    bit_error = 0
    for idx in range(length):
        if a[idx] != b[idx]:
            bit_error += 1
            
    return bit_error / length


def _vec(A):
    # A is numpy array
    return A.flatten(order='F')

    
def bits_to_symbols(x_b, alphabet, k):    
    
    x_b_rev = x_b[::-1]
    
    df_ = alphabet[['m', 'I', 'Q']].copy()
    df_.loc[:, 'IQ'] = df_['I'].astype(str) + df_['Q'].astype(str)
    
    x_sym = []
    while len(x_b_rev) > 0:
        x_b_i = x_b_rev[:k][::-1].zfill(k)
        #print(x_b_rev, x_b_i)
        x_b_rev = x_b_rev[k:] #
        
        # Convert this to symbol from alphabet
        sym_i = df_.loc[df_['IQ'] == x_b_i, 'm'].values[0].astype(int)
        
        # print(x_b_i, sym_i)
        x_sym.append(sym_i)
        
    return np.array(x_sym[::-1])


def symbols_to_bits(x_sym, k, alphabet, is_complex=False):
    if is_complex == False: # Here, symbols are given by number, not by I/Q
        x_bits = ''
        for s in x_sym:
            try:
                i, q = alphabet.loc[alphabet['m'] == s, ['I', 'Q']].values[0]
            except Exception as e:
                print(e)
                # There is no corresponding I/Q, so these are padding, simply append with X
                i, q = 'X', 'X'
                pass
            x_bits += '{}{}'.format(i, q).zfill(k)
        return x_bits
    else:
        # Convert the symbols to number first, then call the function again
        information = []
        x_sym_IQ = x_sym
        # m, n = x_sym_IQ.shape
        x_sym_IQ = x_sym_IQ.flatten()
        for s in x_sym_IQ:
            try:
                information.append(alphabet[np.isclose(alphabet['x'], s)]['m'].values[0])
            except:
                information.append('X')
                pass
        information = np.array(information)
        return symbols_to_bits(information, k, alphabet, is_complex=False)


def bits_to_baseband(x_bits, alphabet, k):
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
    # Next is baseband which is the complex valued symbols
    for i, q in zip(x_b_i, x_b_q):
        sym = alphabet.loc[(alphabet['I'] == i) & (alphabet['Q'] == q), 'x'].values[0]
        x_sym.append(sym)
        
    x_sym = np.array(x_sym)
    
    return x_b_i, x_b_q, x_sym


def compute_crc(x_bits_orig, crc_length, crc_polynomial):
    # Introduce CRC to x
    
    # Make sure the crc polynomial is not longer than the codeword size.
    # Otherwise, an error
    length_crc = len(bin(crc_polynomial)[2:])
        
    x_bits = x_bits_orig.zfill(crc_length)

    crc = 0
    for position, value in enumerate(bin(crc_polynomial)[2:]):
        if value == '1':
            crc = crc ^ int(x_bits[position])
    crc = bin(crc)[2:]
    
    if len(crc) > crc_length:
        raise ValueError("Check CRC length parameter.")
    crc = crc.zfill(crc_length)
    
    return crc


def compute_large_scale_fading(d, f_c, G_t_dB=12, G_r_dB=0, pl_exp=2):
    global np_random
    
    l = c / f_c
    G = linear(G_t_dB) * linear(G_r_dB) * (l / (4 * pi * d)) ** pl_exp
        
    assert (G < 1)
    
    return G


# 3GPP CDL-C Model Parameters
def generate_cdl_c_channel(N_sc, N_r, N_t, subcarrier_spacing, center_frequency):
    global np_random
    # Parameters for CDL-C clusters from 3GPP TR 38.901 Table 7.7.1-3
    delays = np.array([0, 30e-9, 70e-9, 90e-9, 110e-9])  # Delays in seconds
    aoas = np.array([0, 30, 60, 90, 120])  # Angles of arrival in degrees
    zoas = np.array([0, 10, 20, 30, 40])  # Zenith angles of arrival in degrees
    aods = np.array([0, -30, -60, -90, -120])  # Angles of departure in degrees
    zods = np.array([0, -10, -20, -30, -40])  # Zenith angles of departure in degrees
    powers_dB = np.array([0, -3, -6, -9, -12])  # Power levels in dB

    # Calculate the frequency of each subcarrier
    subcarrier_freqs = np.arange(N_sc) * subcarrier_spacing + \
        center_frequency - (N_sc // 2) * subcarrier_spacing

    # Initialize the channel matrix
    H = np.zeros((N_sc, N_r, N_t), dtype=np.complex128)

    for delay, aoa, zoa, aod, zod, power_dB in zip(delays, aoas, zoas, aods, zods, powers_dB):
        # Calculate the phase shift for each cluster based on its angular parameters
        phase_shift = calculate_phase_shift(subcarrier_freqs, delay, aoa, zoa, aod, zod)

        # Generate random channel coefficients for this cluster
        H += np.sqrt(linear(power_dB)) * np_random.randn(N_sc, N_r, N_t) * phase_shift[:, np.newaxis, np.newaxis]

    return H


# 3GPP CDL-E Model Parameters
def generate_cdl_e_channel(N_sc, N_r, N_t, subcarrier_spacing, center_frequency):
    global np_random
    # Parameters for Cluster 0 (LOS, Specular) from 3gpp TR 38900 Table 7.7.1-5. CDL-E.
    delay = 0  # Shortest delay in sec
    aoa = -180  # Angle of arrival in degrees
    zoa = 80.4  # Zenith angle of arrival
    aod = 0  # Angle of departure
    zod = 99.6  # Zenith angle of departure
    power_dB = -0.03  # Highest received power in dBm

    # Calculate the frequency of each subcarrier
    subcarrier_freqs = np.arange(N_sc) * subcarrier_spacing + \
        center_frequency - (N_sc // 2) * subcarrier_spacing

    # Calculate the phase shift based on angular parameters
    phase_shift = calculate_phase_shift(subcarrier_freqs, delay, aoa, zoa, aod, zod)

    # Generate random channel coefficients with the specified power
    H = np.sqrt(linear(power_dB)) * np_random.randn(N_r, N_t) * phase_shift[:, np.newaxis, np.newaxis]

    return H


def calculate_phase_shift(subcarrier_freqs, delay, aoa, zoa, aod, zod):
    # Calculate the phase shift for each subcarrier based on angular parameters
    phase_shift = np.exp(-1j * 2 * np.pi * subcarrier_freqs * delay) * \
        np.exp(-1j * 2 * np.pi * np.sin(np.deg2rad(zoa)) * \
               np.sin(np.deg2rad(aoa)) * \
                   np.sin(np.deg2rad(zod - zoa)))
    return phase_shift


def compute_channel_eigenmodes(H):
    # Use the pilot symbol
    pilot_idx = 0
    v, e = np.linalg.eig(H[pilot_idx,:,:]@H[pilot_idx,:,:].conjugate().T)
    # U, S, Vh = np.linalg.svd(H[pilot_idx, :, :], full_matrices=False, hermitian=True)
    # eigenmodes = S ** 2
    eigenmodes = np.real(v)
    return eigenmodes


# This is not an ideal choice of a channel.
# Ideally you want more correlations.
def create_ricean_channel(N_r, N_t, K, N_sc=1, sigma_dB=8):
    global G # Pathloss in dB   
    
    G_fading = dB(G) - np_random.normal(loc=0, scale=np.sqrt(sigma_dB), size=(N_r, N_t))
    G_fading = np.array([linear(g) for g in G_fading])
    
    mu = np.sqrt(K / (1 + K))
    sigma = np.sqrt(1 / (1 + K))
    
    # Rician fading
    H = np_random.normal(loc=mu, scale=sigma, size=(N_r, N_t)) + \
        1j * np_random.normal(loc=mu, scale=sigma, size=(N_r, N_t))
    
    U, S, Vh = np.linalg.svd(H, full_matrices=False, hermitian=True)    

    # Repeat the channel across all subcarriers
    H = np.tile(H, N_sc).T.reshape(-1, N_r, N_t)
    
    # Normalize channel to unity gain and add large scale gain
    traces = np.trace(H, axis1=1, axis2=2)
    for idx in range(N_sc):
        H[idx, :, :] /= traces[idx]
        H[idx, :, :] *= np.sqrt(G_fading / 2) # element multiplication.
    
    # TODO:  if fading then add to H the coefficients.
    
    H = H.reshape(N_sc, N_r, N_t)
    return H, S ** 2


def create_rayleigh_channel(N_r, N_t, N_sc=1, sigma_dB=8):
    return create_ricean_channel(N_r, N_t, 0, N_sc, sigma_dB)


def _loss_fn_classifier(Y_true, Y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy()
    return cce(Y_true, Y_pred)


def _create_dnn(input_dimension, output_dimension, depth=5, width=10):
    nX = input_dimension
    nY = output_dimension
    
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dimension,)))
    
    for hidden in range(depth):
        model.add(layers.Dense(width, activation='sigmoid'))
   
    model.add(layers.Dense(nY, activation='softmax'))
    
    model.compile(loss=_loss_fn_classifier, optimizer='adam', 
                  metrics=['accuracy', 'categorical_crossentropy']) # Accuracy here is okay.
    
    # Reporting the number of parameters
    print(model.summary())
    
    num_params = model.count_params()
    print('Number of parameters: {}'.format(num_params))
    
    return model


def DNN_detect_symbol(X, y, train_split=0.8, depth=5, width=2, epoch_count=100, batch_size=32):
    global prefer_gpu
    global np_random
    
    use_cuda = len(tf.config.list_physical_devices('GPU')) > 0 and prefer_gpu
    device = "/gpu:0" if use_cuda else "/cpu:0"

    _, nX = X.shape
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, train_size=train_split,
                    random_state=np_random)
  
    le = LabelEncoder()
    le.fit(y_train)
    encoded_y = le.transform(y_train)     
    Y_train = keras.utils.to_categorical(encoded_y)
    encoded_y = le.transform(y_test)
    Y_test = keras.utils.to_categorical(encoded_y)
    
    _, nY = Y_train.shape

    dnn_classifier = _create_dnn(input_dimension=nX, output_dimension=nY,
                                 depth=depth, width=width)
    
    with tf.device(device):
        dnn_classifier.fit(X_train, Y_train, epochs=epoch_count, batch_size=batch_size)
        
    with tf.device(device):
        Y_pred = dnn_classifier.predict(X_test)
        loss, accuracy_score, _ = dnn_classifier.evaluate(X_test, Y_test)

    # Reverse the encoded categories
    y_test = le.inverse_transform(np.argmax(Y_test, axis=1))
    y_pred = le.inverse_transform(np.argmax(Y_pred, axis=1))
      
    return dnn_classifier, accuracy_score, np.c_[y_test, y_pred]


def _unsupervised_detection(x_sym_hat, alphabet):
    global np_random, M_constellation
    
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
    
    bits = []
    for s in range(x_sym_hat_flat.shape[0]):
        bits.append(f'{bits_i[s]}{bits_q[s]}')
        
    bits = np.array(bits).reshape(x_sym_hat.shape)

    return information, symbols, [bits_i, bits_q], bits


def ML_detect_symbol(symbols, alphabet):    
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
    
    bits = []
    for s in range(symbols_flat.shape[0]):
        bits.append(f'{bits_i[s]}{bits_q[s]}')
        
    bits = np.array(bits).reshape(symbols.shape)
    bits_i = bits_i.reshape(symbols.shape)
    bits_q = bits_q.reshape(symbols.shape)
    
    return information, symbols, [bits_i, bits_q], bits
    

def equalize_channel(H_hat, algorithm, rho=None):
    # rho is linear (non dB).  So is Rx_SNR.
    
    N_r, N_t = H_hat.shape
    
    # if algorithm == 'na':
    #     return 1, rho
    
    if algorithm == 'ZF':
        # Note that this pinv can cause problems if H is singular.  It will basically
        # enhance (or amplify) noise.
        # Use MMSE.        
        try:
            W = np.linalg.inv(H_hat)
        except:
            print("WARNING: Singular matrix.  Proceeding with pseudo-inverse may cause unrealistically large numbers.")
            W = np.linalg.pinv(H_hat)
        #Rx_SNR = rho / np.diag(np.real(np.linalg.inv(H_hat.conjugate().T@H_hat)))
    
    if algorithm == 'MMSE':
        assert(rho is not None)
        # W = H_hat.conjugate().T@np.linalg.inv(H_hat@H_hat.conjugate().T + (1./rho)*np.eye(N_r))
        W = np.linalg.inv(H_hat.conjugate().T@H_hat + (1./rho)*np.eye(N_t))@H_hat.conjugate().T
        
    WWH = np.real(W@W.conjugate().T)
    Rx_SNR = rho / np.diag(WWH)
    
    # assert(W.shape == (N_t, N_r))
    
    return W, Rx_SNR

        
def ls_estimate_channel(X_p, Y_p):
    # This is for least square (LS) estimation
    N_t, _ = X_p.shape
    
    if not np.allclose(X_p@X_p.T, np.eye(N_t)):
        raise ValueError("The training sequence is not semi-unitary.  Cannot estimate the channel.")
    
    # This is least square (LS) estimation
    H_hat = Y_p@X_p.conjugate().T
    
    return H_hat
  

def generate_pilot(N_t, n_pilot, random_state=None):
    # Check if the dimensions are valid for the operation
    if n_pilot < N_t:
        raise ValueError("The length of the training sequence should be greater than or equal to the number of transmit antennas.")
    
    # Compute a unitary matrix from a combinatoric of e
    I = np.eye(N_t)
    idx = random_state.choice(range(N_t), size=N_t, replace=False)
    Q = I[:, idx] 
    
    assert(np.allclose(Q@Q.T, np.eye(N_t)))  # Q is indeed unitary, but square.
    
    # # Scale the unitary matrix
    # Q /= np.linalg.norm(Q, ord='fro')
    
    # To make a semi-unitary, we need to post multiply with a rectangular matrix
    # Now we need a rectangular matrix (fat)
    A = np.zeros((N_t, n_pilot), int)
    np.fill_diagonal(A, 1)
    X_p =  Q @ A
    
    # The pilot power should be SNR / noise power
    # What matrix X_pX_p* = I (before scaling)
    assert(np.allclose(X_p@X_p.T, np.eye(N_t)))  # This is it
    
    # The training sequence is X_p.  It has N_t rows and n_pilot columns
    return X_p
    


def compute_overhead(b, codeword_length, k, N_sc, N_s, crc_length):
    effective_payload_length_symbols = int(np.ceil(b / k)) # symbols
    effective_crc_length_symbols = int(np.ceil(crc_length / k)) # The receiver is also aware of the CRC length (in bits)
    padding_symbols = 0

    max_iter = 1e6
    s_transmission = codeword_length // k * N_s # this is initialized to the maximum symbol length
    
    print('Optimizing for padding... ', end='')
    
    iteration = 0
    max_payload = s_transmission - effective_crc_length_symbols - padding_symbols
    for s_transmission in range(max_payload, 1, -1):
        for padding_symbols in range(max_payload - 1):
            iteration += 1
            current_configuration = effective_crc_length_symbols + s_transmission + padding_symbols
            
            n_transmissions = int(np.ceil(effective_payload_length_symbols / current_configuration))
            # print(current_configuration, padding_symbols, effective_crc_length_symbols, s_transmission, N_sc)
            
            if iteration > max_iter:
                raise ValueError("Max iteration reached.  Try to increase the codeword length.")
                
            if current_configuration <= N_sc and current_configuration % N_s == 0:
                assert (current_configuration % N_s == 0)
                print(' done.')
                
                return n_transmissions, current_configuration, s_transmission, padding_symbols, effective_crc_length_symbols 
    
    
    print(' NOT done.')
    return None    


def generate_precoder_combiner(H, rho):
    # Both precoder and combiner need both the channel state info (for parallel trans)
    # and the max power constraint i.e., the power per symbol ||Fx||F ** 2 is constant
    N_sc, N_r, N_t = H.shape
    
    N_s = min(N_r, N_t)
    
    # This is the matrix for parallel streams
    F = np.zeros((N_sc, N_t, N_s), dtype=np.complex128) # precoder
    Gcomb = np.zeros((N_sc, N_s, N_r), dtype=np.complex128) # combiner
    
    for subcarrier_idx in range(N_sc):
        # SVD
        U, S, Vh = np.linalg.svd(H[subcarrier_idx, :, :], full_matrices=True, hermitian=True)        
        # This is the power control matrix        
        # Waterfilling:
        try:
            D = waterfilling(S, rho)
            Dinv = np.linalg.inv(D)
        except Exception as e:
            print(e)
            D = np.eye(S.shape[0])
            Dinv = np.eye(S.shape[0])

        # Precoder
        F[subcarrier_idx, :, :] = Vh.conjugate().T@D

        # Combiner
        Gcomb[subcarrier_idx, :, :] = Dinv@U.conjugate().T
    
    return F, Gcomb, S ** 2
    
    
def waterfilling(Sigma, rho):
    S = np.diag(Sigma)
    
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


def transmit_receive(data, alphabet, channel, equalizer, snr_dB, crc_polynomial, crc_length, n_pilot, perfect_csi=False):
    global quantization_b
    global compress_channel, p_interference, interference_power
    global G, Df
    
    k = np.log2(alphabet.shape[0]).astype(int)
    rho = linear(snr_dB)
    
    SERs = []
    BERs = []    
    block_error = 0
        
    N_sc, N_t, N_r = channel.shape
    
    tti = 1e-3 # in seconds
        
    channel_coherence_time = np.inf
    
    # Number of streams
    N_s = min(N_r, N_t)
        
    # Bit rate
    max_codeword_size = N_sc * k
    
    B = Df # per OFDM resource element
    print(f'Transmission BW per stream = {B:.2f} Hz')
    print(f'Number of OFDM subcarriers per stream = {N_sc}')

    data_orig = copy.deepcopy(data)
    
    b = len(data) # this is in bits

    # The dimension of the pilots is N_t x n_pilot
    P = generate_pilot(N_t, n_pilot, random_state=np_random)
    
    # Power (actually energy) per pilot
    E_P = np.linalg.norm(P, axis=1, ord=2) ** 2
    
    # This is the true channel before any diag
    H_true = copy.deepcopy(channel)
        
    H_0 = copy.deepcopy(H_true)
    H = copy.deepcopy(H_true)    
    F, Gcomb, _ = generate_precoder_combiner(H=H_0, rho=rho)
        
    # Debug purposes
    if perfect_csi:
        F = np.tile(np.eye(N_t), N_sc).T.reshape(N_sc, N_t, N_t)
        Gcomb = np.tile(np.eye(N_t), N_sc).T.reshape(N_sc, N_r, N_r)
    
    codeword_size = max_codeword_size #// 2 # actually should come from the CQI/rho: CQI suggests TBS.
    max_bit_rate_per_stream = codeword_size / tti
    print(f'Transmission maximum bitrate per stream = {max_bit_rate_per_stream:.2f} bps')
    
    n_transmissions, total_frame_length_symbols, s_transmission, padding_symbols, effective_crc_length_symbols = compute_overhead(b, codeword_size, k, N_sc, N_s, crc_length)
    
    # Number of bits per transmission
    bits_per_transmission = b // n_transmissions
    
    ## A transmission block size must be integer.    
    #assert (b % n_transmissions == 0)
    
    # Total data is b bits or b / k symbols.
    assert(b <= k * total_frame_length_symbols * n_transmissions * N_s)

    ############################################################################
    # Now estimate the channel using the pilots and use H_hat.
    H_hat = np.zeros_like(H, dtype=np.complex128)
    for subcarrier_idx in range(N_sc):
        # Additive noise
        N = 1./ np.sqrt(2) * (np_random.normal(0, 1, size=(N_r, n_pilot)) + \
                              1j * np_random.normal(0, 1, size=(N_r, n_pilot)))

        E_noise = np.linalg.norm(N, axis=1, ord=2) ** 2

        # Apply some changes to the noise statistic
        for idx in range(N.shape[0]):
            N[idx, :] /= np.sqrt(E_noise[idx] / 2.) # normalize noise
            N[idx, :] *= np.sqrt(E_P[idx] / rho / 2.) # noise power = signal / SNR

        E_noise_after = np.linalg.norm(N, axis=1, ord=2) ** 2
        # Note E_P / E_noise_after is equal to rho, which is the transmit SNR.

        # Estimate the channel due to the pilot
        # The `reference symbol' is the first n_pilot OFDM symbols being transmitted.
        H_sc = H[subcarrier_idx, :, :n_pilot] # N_sc x N_r x N_t
        
        HP = H_sc@P
        
        T = HP # The noise here somehow is creating a problem.  I will assume perfect estimation hence use H moving forward.

        H_hat_sc = ls_estimate_channel(P, T)
        H_hat[subcarrier_idx, :, :n_pilot] = H_hat_sc
    
    channel_estimation_error = _cplex_mse(_vec(H), _vec(H_hat)).numpy() * 100.
    print()
    print(f'Channel estimation error: {channel_estimation_error:.4f}')
    print()
    
    assert(channel_estimation_error < 0.01)
    ############################################################################
        
    SINR_Rx = []
    Tx_EbN0 = []
    Rx_EbN0 = []
    PL = []
    power_rx = []
    data_rx = []
    
   # noise_powers = []

    if b < 10000:
        print('Warning: Small number of bits can cause curves to differ from theoretical ones due to insufficient number of samples.')

    if n_transmissions < 1000:
        print('Warning: Small number of codeword transmissions can cause BLER values to differ from theoretical ones due to insufficient number of samples.')

    H_reconstructed = copy.deepcopy(H)

    eig = compute_channel_eigenmodes(H)
    eig_rec = compute_channel_eigenmodes(H_reconstructed)
    
    # Debug purposes
    if perfect_csi:
        H_reconstructed = np.eye(N_t, dtype=np.complex128)
        H_reconstructed = np.tile(H_reconstructed, N_sc).T.reshape(N_sc, N_r, N_t)

        H = H_reconstructed
        
        n = np.zeros((N_sc, N_r), dtype=np.complex128)
        interf = np.zeros((N_sc, N_r), dtype=np.complex128)
        p_interference = 0
        
        E_noise = 0
        eig = np.ones(N_t)
    
    print('Channel reconstruction error: {}'.format((np.linalg.norm(_vec(H) - _vec(H_reconstructed), ord=2) ** 2)))
    
    # check H_reconstructed if it is identity?  If so, then the autoencoder is not doing well.
    H_reconstructed_0 = copy.deepcopy(H_reconstructed)
    ####################################
    # These are diagnolized representations.
    GHF_0 = np.zeros((N_sc, N_t, N_s), dtype=np.complex128)    
    GHF_reconstructed_0 = np.zeros((N_sc, N_t, N_s), dtype=np.complex128)
    GHF_true = np.zeros((N_sc, N_t, N_s), dtype=np.complex128)
    
    for idx in range(N_sc):        
        GHF_0[idx, :, :] = Gcomb[idx, :, :]@H_0[idx, :, :]@F[idx, :, :]
        GHF_reconstructed_0[idx, :, :] = Gcomb[idx, :, :]@H_reconstructed_0[idx, :, :]@F[idx, :, :]        
        GHF_true[idx, :, :] = Gcomb[idx, :, :]@H_true[idx, :, :]@F[idx, :, :]
        
    
    # See original channel.
    plot_channel(channel=H, filename='original')
    ####################################
        
    print(f'Transmitting a total of {b} bits.')
    
    total_transmitted_bits = 0
    for tx_idx in np.arange(n_transmissions):
        # reset the channels
        H = copy.deepcopy(H_0)
        H_reconstructed = copy.deepcopy(H_reconstructed_0)
        GHF_reconstructed = copy.deepcopy(GHF_reconstructed_0)
        
        # Only use H_reconstructed moving forward... not H.
        
        print(f'Transmitting codeword {tx_idx+1}/{n_transmissions}')
        # Every transmission is for one codeword, divided up to N_s streams.
                
        # CRC is not applied per codeword.  Only one CRC regardless
        # of transmission rank.  Validated this with Nokia's PM counters.
        
        # 1) Compute CRC based on the original codeword
        # 2) Pad what is left *in between* to fulfill MIMO rank
        
        x_bits_orig = data[tx_idx*bits_per_transmission:(tx_idx+1)*bits_per_transmission]
        
        # Note this is the function causing the diagonlized distortion.
        # If you have 10 bits of data and you are using QAM
        # 1001001011
        # 10 01 00 10 11 (4QAM)
        # 1001 0010 0011 (16QAM)  WRONG because this causes two new bits
        # 0010 0100 1011 (CORRECT)
        # Therefore, the approach is you first reverse the bits
        # Then you bring symbols
        x_info = bits_to_symbols(x_bits_orig, alphabet, k) # Correct
        
        crc = compute_crc(x_bits_orig, crc_length, crc_polynomial)  # Compute CRC in bits.
        
        crc_padded = crc.zfill(effective_crc_length_symbols * k)
        _, _, crc_symbols = bits_to_baseband(crc_padded, alphabet, k)
        
        # Symbols
        x_b_i, x_b_q, x_sym = bits_to_baseband(x_bits_orig, alphabet, k)
        
        # Extra padding is necessary
        # Note this padding cannot be removed (think residual transmission)
        # where full buffer is not possible and extra padding is needed.
        padding_i = (len(x_sym) + len(crc_symbols)) % N_s
        
        payload_sym_len = len(x_sym) + len(crc_symbols) + padding_i        
        if payload_sym_len % N_s != 0:
            padding_i += (payload_sym_len // N_s + 1) * N_s - payload_sym_len
            
        x_sym_crc = np.r_[x_sym, np.zeros(padding_i), crc_symbols]
        
        #x_sym_crc = np.r_[x_sym, np.zeros(padding_symbols), crc_symbols]
        total_transmitted_bits += len(x_sym_crc) * k
        
        # Signal energy
        E_x = np.linalg.norm(x_sym_crc, ord=2) ** 2

        # The next two lines achieve normalization but the colors will be diluted.  Commenting them out is not a problem.
        # x_sym_crc /= np.sqrt(E_x)
        # E_x = np.linalg.norm(x_sym_crc, ord=2) ** 2

        # Subcarrier power (OFDM resource element)
        P_sym = E_x * Df / (N_sc * N_t)
        P_sym_dB = dB(P_sym)

        # Map codewords to the MIMO layers.
        # x_sym_crc has dimensions of N_sc times N_s
        # x_sym_crc = x_sym_crc.reshape(-1, N_s) # this is WRONG
        x_sym_crc = x_sym_crc.reshape(N_s, -1).T # this is correct.
        # NOTE x_sym_crc.reshape(N_s,-1).T is not equal to x_sym_crc.reshape(-1,N_s)
        
        # Number of required symbols is N_sc_payload
        # Number of available symbols is N_sc
        N_sc_payload = x_sym_crc.shape[0]
        if (N_sc_payload > N_sc):
            raise ValueError(f"Number of required subcarriers {N_sc_payload} exceeds {N_sc}.  Either increase N_sc or N_s or k.")
    
        # TODO: Try np.tensordot as an alternative for for loop multiplications

        # Gross code rate
        R = len(x_bits_orig) / (N_s * N_sc_payload * k)
        print(f'Code rate (at the transmitter) = {R:.4f}')
                
        Tx_EbN0_ = snr_dB - dB(k)
        Tx_EbN0.append(Tx_EbN0_)
        
        # print('Average symbol power at the transmitter is: {:.4f} dBm'.format(P_sym_dB.mean()))
        print(f'SNR at the transmitter (per stream): {snr_dB:.4f} dB')
        print(f'EbN0 at the transmitter (per stream): {Tx_EbN0_:.4f} dB')
        
        if Tx_EbN0_ < -1.59:
            print()
            print('** Outage at the transmitter **')
            print()
        
        ###################
        # Channel impact (precoded)
        np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
        n_eig = len(eig)
        print(f'Transmitter computed {n_eig} eigenmodes for channel H at precoding: {eig}.')        
        print(f'The reconstructed channel has eigenmodes: {eig_rec}')
        
        if (len(eig[eig < 1]) > 0):
            print('WARNING: Low eigenmode values detected in channel H.')
        
        # Now precoding.
        for idx in range(N_sc):
            H[idx, :, :] = H[idx, :, :]@F[idx, :, :]
            H_reconstructed[idx, :, :] = H_reconstructed[idx, :, :]@F[idx, :, :]

        # Assume channel is constant across all subcarriers
        # Since the channel coherence time is assumed constant
        if (tx_idx * tti > channel_coherence_time):
            H = H # get a new channel
            H_reconstructed = H_reconstructed
        else:
            H = H # this line has no meaning except to remind us that the channel changes after coherence time.
            H_reconstructed = H_reconstructed
            
        # and only make sure we do not exceed the number of OFDM symbols available to us.
        # since every transmission is one TTI = 1 ms.
        
        # Channel (precoded)
        Hx = np.zeros((N_sc_payload, N_r), dtype=np.complex128) # N_sc x Nr
        for idx in range(N_sc_payload):
            Hx[idx, :] = H_reconstructed[idx, :, :]@x_sym_crc[idx, :] # Impact of the channel on the transmitted symbols
        
        # Power (actually energy) per OFDM symbol
        E_Hx = np.linalg.norm(Hx, axis=1, ord=2) ** 2
        
        # Interference
        interf = 1./ np.sqrt(2) * (np_random.normal(0, 1, size=(N_sc_payload, N_r)) + 
                                   1j * np_random.normal(0, 1, size=(N_sc_payload, N_r)))
        
        E_interf = np.linalg.norm(interf, axis=1, ord=2) ** 2
        
        mask = np_random.binomial(n=1, p=p_interference, size=N_sc_payload)
        
        # Apply some changes to interference
        for idx in range(N_sc_payload):
            interf[idx, :] /= np.sqrt(E_interf[idx]) # normalize interference
            interf[idx, :] *= np.sqrt(linear(interference_power) * 1e-3 / Df) # introduce power in W
            # Finally, with probability p, enable interference on N_sc_payload.
            interf[idx, :] *= mask[idx]
        
        E_interf = np.linalg.norm(interf, axis=1, ord=2) ** 2
        
        # Additive noise
        if not perfect_csi:
            n = 1./ np.sqrt(2) * (np_random.normal(0, 1, size=(N_sc_payload, N_r)) + \
                                  1j * np_random.normal(0, 1, size=(N_sc_payload, N_r)))
            
            E_noise = np.linalg.norm(n, axis=1, ord=2) ** 2
            
            # Apply some changes to the noise statistic
            for idx in range(N_sc_payload):
                n[idx, :] /= np.sqrt(E_noise[idx] / 2.) # normalize noise
                n[idx, :] *= np.sqrt(E_Hx[idx] / rho / 2.) # noise power = signal / SNR

            E_noise = np.linalg.norm(n, axis=1, ord=2) ** 2

        # Keep track of transmit noise
        transmit_noise_power_dB = dB(E_noise.mean()* Df)
        
        # Good check here:
        assert(np.isclose(snr_dB, dB(E_Hx / E_noise)).all())
                
        Y = np.zeros_like(n, dtype=np.complex128)
        for idx in range(N_sc_payload):
            Y[idx, :] = Hx[idx, :] + n[idx, :] + interf[idx, :]
        
        # Apply combiner now.
        # This effectively brings us now to a model y = Sigma x + tilde n
        for idx in range(N_sc_payload):
            Y[idx, :] = Gcomb[idx, :, :]@Y[idx, :]
        #    Y[idx, :] = GcombRec[idx, :, :]@Y[idx, :]
        
        # At the receiver, use the reconstructed channel:
        # H_reconstructed from here onwards.
        
        # For future:  If channel MSE is greater than certain value, then CSI
        # knowledge scenarios kick in (i.e., CSI is no longer known perfectly to
        # both the transmitter/receiver).
        
        # MIMO-OFDM Receiver
        # Now use the estimated channel H_reconstructed
        # and apply the equalizer to remove channel effect
        # Channel equalization
        # Now the received SNR per antenna (per symbol) due to the receiver
        # SNR per antenna and SNR per symbol are the same thing technically.
        
        # Equalization
        W = np.zeros_like(GHF_reconstructed, dtype=np.complex128)
        for idx in range(N_sc_payload):
            W[idx, :, :], _ = equalize_channel(GHF_reconstructed[idx, :, :], algorithm=equalizer, rho=rho)

        # This is x_hat technically.
        
        received_symbols = np.zeros_like(Y, dtype=np.complex128)
        for idx in range(N_sc_payload):
            received_symbols[idx, :] = W[idx,:, :]@GHF_reconstructed[idx,:, :]@x_sym_crc[idx, :] # Gcomb[idx, :, :]@Hx[idx, :] 
        
        P_sym_rx = np.linalg.norm(received_symbols[:N_sc_payload, :].mean(axis=0), ord=2) ** 2
        P_sym_rx_dB = dB(P_sym_rx)
        # print(f'Average reference symbols power at the receiver after equalization is: {P_sym_rx_dB:.4f} dBm')
        if (P_sym_rx_dB <= -120 or P_sym_rx_dB >= 46):
            print(f"WARNING: Unrealistic receive power levels {P_sym_rx_dB:.2f} dBm.  Check channel data.")

        # Compute the average path loss, which is basically the channel effect
        path_loss = np.mean(P_sym_dB - P_sym_rx_dB)
        print(f'Average reference symbols path loss at the receiver after equalization is: {path_loss:.4f} dB')
        
        power_rx.append(P_sym_rx_dB)
        PL.append(path_loss)
        
        # Thus z = Wy = W(Hx + v)
        #        = WHx + Wv
        #        = x_hat + Wv 
        z = np.zeros_like(Y, dtype=np.complex128)
        for idx in range(N_sc_payload):
            z[idx, :] = W[idx, :, :]@Y[idx, :]
        
        x_sym_crc_hat = z[:N_sc_payload]
        
        # the symbols are recovered properly        
        assert not perfect_csi or (x_sym_crc_hat == x_sym_crc).all()
        
        # Now how to extract x_hat from z?        
        # Detection to extract the signal from the signal plus noise mix.
        x_sym_hat = _vec(x_sym_crc_hat)
        
        if len(crc_symbols) > 0:
            x_sym_hat = x_sym_hat[:-len(crc_symbols)] # payload including padding
        
        # Remove the padding symbols here
        if padding_i > 0:
            x_sym_hat = x_sym_hat[:-padding_i] # no CRC and no padding.
        
        # Only useful data
        x_sym_hat = x_sym_hat[:len(x_sym)]
        
        # At this point, payload length is equal
        assert (len(x_sym_hat) == len(x_sym))
        
        if len(x_sym_hat) == 0:
            print("WARNING: Empty payload.")
            SINR_Rx.append(np.nan)
            Rx_EbN0.append(np.nan)
            SERs.append(None)
            BERs.append(None)
            continue

        # Now let us find the SNR and Eb/N0 *after* equalization
        
        # Average received SNR
        # received_noise_plus_interference = z - received_symbols
        # received_noise_plus_interference_power = np.linalg.norm(received_noise_plus_interference[:N_sc_payload, :].mean(axis=0), ord=2) ** 2        
        
        # Pnoise_and_intf = Pz - Pchannel x
        received_noise_plus_interference_power = (np.linalg.norm(z[:N_sc_payload, :].mean(axis=0), ord=2) ** 2).round(2) - P_sym_rx.round(2)
        received_noise_plus_interference_power_dB = dB(received_noise_plus_interference_power)
        
        print(f'Average received symbol power at the receiver after equalization is: {P_sym_rx_dB:.2f} dBm')
        print(f'Average noise plus interference power at the receiver: {received_noise_plus_interference_power_dB:.2f} dBm')
            
        if (received_noise_plus_interference_power_dB > 35):
            print(f'CRITICAL: Error at SNR {snr_dB} dB.  Noise exceeding 35 dBm is unrealistic and is ignored.')
            
        Rx_SINR_eq = dB(P_sym_rx / received_noise_plus_interference_power)
       
        SINR_Rx.append(Rx_SINR_eq)

        # Detection of symbols (symbol star is the centroid of the constellation)
        x_info_hat, _, _, x_bits_hat = ML_detect_symbol(x_sym_hat, alphabet)
        # x_info_hat, _, _, x_bits_hat = _unsupervised_detection(x_sym_hat, alphabet)
        x_bits_hat = ''.join(x_bits_hat)
        
        ## Remove any potential leading zeros
        x_bits_orig = x_bits_orig[-bits_per_transmission:]
        x_bits_hat = x_bits_hat[-bits_per_transmission:]
                
        # # To test the performance of DNN in detecting symbols:
        # X = np.c_[np.real(x_sym_hat), np.imag(x_sym_hat)]
        # y = x_info_hat
        # model_dnn, dnn_accuracy_score, _ = DNN_detect_symbol(X=X, y=y)
                
        # Compute CRC on the received frame
        crc_comp = compute_crc(x_bits_hat, crc_length, crc_polynomial)
        ########################################################
        # Error statistics
        # Block error
        
        # If CRC1 xor CRC2 is not zero, then error.
        if int(crc, 2) ^ int(crc_comp, 2) != 0:
            block_error += 1
    
        symbol_error = 1 - np.mean(x_info_hat == x_info)
        
        SERs.append(symbol_error)
        
        x_hat_b_i, x_hat_b_q, _ = bits_to_baseband(x_bits_hat, alphabet, k)
        
        ber_i = _ber(x_hat_b_i, x_b_i) 
        ber_q = _ber(x_hat_b_q, x_b_q)
                
        data_rx.append(x_bits_hat)
            
        ber = np.mean([ber_i, ber_q])
        BERs.append(ber)
        
        # Code rate
        R = (1 - ber) * len(x_bits_orig) / (N_s * N_sc_payload * k) # 1 - overhead / codeword_size
        print(f'Code rate (at the receiver) = {R:.4f}')
        
        # Compute the average EbN0 at the receiver
        C = k * R
        Rx_EbN0_ = Rx_SINR_eq - 10*np.log10(C)
        Rx_EbN0.append(Rx_EbN0_)

        # print(f'Average reference symbols SNR at the receiver before equalization: {Rx_SNR_:.4f} dB')
        print(f'Average reference symbols SINR at the receiver after equalization: {Rx_SINR_eq:.4f} dB')
        print('EbN0 at the receiver: {:.4f} dB'.format(Rx_EbN0_))
                
        if Rx_EbN0_ < -1.59:
            print()
            print('** An outage detected at the receiver **')                
            print()
    # for

    overhead = total_transmitted_bits - b
    print(f"Total transmitted bits: {total_transmitted_bits} bits.")    
    print(f"Overhead bits: {overhead} ({overhead / b * 100:.2f}%).")
    
    BLER = block_error / n_transmissions
    # print(f'For this transmission, the BLER was {BLER:.3f}')
    print()
    print()
    # Now extract from every transmission
    data_rx_ = ''.join(data_rx)
    
    assert len(data_rx_) == b
    
    return np.arange(n_transmissions), P_sym_dB.mean(), power_rx, PL, SERs, SINR_Rx, BERs, BLER, Tx_EbN0, Rx_EbN0, channel_estimation_error, data_rx_

    
def dB(x):
    return 10 * np.log10(x)


def linear(x):
    return 10 ** (x / 10.)


def read_bitmap(file, word_length=8):
    global payload_size
    
    if file is not None:
        # This is a 32x32 pixel image
        im_orig = plt.imread(file)
        
        im = im_orig.flatten()
        im = [bin(a)[2:].zfill(word_length) for a in im] # all fields when converted to binary need to have word length.
    
        im = ''.join(im) # This is now a string of bits

    else:
        # These lines are for random bits 
        im  = np_random.binomial(1, 0.5, size=payload_size)
        s = ''
        for a in im:
            s = s + str(a)
        im = s
    
    return im


def _convert_to_bytes_decimal(data, word_length=8):
    n = len(data) // word_length
    dim = int(np.sqrt(n / 3)) # this is an RGB channel.
    
    data_vector = []
    for i in range(n):
        d = str(data[i*word_length:(i+1)*word_length])
        d = int(d, 2)
        # print(d)
        data_vector.append(d)
    
    data_vector = np.array(data_vector, dtype='uint8')
    # Truncate if needed
    data_vector = data_vector[:dim * dim * 3]

    # Now reshape
    data_vector = data_vector.reshape(dim, dim, 3)
    
    return data_vector


def _plot_8bit_bitmaps(data1, data2):
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)
    
    ax1.imshow(_convert_to_bytes_decimal(data1))
    ax2.imshow(_convert_to_bytes_decimal(data2))
    
    ax1.axis('off')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    
    
def plot_learning(history):    
    fig, ax = plt.subplots(figsize=(9,6))

    plt.plot(history.history['val_loss'], 'r', lw=2, label='Validation loss')
    plt.plot(history.history['loss'], 'b', lw=2, label='Loss')

    plt.grid(which='both', axis='both')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    tikzplotlib.save('history_dnn.tikz')
    plt.show()
    plt.close(fig) 
    
    
def generate_plot(df, xlabel, ylabel, semilogy=True):
    cols = list(set([xlabel, ylabel, 'Tx_SNR']))
    df = df[cols]
    df_plot = df.groupby('Tx_SNR').mean().reset_index()

    fig, ax = plt.subplots(figsize=(9,6))
    if semilogy:
        ax.set_yscale('log')
    ax.tick_params(axis=u'both', which=u'both')
    plt.plot(df_plot[xlabel].values, df_plot[ylabel].values, '--bo', alpha=0.7, 
             markeredgecolor='k', markerfacecolor='r', markersize=6)

    plt.grid(which='both', axis='both')
    plt.xticks(Tx_SNRs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    tikzplotlib.save(f'output_{ylabel}_vs_{xlabel}.tikz')
    plt.show()
    plt.close(fig)
    

def plot_channel(channel, filename):
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
    plt.savefig(f'channel_{filename}.pdf', format='pdf')
    plt.show()
    plt.close(fig)


def run_simulation(file_name, channel_matrix, equalizer, constellation, Tx_SNRs, crc_polynomial, crc_length, n_pilot):
    global M_constellation
    
    alphabet = create_constellation(constellation=constellation, M=M_constellation)
    _plot_constellation(constellation=alphabet)
    data = read_bitmap(file_name)

    _plot_8bit_bitmaps(data, data)
    df_output = pd.DataFrame()
    for snr in Tx_SNRs:
        c_i, P_x_Tx_i, P_x_Rx_i, PL_i, SER_i, Rx_SINR_i, BER_i, BLER_i, Tx_EbN0_i, Rx_EbN0_i, channel_mse_i, data_received = \
            transmit_receive(data, alphabet, channel_matrix, equalizer, 
                         snr, crc_polynomial, crc_length, n_pilot, perfect_csi=False)
        
        _plot_8bit_bitmaps(data, data_received)
        
        count = 0
        for x, y in zip(data, data_received):
            if x != y:
                count += 1
        
        print(f'Image bit errors: {count} ({100.*count/len(data):.4f}%).')
        
        df_output_ = pd.DataFrame(data={'Codeword': c_i})
        df_output_['P_x_Tx'] = P_x_Tx_i
        df_output_['P_x_Rx'] = P_x_Rx_i
        df_output_['PL'] = PL_i        
        df_output_['Tx_SNR'] = snr
        df_output_['Rx_SINR'] = Rx_SINR_i
        df_output_['SER'] = SER_i
        df_output_['Avg_BER'] = BER_i
        df_output_['BLER'] = BLER_i
        df_output_['Tx_EbN0'] = Tx_EbN0_i
        df_output_['Rx_EbN0'] = Rx_EbN0_i
        df_output_['Channel_MSE'] = channel_mse_i

        if df_output.shape[0] == 0:
            df_output = df_output_.copy()
        else:
            df_output = pd.concat([df_output_, df_output], axis=0, ignore_index=True)
        
    df_output = df_output.reset_index(drop=True)
    
    return df_output


def create_deepMIMO_channel(UE_idx=1, freq=3.5, bw=15, N_sc=64, N_r=1, N_t=32, normalize=False):
    global max_UE_count
    global n_pilot
    
    import DeepMIMO

    # Load the default parameters
    parameters = DeepMIMO.default_params()

    # Set scenario name
    parameters['scenario'] = 'O1_3p5'

    # Set the main folder containing extracted scenarios
    parameters['dataset_folder'] = r'C:\\Users\\farismismar\\Desktop\\code\\scenarios'

    parameters['num_paths'] = 2

    # User rows 1-100
    parameters['user_row_first'] = 1
    parameters['user_row_last'] = max_UE_count

    # Activate only the first basestation
    parameters['active_BS'] = np.array([1]) 

    parameters['OFDM']['bandwidth'] = bw / 1e3 # bw parameter expected is in GHz
    parameters['OFDM']['subcarriers'] = N_sc # OFDM with 512 subcarriers
    parameters['OFDM']['subcarriers_limit'] = n_pilot # 4 symbols for pilot

    parameters['ue_antenna']['shape'] = np.array([1, N_r, 1]) # UE antennas
    parameters['bs_antenna']['shape'] = np.array([1, N_t, 1]) # ULA of N_t elements

    # Generate data
    dataset = DeepMIMO.generate_data(parameters)

    bs_idx = 0
    # UE_idx = 0

    # # Shape of BS 0 - UE 0 channel
    # print(dataset[bs_idx]['user']['channel'][UE_idx].shape)

    # loc_x = dataset[bs_idx]['user']['location'][:, 0]
    # loc_y = dataset[bs_idx]['user']['location'][:, 1]
    # loc_z = dataset[bs_idx]['user']['location'][:, 2]
    # pathloss = dataset[bs_idx]['user']['pathloss']
    
    H = dataset[bs_idx]['user']['channel'][UE_idx] # shape is Nt, Nr, Nsc.
    
    # Reshape channel
    H = H.reshape(N_sc, N_t, N_r)
        
    if normalize:
        # Normalize channel so the trace is equal to N*t * N_r
        traces = np.trace(H, axis1=1, axis2=2)    
        for idx in range(N_sc):
            H[idx, :, :] /= traces[idx] / np.sqrt(N_t*N_r)
       
    return H

# 1) Create a channel
# G = compute_large_scale_fading(d=65, f_c=freq)
# channel_H, eig_H = create_ricean_channel(N_r=N_r, N_t=N_t, N_sc=N_sc, K=K_factor, sigma_dB=shadowing_std) # Performance plots look awesome 

# channel_H = generate_cdl_e_channel(N_sc=N_sc, N_t=N_t, N_r=N_r, subcarrier_spacing=Df, center_frequency=freq)
channel_H = generate_cdl_c_channel(N_sc=N_sc, N_t=N_t, N_r=N_r, subcarrier_spacing=Df, center_frequency=freq)
# channel_H = create_deepMIMO_channel(UE_idx=1, freq=freq, N_sc=N_sc, N_r=N_r, N_t=N_t, normalize=False)

# 2) Run the simulation on this channel
start_time = time.time()
df_output = run_simulation(file_name, channel_H, MIMO_equalizer, constellation, 
                           Tx_SNRs, crc_polynomial, crc_length, n_pilot)
end_time = time.time()

print('Simulation time: {:.2f} mins'.format((end_time - start_time) / 60.))
df_output.to_csv('output.csv', index=False)

# 3) Generate plot
xlabel = 'Tx_SNR' #'Tx_SNR' # rho values in dB
ylabel = 'Avg_BER'

generate_plot(df=df_output, xlabel=xlabel, ylabel=ylabel)

xlabel = 'Tx_SNR' #'Tx_SNR' # rho values in dB
ylabel = 'BLER'

generate_plot(df=df_output, xlabel=xlabel, ylabel=ylabel)