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

import pdb

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
from tensorflow.keras import layers, optimizers
from tensorflow.compat.v1 import set_random_seed

from sklearn.preprocessing import MinMaxScaler

# System parameters
file_name = 'faris.bmp' # either a file name or a payload
payload_size = 0 # 30000 # bits
constellation = 'QAM'
M_constellation = 64
MIMO_equalizer = 'ZF'
seed = 7
n_pilot = 4
N_sc = 64
N_r = 4
N_t = 4
freq = 3.5 # GHz
quantization_b = np.inf
shadowing_std = 8  # dB
K_factor = 4

crc_polynomial = 0b1001_0011
crc_length = 24 # bits

Tx_SNRs = [-3,0,3,10,15,20,25,30,35] # in dB

prefer_gpu = True
##################

__release_date__ = '2024-03-16'
__ver__ = '0.50'

##################
plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = "14"

np_random = np.random.RandomState(seed=seed)
k_constellation = int(np.log2(M_constellation))


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


def quantize(x, b):
    if b == np.inf:
        return x
        
    m, n = x.shape
    x = x.flatten()
    
    x_re = np.real(x)
    x_im = np.imag(x)

    x_re_b = _lloyd_max_quantization(x_re, b)
    x_im_b = _lloyd_max_quantization(x_im, b)
    
    x_b = x_re_b + 1j * x_im_b
    
    return x_b.reshape((m, n))


def _lloyd_max_quantization(x, b, max_iteration=100):
    # derives the quantized vector
    # https://gist.github.com/PrieureDeSion
    # https://github.com/stillame96/lloyd-max-quantizer
    from utils import normal_dist, expected_normal_dist, MSE_loss, LloydMaxQuantizer
    
    repre = LloydMaxQuantizer.start_repre(x, b)
    min_loss = 1.0

    for i in range(max_iteration):
        thre = LloydMaxQuantizer.threshold(repre)
        # In case wanting to use with another mean or variance,
        # need to change mean and variance in utils.py file
        repre = LloydMaxQuantizer.represent(thre, expected_normal_dist, normal_dist)
        x_hat_q = LloydMaxQuantizer.quant(x, thre, repre)
        loss = MSE_loss(x, x_hat_q)

        # # Print every 10 loops
        # if(i%10 == 0 and i != 0):
        #     print('iteration: ' + str(i))
        #     print('thre: ' + str(thre))
        #     print('repre: ' + str(repre))
        #     print('loss: ' + str(loss))
        #     print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        # Keep the threhold and representation that has the lowest MSE loss.
        if(min_loss > loss):
            min_loss = loss
            min_thre = thre
            min_repre = repre

    # print('min loss: ' + str(min_loss))
    # print('min thresholds: ' + str(min_thre))
    # print('min representative levels: ' + str(min_repre))
    
    # x_hat_q with the lowest amount of loss.
    best_x_hat_q = LloydMaxQuantizer.quant(x, min_thre, min_repre)
    
    return best_x_hat_q


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
    iter_ct = int(np.ceil(len(x_b) / k))
    
    df_ = alphabet[['m', 'I', 'Q']].copy()
    df_.loc[:, 'IQ'] = df_['I'].astype(str) + df_['Q'].astype(str)
    
    x_sym = []
    for i in range(iter_ct):
        bits_i = x_b[i*k:(i+1)*k] # read the next ith stride of k bits
        # Convert this to symbol from alphabet
        sym_i = df_.loc[df_['IQ'] == bits_i.zfill(k), 'm'].values[0].astype(int)
        # print(bits_i, sym_i)
        x_sym.append(sym_i)
        
    return np.array(x_sym)


def symbols_to_bits(x_sym, k, alphabet, is_complex=False):    
    if is_complex == False: # Here, symbols are given by number, not by I/Q
        x_bits = ''
        for s in x_sym:
            try:
                i, q = alphabet.loc[alphabet['m'] == s, ['I', 'Q']].values[0]
            except:
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
    x_b_i = []
    x_b_q = []
    for idx in range(len(x_bits) // k):
        codeword = x_bits[idx*k:(idx+1)*k]    
        x_b_i.append(codeword[:(k//2)])
        x_b_q.append(codeword[(k//2):])

    x_sym = []
    # Next is baseband which is the complex valued symbols
    for i, q in zip(x_b_i, x_b_q):
        sym = alphabet.loc[(alphabet['I'] == i) & (alphabet['Q'] == q), 'x'].values[0]
        x_sym.append(sym)
        
    x_sym = np.array(x_sym)
    
    return x_b_i, x_b_q, x_sym


def compute_crc(x_bits_orig, codeword_size, crc_polynomial, crc_length):
    # Introduce CRC to x
    
    # Make sure the crc polynomial is not longer than the codeword size.
    # Otherwise, an error
    length_crc = len(bin(crc_polynomial)[2:])
    
    if codeword_size < length_crc:
        raise ValueError(f'The codeword size should be {length_crc} bits')
        
    x_bits = x_bits_orig.zfill(codeword_size)

    crc = 0
    for position, value in enumerate(bin(crc_polynomial)[2:]):
        if value == '1':
            crc = crc ^ int(x_bits[position])
    crc = bin(crc)[2:]
    
    if len(crc) > crc_length:
        raise ValueError("Check CRC length parameter.")
    crc = crc.zfill(crc_length)
    
    return crc


def create_ricean_channel(N_r, N_t, K, N_sym=1, sigma_dB=8):
    global G # Pathloss in dB   
    
    G_fading = dB(G) - np_random.normal(loc=0, scale=np.sqrt(sigma_dB), size=(N_sym, N_r, N_t))       
    G_fading = np.array([linear(g) for g in G_fading])
    
    mu = np.sqrt(K / (1 + K))
    sigma = np.sqrt(1 / (1 + K))
    
    # Rician fading
    # H = np_random.normal(loc=mu, scale=sigma, size=(N_sym, N_r, N_t)) + \
    #     1j * np_random.normal(loc=mu, scale=sigma, size=(N_sym, N_r, N_t))
    
    H = np_random.normal(loc=mu, scale=sigma, size=(N_r, N_t)) + \
        1j * np_random.normal(loc=mu, scale=sigma, size=(N_r, N_t))
     
    # Repeat the channel across all subcarriers
    H = np.tile(H, N_sym).T.reshape(-1, N_r, N_t)
    
    # Normalize channel to unity gain
    traces = np.trace(H, axis1=1, axis2=2)
    for idx in range(N_sym):
        H[idx, :, :] /= traces[idx]
    
    # Introduce large scale gain
    H *= np.sqrt(G / 2)
    
    # TODO:  if fading then add to H the coefficients.

    H = H.reshape(N_sym, N_r, N_t)
    return H


def create_rayleigh_channel(N_r, N_t, N_sym=1, sigma_dB=8):
    global G # Pathloss in dB
    
    G_fading = dB(G) - np_random.normal(loc=0, scale=np.sqrt(sigma_dB), size=(N_r, N_t))
    G_fading = np.array([linear(g) for g in G_fading])
    
    # Rayleigh fading with G being the large scale fading
    H = np_random.normal(0, 1, size=(N_r, N_t)) + \
                          1j * np_random.normal(0, 1, size=(N_r, N_t))
     
    # Repeat the channel across all subcarriers
    H = np.tile(H, N_sym).T.reshape(-1, N_r, N_t)
    
    # Normalize channel to unity gain
    traces = np.trace(H, axis1=1, axis2=2)
    for idx in range(N_sym):
        H[idx, :, :] /= traces[idx]
    
    # Introduce large scale gain
    H *= np.sqrt(G / 2)
    
    # TODO:  if fading then add to H the coefficients.
    
    H = H.reshape(N_sym, N_r, N_t)
    return H


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
    
    if algorithm == 'ZF':
        W = np.linalg.pinv(H_hat)
    if algorithm == 'MMSE':
        print("WARNING:  The MMSE algorithm is not correct.")
        assert(rho is not None)
        W = H_hat.conjugate().T@np.linalg.inv(H_hat@H_hat.conjugate().T + (1./rho)*np.eye(N_r))        
    
    WWH = np.real(W@W.conjugate().T)
    Rx_SNR = rho / np.diag(WWH)
    
    assert(W.shape == (N_t, N_r))
    
    return W, Rx_SNR

        
def estimate_channel(X_p, Y_p, noise_power, algorithm, random_state=None):
    # This is for least square (LS) estimation    
    N_t, _ = X_p.shape
    
    if not np.allclose(X_p@X_p.T, np.eye(N_t)):
        raise ValueError("The training sequence is not semi-unitary.  Cannot estimate the channel.")
    
    if algorithm == 'LS':
        # This is least square (LS) estimation
        H_hat = Y_p@X_p.conjugate().T
    
    return H_hat
  

def generate_pilot(N_r, N_t, n_pilot, N_sc, random_state=None):
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
    
    # X_p has a dimension of N_sym x N_t x n_pilot
    X_p = np.tile(X_p.T, N_sc).T.reshape(-1, N_t, n_pilot) # ok
    
    return X_p
    

def channel_eigenmodes(H):
    # HH = H@H.conjugate().T
    # eigenvalues, eigenvectors = np.linalg.eig(HH)
    
    U, S, Vh = np.linalg.svd(H, full_matrices=False)
    eigenmodes = S ** 2
    return eigenmodes


def compute_overhead(b, codeword_size, k, N_s, crc_length):
    global N_sc
    
    # Find the right number of transmissions and the right transmission size
    
    effective_crc_length = int(k * np.ceil(crc_length / k)) # The receiver is also aware of the CRC length (in bits)    
    effective_crc_length_symbols = effective_crc_length // k # in symbols
  
    s_transmission = (N_sc - effective_crc_length_symbols) * N_s
    pad_length = int(N_s * np.ceil((s_transmission + effective_crc_length_symbols) / N_s)) - (s_transmission + effective_crc_length_symbols) # in symbols

    n_transmissions = int(np.ceil(b / (s_transmission * k))) # from subcarriers

    return n_transmissions, s_transmission, pad_length, effective_crc_length 

    
def transmit_receive(data, alphabet, channel, equalizer, snr_dB, crc_polynomial, crc_length, n_pilot, perfect_csi=False):
    global quantization_b
        
    k = np.log2(alphabet.shape[0]).astype(int)
    rho = linear(snr_dB)
    
    SERs = []
    BERs = []
    block_error = 0
        
    N_sc, N_t, N_r = channel.shape
    
    Df = 15e3 # subcarrier in Hz
    tti = 1e-3 # in seconds
        
    # Number of streams
    N_s = min(N_r, N_t)
        
    # Bit rate
    max_codeword_size = N_sc * k # per stream of actual payload (no CRC)
    max_bit_rate_per_stream = max_codeword_size / tti

    print(f'Transmission maximum bitrate per stream = {max_bit_rate_per_stream:.2f} bps')
    
    B = Df # per OFDM resource element
    print(f'Transmission BW per stream = {B:.2f} Hz')
    print(f'Number of OFDM subcarriers per stream = {N_sc}')
        
    b = len(data) # this is in bits
    
    n_transmissions, s_transmission, pad_length, effective_crc_length = compute_overhead(b, max_codeword_size, k, N_s, crc_length)
    codeword_size = k * s_transmission
        
    effective_crc_length_symbols = effective_crc_length // k # in symbols
    
    assert(b <= s_transmission * k * n_transmissions * N_s)
    
    # Code rate = 1 - (pad_length + effective_crc_length_symbols) / s_transmission
    
    x_info_complete = bits_to_symbols(data, alphabet, k)
    
    # Pilot symbols have a dimension of N_sym x N_t x n_pilot
    P = generate_pilot(N_r, N_t, n_pilot, N_sc, random_state=np_random)

    SNR_Rx = []
    Tx_EbN0 = []
    Rx_EbN0 = []
    PL = []
    data_rx = []
    channel_mse = []
    
    if b < 10000:
        print('Warning: Small number of bits can cause curves to differ from theoretical ones due to insufficient number of samples.')

    if n_transmissions < 10000:
        print('Warning: Small number of codeword transmissions can cause BLER values to differ from theoretical ones due to insufficient number of samples.')

    # See channel before
    plot_channel(channel)
    
    H_reconstructed = channel    
    
    print()
    print(f'Transmitting a total of {b} bits.')
    for tx_idx in np.arange(n_transmissions):
        print(f'Transmitting codeword {tx_idx + 1}/{n_transmissions}')
        # Every transmission is for one codeword, divided up to N_s streams.
        x_info = x_info_complete[tx_idx*s_transmission:(tx_idx+1)*s_transmission] # in symbols
        
        # 1) Compute CRC based on the original codeword
        # 2) Pad what is left *in between* to fulfill MIMO rank
        x_bits_orig = symbols_to_bits(x_info, k, alphabet)          # correct
        crc = compute_crc(x_bits_orig, codeword_size, crc_polynomial, crc_length)  # Compute CRC in bits.
        
        crc_padded = crc.zfill(effective_crc_length)
        _, _, crc_symbols = bits_to_baseband(crc_padded, alphabet, k)
                
        # Symbols
        x_b_i, x_b_q, x_sym = bits_to_baseband(x_bits_orig, alphabet, k)
        
        # Map codewords to the MIMO layers.
        x_sym_crc = np.r_[x_sym, np.zeros(pad_length), crc_symbols]
        
        # x_sym_crc has dimensions of N_sc times N_s
        #x_sym_crc = x_sym_crc.reshape(-1, N_s) # this is WRONG
        x_sym_crc = x_sym_crc.reshape(N_s, -1).T # this is correct.        
        # NOTE x_sym_crc.reshape(N_s,-1).T is not equal to x_sym_crc.reshape(-1,N_s)
        
        N_sc_payload = x_sym_crc.shape[0]
        
        if (N_sc_payload > N_sc):
            raise ValueError(f"Number of required subcarriers {N_sc_payload} exceeds {N_sc}.  Increase it or increase modulation order.")
            
        # Number of required symbols is N_sc_payload
        # Number of available symbols is N_sc        
        
        # Code rate
        R = len(x_bits_orig) / (N_s * N_sc_payload * k) # 1 - overhead / codeword_size
        print(f'Code rate = {R:.4f}')
        
        if x_sym_crc.shape[1] > N_sc:
            raise ValueError("Increase number of subcarriers to at least {} or increase codeword size.".format(x_sym_crc.shape[1]))
        
        # Signal energy
        E_x = np.linalg.norm(x_sym_crc, axis=1, ord=2) ** 2  # do not normalize symbol energy (Normalization does not work for M-QAM, M > 4).
        
        # Symbol power (OFDM resource element)
        P_sym = E_x * Df / (N_sc * N_t)
        P_sym_dB = dB(P_sym)
      
        # Noise power
        noise_power = P_sym_dB.mean() - snr_dB
        
        Tx_EbN0_ = snr_dB - dB(k)
        Tx_EbN0.append(Tx_EbN0_)
        
        print('Average symbol power at the transmitter is: {:.4f} dBm'.format(P_sym_dB.mean()))
        print(f'SNR at the transmitter (per stream): {snr_dB:.4f} dB')
        print(f'EbN0 at the transmitter (per stream): {Tx_EbN0_:.4f} dB')
        
        if Tx_EbN0_ < -1.59:
            print()
            print('** Outage at the transmitter **')
            print()
                
        # TODO: Introduce a precoder
        F = np.eye(N_t)
        
        # Debug purposes
        if perfect_csi:
            channel = np.eye(N_t)
            channel = np.tile(channel, N_sc).T.reshape(N_sc, N_r, N_t)
        
        ##############
        # Channel
        H = H_reconstructed[:N_sc_payload, :, :]
        
        # Assume channel is constant across all subcarriers
        eig = channel_eigenmodes(H.mean(axis=0))
        n_eig = len(eig)    
        print(f'Transmitter computed {n_eig} eigenmodes for channel H: {eig}.')
        
        # Since the channel coherence time is assumed constant
        H = H # this line has no meaning except to remind us that the channel changes after coherence time.
        # and only make sure we do not exceed the number of OFDM symbols available to us.
        # since every transmission is one TTI = 1 ms.
        
        # TODO: If N_t != N_r, the precoder is needed, and the multiplication H@x needs to be revisited.
        # Channel
        Hx = np.zeros((N_sc_payload, N_r), dtype=np.complex128) # N_sc x Nr
        for idx in range(N_sc_payload):
            Hx[idx, :] = H[idx, :, :]@x_sym_crc[idx,:] # Impact of the channel on the transmitted symbols
        
        # Power (actually energy) per OFDM symbol
        E_Hx = np.linalg.norm(Hx, axis=1, ord=2) ** 2
        
        # Additive noise
        a0, b0 = Hx.shape
        b0 += n_pilot # add pilot subcarriers
        n = 1./ np.sqrt(2) * (np_random.normal(0, 1, size=(a0,b0)) + \
                              1j * np_random.normal(0, 1, size=(a0,b0)))
        del a0, b0
        E_noise = np.linalg.norm(n, axis=1, ord=2) ** 2
        
        # Apply some changes to the noise statistic
        for idx in range(n.shape[0]):
            n[idx, :] /= np.sqrt(E_noise[idx] / 2.) # normalize noise
            n[idx, :] *= np.sqrt(E_Hx[idx] / rho / 2.) # noise power = signal / SNR
            
        E_noise = np.linalg.norm(n, axis=1, ord=2) ** 2
        
        if perfect_csi:
            n = np.zeros_like(n)
            E_noise = 0
            
        Y = Hx + n[:, :Hx.shape[1]]
        
        # If system is perfect, these two are equal        
        x_bits = symbols_to_bits(x_sym_crc, k, alphabet, is_complex=True)
        y_bits = symbols_to_bits(Y, k, alphabet, is_complex=True)
        # assert(x_bits == y_bits)
            
        # Useless statistics
        P_sym_rx = E_Hx * Df / N_sc
        P_sym_rx_dB = dB(P_sym_rx).mean()
        
        print(f'Average symbol power at the receiver is: {P_sym_rx_dB:.4f} dBm')
        
        # Channel
        HP = np.zeros(shape=(N_sc_payload, N_r, n_pilot), dtype=np.complex64)        
        for idx in range(HP.shape[0]):
            HP[idx, :, :] = H[idx, :, :]@P[idx, :, :]
        
        # The `reference symbol' is the first 4 OFDM symbols.
        H_hat = np.zeros_like(H, dtype=np.complex128)
        n_rs = min(N_sc_payload, 4)
        for idx_rs in range(n_rs):
            T_i = HP[idx_rs, :, :] + n[idx_rs, :n_pilot]
            P_i = P[idx_rs, :, :]
            
            # Estimate the channel
            H_hat_i = estimate_channel(P_i, T_i, noise_power=noise_power, algorithm='LS', random_state=np_random)
            H_hat += H_hat_i

        H_hat /= n_rs
        
        error_vector = _vec(H.mean(axis=0)) - _vec(H_hat.mean(axis=0))
        
        channel_estimation_mse = np.linalg.norm(error_vector, 2) ** 2 / np.product(H_hat.shape)
        print()
        print(f'Channel estimation MSE: {channel_estimation_mse:.4f}')
        print()
        channel_mse.append(channel_estimation_mse)
     
        # For future:  If channel MSE is greater than certain value, then CSI
        # knowledge scenarios kick in (i.e., CSI is no longer known perfectly to
        # both the transmitter/receiver).
        
        # Average the channel here.
        H_hat = H_hat.mean(axis=0)
        
        # Introduce quantization for Y only
        Y_unquantized = Y        
        Y = quantize(Y_unquantized, quantization_b)
    
        # MIMO-OFDM Receiver
        # The received symbol power *before* equalization impact        
        P_sym_rx = P_sym[idx] * np.linalg.norm(H_hat, ord='fro') ** 2        
        P_sym_rx_dB = dB(P_sym_rx)
        Rx_SNR_ = dB(rho * np.linalg.norm(H_hat, ord='fro') ** 2)

        # Compute the average path loss, which is basically the channel effect
        PL.append(np.mean(P_sym_dB - P_sym_rx_dB))
        
        # Equalizer to remove channel effect
        # Channel equalization
        # Now the received SNR per antenna (per symbol) due to the receiver
        # SNR per antenna and SNR per symbol are the same thing technically.
        W, _ = equalize_channel(H_hat, algorithm=equalizer, rho=rho)
        
        # An optimal equalizer should fulfill WH_hat = I_{N_t} for every subcarrier.
        # W@H_hat[0, :, :]
        
        # Thus z = x_hat + v 
        #        = x_hat + W n
        z = np.zeros_like(x_sym_crc)
        for idx in range(N_sc_payload):
            z[idx, :] = W@Y[idx, :]
            
        x_sym_crc_hat = z
 
        # the symbols are recovered properly
        # assert (x_sym_crc_hat == x_sym_crc).all()
        #pdb.set_trace()
        
        # Now how to extract x_hat from z?        
        # Detection to extract the signal from the signal plus noise mix.         
        x_sym_hat = _vec(x_sym_crc_hat)[:-effective_crc_length_symbols] # payload including padding
 
        # Remove the padding, which is essentially defined by the last data not on N_t boundary
        if pad_length > 0:
            x_sym_hat = x_sym_hat[:-pad_length] # no CRC and no padding.
        
        if len(x_sym_hat) == 0:
            print("WARNING: Empty payload.")
            SNR_Rx.append(np.nan)
            Rx_EbN0.append(np.nan)
            SERs.append(None)
            BERs.append(None)
            continue

        # Now let us find the SNR and Eb/N0 *after* equalization        
        # Average received SNR
        # Rx_SNR_eq = dB(Rx_SNR_eq.mean())
        Rx_SNR_eq = dB(rho * np.linalg.norm(W@H_hat, ord='fro') ** 2)
        SNR_Rx.append(Rx_SNR_eq)
                
        # Compute the average EbN0 at the receiver
        C = k * R
        Rx_EbN0_ = Rx_SNR_eq - 10*np.log10(C)
        Rx_EbN0.append(Rx_EbN0_)

        print(f'Average signal SNR at the receiver before equalization: {Rx_SNR_:.4f} dB')
        print(f'Average signal SNR at the receiver after equalization: {Rx_SNR_eq:.4f} dB')
        print('EbN0 at the receiver: {:.4f} dB'.format(Rx_EbN0_))
                
        if Rx_EbN0_ < -1.59:
            print('** An outage detected at the receiver **')                
            print()
        
        # Detection of symbols (symbol star is the centroid of the constellation)
        x_info_hat, _, _, x_bits_hat = ML_detect_symbol(x_sym_hat, alphabet)
        # x_info_hat, _, _, x_bits_hat = _unsupervised_detection(x_sym_hat, alphabet)
        x_bits_hat = ''.join(x_bits_hat)

        # # To test the performance of DNN in detecting symbols:
        # X = np.c_[np.real(x_sym_hat), np.imag(x_sym_hat)]
        # y = x_info_hat
        # model_dnn, dnn_accuracy_score, _ = DNN_detect_symbol(X=X, y=y)
                
        # Compute CRC on the received frame
        crc_comp = compute_crc(x_bits_hat, codeword_size, crc_polynomial, crc_length)
        
        ########################################################
        # Error statistics
        # Block error
        if int(crc) != int(crc_comp):
            block_error += 1
        
        symbol_error = 1 - np.mean(x_info_hat == x_info)
        
        SERs.append(symbol_error)

        x_hat_b = symbols_to_bits(x_info_hat, k, alphabet, is_complex=False)
        x_hat_b_i, x_hat_b_q, _ = bits_to_baseband(x_hat_b, alphabet, k)

        ber_i = _ber(x_hat_b_i, x_b_i) 
        ber_q = _ber(x_hat_b_q, x_b_q)
        
        # System should preserve the number of bits.
        assert(len(x_bits_orig) == len(x_bits_hat))
        
        data_rx.append(x_bits_hat)
        ber = np.mean([ber_i, ber_q])
        BERs.append(ber)        
        # for

    total_transmitted_bits = N_s * codeword_size * n_transmissions
    print(f"Total transmitted bits: {total_transmitted_bits} bits.")

    BLER = block_error / n_transmissions

    # Now extract from every transmission 
    data_rx_ = ''.join(data_rx)    
        
    return np.arange(n_transmissions), P_sym_dB.mean(), SERs, SNR_Rx, PL, BERs, BLER, Tx_EbN0, Rx_EbN0, channel_mse, data_rx_

    
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
    dim = int(np.sqrt(n / 3))
    
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


def _plot_bitmaps(data1, data2):
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)
        
    ax1.imshow(_convert_to_bytes_decimal(data1))
    ax2.imshow(_convert_to_bytes_decimal(data2))
    
    ax1.axis('off')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    
    
def generate_plot(df, xlabel, ylabel):
    cols = list(set([xlabel, ylabel, 'Tx_SNR']))
    df = df[cols]
    df_plot = df.groupby('Tx_SNR').mean().reset_index()

    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_yscale('log')
    ax.tick_params(axis=u'both', which=u'both')
    plt.plot(df_plot[xlabel].values, df_plot[ylabel].values, '--bo', alpha=0.7, 
             markeredgecolor='k', markerfacecolor='r', markersize=6)

    plt.grid(which='both', axis='both')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    tikzplotlib.save('output.tikz')
    plt.show()
    plt.close(fig)
    

def plot_channel(channel):
    global N_r, N_t
    
    try:
        H = channel.reshape(-1, N_r, N_t)

        xlabel = 'TX Antennas'
        ylabel = 'Subcarriers'
        
        fig, ax = plt.subplots(figsize=(12, 6))
    
        plt.imshow(np.abs(H / np.max(H)) ** 2, aspect='auto')
            
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        plt.tight_layout()
        tikzplotlib.save('channel.tikz')
        plt.show()
        plt.close(fig)
        
    except:
        return

    
def compute_large_scale_fading(d, f_c, G_t=1, G_r=1, pl_exp=2):
    global np_random
    
    l = c / (f_c * 1e9)
    G = G_t * G_r * (l / (4 * pi * d)) ** pl_exp
    
    assert (G < 1)
    
    return G


def run_simulation(file_name, channel_matrix, equalizer, constellation, Tx_SNRs, crc_polynomial, crc_length, n_pilot):
    global M_constellation
    
    alphabet = create_constellation(constellation=constellation, M=M_constellation)
    _plot_constellation(constellation=alphabet)
    data = read_bitmap(file_name)

    # _plot_bitmaps(data, data)
    df_output = pd.DataFrame()
    for snr in Tx_SNRs:
        c_i, P_x_Tx_i, SER_i, Rx_SNR_i, PL_i, BER_i, BLER_i, Tx_EbN0_i, Rx_EbN0_i, channel_mse_i, data_received = \
            transmit_receive(data, alphabet, channel_matrix, equalizer, 
                         snr, crc_polynomial, crc_length, n_pilot, perfect_csi=False)
        
        _plot_bitmaps(data, data_received)
        
        df_output_ = pd.DataFrame(data={'Codeword': c_i})
        df_output_['P_x_Tx'] = P_x_Tx_i
        df_output_['SER'] = SER_i        
        df_output_['Tx_SNR'] = snr        
        df_output_['Rx_SNR'] = Rx_SNR_i
        df_output_['PL'] = PL_i
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


# 1) Create a channel
G = compute_large_scale_fading(d=100, f_c=freq)
H = create_ricean_channel(N_r=N_r, N_t=N_t, N_sym=N_sc, K=K_factor, sigma_dB=shadowing_std)

# 2) Run the simulation on this channel
df_output = run_simulation(file_name, H, MIMO_equalizer, 
                           constellation, Tx_SNRs, crc_polynomial, crc_length,
                           n_pilot)
df_output.to_csv('output.csv', index=False)

# 3) Generate plot
xlabel = 'Rx_EbN0'
ylabel = 'Avg_BER'

generate_plot(df=df_output, xlabel=xlabel, ylabel=ylabel)