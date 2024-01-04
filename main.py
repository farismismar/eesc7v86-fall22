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
file_name = 'faris.bmp' # Payload to be transmitted
constellation = 'QPSK'
M_constellation = 4
seed = 7
codeword_size = 4 # bits
n_pilot = 4
N_r = 1
N_t = 1
f_c = 1.8e6 # in Hz

# Note that the polynomial size is equal to the codeword size.
crc_polynomial = 0b1010
crc_length = 2 # bits

sigmas = np.sqrt([0.001, 0.01, 0.1, 1, 10]) # square root of noise power (W) 10 log(kTB + Nf)

prefer_gpu = True
##################

__release_date__ = '2024-01-03'
__ver__ = '0.3'

##################
plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = "14"

np_random = np.random.RandomState(seed=seed)
k_constellation = int(np.log2(M_constellation))


def create_constellation(constellation, M):
    if (constellation == 'QPSK'):
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
    centroids = pd.DataFrame(columns=['m', 'x_I', 'x_Q'])

    for m in np.arange(M):
        centroid_ = pd.DataFrame(data={'m': int(m),
                                       'x_I': np.sqrt(1 / 2) * np.cos(2*np.pi/M*m + np.pi/M),
                                       'x_Q': np.sqrt(1 / 2) * np.sin(2*np.pi/M*m + np.pi/M)}, index=[m])
        if centroids.shape[0] == 0:
            centroids = centroid_.copy()
        else:
            centroids = pd.concat([centroids, centroid_], ignore_index=True)
    
    gray = centroids['m'].apply(lambda x: decimal_to_gray(x, k))
    centroids['I'] = gray.str[:(k//2)]
    centroids['Q'] = gray.str[(k//2):]

    centroids.loc[:, 'x'] = centroids.loc[:, 'x_I'] + 1j * centroids.loc[:, 'x_Q']
    
    # Normalize the transmitted symbols    
    # The average power is normalized to unity
    P_average = np.mean(np.abs(centroids.loc[:, 'x']) ** 2)
    centroids.loc[:, 'x'] /= np.sqrt(P_average)
    
    return centroids


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
    plt.close(fig)
    

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


def compute_crc(x_bits_orig, crc_polynomial, crc_length):
    # Introduce CRC to x
    global codeword_size
    
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


def create_rayleigh_channel(N_r, N_t):
    global G
    # Rayleigh fading with G being the large scale fading
    H = np.sqrt(G / 2) * (np_random.normal(0, 1, size=(N_r, N_t)) + \
                          1j * np_random.normal(0, 1, size=(N_r, N_t)))
    
    # Normalize the channel so it has unity power gain
    H /= np.linalg.norm(H, ord='fro')
    
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
    

def equalize_channel(H):
    global N_t, N_r, G
    
    # ZF equalization
    W = 1/np.sqrt(G) * np.linalg.pinv(H)
    
    assert(W.shape == (N_r, N_t))
    
    return W

        
def estimate_channel(X_p, Y_p, noise_power, random_state=None):
    global G
    # This is for least square (LS) estimation
    # and the linear minimum mean squared error (L-MMSE):
    N_t, _ = X_p.shape
    
    if not np.allclose(X_p@X_p.T, np.eye(N_t)):
        raise ValueError("The training sequence is not semi-unitary.  Cannot estimate the channel.")
    
    # This is least square (LS) estimation
    H_hat_ls = 1. / np.sqrt(G) * Y_p@X_p.conjugate().T
    
    # This is the L-MMSE estimation:
    H_hat = np.sqrt(G) * Y_p@X_p.conjugate().T@np.linalg.inv((G + noise_power)*np.eye(N_t))
    
    return H_hat
  

def generate_pilot(N_r, N_t, n_pilot, random_state=None):
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
    
    # What matrix X_pX_p* = I (before scaling)
    assert(np.allclose(X_p@X_p.T, np.eye(N_t)))  # This is it
    
    # The training sequence is X_p.  It has N_t rows and n_pilot columns
    return X_p
    

def transmit_receive(data, codeword_size, alphabet, H, k, noise_power, crc_polynomial, crc_length, n_pilot, perfect_csi=False):
    global G
    
    if codeword_size < k:
        raise ValueError("Codeword size is too small for the chosen modulation")
        
    SERs = []
    BERs = []
    block_error = 0
    
    N_r, N_t = H.shape
    
    Df = 15e3 # subcarrier in Hz
    tti = 1e-3 # in seconds
    
    # Effective codeword size, must coincide with integer number of symbols
    codeword_size = int(np.ceil(codeword_size / k) * k)
    
    # Bit rate 
    bit_rate = codeword_size / tti

    # Number of streams
    N_s = min(N_r, N_t)
    bit_rate_per_stream = bit_rate / N_s

    print(f'Transmission maximum bitrate per stream = {bit_rate_per_stream:.2f} bps')
    
    # Find the correct number of subcarriers required for this bit rate
    # assuming 1:1 code rate.
    Nsc = np.ceil(bit_rate_per_stream / (k * Df))    # Number of OFDM subcarriers
    B = Nsc * Df # Transmit bandwidth
    print(f'Transmission BW per stream = {B:.2f} Hz')
    
    # Find the noise power spectral density
    N0 = noise_power * B
    
    # Note that bit_rate / B cannot exceed k.
    # Thus if bandwidth became B N_s due to spatial multiplexing, then the bit rate also scales by N_s.
    assert(k >= bit_rate_per_stream / B)
    
    b = len(data)
    n_transmissions = int(np.ceil(b / (codeword_size * N_t)))
    
    x_info_complete = bits_to_symbols(data, alphabet, k)
    
    SNR_Rx = []
    SNR_Tx = []
    data_rx = []
    Tx_EbN0 = []
    Rx_EbN0 = []
    Es_Tx = []
    Es_Rx = []
    PL = []
    
    if n_transmissions < 10000:
        print('Warning: Small number of transmissions can cause curves to look incorrect due to insufficient number of samples.')

    print(f'Transmitting a total of {b} bits.')
    for codeword in np.arange(n_transmissions):
        print(f'Transmitting codeword {codeword + 1}/{n_transmissions}')
        # Every transmission is for one codeword, divided up to N_t streams.
        x_info = x_info_complete[codeword*N_t*(codeword_size // k):(codeword+1)*N_t*(codeword_size // k)]
        
        # 1) Compute CRC based on the original codeword
        # 2) Pad what is left *in between* to fulfill MIMO rank
        x_bits_orig = symbols_to_bits(x_info, k, alphabet)          # correct
        crc = compute_crc(x_bits_orig, crc_polynomial, crc_length)  # Compute CRC in bits.
        
        effective_crc_length = int(k * np.ceil(crc_length / k)) # The receiver is also aware of the CRC length (in bits)
        crc_padded = crc.zfill(effective_crc_length)
        _, _, crc_symbols = bits_to_baseband(crc_padded, alphabet, k)
        effective_crc_length_symbols = effective_crc_length // k # in symbols
                
        # Symbols
        x_b_i, x_b_q, x_sym = bits_to_baseband(x_bits_orig, alphabet, k)
        
        # Map codewords to the MIMO layers.
        pad_length = int(N_t * np.ceil((len(x_sym) + effective_crc_length_symbols) / N_t)) - len(x_sym) - effective_crc_length_symbols # in symbols
        x_sym_crc = np.r_[x_sym, np.zeros(pad_length), crc_symbols]
        
        # Symbol power (energy)
        x_sym_crc = x_sym_crc.reshape(-1, N_t).T # Do not be tempted to do the obvious!
        Es = np.linalg.norm(x_sym_crc, ord=2, axis=0).mean()
        Es_Tx.append(10*np.log10(Es) + 30) # in dBmJoules
        # For every vector x, the power should be Es N_t = N_t.
        # np.linalg.norm(x_sym_crc, ord=2, axis=0) 
        
        Tx_SNR_ = 10*np.log10(Es * B / (N_t * noise_power))
        SNR_Tx.append(Tx_SNR_)
        
        Tx_EbN0_ = 10 * np.log10(Es * B / (k * N_t * noise_power))
        Tx_EbN0.append(Tx_EbN0_)
        
        print(f'Symbol SNR at the transmitter (per stream): {Tx_SNR_:.4f} dB')
        print(f'EbN0 at the transmitter (per stream): {Tx_EbN0_:.4f} dB')
                
        # TODO: Introduce a precoder
        F = np.eye(N_t)
        
        # Additive noise sampled from a complex Gaussian
        noise_dimension = max(n_pilot, x_sym_crc.shape[1])
        n = np_random.normal(0, scale=np.sqrt(noise_power)/np.sqrt(2), size=(N_r, noise_dimension)) + \
            1j * np_random.normal(0, scale=np.sqrt(noise_power)/np.sqrt(2), size=(N_r, noise_dimension))
        
        # Debug purposes
        if perfect_csi:
            n = np.zeros((N_r, noise_dimension))
            H = np.eye(N_t)

        # TODO: Introduce quantization for both Y and Y pilot
        
        # Channel
        HFx = H@F@x_sym_crc
        Y = np.sqrt(G) * HFx + n[:, :x_sym_crc.shape[1]]
        
        # Pilot contribution (known sequence)
        # Generate pilot
        P = generate_pilot(N_r, N_t, n_pilot, random_state=np_random)
        
        # Channel
        HFP = H@F@P
        T = np.sqrt(G) * HFP + n[:, :n_pilot]
    
        # Estimate the channel
        H_hat = estimate_channel(P, T, noise_power, random_state=np_random)
        
        error_vector = _vec(H) - _vec(H_hat)
        
        channel_estimation_mse = np.linalg.norm(error_vector, 2) ** 2 / (N_t * N_r)
        print(f'Channel estimation MSE: {channel_estimation_mse:.4f}')
        
        # Channel equalization using ZF
        W = equalize_channel(H_reconstructed)
        W /= np.trace(W)
        
        # TODO: How does F impact W?  Idea removes N_t in the denom.
        
        # The optimal equalizer should fulfill WH = I_{N_t} * sqrt(G)]
        x_sym_crc_hat = W@Y
        # np.allclose(x_sym_crc_hat, x_sym_crc)
        
        # crc_sym_hat = _vec(x_sym_crc_hat)[-effective_crc_length_symbols:] # only transmitted CRC
        x_sym_hat = _vec(x_sym_crc_hat)[:-effective_crc_length_symbols] # payload including padding

        # Remove the padding, which is essentially defined by the last data not on N_t boundary
        if pad_length > 0:
            x_sym_hat = x_sym_hat[:-pad_length] # no CRC and no padding.

        # np.allclose(x_sym_hat, x_sym)
            
        # Detection of symbols (symbol star is the centroid of the constellation)        
        x_info_hat, _, _, x_bits_hat = ML_detect_symbol(x_sym_hat, alphabet)
        # x_info_hat, _, _, x_bits_hat = _unsupervised_detection(x_sym_hat, alphabet)
        x_bits_hat = ''.join(x_bits_hat)

        # # To test the performance of DNN in detecting symbols:
        # X = np.c_[np.real(x_sym_hat), np.imag(x_sym_hat)]
        # y = x_info_hat
        # model_dnn, dnn_accuracy_score, _ = DNN_detect_symbol(X=X, y=y)
        
        # TODO:  This needs to be fixed:
        received_noise_power = noise_power * np.linalg.norm(W, 'fro') ** 2 # equalization enhanced noise
        
        # TODO:  Check the receiver statistics.
        # Compute the path loss which is the channel gain multiplied by the large scale fading gain.
        
        # Compute the received symbol SNR all before equalization
        Rx_SNR_ = 10*np.log10(G * Es * B / (N_t * noise_power))
        
        PL.append(Rx_SNR_ - Tx_SNR_)
        # Compute the EbN0 at the receiver and just before equalization
        Rx_EbN0_ = 10 * np.log10(G * Es * B / (N_t * noise_power * bit_rate_per_stream))
        
        print(f'Symbol SNR at the receiver (per stream): {Rx_SNR_:.4f} dB')
        print(f'EbN0 at the receiver (per stream): {Rx_EbN0_:.4f} dB')
        
        # Compute the received symbol SNR all after equalization
        Rx_SNR_ = 10*np.log10(Es * B / (N_t * received_noise_power))
        SNR_Rx.append(Rx_SNR_)

        # Compute the EbN0 at the receiver and just before equalization
        Rx_EbN0_ = 10 * np.log10(Es * B / (N_t * received_noise_power * bit_rate_per_stream))
        Rx_EbN0.append(Rx_EbN0_)        
        
        print(f'Symbol SNR at the receiver (per stream): {Rx_SNR_:.4f} dB')
        print(f'EbN0 at the receiver (per stream): {Rx_EbN0_:.4f} dB')
        
        if Rx_EbN0_ < -1.59:
            print('** Outage at the receiver **')
        
        # Compute CRC on the received frame
        crc_comp = compute_crc(x_bits_hat, crc_polynomial, crc_length)

        ########################################################
        # Error statistics
        # Block error
        if int(crc) != int(crc_comp):
            block_error += 1
            
        symbol_error = 1 - np.mean(x_info_hat == x_info)
        SERs.append(symbol_error)
        
        #x_hat_b = symbols_to_bits(x_info_hat, k, alphabet, is_complex=False)
        x_hat_b_i, x_hat_b_q, _ = bits_to_baseband(x_bits_hat, alphabet, k)

        ber_i = 1 - np.mean(x_hat_b_i == x_b_i)
        ber_q = 1 - np.mean(x_hat_b_q == x_b_q)
        
        # System should preserve the number of bits.
        assert(len(x_bits_orig) == len(x_bits_hat))
        
        data_rx.append(x_bits_hat)
        ber = np.mean([ber_i, ber_q])
        BERs.append(ber)
        # for

    total_transmitted_bits = N_t * codeword_size * n_transmissions
    print(f"Total transmitted bits: {total_transmitted_bits} bits.")

    BLER = block_error / n_transmissions

    # Now extract from every transmission 
    data_rx_ = ''.join(data_rx)
    
    # Summarized
    # Es_Tx = np.mean(Es_Tx)
    # SER = np.mean(SERs)
    # BER = np.mean(BERs)
    
    # SNR_Tx = np.mean(SNR_Tx)
    # SNR_Rx = np.mean(SNR_Rx)
    # Tx_EbN0 = np.mean(Tx_EbN0)
    # Rx_EbN0 = np.mean(Rx_EbN0)
    
    return np.arange(n_transmissions), Es, SERs, SNR_Tx, SNR_Rx, PL, BERs, BLER, Tx_EbN0, Rx_EbN0, bit_rate_per_stream, B, data_rx_


def read_bitmap(file, word_length=8):
    # This is a 32x32 pixel image
    im_orig = plt.imread(file)
    
    im = im_orig.flatten()
    im = [bin(a)[2:].zfill(word_length) for a in im] # all fields when converted to binary need to have word length.

    im = ''.join(im) # This is now a string of bits

    # These lines are for random bits 
    im  = np_random.binomial(1,0.5, 40000)
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
    df_plot = df_output.groupby('noise_power').mean().reset_index()

    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_xscale('log')
    plt.plot(df_plot[xlabel].values, df_plot[ylabel].values, '--bo', alpha=0.7, 
             markeredgecolor='k', markerfacecolor='r', markersize=6)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    

def compute_large_scale_fading(d, f_c, pl_exp=2):
    l = c / f_c
    G = (4 * pi * d / l) ** pl_exp

    return G
        
def run_simulation(file_name, codeword_size, h, constellation, k_constellation, sigmas, crc_polynomial, crc_length, n_pilot):
    alphabet = create_constellation(constellation=constellation, M=int(2 ** k_constellation))
    data = read_bitmap(file_name)

    # _plot_bitmaps(data, data)
    df_output = pd.DataFrame()
    for sigma in sigmas:
        c_i, Es_Tx_i, SER_i, Tx_SNR_i, Rx_SNR_i, PL_i, BER_i, BLER_i, Tx_EbN0_i, Rx_EbN0_i, bit_rate, bandwidth, data_received = \
            transmit_receive(data, codeword_size, alphabet, h, k_constellation, 
                         sigma ** 2, crc_polynomial, crc_length, n_pilot, perfect_csi=False)
        df_output_ = pd.DataFrame(data={'Codeword': c_i})
        df_output_['Es'] = Es_Tx_i        
        df_output_['SER'] = SER_i
        df_output_['Tx_SNR'] = Tx_SNR_i
        df_output_['Rx_SNR'] = Rx_SNR_i
        df_output_['PL'] = PL_i
        df_output_['Avg_BER'] = BER_i
        df_output_['BLER'] = BLER_i
        df_output_['Tx_EbN0'] = Tx_EbN0_i
        df_output_['Rx_EbN0'] = Rx_EbN0_i
        df_output_['Bit_Rate'] = bit_rate
        df_output_['BW'] = bandwidth
        df_output_['noise_power'] = sigma ** 2
        
        _plot_bitmaps(data, data_received)
        
        if df_output.shape[0] == 0:
            df_output = df_output_.copy()
        else:
            df_output = pd.concat([df_output_, df_output], axis=0, ignore_index=True)
        
    df_output = df_output.reset_index(drop=True)
    
    return df_output


# 1) Create a channel
G = compute_large_scale_fading(d=1, f_c=f_c)
H = create_rayleigh_channel(N_r=N_r, N_t=N_t)

# 2) Run the simulation on this channel
df_output = run_simulation(file_name, codeword_size, H, constellation, 
                           k_constellation, sigmas, crc_polynomial, crc_length,
                           n_pilot)
df_output.to_csv('output.csv', index=False)

# 3) Generate plot
xlabel = 'Tx_EbN0'
ylabel = 'Avg_BER'

generate_plot(df=df_output, xlabel=xlabel, ylabel=ylabel)
