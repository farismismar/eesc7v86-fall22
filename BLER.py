	# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:19:40 2023

@author: farismismar
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if os.name == 'nt':
    os.add_dll_directory("/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# The GPU ID to use, usually either "0" or "1" based on previous line.
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from keras import backend as K
from scipy.linalg import hankel

file_name = 'faris.bmp'

# System parameters
constellation = 'QAM'
M_constellation = 64
seed = 7
codeword_size = 16 # bits
k_constellation = int(np.log2(M_constellation))
n_pilot = 4
N_r = 2
N_t = 2

plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = "14"

# Note that the polynomial size is equal to the codeword size.
crc_polynomial = 0b0001_0010_0000_0010
crc_length = 2 # bits

sigmas = np.sqrt(np.logspace(-1, 3, num=8)) # square root of noise power
##################

np_random = np.random.RandomState(seed=seed)

def create_constellation(constellation, M):
    if (constellation == 'QPSK'):
        return create_constellation_psk(M)
    elif (constellation == 'QAM'):
        return create_constellation_qam(M)
    else:
        return None
     
# Constellation based on Gray code
# OK
def create_constellation_psk(M):
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
    
    # Normalize the transmitted symbols
    signal_power = np.sum(centroids['x_I'] ** 2 + centroids['x_Q'] ** 2) / M

    centroids[['x_I', 'x_Q']] /= np.sqrt(signal_power)
    centroids.loc[:, 'x'] = centroids.loc[:, 'x_I'] + 1j * centroids.loc[:, 'x_Q']
    
    gray = centroids['m'].apply(lambda x: decimal_to_gray(x, k))
    centroids['I'] = gray.str[:(k//2)]
    centroids['Q'] = gray.str[(k//2):]
    
    return centroids


# Constellation based on Gray code
def create_constellation_qam(M):
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
    
    # Normalize the transmitted symbols
    signal_power = np.sum(constellation['x_I'] ** 2 + \
                           constellation['x_Q'] ** 2) / M
    
    constellation[['x_I', 'x_Q']] /= np.sqrt(signal_power)
    constellation.loc[:, 'x'] = constellation.loc[:, 'x_I'] + 1j * constellation.loc[:, 'x_Q']
    
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
        sym_i = df_.loc[df_['IQ'] == bits_i, 'm'].values[0].astype(int)
        # print(bits_i, sym_i)
        x_sym.append(sym_i)
        
    return np.array(x_sym)


# TODO: The next two functions are a bit off?
def symbols_to_bits(x_sym, k, alphabet, is_complex=False):
    if is_complex == False:
        x_bits = ''
        for s in x_sym:
            i, q = alphabet.loc[alphabet['m'] == s, ['I', 'Q']].values[0]
            x_bits += '{}{}'.format(i, q).zfill(k)
        return x_bits
    
    pdb.set_trace()
# if is_complex == False:
    #     x_streams = []
    #     for stream in x_sym.T:
    #         x_bits = ''
    #         for s in stream:
    #             i, q = alphabet.loc[alphabet['m'] == s, ['I', 'Q']].values[0]
    #             x_bits += '{}{}'.format(i, q).zfill(k)
    #         x_streams.append(x_bits)
    #     return x_streams
    # else:
    #     pdb.set_trace()
    #     x_b_i = np.sign(np.real(x_sym))
    #     x_b_q = np.sign(np.imag(x_sym))

    #     x_bits = []
    #     for i, q in zip(x_b_i, x_b_q):
    #         x_bits.append('{}{}'.format(int(0.5*i + 0.5), int(0.5*q + 0.5)))
    #     return x_b_i, x_b_q, ''.join(x_bits)


# This function is wrong.
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


def compute_crc(x_bits_orig, crc_polynomial):
    # Introduce CRC to x
    crc = 0
    for position, value in enumerate(bin(crc_polynomial)[2:]):
        if value == '1':
            crc = crc ^ int(x_bits_orig[position])
    crc = bin(crc)[2:]
    
    return crc


def create_rayleigh_channel(N_r, N_t):
    G = 1 # large scale fading constant
    # Rayleigh fading
    H = np.sqrt(G / 2) * (np_random.normal(0, 1, size=(N_r, N_t)) + \
                          1j * np_random.normal(0, 1, size=(N_r, N_t)))

    return H


def ML_detect_symbol(x_sym_hat, alphabet):
    # This function returns argmin |x - s_m| based on AWGN ML detection
    df = pd.DataFrame(data={0: abs(x_sym_hat - alphabet.loc[alphabet['m'] == 0, 'x'].values[0]),
                            1: abs(x_sym_hat - alphabet.loc[alphabet['m'] == 1, 'x'].values[0]),
                            2: abs(x_sym_hat - alphabet.loc[alphabet['m'] == 2, 'x'].values[0]),
                            3: abs(x_sym_hat - alphabet.loc[alphabet['m'] == 3, 'x'].values[0])})
    
    return df.idxmin(axis=1).values



def _estimate_channel(X_p, Y_p, random_state=None):
    # This is for least square (LS) estimation
    
    N_t, _ = X_p.shape    
    if not np.allclose(X_p@X_p.T, np.eye(N_t)):
        raise ValueError("The training sequence is not semi-unitary.  Cannot estimate the channel.")
    
    # This is least square (LS) estimation
    H_hat = Y_p@X_p.conjugate().T
    
    return H_hat
  

def _generate_pilot(N_r, N_t, n_pilot, random_state=None):
    # Check if the dimensions are valid for the operation
    if n_pilot < N_t:
        raise ValueError("The length of the training sequence should be greater than or equal to the number of transmit antennas.")
    
    # Compute a unitary matrix from a combinatoric of e
    I = np.eye(N_t)
    idx = random_state.choice(range(N_t), size=N_t, replace=False)
    Q = I[:, idx] 
    
    assert(np.allclose(Q@Q.T, np.eye(N_t)))  # Q is indeed unitary, but square.
    
    # To make a semi-unitary, we need to post multiply with a rectangular matrix
    # Now we need a rectangular matrix (fat)
    A = np.zeros((N_t, n_pilot), int)
    np.fill_diagonal(A, 1)
    X_p =  Q @ A
        
    # What matrix X_pX_p* = I
    assert(np.allclose(X_p@X_p.T, np.eye(N_t)))  # This is it
    
    # The training sequence is X_p.  It has N_t rows and n_pilot columns
    return X_p
    
    
def _cplex_mse(y_true, y_pred):
    y_true = K.cast(y_true, tf.complex64) 
    y_pred = K.cast(y_pred, tf.complex64) 
    y_true = tf.reshape(y_true, [1, -1])
    y_pred = tf.reshape(y_pred, [1, -1])
    return K.mean(K.square(K.abs(y_true - y_pred)))
    

def _compress_channel(H, compression_ratio, epochs=10, batch_size=16, 
                      training_split=0.5):
    from gan_inversion import Autoencoder
    global seed
    
    N_r, N_t = H.shape
    
    # Normalize the channel such that the sq Frobenius norm is N_t N_r  
    H /= np.linalg.norm(H, ord='fro') / np.sqrt(N_t*N_r)    
    
    latent_dim = H.shape[1]
    
    compressed_latent_dim = int((1 - compression_ratio) * latent_dim)
    training_size = int(training_split * H.shape[0])
    
    autoencoder_re = Autoencoder(compressed_latent_dim, 
                              shape=H.shape[1:], seed=seed)
    autoencoder_re.compile(optimizer='adam', loss=_cplex_mse)
    
    # For the encoder to learn to compress, pass x_train as an input and target
    # Decoder will learn how to reconstruct original 
    x_train = np.real(H[:training_size, 1])
    x_test = np.real(H[training_size:, 1])
    
    history_re = autoencoder_re.fit(x_train, x_train,
                    epochs=epochs, batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    
    # Encoded samples are compressed and the latent vector has a lower
    # dimension as the result of compression.
    h_re_compressed = autoencoder_re.encoder.predict(x_test)
    
    # Decoder tries to reconstruct the true signal
    h_re_reconstructed = autoencoder_re.decoder.predict(h_re_compressed)
    
    # Repeat but for imaginary part.  See p6 https://arxiv.org/pdf/1905.03761.pdf
    autoencoder_im = Autoencoder(compressed_latent_dim, 
                              shape=H.shape[1:], seed=seed)
    autoencoder_im.compile(optimizer='adam', loss=_cplex_mse)
    
    x_train = np.imag(H[:training_size, 1])
    x_test = np.imag(H[training_size:, 1])
    
    history_im = autoencoder_im.fit(x_train, x_train,
                    epochs=epochs, batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    h_im_compressed = autoencoder_im.encoder.predict(x_test)
    h_im_reconstructed = autoencoder_im.decoder.predict(h_im_compressed)

    h_compressed = h_re_compressed + 1j * h_im_compressed
    h_reconstructed = h_re_reconstructed + 1j * h_im_reconstructed
    del h_re_compressed, h_im_compressed
    del h_re_reconstructed, h_im_reconstructed
    
    error = _cplex_mse(H[training_size:, :], h_reconstructed)
    
    return h_compressed, h_reconstructed, error.numpy()


def transmit_receive(data, codeword_size, alphabet, H, k, noise_power, crc_polynomial, crc_length, n_pilot):
    SERs = []
    BERs = []
    block_error = 0
    
    N_r, N_t = H.shape
    
    Df = 15e3 # subcarrier in Hz
    tti = 1e-3 # in seconds
    bit_rate = codeword_size / tti * N_t
    print(f'Transmission maximum rate = {bit_rate:.2f} bps')
    
    # Find the correct number of subcarriers required for this bit rate
    # assuming 1:1 code rate.
    Nsc = np.ceil(bit_rate / (k * Df))    # Number of subcarriers
    B = Nsc * Df # Transmit bandwidth
    print(f'Transmission BW = {B:.2f} Hz')
    
    Es = np.linalg.norm(alphabet['x'], ord=2) ** 2 / alphabet.shape[0]
    Tx_SNR = 10*np.log10(Es / (N_t * noise_power))
    
    N0 = noise_power / B
    Tx_EbN0 = 10 * np.log10((Es/(N_t * k)) / N0)
    
    print(f'SNR at the transmitter (per stream): {Tx_SNR:.4f} dB')
    print(f'EbN0 at the transmitter (per stream): {Tx_EbN0:.4f} dB')
    b = len(data)
    n_transmissions = int(np.ceil(np.ceil(b / codeword_size) / N_t))
    
    x_info_complete = bits_to_symbols(data, alphabet, k)
    
    data_rx = []
    
    print(f'Transmitting a total of {b} bits.')
    Rx_EbN0 = []
    for codeword in np.arange(n_transmissions):
        print(f'Transmitting codeword {codeword + 1}/{n_transmissions}')
        # Every transmission takes up N_t symbols each symbol has codeword bits
        x_info = x_info_complete[codeword*N_t*(codeword_size // k):(codeword+1)*N_t*(codeword_size // k)]
        
        # CRC is added to the original data, and not per MIMO stream.
        x_bits_orig = symbols_to_bits(x_info, k, alphabet)  # correct
        crc = compute_crc(x_bits_orig, crc_polynomial)      # Compute CRC.
        effective_crc_length = int(k * np.ceil(crc_length / k))
        
        # If CRC less than bps of constellation and N_t, pad it first.
        x_bits_crc = x_bits_orig + crc.zfill(effective_crc_length)        
        
        # Bits to I/Q
        x_b_i, x_b_q, x_sym_crc = bits_to_baseband(x_bits_crc, alphabet, k)
        
        # Number of zero pads due to transmission rank
        pad_length = int(N_t * np.ceil(len(x_sym_crc) / N_t)) - len(x_sym_crc)
        x_sym_crc = np.r_[x_sym_crc, np.zeros(pad_length)]
        
        x_sym_crc = x_sym_crc.reshape(-1, N_t).T
        
        # Additive noise
        n = np_random.normal(0, scale=noise_power/np.sqrt(2), size=(N_r, n_pilot)) + \
            1j * np_random.normal(0, scale=noise_power/np.sqrt(2), size=(N_r, n_pilot))
        
        # Debug purposes
        n = np.zeros((N_r, n_pilot))
        H = np.eye(N_t)
        
        # Channel
        Y = np.empty(N_r)
        for s in range(N_t):
            y_s = np.matmul(H, x_sym_crc[:,s]) + n[:,0]
            Y = np.c_[Y, y_s]
        Y = Y[:, 1:] # get rid of that "empty" column.
        
        # x_sym_crc must be a column vector of N_t
        assert(x_sym_crc.shape[0] == N_t)
        
        # Pilot contribution (known sequence)
        # Generate pilot
        P = _generate_pilot(N_r, N_t, n_pilot, random_state=np_random)
        
        # Channel
        T = H@P + n
    
        # Estimate the channel
        H_hat = _estimate_channel(P, T, random_state=np_random)
        
        error_vector = _vec(H) - _vec(H_hat)
        
        channel_estimation_mse = np.linalg.norm(error_vector, 2) ** 2 / (N_t * N_r)
        print(f'Channel estimation MSE: {channel_estimation_mse:.4f}')
        
        # Now compress h_hat
        # TODO: with compression, toggle comments below
        # H_compress, H_reconstructed, comp_channel_error = _compress_channel(H_hat, compression_ratio=0)
        H_reconstructed = H_hat
        comp_channel_error = np.nan
        
        # Channel equalization using matched filter
        W = np.conjugate(H_reconstructed) / (np.linalg.norm(H_reconstructed, 2) ** 2)
        assert(W.shape == (N_r, N_r))
        
        pdb.set_trace()        
        x_sym_crc_hat = W @ Y # Something is wrong with the dimensions.
        
        
        # Back to bits
        _, _, x_bits_crc_hat = symbols_to_bits(x_sym_crc_hat, k, alphabet,
                                                       is_complex=True)
        # Remove CRC
        x_bits_hat, crc = x_bits_crc_hat[:-crc_length], x_bits_crc_hat[-crc_length:]
        
        # Compute the EbN0 at the receiver
        received_noise_power = noise_power * np.real(np.mean(np.vdot(w, w))) # equalization enhanced noise
        N0 = received_noise_power / B
        Rx_EbN0.append(10 * np.log10((Es/k) / N0))
        
        # Compute CRC on the received frame
        _, crc_comp = add_crc(x_bits_hat, crc_polynomial, crc_length)
        
        if int(crc) != int(crc_comp):
            block_error += 1
            
        # Now back to symbol without the CRC
        x_sym_hat = x_sym_crc_hat[:-crc_length // k]
        
        ########################################################
        # Symbol ML detection
        x_info_hat = ML_detect_symbol(x_sym_hat, alphabet)
        
        ########################################################
        
        # Error statistics
        symbol_error = 1 - np.mean(x_info_hat == x_info)
        SERs.append(symbol_error)
        
        x_hat_b_i, x_hat_b_q, x_hat_b = symbols_to_bits(x_sym_hat, k, alphabet, is_complex=True)
        ber_i = 1 - np.mean(x_hat_b_i == x_b_i[:-crc_length // 2])
        ber_q = 1 - np.mean(x_hat_b_q == x_b_q[:-crc_length // 2])
        
        data_rx.append(x_hat_b)
        ber = np.mean([ber_i, ber_q])
        BERs.append(ber)
        # for
        
    BLER = block_error / n_transmissions

    data_rx_ = [np.array([d[:8], d[-8:]]) for d in data_rx]
    data_rx_ = np.array(data_rx_).flatten()
    data_rx_ = ''.join(data_rx_)
    
    return SERs, BERs, BLER, Tx_EbN0, Rx_EbN0, data_rx_, comp_channel_error


def read_bitmap(file):
    # This is a 32x32 pixel image
    im_orig = plt.imread(file)
    #im_orig = im_orig[:32, :32, :]  # 32x32x3
    
    im = im_orig.reshape(1, -1)[0]
    im = [bin(a)[2:].zfill(8) for a in im]
    im = ''.join(im) # This is now a string of bits
    
    return im


def _convert_to_bytes_decimal(data):
    n = len(data) // 8
    dim = int(np.sqrt(n / 3))
    
    data_vector = []
    for i in range(n):
        d = str(data[i*8:(i+1)*8])
        d = int(d, 2)
        # print(d)
        data_vector.append(d)
    
    data_vector = np.array(data_vector, dtype='uint8')
    # Truncate if needed
    data_vector = data_vector[:dim * dim * 3]

    # Now reshape
    data_vector = data_vector.reshape(dim, dim, 3)
    
    return data_vector


def plot_bitmaps(data1, data2):
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
    
    
def run_simulation(file_name, codeword_size, h, constellation, k_constellation, sigmas, crc_polynomial, crc_length, n_pilot):
    alphabet = create_constellation(constellation=constellation, M=int(2 ** k_constellation))
    data = read_bitmap(file_name)

    plot_bitmaps(data, data)
    
    df_output = pd.DataFrame(columns=['noise_power', 'Rx_EbN0', 'Avg_SER', 'Avg_BER', 'BLER', 'Channel_Error'])
    for sigma in sigmas:
        SER_i, BER_i, BLER_i, _, Rx_EbN0_i, data_received, chan_err_comp = transmit_receive(data, codeword_size, alphabet, h, k_constellation, sigma ** 2, crc_polynomial, crc_length, n_pilot)
        df_output_ = pd.DataFrame(data={'Avg_SER': SER_i})
        df_output_['Avg_BER'] = BER_i
        df_output_['noise_power'] = sigma ** 2
        df_output_['BLER'] = BLER_i
        df_output_['Rx_EbN0'] = Rx_EbN0_i
        df_output_['Channel_Error'] = chan_err_comp
        
        plot_bitmaps(data, data_received)
        
        if df_output.shape[0] == 0:
            df_output = df_output_.copy()
        else:
            df_output = pd.concat([df_output_, df_output], axis=0, ignore_index=True)
        
        print(f'Block error rate: {BLER_i:.4f}.')
        print('Average symbol error rate: {:.4f}.'.format(np.mean(SER_i)))
        print('Average bit error rate: {:.4f}.'.format(np.mean(BER_i)))

    return df_output


# 1) Create a channel
H = create_rayleigh_channel(N_r=N_r, N_t=N_t) # length=(crc_length + codeword_size) // 2)

# 2) Run the simulation on this channel
df_output = run_simulation(file_name, codeword_size, H, constellation, 
                           k_constellation, sigmas, crc_polynomial, crc_length,
                           n_pilot)
df_output.to_csv('output.csv', index=False)

# 3) Generate plot
xlabel = 'BLER'
ylabel = 'Rx_EbN0'

generate_plot(df=df_output, xlabel=xlabel, ylabel=ylabel)
