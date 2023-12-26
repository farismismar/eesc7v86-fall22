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
import matplotlib.pyplot as plt

file_name = 'faris.bmp'

# System parameters
compress_channel = True
constellation = 'QAM'
M_constellation = 64
seed = 7
codeword_size = 16 # bits
n_pilot = 20
N_r = 16
N_t = 16

# Note that the polynomial size is equal to the codeword size.
crc_polynomial = 0b0001_0010_0000_0010
crc_length = 2 # bits

sigmas = np.sqrt(np.logspace(-4, -1, num=6)) # square root of noise power
##################

plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = "14"

np_random = np.random.RandomState(seed=seed)
k_constellation = int(np.log2(M_constellation))

def create_constellation(constellation, M):
    if (constellation == 'QPSK'):
        return create_constellation_psk(M)
    elif (constellation == 'QAM'):
        return create_constellation_qam(M)
    else:
        return None
     
# Constellation based on Gray code
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
    crc = 0
    for position, value in enumerate(bin(crc_polynomial)[2:]):
        if value == '1':
            crc = crc ^ int(x_bits_orig[position])
    crc = bin(crc)[2:]
    
    if len(crc) > crc_length:
        raise ValueError("Check CRC length parameter.")
    crc = crc.zfill(crc_length)
    
    return crc


def create_rayleigh_channel(N_r, N_t, G=1):
    # Rayleigh fading with G being the large scale fading
    H = np.sqrt(G / 2) * (np_random.normal(0, 1, size=(N_r, N_t)) + \
                          1j * np_random.normal(0, 1, size=(N_r, N_t)))
    # Normalize the channel
    H /= np.linalg.norm(H, ord='fro') / np.sqrt(N_t*N_r)
    
    return H


def ML_detect_symbol(x_sym_hat, alphabet):
    df_information = pd.DataFrame()
    x_sym_hat_flat = x_sym_hat.flatten()
    
    for s in range(x_sym_hat_flat.shape[0]):
        x_hat = x_sym_hat_flat[s]
        # This function returns argmin |x - s_m| based on AWGN ML detection
        # for any arbitrary constellation denoted by the alphabet
        distances = alphabet['x'].apply(lambda x: abs(x - x_hat) ** 2)
        m_star = distances.idxmin(axis=0)
        
        df_i = pd.DataFrame(data={'m': m_star,
                                  'x': alphabet.loc[alphabet['m'] == m_star, 'x'],
                                  'I': alphabet.loc[alphabet['m'] == m_star, 'I'],
                                  'Q': alphabet.loc[alphabet['m'] == m_star, 'Q']})
        
        df_information = pd.concat([df_information, df_i], axis=0, ignore_index=True)
    
    information = df_information['m'].values.reshape(x_sym_hat.shape)
    
    # Now simply compute other elements.
    symbols = df_information['x'].values.reshape(x_sym_hat.shape)
    bits_i = df_information['I'].values
    bits_q = df_information['Q'].values
    
    bits = []
    for s in range(x_sym_hat_flat.shape[0]):
        bits.append(f'{bits_i[s]}{bits_q[s]}')
        
    bits = np.array(bits).reshape(x_sym_hat.shape)
    bits_i = bits_i.reshape(x_sym_hat.shape)
    bits_q = bits_q.reshape(x_sym_hat.shape)
    
    return information, symbols, [bits_i, bits_q], bits


def _equalize_channel(H):
    global N_t, N_r
    
    # ZF equalization
    W = np.linalg.pinv(H)
    
    assert(W.shape == (N_r, N_t))
    
    return W

        
def _estimate_channel(X_p, Y_p, noise_power, random_state=None):
    # This is for least square (LS) estimation
    # and the linear minimum mean squared error (L-MMSE):
    N_t, _ = X_p.shape
    
    if not np.allclose(X_p@X_p.T, np.eye(N_t)):
        raise ValueError("The training sequence is not semi-unitary.  Cannot estimate the channel.")
    
    # This is least square (LS) estimation
    H_hat_ls = Y_p@X_p.conjugate().T
    
    # This is the L-MMSE estimation:
    H_hat = H_hat_ls@np.linalg.inv(np.eye(N_t) * (1 + noise_power))
    
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
    SERs = []
    BERs = []
    block_error = 0
    
    N_r, N_t = H.shape
    
    Df = 15e3 # subcarrier in Hz
    tti = 1e-3 # in seconds
    
    # Effective codeword size, must coincide with integer number of symbols
    codeword_size = int(np.ceil(codeword_size / k) * k)
    
    bit_rate = codeword_size / tti * N_t
    print(f'Transmission maximum rate = {bit_rate:.2f} bps')
    
    # Find the correct number of subcarriers required for this bit rate
    # assuming 1:1 code rate.
    Nsc = np.ceil(bit_rate / (k * Df))    # Number of OFDM subcarriers
    B = Nsc * Df # Transmit bandwidth
    print(f'Transmission BW = {B:.2f} Hz')
    
    Es = np.linalg.norm(alphabet['x'], ord=2) ** 2 / alphabet.shape[0]
    Tx_SNR = 10*np.log10(Es / (N_t * noise_power))
    
    N0 = noise_power / B
    Tx_EbN0 = 10 * np.log10((Es/(N_t * k)) / N0)
    print(f'SNR at the transmitter (per stream): {Tx_SNR:.4f} dB')
    print(f'EbN0 at the transmitter (per stream): {Tx_EbN0:.4f} dB')
    
    b = len(data)
    n_transmissions = int(np.ceil(b / (codeword_size * N_t)))
    
    # Find G from the normalized channel
    G = np.trace(H.conjugate().T@H) / (N_r * N_t)
    
    x_info_complete = bits_to_symbols(data, alphabet, k)
    
    data_rx = []    
    print(f'Transmitting a total of {b} bits.')
    Rx_EbN0 = []
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

        x_sym_crc = x_sym_crc.reshape(-1, N_t).T
        
        # Additive noise
        n = np_random.normal(0, scale=noise_power/np.sqrt(2), size=(N_r, n_pilot)) + \
            1j * np_random.normal(0, scale=noise_power/np.sqrt(2), size=(N_r, n_pilot))
        
        # Debug purposes
        if perfect_csi:
            n = np.zeros((N_r, n_pilot))
            H = np.eye(N_t)
            
        # Channel
        Y = H@x_sym_crc + n[:, :x_sym_crc.shape[1]]
        
        # Pilot contribution (known sequence)
        # Generate pilot
        P = _generate_pilot(N_r, N_t, n_pilot, random_state=np_random)
        
        # Channel
        T = H@P + n
    
        # Estimate the channel
        H_hat = _estimate_channel(P, T, noise_power, random_state=np_random)
        
        error_vector = _vec(H) - _vec(H_hat)
        
        channel_estimation_mse = np.linalg.norm(error_vector, 2) ** 2 / (N_t * N_r)
        print(f'Channel estimation MSE: {channel_estimation_mse:.4f}')
        
        # Channel equalization using ZF
        W = _equalize_channel(H_hat)
        
        # The optimal equalizer should fulfill W* H = I_{N_t}]
        x_sym_crc_hat = W.conjugate().transpose() @ Y
        
        # crc_sym_hat = _vec(x_sym_crc_hat)[-effective_crc_length_symbols:] # only transmitted CRC
        x_sym_hat = _vec(x_sym_crc_hat)[:-effective_crc_length_symbols] # payload including padding
        
        # Remove the padding, which is essentially defined by the last data not on N_t boundary
        if pad_length > 0:
            x_sym_hat = x_sym_hat[:-pad_length] # no CRC and no padding.
        
        # Detection of symbols (symbol star is the centroid of the constellation)        
        x_info_hat, _, _, x_bits_hat = ML_detect_symbol(x_sym_hat, alphabet)
        x_bits_hat = ''.join(x_bits_hat)
        
        # Compute the EbN0 at the receiver
        received_noise_power = noise_power * np.linalg.norm(W, 'fro') ** 2 # equalization enhanced noise
        N0 = received_noise_power / B
        Rx_EbN0.append(10 * np.log10((Es/k) / N0))
        
        # Compute CRC on the received frame
        crc_comp = compute_crc(x_bits_hat, crc_polynomial, crc_length)
        
        if int(crc) != int(crc_comp):
            block_error += 1
            
        ########################################################
        # Error statistics
        symbol_error = 1 - np.mean(x_info_hat == x_info)
        SERs.append(symbol_error)
        
        x_hat_b = symbols_to_bits(x_info_hat, k, alphabet, is_complex=False)
        x_hat_b_i, x_hat_b_q, _ = bits_to_baseband(x_hat_b, alphabet, k)

        ber_i = 1 - np.mean(x_hat_b_i == x_b_i)
        ber_q = 1 - np.mean(x_hat_b_q == x_b_q)
        
        # System should preserve the number of bits.
        assert(len(x_bits_orig) == len(x_hat_b))
        
        data_rx.append(x_hat_b)
        ber = np.mean([ber_i, ber_q])
        BERs.append(ber)
        # for

    total_transmitted_bits = N_t * codeword_size * n_transmissions
    print(f"Total transmitted bits: {total_transmitted_bits} bits.")

    BLER = block_error / n_transmissions

    # Now extract from every transmission 
    data_rx_ = ''.join(data_rx)
    
    return SERs, BERs, BLER, Tx_EbN0, Rx_EbN0, data_rx_


def read_bitmap(file):
    # This is a 32x32 pixel image
    im_orig = plt.imread(file)
    
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
    
    df_output = pd.DataFrame(columns=['noise_power', 'Rx_EbN0', 'Avg_SER', 'Avg_BER', 'BLER'])
    for sigma in sigmas:
        SER_i, BER_i, BLER_i, _, Rx_EbN0_i, data_received = \
            transmit_receive(data, codeword_size, alphabet, h, k_constellation, sigma ** 2, crc_polynomial, crc_length, n_pilot)
        df_output_ = pd.DataFrame(data={'Avg_SER': SER_i})
        df_output_['Avg_BER'] = BER_i
        df_output_['noise_power'] = sigma ** 2
        df_output_['BLER'] = BLER_i
        df_output_['Rx_EbN0'] = Rx_EbN0_i
        
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
H = create_rayleigh_channel(N_r=N_r, N_t=N_t, G=1)

# 2) Run the simulation on this channel
df_output = run_simulation(file_name, codeword_size, H, constellation, 
                           k_constellation, sigmas, crc_polynomial, crc_length,
                           n_pilot)
df_output.to_csv('output.csv', index=False)

# 3) Generate plot
xlabel = 'BLER'
ylabel = 'Rx_EbN0'

generate_plot(df=df_output, xlabel=xlabel, ylabel=ylabel)
