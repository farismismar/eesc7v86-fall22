	# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:19:40 2023

@author: farismismar
"""

import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt

file_name = 'faris.bmp'

# System parameters
seed = 7
codeword_size = 16 # bits
k_QPSK = 2
n_pilot = 4

plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = "14"

# Note that the polynomial size is equal to the codeword size.
crc_polynomial = 0b0001_0010_0000_0010
crc_length = 2 # bits

sigmas = np.sqrt(np.logspace(-1, 3)) # square root of noise power
##################

np_random = np.random.RandomState(seed=seed)

def create_constellation(M):
    centroids = pd.DataFrame(columns=['m', 'x_I', 'x_Q'])

    for m in np.arange(M):
        centroid_ = pd.DataFrame(data={'m': m,
                                       'x_I': np.sqrt(1 / 2) * np.cos(2*np.pi/M*m + np.pi/M),
                                       'x_Q': np.sqrt(1 / 2) * np.sin(2*np.pi/M*m + np.pi/M)}, index=[m])
        if centroids.shape[0] == 0:
            centroids = centroid_.copy()
        else:
            centroids = pd.concat([centroids, centroid_], ignore_index=True)
    
    # Normalize the transmitted symbols
    signal_power = np.mean(centroids['x_I'] ** 2 + centroids['x_Q'] ** 2)

    centroids.iloc[:, 1:] /= np.sqrt(signal_power)
    
    centroids.loc[:, 'm'] = centroids.loc[:, 'm'].astype(int)
    centroids.loc[:, 'x'] = centroids.loc[:, 'x_I'] + 1j * centroids.loc[:, 'x_Q']
    
    centroids['I'] = np.sign(centroids['x_I'])
    centroids['Q'] = np.sign(centroids['x_Q'])
    
    return centroids


def symbols_to_bits(x_sym, k, alphabet, is_complex=False):
    if is_complex == False:
        x_bits = ''
        for s in x_sym:
            i, q = alphabet.loc[alphabet['m'] == s, ['I', 'Q']].values[0]
            x_bits += '{}{}'.format(int(0.5*i + 0.5), int(0.5*q + 0.5)).zfill(k)
            
        return x_bits
    else:
        x_b_i = np.sign(np.real(x_sym))
        x_b_q = np.sign(np.imag(x_sym))

        x_bits = []
        for i, q in zip(x_b_i, x_b_q):
            x_bits.append('{}{}'.format(int(0.5*i + 0.5), int(0.5*q + 0.5)))
        return x_b_i, x_b_q, ''.join(x_bits)


def bits_to_symbols(x_b, alphabet, k):
    iter_ct = int(np.ceil(len(x_b) / k))

    df_ = alphabet[['m', 'I', 'Q']].replace(-1, 0).astype(int)
    df_['IQ'] = df_['I'].astype(str) + df_['Q'].astype(str)
    
    x_sym = []
    for i in range(iter_ct):
        bits_i = x_b[i*k:(i+1)*k] # read the next ith stride of k bits
        # Convert this to symbol from alphabet
        sym_i = df_.loc[df_['IQ'] == bits_i, 'm'].values[0].astype(int)
        # print(bits_i, sym_i)
        x_sym.append(sym_i)
        
    return np.array(x_sym)


def add_crc(x_bits_orig, crc_polynomial, crc_length):
    # Introduce CRC to x
    crc = 0
    for position, value in enumerate(bin(crc_polynomial)[2:]):
        if value == '1':
            crc = crc ^ int(x_bits_orig[position])
    
    crc = bin(crc)[2:].zfill(crc_length)
    x_bits = x_bits_orig + crc
    
    return x_bits, crc


def bits_to_baseband(x_bits):
    x_b_i = []
    x_b_q = []
    for i, q in zip(x_bits[::2], x_bits[1::2]):
        x_b_i.append(2*int(i) - 1)
        x_b_q.append(2*int(q) - 1)
    x_sym = (x_b_i + 1j * np.array(x_b_q)) / np.sqrt(2)
    
    return x_b_i, x_b_q, x_sym


def create_rayleigh_channel(length):
    G = 1 # large scale fading constant
    # Rayleigh fading
    h = np.sqrt(G / 2) * (np_random.normal(0, 1, size=length) + \
                          1j * np_random.normal(0, 1, size=length))

    return h


def ML_detect_symbol(x_sym_hat, alphabet):
    # This function returns argmin |x - s_m| based on AWGN ML detection
    df = pd.DataFrame(data={0: abs(x_sym_hat - alphabet.loc[alphabet['m'] == 0, 'x'].values[0]),
                            1: abs(x_sym_hat - alphabet.loc[alphabet['m'] == 1, 'x'].values[0]),
                            2: abs(x_sym_hat - alphabet.loc[alphabet['m'] == 2, 'x'].values[0]),
                            3: abs(x_sym_hat - alphabet.loc[alphabet['m'] == 3, 'x'].values[0])})
    
    return df.idxmin(axis=1).values


def _estimate_channel(x_pilot, y_pilot):
    h = 1
    return h


def transmit_receive(data, codeword_size, alphabet, h, k, noise_power, crc_polynomial, crc_length, n_pilot):
    SERs = []
    BERs = []
    block_error = 0
    
    Df = 15e3 # subcarrier in Hz
    tti = 1e-3 # in seconds
    bit_rate = codeword_size / tti
    print(f'Transmission rate = {bit_rate:.2f} bps')
    
    # Find the correct number of subcarriers required for this bit rate
    # assuming 1:1 code rate.
    Nsc = np.ceil(bit_rate / (k * Df))    # Number of subcarriers
    B = Nsc * Df # Transmit bandwidth
    print(f'Transmission BW = {B:.2f} Hz')
    
    Es = np.linalg.norm(alphabet['x'], ord=2) ** 2 / alphabet.shape[0]
    Tx_SNR = 10*np.log10(Es / noise_power)
    
    N0 = noise_power / B
    Tx_EbN0 = 10 * np.log10((Es/k) / N0)
    
    print(f'SNR at the transmitter: {Tx_SNR:.4f} dB')
    print(f'EbN0 at the transmitter: {Tx_EbN0:.4f} dB')
    b = len(data)
    n_transmissions = int(np.ceil(b / codeword_size))
    x_info_complete = bits_to_symbols(data, alphabet, k)
    
    data_rx = []
    
    print(f'Transmitting a total of {b} bits.')
    Rx_EbN0 = []
    for codeword in np.arange(n_transmissions):
        print(f'Transmitting codeword {codeword + 1}/{n_transmissions}')
        x_info = x_info_complete[codeword*(codeword_size // k):(codeword+1)*(codeword_size // k)] #np_random.choice(alphabet['m'], size=codeword_size // k)
        x_bits_orig = symbols_to_bits(x_info, k, alphabet)
        # Padding with zeros
        x_bits_orig = x_bits_orig.zfill(codeword_size)
        x_bits_crc, _ = add_crc(x_bits_orig, crc_polynomial, crc_length)
        
        assert(len(x_bits_crc) == crc_length + codeword_size)
        
        # Bits to I/Q
        x_b_i, x_b_q, x_sym_crc = bits_to_baseband(x_bits_crc)
        
        # Additive noise
        n = np_random.normal(0, scale=noise_power/np.sqrt(2), size=len(x_sym_crc)) + \
            1j * np_random.normal(0, scale=noise_power/np.sqrt(2), size=len(x_sym_crc)) 
    
        # Channel
        y = h*x_sym_crc + n
    
        # Extract the pilot
        x_p = x_sym_crc[:n_pilot]
        y_p = y[:n_pilot]
        
        # Channel estimation from the received signal
        h_hat = _estimate_channel(x_p, y_p)
        
        # Channel equalization using matched filter
        w = np.conjugate(h_hat) / (np.abs(h_hat) ** 2)
        x_sym_crc_hat = w * y
        
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
    
    return SERs, BERs, BLER, Tx_EbN0, Rx_EbN0, data_rx_


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
    
    
def run_simulation(file_name, codeword_size, h, k_QPSK, sigmas, crc_polynomial, crc_length, n_pilot):
    alphabet = create_constellation(M=int(2 ** k_QPSK))
    data = read_bitmap(file_name)

    plot_bitmaps(data, data)
    
    df_output = pd.DataFrame(columns=['noise_power', 'Rx_EbN0', 'Avg_SER', 'Avg_BER', 'BLER'])
    for sigma in sigmas:
        SER_i, BER_i, BLER_i, _, Rx_EbN0_i, data_received = transmit_receive(data, codeword_size, alphabet, h, k_QPSK, sigma ** 2, crc_polynomial, crc_length, n_pilot)
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
h = create_rayleigh_channel(length=(crc_length + codeword_size) // k_QPSK)

# 2) Run the simulation on this channel
df_output = run_simulation(file_name, codeword_size, h, k_QPSK, 
                           sigmas, crc_polynomial, crc_length, n_pilot)
df_output.to_csv('output.csv', index=False)

# 3) Generate plot
xlabel = 'noise_power'
ylabel = 'BLER'

generate_plot(df=df_output, xlabel=xlabel, ylabel=ylabel)
#############

# Check this
# https://hific.github.io/