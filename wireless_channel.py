# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:43:23 2021

@author: Faris Mismar
"""

import os

import random
import numpy as np
import pandas as pd

import time
import pdb

from sklearn.linear_model import LinearRegression
#from scipy import stats

from sklearn.cluster import KMeans

from dnn import DeepNeuralNetworkClassifier
from ensemble import EnsembleClassifier
from utils import Utils

import matplotlib
import matplotlib.pyplot as plt

class MachineLearningWireless:
    def __init__(self, random_state=None, prefer_gpu=True):
    
        self.prefer_gpu = prefer_gpu
        
        self.H = None # Channel
        self.F = None # Precoder
        self.b = None # Quant resolution
        
        self._reset_random_state(random_state)
        
        
    def _reset_random_state(self, seed):
                
        if seed is None:
            self.seed = np.random.mtrand._rand
        else:
            self.seed = seed
        
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        random.seed(self.seed) # gives a DeprecationWarning
        self.np_random_state = np.random.RandomState(self.seed)
        
    
    def create_channel(self, N_t, N_r, noise_variance=0, large_scale_gain=1, fading='Rayleigh', modulation='BPSK'):
        self.N_t = N_t
        self.N_r = N_r
        
        self.noise_variance = noise_variance
        self.G = large_scale_gain
        self.fading = fading
        self.modulation = modulation
        np_random_state = self.np_random_state
        
        precision = 6
        
        # This will hold the transmitted symbols
        constellation = pd.DataFrame(columns=['m', 'I', 'Q'])
        
        # TODO: is this necessary?
        E_g = 1 # Pulse with unity energy
        
        # Constructing the constellation following Proakis 
        if modulation == 'BPSK':
            True
        elif modulation == 'QPSK':
            M = 4
            for m in range(M):
                constellation = constellation.append({'m': m,
                        'I': np.sqrt(E_g / 2) * np.cos(2*np.pi/M*m + np.pi/M),
                        'Q': np.sqrt(E_g / 2) * np.sin(2*np.pi/M*m + np.pi/M)},
                        ignore_index=True)
        else:
            print('Only modulations supported are BPSK and QPSK.')
            return None        
        
        # Rayleigh fading channel (assume within coherence time)
        if fading == 'Rayleigh':
            H = 1. / np.sqrt(2) * (np_random_state.normal(loc=0, scale=1, size=N_t*N_r) + \
                                   1j * np_random_state.normal(loc=0, scale=1, size=N_t*N_r))
            H = H.reshape(N_r, N_t)
        
        # Normalize the channel such that the sq Frobenius norm is N_t N_r        
        H /= np.linalg.norm(H, ord='fro') / np.sqrt(N_t*N_r)
        self.H = H
        
        # Precoder F, whose Frobenius norm squared is N_t
        self.F = np.eye(N_t) # We are not using F everywhere it should be really.
        
        # Normalize the transmitted symbols
        signal_power = np.mean(constellation['I'] ** 2 + constellation['Q'] ** 2)
        constellation.iloc[:, 1:] /= np.sqrt(signal_power)     
        constellation = constellation.round(precision)
        
        constellation.loc[:, 'm'] = constellation.loc[:, 'm'].astype(int)
        constellation['bits'] = np.sign(constellation['I']).apply(lambda x: str(max(0, int(x)))) + \
            np.sign(constellation['Q']).apply(lambda x: str(max(0, int(x))))
        
        return constellation, H

    
    def set_simulation_duration(self, N_symbols=100):
        # Simulation duration in number of symbols
        self.N_symbols = N_symbols
        return self


    def construct_data(self, constellation):
        N_t = self.N_t
        N_r = self.N_r
        H = self.H 
        
        if H is None:
            print('Did you run create_channel first?')
            return None
        
        F = self.F
        
        np_random_state = self.np_random_state
        N_symbols = self.N_symbols
        
        noise_variance = self.noise_variance

        df = pd.DataFrame(columns=['m', 'x_I', 'x_Q', 'HFx_I', 'HFx_Q', 'y_I', 'y_Q', 'n_I', 'n_Q'])
        
        for sym in range(N_symbols):
            df_N_t = constellation.sample(n=N_t, replace=True, random_state=np_random_state)
            
            m = df_N_t['m']
            x_I = df_N_t['I']
            x_Q = df_N_t['Q']
            
            x = x_I + 1j * x_Q
            
            power_x = np.mean(abs(x_I) ** 2 + abs(x_Q) ** 2) # Mean symbol power
            x /= np.sqrt(power_x) # Normalize power dividing by the sqrt(energy)
 
            # Circular Gaussian with a total variance of noise_variance
            # Noise power in Watts
            n_I = np_random_state.normal(loc=0., scale=np.sqrt(noise_variance / 2.), 
                                   size=N_r)
            n_Q = np_random_state.normal(loc=0., scale=np.sqrt(noise_variance / 2.), 
                                   size=N_r)
            n = n_I + 1j * n_Q
            
            # Channel effect
            G = self.G # large scale channel effect
            HFx = np.sqrt(G) * np.matmul(np.matmul(H, F), x)
            
            # Distribute power equally over transmit antennas
            normalized_HFx = np.sqrt(power_x/N_t) * HFx
            y = normalized_HFx + n
            
            x_I = np.real(x)
            x_Q = np.imag(x)
            HFx_I = np.real(normalized_HFx)
            HFx_Q = np.imag(normalized_HFx)
            y_I = np.real(y)
            y_Q = np.imag(y)
            m = np.array(m)
            
            to_append = pd.Series([m, x_I, x_Q, HFx_I, HFx_Q, y_I, y_Q, n_I, n_Q], index=df.columns)
            df = df.append(to_append, ignore_index=True)
        
        return df


    def wrangle_data(self, df):
        N_t = self.N_t
        N_r = self.N_r
        
        ### Data wrangling: apply knowledge to enhance data for the learning
        col_names = df.columns
        cols_suffixes_x = [f'_{i + 1}' for i in range(N_t)]
        cols_suffixes_y = [f'_{i + 1}' for i in range(N_r)]
        df_wrangled = pd.DataFrame()
        
        for col in col_names:
            if (col[0] == 'x') or (col[0] == 'm'):
                cols_suffixes = cols_suffixes_x
            elif (col[0] == 'y') or (col[0] == 'n'):
                cols_suffixes = cols_suffixes_y
            else:
                print('WARNING: unknown column.  Skipping.')
                continue
            new_cols = [f'{col}{c}' for c in cols_suffixes]
            df_wrangled_c = pd.DataFrame(df[col].tolist(), columns=new_cols)
            df_wrangled = pd.concat([df_wrangled, df_wrangled_c], axis=1, 
                                    ignore_index=False)

        X_true = df_wrangled.filter(regex='[mx]_')
        n = df_wrangled.filter(regex='n_')
        
        return df_wrangled, X_true, n


    def estimate_channel(self, df_wrangled, N_pilot, noise_power, estimator='least_squares'):       
        # Note that Tr(H_hat) == N_t -- Check MIMO
        
        X = df_wrangled.filter(regex='x_I').values + 1j * df_wrangled.filter(regex='x_Q').values
        Y = df_wrangled.filter(regex='y_I').values + 1j * df_wrangled.filter(regex='y_Q').values
        
        # Use a pilot to estimate the channel
        X = X[:N_pilot]
        Y = Y[:N_pilot]
        
        H_hat_LS = np.matmul(np.linalg.pinv(X), Y).T
        
        if estimator == 'ideal':
            H_hat = self.H
        elif estimator == 'least_squares':
            H_hat = H_hat_LS
        elif estimator == 'lmmse':
            # TODO: find the LMMSE estimate
            H_hat = noise_power * np.eye(N_t) * H_hat_LS
        else:
            print('WARNING: unknown estimation algorithm.  Defaulting to least squares.')
            H_hat = H_hat_LS
            
        return H_hat
    
    
    def estimate_channel_learning(self, df_wrangled, N_pilot, how='linear_regression'):
        if how != 'linear_regression':
            return None, np.nan
        
        N_t = self.N_t
        N_r = self.N_r
        
        df_pilot = df_wrangled.iloc[:N_pilot, :]
        
        ### Learning: use linear regression ###########################################
        reg = LinearRegression(n_jobs=-1, fit_intercept=True)
        X = df_pilot.filter(regex='x')
        #print(X.columns) # to find the order and hence understand the coefficients hij
        
        H_hat = np.zeros((1, N_t))
        
        # We can use the in-phase component only for estimation.  Why?
        cols = df_pilot.filter(regex='y_I').columns
        
        for qty in cols:
            y = df_pilot[qty]
            # These are manual derivations to validate
            # For 1x1
            # y_I_1 = h11_I x_I_1 - h11_Q x_Q_1
            
            # For 2x2
            # y_I_1 = (h11_I x_I_1 + h12_I x_I_2) - h11_Q x_Q_1 - h12_Q x_Q_2 + n_I_1
            # y_Q_1 = (h11_Q x_I_1 + h12_Q x_I_2) + h11_I x_Q_1 + h12_I x_Q_2 + n_Q_1
            
            reg.fit(X, y)
            rsq = reg.score(X, y)
            print(f'Fitting score: {rsq:.2f}')
            
            h_hat_row = np.complex128(reg.coef_)
            if 'I' in qty:
                h_hat_row[N_t:] *= -1j # this is for in-phase
            else:
                # this is for quadrature
                print('WARNING: We use the in-phase component only.  Skipping.')
                continue
            
            # As long as the order is x in-phase then x quadrature, this line works
            # for rank-2.  How is it for rank 4?
            if 'I' in qty:
                h_hat_row = h_hat_row.reshape(2, -1) # for I/Q
                h_hat_row = h_hat_row.sum(axis=0)
            else:
                h_hat_row = h_hat_row[:(N_t//2)] + h_hat_row[(N_t//2):]
            H_hat = np.r_[H_hat, [h_hat_row]] # append a row
        H_hat = H_hat[1:] # remove the placeholder
        
        return H_hat


    def equalize(self, estimated_channel, symbols, noise_process, equalizer='MMSE'):
        # TODO: what is the difference between a combiner and an equalizer?
        # Where does MRC fit into this?
        
        # W is N_t x N_r        
        G = self.G
        noise_variance = self.noise_variance # note this should equal np.var(noise_process)
        N_r = self.N_r
        N_t = self.N_t
        
        H = estimated_channel
        HH = H.conj().T

        if (equalizer == 'ZF'):
            W = 1 / np.sqrt(G) * np.linalg.pinv(H) # fast short-hand notation
            #W4 = np.matmul(HH, np.linalg.pinv(np.matmul(H, HH)))
        elif (equalizer == 'MMSE'):
            W = 1 / np.sqrt(G) * np.matmul(HH, np.linalg.pinv(np.matmul(H, HH) + noise_variance * np.eye(N_r)))
        else:
            W = None

        # Compute the enhanced noise process due to equalization
        df_noise = pd.DataFrame()
        for stream in range(1, N_r + 1):
            n = noise_process.filter(regex=f'{stream}')
            n = n.filter(regex='I').values + 1j * n.filter(regex='Q')
            n.columns = [f'n_{stream}']
            df_noise = pd.concat([df_noise, n], axis=1)

        v = [] 
        for idx, row in df_noise.iterrows():
            n_i = row.values.reshape(N_r, 1) # column vector
            # v = W* n (enhanced noise)
            v_i = np.linalg.multi_dot([W, n_i])
            v.append(v_i)
        
        df_1 = pd.DataFrame([list(np.real(x)) for x in v])
        df_1.columns = [f'v_I_{stream}' for stream in range(1, N_t + 1)]
        df_1 = df_1.applymap(lambda x: x[0])
        
        df_2 = pd.DataFrame([list(np.imag(x)) for x in v])
        df_2.columns = [f'v_Q_{stream}' for stream in range(1, N_t + 1)]
        df_2 = df_2.applymap(lambda x: x[0])
        
        df_enhanced_noise = pd.concat([df_1, df_2], axis=1)
        
        # Now do the same thing for R        
        # Can there be a better construct?
        df_symbols = pd.DataFrame()
        for stream in range(1, N_r + 1):
            y = symbols.filter(regex=f'_{stream}').filter(regex='y_')
            y = y.filter(regex='I').values + 1j * y.filter(regex='Q')
            y.columns = [f'y_{stream}']
            df_symbols = pd.concat([df_symbols, y], axis=1)
            
        R = [] 
        for idx, row in df_symbols.iterrows():
            y_i = row.values.reshape(N_r, 1)
            # R = W* x Y
            r_i = np.linalg.multi_dot([W, y_i])
            R.append(r_i)
    
        df_1 = pd.DataFrame([list(np.real(x)) for x in R])
        df_1.columns = [f'r_I_{stream}' for stream in range(1, N_t + 1)]
        df_1 = df_1.applymap(lambda x: x[0])
        
        df_2 = pd.DataFrame([list(np.imag(x)) for x in R])
        df_2.columns = [f'r_Q_{stream}' for stream in range(1, N_t + 1)]
        df_2 = df_2.applymap(lambda x: x[0])
        
        df = pd.concat([symbols, df_1, df_2, df_enhanced_noise], axis=1)
        
        return W, df, df_enhanced_noise
    
    
    def quantize(self, df, b=1):
        self.b = b
        # Just append X_hat
        X_hat = df.filter(regex='r_')
         
        new_columns = [c.replace('r', 'x_hat') for c in X_hat.columns]
        X_hat.columns = new_columns
        
        if b == np.inf:
            df = pd.concat([df, X_hat], axis=1)
        elif b == 1:
            X_hat = X_hat.applymap(lambda x: np.sign(x))
            df = pd.concat([df, X_hat], axis=1)
        else:
            print('WARNING: We only support 1-bit quantization.  Skipping.')
                
        return df
    
    
    def unsupervised_detection(self, df, constellation):
        seed = self.seed
        
        M = constellation.shape[0]
        
        # Intialize k-means centroid location deterministcally as a constellation
        kmeans = KMeans(n_clusters=M, init=constellation[['I', 'Q']], n_init=1, 
                        random_state=seed)

        kmeans.fit(constellation[['I', 'Q']])    

        df_predictions = pd.DataFrame()
        for streams in range(1, N_t + 1):
            df_ = df.filter(regex=f'{streams}')
            X_ = pd.DataFrame(data={'I': df_.filter(regex='x_hat_I').squeeze(),
                                    'Q': df_.filter(regex='x_hat_Q').squeeze()}, index=df_.index)
 
            m_true = df_.filter(regex=f'm_{streams}')
            m_predict = kmeans.predict(X_)
            
            df_predictions_s = pd.DataFrame(data={'m_pred_unsup': m_predict})
            df_predictions_s['stream'] = streams
            df_predictions_s['m_true'] = m_true
            df_predictions_s = df_predictions_s.join(X_) # add the estimated symbols
            
            df_predictions = pd.concat([df_predictions, df_predictions_s], axis=0)
    
        df_predictions['match'] = (df_predictions['m_true'] == df_predictions['m_pred_unsup']).astype(int)
    
        df_pred_branch = df_predictions.groupby(['stream']).mean()['match'].reset_index()
        
        self.plot_clusters(df_predictions[['stream', 'm_pred_unsup', 'I', 'Q']], constellation)
    
        # This is the correct way to calculate accuracy.  Why?
        weighted_average_accuracy = df_predictions['match'].mean()
        
        # This should match the weighted accuracy if streams are equally likely
        arithmetic_average_accuracy = df_pred_branch['match'].mean()
        
        return df_predictions, weighted_average_accuracy
    
    
    # Do some plotting
    def plot_clusters(self, X, centroids):
        M = X['m_pred_unsup'].max()
        limit = np.sqrt(M)
        
        plt.rcParams['font.family'] = "Arial"
        plt.rcParams['font.size'] = "14"
        
        fig = plt.figure(figsize=(8,5))
        for s in sorted(X['stream'].unique()):
            X_ = X[X['stream'] == s]
            
            plt.scatter(X_['I'], X_['Q'], c=X_['m_pred_unsup'], cmap='RdYlGn', alpha=0.5)
            plt.scatter(centroids['I'], centroids['Q'], c=centroids['m'], cmap='RdYlGn', alpha=0.5, edgecolor='k')
            plt.grid(which='both')
            plt.xlabel('$I$')
            plt.ylabel('$Q$')
            plt.title(f'I/Q for Stream {s}')
            plt.xlim(-limit - 1,limit + 1)
            plt.ylim(-limit - 1,limit + 1)
            plt.tight_layout()
            plt.close(fig)
            plt.show()


    def compute_receive_snr(self, signal_process, noise_process, dB=False):
        N_r = self.N_r
        
        # Compute average SNR per receive branch N_r
        # before equalization
        df_SNRs = pd.DataFrame()
        for stream in range(1, N_r + 1):
            # If y = HFx + n
            # Then SNR = power of HFx / power of n
            sig = signal_process.filter(regex=f'_{stream}')
            noise = noise_process.filter(regex=f'_{stream}')

            received_power = sig.iloc[:, 0] ** 2 + sig.iloc[:, 1] ** 2 # symbol power I/Q
            noise_power = noise.iloc[:, 0] ** 2 + noise.iloc[:, 1] ** 2 # symbol power I/Q
            signal_power = received_power - noise_power
            
            snr = signal_power / noise_power
            df_SNR_j = pd.DataFrame(data={f'SNR_{stream}': snr}, index=received_power.index)
            df_SNRs = pd.concat([df_SNRs, df_SNR_j], axis=1)
        
        df_average = df_SNRs.mean(axis=0)
        
        if dB == True:
            df_SNRs = df_SNRs.applymap(lambda x: 10 * np.log10(x))
            df_average = df_average.apply(lambda x: 10 * np.log10(x))
            
        return df_average, df_SNRs
    
    
    def mse(self, true, estimate):
        error_vector = true.reshape(-1, 1) - estimate.reshape(-1, 1)
        N_pilot = error_vector.shape[0]
        MSE = np.linalg.norm(np.abs(error_vector) ** 2, 2) / N_pilot
        return MSE
    

    def block_error(self, df, constellation, df_BERs, codeword_length):
        # Convert detected symbols to their respective bits per branch
        # Apply CRC on codewords
        # Let CRC generator polynomial be x^3 + x + 1 = 1011
        gen_poly = '1011'
        len_poly = len(gen_poly)
        
        # LTE specifies polynomials in 36.212
        modulation = self.modulation
        bits = constellation[['m', 'bits']]
        
        df_BLERs = pd.DataFrame()
        for stream in range(1, N_t + 1):
            true_s = df[f'm_{stream}'].to_frame()
            pred_s = df[f'm_hat_{stream}'].to_frame()
           
            true_bits_s = true_s.merge(bits, how='left', left_on=f'm_{stream}', right_on='m')
            pred_bits_s = pred_s.merge(bits, how='left', left_on=f'm_hat_{stream}', right_on='m')
        
            pred_bits = pred_bits_s['bits'].str.cat(sep='')
            true_bits = true_bits_s['bits'].str.cat(sep='')
           
            assert(len(pred_bits) == len(true_bits))
            
            true_codewords = []
            pred_codewords = []
            for idx in range(0, len(pred_bits), codeword_length):
                # append the CRC of 0's repeated for length of the generator poly
                codeword = pred_bits[idx:idx+codeword_length] + '0' * len_poly
                pred_codewords.append(codeword)
                
                codeword = true_bits[idx:idx+codeword_length] + '0' * len_poly
                true_codewords.append(codeword)
                
            true_remainder = [int(p, 2) % int(gen_poly, 2) for p in true_codewords]
            pred_remainder = [int(p, 2) % int(gen_poly, 2) for p in pred_codewords]
            
            # BLER is if the CRC fails
            BLER_s = [int(i != j) for i, j in zip(true_remainder, pred_remainder)]
            BLER_s = pd.DataFrame(data={f'BLER_{stream}': BLER_s})
                                        
            # Now append to the df_BLERs
            df_BLERs = pd.concat([df_BLERs, BLER_s], axis=1)
            
        df_average_BLER = df_BLERs.mean(axis=0)
        
        return df_average_BLER, df_BLERs      
        
            
    def symbol_error(self, df):
        modulation = self.modulation
        
        df_SERs = pd.DataFrame()
        for stream in range(1, N_t + 1):
            true_s = df.filter(regex=f'm_{stream}')
            pred_s = df.filter(regex=f'm_hat_{stream}')
            
            error = (true_s.values != pred_s).astype(int)
            error.columns = [f'SER_{stream}']
          
            df_SERs = pd.concat([df_SERs, error], axis=1)
        
        df_average_SER = df_SERs.mean(axis=0)
        
        if modulation == 'QPSK':
            new_columns = [c.replace('SER', 'BER') for c in df_SERs.columns]
            df_BERs = df_SERs.applymap(lambda x: 1 - np.sqrt(1 - x))
            df_BERs.columns = new_columns
            df_average_BER = df_BERs.mean(axis=0)
            
            return df_average_SER, df_SERs, df_average_BER, df_BERs
                    
        return df_average_SER, df_SERs, None, None
    
    
    # TODO: bad detection?
    def ml_detection(self, df, constellation):
        # This is the baseline symbol
        s_m = constellation['I'] + 1j * constellation['Q']
        
        # Construct p(y - x_m):
        # # Keep in mind that centroids are sorted from m = 0 to M - 1.
        # for m in constellation['m'].unique():
        #     # Basically r_I - I and r_Q - Q (or r - s_m) where s_m is mean
            
           
        #     # Following 4.2-15
        #     p_rI_given_m = stats.norm(constellation.loc[centroids['m'] == m, 'I'].values[0], noise_power / 2.)
        #     p_rQ_given_m = stats.norm(constellation.loc[centroids['m'] == m, 'Q'].values[0], noise_power / 2.)
                   
        for stream in range(1, N_t + 1):
            df_s = df.filter(regex=f'_{stream}')
        #     # Now compute p(x_hat | s_m) per branch
        #    # ###########################         
        #    #  m_hat = []
        #    #  for idx, row in df_s.iterrows():
        #    #      log_density_m = []
            
    
        #    #          # Now find the likelihood per I/Q given x_m
        #    #          log_dens_m_I = np.log(p_rI_given_m.pdf(mean_rI_m))
        #    #          log_dens_m_Q = np.log(p_rQ_given_m.pdf(mean_rQ_m))
        #    #          log_density = (log_dens_m_I + log_dens_m_Q)
        #    #          log_density_m.append(log_density)
                    
        #    #      log_density_m = np.array(log_density_m).round(4)
        #    #      m_hat.append(log_density_m.argmax()) # DigiComm p171
         
        #    #  df[f'm_hat_{stream}'] = m_hat 
        #    #  ######################
    
            m_hat = []
            for idx, row in df_s.iterrows():
                x_hat = row.filter(regex='x_hat_I').values + 1j * row.filter(regex='x_hat_Q').values
    
                diff = x_hat - s_m # distance from constellation at baseband
                distances = np.abs(diff)
                m_hat.append(distances.idxmin()) # from Digi Comm p171
                
            df[f'm_hat_{stream}'] = m_hat 
            
        return df
        
    def dnn_detection(self, df, train_size, n_epochs, batch_size):

        X = df.filter(regex='y_|n_|r_|v_|x_hat') # everything except m and true x
        y = df.filter(regex='m_[0-9]+') # clearly m_hat is not welcome :)
        
        dnn_classifier = DeepNeuralNetworkClassifier(
            seed=self.seed, prefer_gpu=self.prefer_gpu)
        
        df_predictions = pd.DataFrame()
        for streams in range(1, N_t + 1):
       
            X_ = X.filter(regex=f'{streams}')
            y_ = y.filter(regex=f'{streams}')
            
            X_test_, y_test_ = dnn_classifier.train(X_, y_,
                        train_size=train_size, n_epochs=n_epochs, 
                        batch_size=batch_size)
            acc, y_pred_ = dnn_classifier.test(X_test_, y_test_)
            
            df_predictions_s = pd.DataFrame(data={'stream': streams,
                                                  'm_true': y_test_,
                                                  'm_pred_dnn': y_pred_})
            df_predictions = pd.concat([df_predictions, df_predictions_s], axis=0)
    
        df_predictions['match'] = (df_predictions['m_true'] == df_predictions['m_pred_dnn']).astype(int)
    
        df_pred_branch = df_predictions.groupby(['stream']).mean()['match'].reset_index()

        # This is the correct way to calculate accuracy.  Why?
        weighted_average_accuracy = df_predictions['match'].mean()
        
        # This should match the weighted accuracy if streams are equally likely
        arithmetic_average_accuracy = df_pred_branch['match'].mean()
        
        return df_predictions, weighted_average_accuracy
    
        
    def ensemble_detection(self, df, train_size):

        X = df.filter(regex='y_|n_|r_|v_|x_hat') # everything except m and true x
        y = df.filter(regex='m_[0-9]+') # clearly m_hat is not welcome :)
        
        classifier = EnsembleClassifier(
            seed=self.seed, prefer_gpu=self.prefer_gpu, is_booster=False)
        
        df_predictions = pd.DataFrame()
        for streams in range(1, N_t + 1):
       
            X_ = X.filter(regex=f'{streams}')
            y_ = y.filter(regex=f'{streams}')
            
            X_test_, y_test_ = classifier.train(X_, y_, train_size=train_size)
            acc, y_pred_ = classifier.test(X_test_, y_test_)
            
            df_predictions_s = pd.DataFrame(data={'stream': streams,
                                                  'm_true': y_test_.values.ravel(),
                                                  'm_pred_rf': y_pred_})
            df_predictions = pd.concat([df_predictions, df_predictions_s], axis=0)
    
        df_predictions['match'] = (df_predictions['m_true'] == df_predictions['m_pred_rf']).astype(int)
    
        df_pred_branch = df_predictions.groupby(['stream']).mean()['match'].reset_index()

        # This is the correct way to calculate accuracy.  Why?
        weighted_average_accuracy = df_predictions['match'].mean()
        
        # This should match the weighted accuracy if streams are equally likely
        arithmetic_average_accuracy = df_pred_branch['match'].mean()
        
        return df_predictions, weighted_average_accuracy
    
## Simulation parameters ###########
noise_power = 1e-3 # in Watts
N_t = 4
N_r = 2
N_symbols = 256
N_pilot = 77 # for channel estimation
seed = 0
G = 1 # linear of large scale gain
myUtils = Utils(seed=seed)
####################################

mlw = MachineLearningWireless(random_state=seed, prefer_gpu=True)

mlw.set_simulation_duration(N_symbols=N_symbols)
constellation, H = mlw.create_channel(N_t=N_t, N_r=N_r, fading='Rayleigh', modulation='QPSK',
                   noise_variance=noise_power, 
                   large_scale_gain=G)
df = mlw.construct_data(constellation)
df, X_true, n = mlw.wrangle_data(df)

#######################################
# Question: how does the LS and linear regression perform for a given pilot?
mse_LS = []
mse_MachineLearning = []
mse_npilots = []

N_pilots = np.linspace(50, 250, 10).astype(int)
seeds = np.arange(15)
for s in seeds:
    for N_pilot in N_pilots:
        mlw._reset_random_state(seed=s)
        H_hat_ml = mlw.estimate_channel_learning(df, N_pilot, how='linear_regression')
        H_hat = mlw.estimate_channel(df, N_pilot, noise_power, estimator='least_squares')
        W, df_equalized, v = mlw.equalize(estimated_channel=H_hat_ml, 
                                          symbols=df, noise_process=n, 
                                          equalizer='MMSE')
        # average_receive_SNR, df_receive_SNR = mlw.compute_receive_snr(signal_process=df_equalized.filter(regex='r_'),
        #                     noise_process=df_equalized.filter(regex='v_'), dB=True)
        mse_LS.append(mlw.mse(H, H_hat))
        mse_MachineLearning.append(mlw.mse(H, H_hat_ml))
        mse_npilots.append(N_pilot)
        
df_summary = pd.DataFrame(data={'N_pilot': mse_npilots,
                                'MSE_LS': mse_LS,
                                'MSE_ML': mse_MachineLearning})

df_summary = df_summary.groupby('N_pilot').mean().reset_index()

myUtils.plotXY_comparison(x=df_summary['N_pilot'], y1=df_summary['MSE_LS'], y2=df_summary['MSE_ML'], 
                          xlabel='Pilot size [syms]', y1label='LS estimation', 
                          y2label='LinearReg estimation', 
                          logy=True,
                          title=f'MSE vs Pilot ({N_symbols} symbols)')

#######################################
# Question: what is the BER/BLER
df_quantized = mlw.quantize(df_equalized, b=np.inf)
df_detection = mlw.ml_detection(df=df_quantized, constellation=constellation)
average_receive_SNR, df_receive_SNR = mlw.compute_receive_snr(signal_process=df_detection.filter(regex='r_'),
                    noise_process=df_detection.filter(regex='v_'), dB=True)
average_SER, df_SERs, average_BER, df_BERs = mlw.symbol_error(df_detection)
average_BLER, df_BLERs = mlw.block_error(df_detection, constellation, df_BERs, codeword_length=10)

print('Average SNR post detection {} dB'.format(average_receive_SNR.values))
print('Average BER {}'.format(average_BER.values))
print('Average BLER {}'.format(average_BLER.values))

#######################################
# Question: how does training data size impact accuracy?
train_sizes = np.arange(0.1, 1, 0.1)
accuracy_ensemble = []
accuracy_dnn = []
for t in train_sizes:
    df, average_accuracy_ensemble = mlw.ensemble_detection(df_detection, train_size=t)
    accuracy_ensemble.append(average_accuracy_ensemble)

for t in train_sizes:
    # Good results show from train_size=0.3 and onwards.
    df, average_accuracy_dnn = mlw.dnn_detection(df_detection, train_size=t, n_epochs=128, batch_size=8)
    accuracy_dnn.append(average_accuracy_dnn)

myUtils.plotXY(x=train_sizes, y=accuracy_ensemble, xlabel='Pilot [%]', ylabel='Acc', 
              title='Ensemble')
myUtils.plotXY(x=train_sizes, y=accuracy_dnn, xlabel='Pilot [%]', ylabel='Acc', 
              title='DNN')
myUtils.plotXY_comparison(x=train_sizes, y1=accuracy_ensemble, y2=accuracy_dnn,
                          xlabel='Pilot [%]', y1label='Ensemble', y2label='DNN',
                          title='Ensemble vs DNN')

# Unsupervised: no pilot needed.  Only memory of the constellation.
df_unsup, average_acc_unsup = mlw.unsupervised_detection(df_detection, constellation)

#######################################
# Question: what is the CDF of the symbols like?
df_exploration = df_detection[['x_hat_I_1', 'x_hat_Q_1']]
df_1 = df_exploration['x_hat_I_1'].to_frame()
df_1['IQ'] = 'I'
df_1.columns = ['x_hat', 'IQ']

df_2 = df_exploration['x_hat_Q_1'].to_frame()
df_2['IQ'] = 'Q'
df_2.columns = ['x_hat', 'IQ']

df = pd.concat([df_1, df_2], axis=0)
myUtils.plot_cdfs(df, 'x_hat', 'IQ')

#######################################
# Question: what is the CDF of the SNR, SER, BER, and BLER like?
df_receive_SNR = df_receive_SNR.melt()
myUtils.plot_cdfs(df_receive_SNR, measure='value', category='variable')

df_SERs = df_SERs.melt()
myUtils.plot_cdfs(df_SERs, measure='value', category='variable')

df_BERs = df_BERs.melt()
myUtils.plot_cdfs(df_BERs, measure='value', category='variable')

df_BLERs = df_BLERs.melt()
myUtils.plot_cdfs(df_BLERs, measure='value', category='variable')