#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:22:29 2024

@author: farismismar
"""

import numpy as np
import pandas as pd

import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import os

from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint

from autoencoder import Autoencoder
from environment import radio_environment
from DQNLearningAgent import DQNLearningAgent as DQNAgent
from QLearningAgent import QLearningAgent as TabularAgent

import tensorflow as tf

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

constellation = 'QAM'                    # QAM or PSK
M_constellation = 4                      # Square constellations only.
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

def rotation_channel(X, theta=0, SNR_dB=30, noise='shot'):
    num_samples = X.shape[0]
    SNR = mlw.linear(SNR_dB)

    # Rotate the symbols by an angle theta (counterclockwise)
    data = X * np.exp(1j * theta)

    real = np.real(data)
    imag = np.imag(data)

    # Now rotate real and imag
    # Noise power = 1/SNR because symbol power is 1.
    noise_power = 1. / SNR

    if noise == 'shot':
        lam = 1  # Electrons arrival rate = I Delta t / q (t in femtoseconds)---anything below 1 is physically meaningless
        noise_dc = random_state.poisson(lam=lam, size=num_samples)
        noise_ac = np.sqrt(noise_power / 2) * (random_state.normal(loc=0, scale=1, size=num_samples) + \
                                               1j * random_state.normal(loc=0, scale=1, size=num_samples))
        noise = noise_ac + noise_dc

        noise_real = real + np.real(noise)
        noise_imag = imag + np.imag(noise)
    else:
        if noise not in ['gaussian', 'johnson']:
            print("WARNING: Assuming Gaussian noise.")

        noise_real = real + random_state.normal(0, np.sqrt(noise_power / 2.), num_samples)
        noise_imag = imag + random_state.normal(0, np.sqrt(noise_power / 2.), num_samples)

    return np.c_[noise_real, noise_imag], np.c_[np.real(X), np.imag(X)]


def equalize_rotation_channel_CNN(mlw, pl, theta, SNR_dB, noise='shot', epochs=100, batch_size=64, training_ratio=0.95):
    global random_state
    global max_transmissions

    constellation = mlw.constellation
    M_constellation = mlw.M_constellation
    N_sc = mlw.N_sc

    if M_constellation != 4 and constellation not in ['QPSK', 'QAM']:
        raise ValueError("Error:  This is suitable only for QPSK.")

    alphabet = mlw.create_constellation(constellation=constellation, M=M_constellation)
    training_size = 100 * N_sc  # in symbols

    # These are pilots.
    mlw.N_sc = training_size
    X_information, X_clean, _, _, _, _ = mlw.generate_transmit_symbols(alphabet, n_transmissions=max_transmissions, mode='random')
    mlw.N_sc = N_sc

    # Extract one stream from X_clean only.
    X_clean = X_clean[:, 0]

    X_clean = X_clean.flatten() # ravel()
    X, y = rotation_channel(X_clean, theta=theta, SNR_dB=SNR_dB, noise=noise)  # SNR dB and a rotation of 7.5 deg.

    # Create train and test data
    train_idx = random_state.choice(np.arange(training_size), int(training_ratio * training_size), replace=False)
    test_idx = np.setdiff1d(np.arange(training_size), train_idx)

    X_train, y_train = X[train_idx, :], y[train_idx, :]
    X_test, y_test = X[test_idx, :], y[test_idx, :]

    # Plot the clean one
    constellation = pd.DataFrame({'x_I': np.real(X_clean),
                                  'x_Q': np.imag(X_clean)})
    pl.plot_constellation(constellation, filename='clean_symbols')

    # Then the noisy one
    constellation = pd.DataFrame({'x_I': X_test[:, 0],
                                  'x_Q': X_test[:, 1]})
    pl.plot_constellation(constellation, filename='noisy_symbols')

    # Reshape data for CNN
    X_train = X_train.reshape(-1, 2, 1)  # Reshaping to have 2 features (real and imag) per sample
    X_test = X_test.reshape(-1, 2, 1)

    # Build the CNN model
    model = mlw.create_cnn(learning_rate=1e-2)

    # Find the best model
    checkpoint = ModelCheckpoint('CNN_equalization_weights.keras',
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min',
                             verbose=1)
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
    pl.plot_keras_learning(history, filename='CNN_equalization_learning')

    model.load_weights('CNN_equalization_weights.keras')

    # Evaluate the model
    loss, mae = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test MAE: {mae}')

    # See if this is an equalized instance
    y_pred = model.predict(X_test)

    constellation = pd.DataFrame({'x_I': y_pred[:, 0],
                                  'x_Q': y_pred[:, 1]})
    pl.plot_constellation(constellation, filename='equalized_symbols')

    return X_test, y_test, y_pred



def tabular_reinforcement_learning(max_episodes_to_run, max_timesteps_per_episode, pl, plotting=False):
    global seed, prefer_gpu

    action_size = 3  # control command index up, nothing, index down

    env = radio_environment(action_size=action_size, min_reward=-1, max_reward=10, target=11,
                            max_step_count=max_timesteps_per_episode, seed=seed)

    agent = TabularAgent(state_size=env.observation_space.shape[0], action_size=action_size, seed=seed)

    successful = False
    episode_successful = []  # a list to save the optimal episodes

    optimal_episode = None
    optimal_reward = -np.inf

    print('Ep. | TS | Action | Power Control Index | Current SINR | Recv. SINR | Reward')
    print('-'*80)

    # These will keep track of the average Q values and losses experienced
    # per time step per episode.
    Q_values = []
    losses = []

    optimal_sinr_progress = []
    optimal_power_control_progress = []
    optimal_actions = []

    # Implement the Q-learning algorithm
    # Each episode z has a number of timesteps t
    for episode_index in 1 + np.arange(max_episodes_to_run):
        observation = env.reset()

        total_reward = 0
        done = False
        episode_loss = []
        episode_q = []

        # Read current envirnoment
        (current_sinr, _, pc_index) = observation
        action = agent.begin_episode(observation)

        actions = [action]
        sinr_progress = [current_sinr]  # needed for the SINR based on the episode.
        power_control_progress = [pc_index]

        for timestep_index in 1 + np.arange(max_timesteps_per_episode):
            # Take a step
            next_observation, reward, done, abort = env.step(action)
            (current_sinr, received_sinr, pc_index) = next_observation

            loss, q = agent.get_performance()
            episode_loss.append(loss)
            episode_q.append(q)

            # make next_state the new current state for the next frame.
            observation = next_observation
            total_reward += reward

            successful = done and (abort == False)

            # Let us know how we did.
            print(f'{episode_index}/{max_episodes_to_run} | {timestep_index} | {action} | {pc_index} | {current_sinr:.2f} dB | {received_sinr:.2f} dB | {total_reward:.2f} | ', end='')

            sinr_progress.append(received_sinr)
            power_control_progress.append(pc_index)

            if abort == True:
                print('Episode aborted.')
                break

            if successful:
                print('Episode successful.')
                break

            # Update for the next time step
            action = agent.act(observation, reward)
            actions.append(action)
            print()
        # End For

        # Everytime an episode ends, compute its statistics.
        loss_z = np.mean(episode_loss)
        q_z = np.mean(episode_q)

        losses.append(loss_z)
        Q_values.append(q_z)

        if successful:
            episode_successful.append(episode_index)
            if (total_reward >= optimal_reward):
                optimal_reward, optimal_episode = total_reward, episode_index
                optimal_sinr_progress = sinr_progress
                optimal_power_control_progress = power_control_progress
                optimal_actions = actions
            optimal = 'Episode {}/{} has generated the highest reward {:.2f} yet.'.format(optimal_episode, max_episodes_to_run, optimal_reward)
            print(optimal)
        else:
            reward = 0
        print('-'*80)
    # End For (episodes)

    # Plot the current episode
    if plotting:
        pl.plot_Q_learning_performance(losses, max_episodes_to_run, is_loss=True, filename='tabular_loss')
        pl.plot_Q_learning_performance(Q_values, max_episodes_to_run, is_loss=False, filename='tabular')

        if len(episode_successful) > 0:
            pl.plot_environment_measurements(optimal_sinr_progress, max_timesteps_per_episode + 1, measurement='SINR_dB', filename='tabular')
            pl.plot_agent_actions(optimal_actions, max_timesteps_per_episode, filename='tabular')

    if (len(episode_successful) == 0):
        print("Goal cannot be reached after {} episodes.  Try to increase maximum episodes.".format(max_episodes_to_run))

    return Q_values, losses, optimal_episode, optimal_reward, optimal_sinr_progress, optimal_power_control_progress


def deep_reinforcement_learning(max_episodes_to_run, max_timesteps_per_episode, pl, batch_size=16, plotting=False):
    global seed, prefer_gpu

    action_size = 3  # control command index up, nothing, index down

    env = radio_environment(action_size=action_size, min_reward=-1, max_reward=10, target=11,
                            max_step_count=max_timesteps_per_episode, seed=seed)

    agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=action_size, prefer_gpu=prefer_gpu, seed=seed)

    successful = False
    episode_successful = []  # a list to save the optimal episodes

    optimal_episode = None
    optimal_reward = -np.inf

    print('Ep. | TS | Action | Power Control Index | Current SINR | Recv. SINR | Reward')
    print('-'*80)

    # These will keep track of the average Q values and losses experienced
    # per time step per episode.
    Q_values = []
    losses = []

    optimal_sinr_progress = []
    optimal_power_control_progress = []
    optimal_actions = []

    # Implement the Q-learning algorithm
    # Each episode z has a number of timesteps t
    for episode_index in 1 + np.arange(max_episodes_to_run):
        observation = env.reset()

        total_reward = 0
        done = False
        episode_loss = []
        episode_q = []

        # Read current envirnoment
        (current_sinr, _, pc_index) = observation
        action = agent.begin_episode(observation)

        actions = [action]
        sinr_progress = [current_sinr]  # needed for the SINR based on the episode.
        power_control_progress = [pc_index]

        for timestep_index in 1 + np.arange(max_timesteps_per_episode):
            # Take a step
            next_observation, reward, done, abort = env.step(action)
            (current_sinr, received_sinr, pc_index) = next_observation

            # Remember the previous state, action, reward, and done
            agent.remember(observation, action, reward, next_observation, done)

            # Sample replay batch from memory
            sample_size = min(len(agent.memory), batch_size)

            # Learn control policy
            loss, q = agent.replay(sample_size)

            episode_loss.append(loss)
            episode_q.append(q)

            # make next_state the new current state for the next frame.
            observation = next_observation
            total_reward += reward

            successful = done and (abort == False)

            # Let us know how we did.
            print(f'{episode_index}/{max_episodes_to_run} | {timestep_index} | {action} | {pc_index} | {current_sinr:.2f} dB | {received_sinr:.2f} dB | {total_reward:.2f} | ', end='')

            sinr_progress.append(received_sinr)
            power_control_progress.append(pc_index)

            if abort == True:
                print('Episode aborted.')
                break

            if successful:
                print('Episode successful.')
                break

            # Update for the next time step
            action = agent.act(observation)
            actions.append(action)
            print()
        # End For

        # Every time an episode ends, compute its statistics.
        loss_z = np.mean(episode_loss)
        q_z = np.mean(episode_q)

        losses.append(loss_z)
        Q_values.append(q_z)

        if successful:
            episode_successful.append(episode_index)
            if (total_reward >= optimal_reward):
                optimal_reward, optimal_episode = total_reward, episode_index
                optimal_sinr_progress = sinr_progress
                optimal_power_control_progress = power_control_progress
                optimal_actions = actions
            optimal = 'Episode {}/{} has generated the highest reward {:.2f} yet.'.format(optimal_episode, max_episodes_to_run, optimal_reward)
            print(optimal)
        else:
            reward = 0
        print('-'*80)
    # End For (episodes)

    # Plot the current episode
    if plotting:
        pl.plot_Q_learning_performance(losses, max_episodes_to_run, is_loss=True, filename='dqn_loss')
        pl.plot_Q_learning_performance(Q_values, max_episodes_to_run, is_loss=False, filename='dqn')

        if len(episode_successful) > 0:
            pl.plot_environment_measurements(optimal_sinr_progress, max_timesteps_per_episode + 1, measurement='SINR_dB', filename='dqn')
            pl.plot_agent_actions(optimal_actions, max_timesteps_per_episode, filename='dqn')

    if (len(episode_successful) == 0):
        print("Goal cannot be reached after {} episodes.  Try to increase maximum episodes.".format(max_episodes_to_run))

    tf.keras.backend.clear_session() # free up GPU memory

    return Q_values, losses, optimal_episode, optimal_reward, optimal_sinr_progress, optimal_power_control_progress


def predict_trajectory_with_LSTM(mlw, df, target_variable, depth=1, width=2,
                                 lookahead_time=1, max_lookback=3,
                                 training_size=0.7, batch_size=16,
                                 epoch_count=10):
    global random_state

    if df is None:
        M = 50  # 50 records
        T = 10  # each record has 10 time steps
        df = pd.DataFrame(data={'Time': np.arange(M*T)})
        df['beam_index'] = random_state.randint(0,5, size=M*T)
        df['SINR'] = df['beam_index'] * 5 - random_state.uniform(0,10, size=M*T)
        df['RSRP'] = df['SINR'] - random_state.uniform(-94, -85, size=M*T)

        target_variable = 'beam_index'

    df_engf, target_var = timeseries_engineer_features(df, target_variable, lookahead_time, max_lookback, dropna=False)

    try:
        X_train, X_test, Y_train, Y_test, y_test_true = timeseries_train_test_split(df_engf,
                                                                 target_var, time_steps_per_block=T, train_size=training_size)
    except Exception as e:
        print(f'Critical: Failed to split data due to {e}.')
        raise e

    learning_rate = 1e-2
    start_time = time.time()
    # mc = ModelCheckpoint(model_filename, monitor='val_accuracy', mode='max', save_best_only=True)
    _, [training_accuracy_score, test_accuracy_score], y_test_pred = mlw._train_lstm(X_train, X_test, Y_train, Y_test,
                                            lookbacks=max_lookback, depth=depth, width=width,
                                            epoch_count=epoch_count, batch_size=batch_size, learning_rate=learning_rate)

    print(f'LSTM training accuracy is {training_accuracy_score:.2f}.')
    print(f'LSTM test accuracy is {test_accuracy_score:.2f}.')

    end_time = time.time()
    print("Training took {:.2f} mins.".format((end_time - start_time) / 60.))

    return y_test_pred, test_accuracy_score


def timeseries_train_test_split(df, label, time_steps_per_block, train_size):
    # Avoid truncating training data or test data...
    y = df[label] # must be categorical
    X = df.drop(label, axis=1)

    # Split on border of time
    m = int(X.shape[0] / time_steps_per_block * train_size)
    train_rows = int(m * time_steps_per_block)

    test_offset = ((X.shape[0] - train_rows) // time_steps_per_block) * time_steps_per_block
    X_train = X.iloc[:train_rows, :]
    X_test = X.iloc[train_rows:(train_rows+test_offset), :]
    dummy_Y = keras.utils.to_categorical(y)

    Y_train = dummy_Y[:train_rows]

    Y_test = dummy_Y[train_rows:(train_rows+test_offset)]
    y_test_true = y[train_rows:(train_rows+test_offset)]

    return X_train, X_test, Y_train, Y_test, y_test_true


def timeseries_engineer_features(df, target_variable, lookahead, max_lookback, dropna=False):
    df_ = df.set_index('Time')
    df_y = df_[target_variable].to_frame()

    # This is needed in case the idea is to predict y(t), otherwise
    # the data will contain y_t in the data and y_t+0, which are the same
    # and predictions will be trivial.
    if lookahead == 0:
        df_ = df_.drop(target_variable, axis=1)

    df_postamble = df_.add_suffix('_t')
    df_postamble = pd.concat([df_postamble, pd.DataFrame(np.zeros_like(df_), index=df_.index, columns=df_.columns).add_suffix('_d')], axis=1)

    df_shifted = pd.DataFrame()

    # Column order is important
    for i in range(max_lookback, 0, -1):
        df_shifted_i = df_.shift(i).add_suffix('_t-{}'.format(i))
        df_diff_i = df_.diff(i).add_suffix('_d-{}'.format(i)) # difference with previous time

        df_shifted = pd.concat([df_shifted, df_shifted_i, df_diff_i], axis=1)

    df_y_shifted = df_y.shift(-lookahead).add_suffix('_t+{}'.format(lookahead))
    df_output = pd.concat([df_shifted, df_postamble, df_y_shifted], axis=1)

    if dropna:
        df_output.dropna(inplace=True)
    else:
        # Do not drop data in a time series.  Instead, fill last value
        df_output.bfill(inplace=True)

        # Then fill the first value!
        df_output.ffill(inplace=True)

        # Drop whatever is left.
        df_output.dropna(how='any', axis=1, inplace=True)

    # Whatever it is, no more nulls shall pass!
    assert(df_output.isnull().sum().sum() == 0)

    engineered_target_variable = f'{target_variable}_t+{lookahead}'

    return df_output, engineered_target_variable


def denoise_signal(x_noisy, x_original, epochs=100, batch_size=16, learning_rate=1e-4, plotting=False):
    global seed

    # This is for real-valued vectors only.
    assert len(x_noisy.shape) == 1

    latent_dim = 128
    autoencoder = Autoencoder(latent_dim, shape=x_original.shape, seed=seed)
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=losses.MeanSquaredError())

    # First train on same clean data
    history = autoencoder.fit(x_original, x_original,
                             epochs=epochs, batch_size=batch_size,
                             shuffle=True)

    # Next, train on noisy data.
    history = autoencoder.fit(x_noisy, x_original,
                             epochs=epochs, batch_size=batch_size,
                             shuffle=True)

    # Now try to remove the noise.
    encoded_x = autoencoder.encoder(x_noisy).numpy()
    denoised_x = autoencoder.decoder(encoded_x).numpy()[:,0]

    if plotting:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(x_original, '--b', label='orig')
        ax.plot(x_noisy, 'r', alpha=0.5, label='noisy')
        ax.plot(denoised_x, 'k',  alpha=0.5, label='denoised')

        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        plt.savefig(f'{output_path}/denoising.pdf', format='pdf', dpi=fig.dpi)
        #tikzplotlib.save(f'{output_path}/denoising.tikz')

        plt.show()
        plt.close(fig)

    return denoised_x


##########################################
# Scenario 2: Data-driven use cases
# ########################################

pl = plotter.plotter(output_path=output_path)
mlw = simulator.simulator(N_sc=N_sc, N_r=N_r, N_t=N_t, f_c=f_c,
                 transmit_power=transmit_power, precoder=precoder,
                 channel_type=channel_type, pathloss_model=pathloss_model,
                 noise_figure=noise_figure, max_transmissions=max_transmissions,
                 constellation=constellation, M_constellation=M_constellation,
                 quantization_b=quantization_b, Df=subcarrier_spacing,
                 crc_generator=crc_generator, K_factor=K_factor, 
                 shadowing_std_dev=shadowing_std_dev, prefer_gpu=prefer_gpu,
                 random_state=random_state)

###############################################################################
# CNN for Channel Equalization
X_test, y_test, y_pred = equalize_rotation_channel_CNN(mlw, pl, theta=np.pi/24, SNR_dB=30,
                                                           epochs=102, batch_size=128,
                                                           training_ratio=0.85)
###############################################################################

# Linear regresion for channel estimation.
################################################################################
SNR_dB = 20
noise_power = 1 / mlw.linear(SNR_dB)

X = random_state.uniform(-1, 1, N_sc)  # This is BPSK
X /= np.sqrt(mlw.signal_power(X))
X = np.repeat(X, N_t, axis=np.newaxis).reshape((N_sc, N_t))

H = mlw.create_channel(N_sc, N_r, N_t, channel=channel_type)

# First the real part
re_H = np.real(H)
y = np.zeros((N_sc, N_r))
for idx in range(N_sc):
    y[idx, :] = np.dot(re_H[idx, :, :], X[idx, :]) + random_state.normal(0, noise_power, N_r)

H_est = mlw._estimate_channel_linear_regression(X, y)

estimation_error = mlw.mse(re_H, H_est)
print(f'Using linear regression, estimation error of the real part is: {estimation_error:.4f}.')

# Second the imaginary part
im_H = np.imag(H)
y = np.zeros((N_sc, N_r))
for idx in range(N_sc):
    y[idx, :] = np.dot(im_H[idx, :, :], X[idx, :]) + random_state.normal(0, noise_power, N_r)

H_est = mlw._estimate_channel_linear_regression(X, y)

estimation_error = mlw.mse(im_H, H_est)
print(f'Using linear regression, estimation error of the imaginary is: {estimation_error:.4f}.')
###############################################################################

# Time series prediction with LSTM
################################################################################
y_test_pred, test_accuracy_score = predict_trajectory_with_LSTM(mlw, df=None,
                             target_variable='', depth=0, width=5,
                             lookahead_time=1, max_lookback=10,
                             training_size=0.7, batch_size=64,
                             epoch_count=128)
###############################################################################

# Tabular reinforcement learning simulation
###############################################################################
Q_values, losses, optimal_episode, optimal_reward, \
    optimal_environment_progress, optimal_action_progress = \
        tabular_reinforcement_learning(max_episodes_to_run=100,
                                        max_timesteps_per_episode=15, pl=pl,
                                        plotting=True)
###############################################################################

# Deep reinforcement learning simulation
###############################################################################
Q_values, losses, optimal_episode, optimal_reward, \
    optimal_environment_progress, optimal_action_progress = \
    deep_reinforcement_learning(max_episodes_to_run=200,
                                 max_timesteps_per_episode=15, pl=pl,
                                 plotting=True)
###############################################################################
