#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:22:29 2024

@author: farismismar
"""

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

class plotter:
    __ver__ = '0.9.3'
    __date__ = '2025-05-15'
    
    def __init__(self, output_path=None, create=True):
        self.output_path = './'

        if output_path is not None:
            self.output_path = output_path
            if create and not os.path.exists(output_path):
                os.makedirs(output_path)

    def plot_scatter(self, df, xlabel, ylabel, filename=None):
        output_path = self.output_path

        fig, ax = plt.subplots(figsize=(9, 6))
        plt.scatter(df[xlabel], df[ylabel], s=10, c='r', edgecolors='none', alpha=0.2)

        plt.grid(which='both')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()

        if filename is not None:
            plt.savefig(f'{output_path}/scatter_{ylabel}_{xlabel}_{filename}.pdf', format='pdf', dpi=fig.dpi)
    
        # plt.show()
        plt.close(fig)


    def plot_performance(self, df, xlabel, ylabel, semilogy=True, filename=None):
        output_path = self.output_path

        cols = list(set([xlabel, ylabel, 'snr_dB']))
        df = df[cols]
        df_plot = df.groupby('snr_dB').mean().reset_index()

        fig, ax = plt.subplots(figsize=(9, 6))
        if semilogy:
            ax.set_yscale('log')
        ax.tick_params(axis=u'both', which=u'both')
        plt.plot(df_plot[xlabel].values, df_plot[ylabel].values, '--bo', alpha=0.7,
                 markeredgecolor='k', markerfacecolor='r', markersize=6)

        plt.grid(which='both', axis='both')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()

        if filename is not None:
            plt.savefig(f'{output_path}/performance_{ylabel}_{filename}.pdf', format='pdf', dpi=fig.dpi)
            
        # plt.show()
        plt.close(fig)


    def plot_pdf(self, X, text=None, var=None, algorithm='empirical', num_bins=200, filename=None):
        output_path = self.output_path

        X_re = np.real(X)
        X_im = np.imag(X)

        if text is None:
            text = ''
        else:
            text = f'-{text}'

        is_complex = True
        if np.sum(X_im) == 0:
            is_complex = False

        fig, ax = plt.subplots(figsize=(12, 6))

        if algorithm == 'empirical':
            for label, var in zip(['Re[X]', 'Im[X]'], [X_re, X_im]):
                # Is this a real quantity?
                if not is_complex and label == 'Im[X]':
                    continue
                counts, bin_edges = np.histogram(var, bins=num_bins, density=True)
                pdf = counts / counts.sum()
                bin_edges = np.insert(bin_edges, 0, bin_edges[0] - (bin_edges[2] - bin_edges[1]))
                ax.plot(bin_edges[2:], pdf, '-', linewidth=1.5, label=f'{label}{text}')
            ax.legend()

        if algorithm == 'KDE':
            df_re = pd.DataFrame(X_re).add_suffix(f'-{text}-Re')
            df_im = pd.DataFrame(X_im).add_suffix(f'-{text}-Im')

            df = df_re.copy()
            if is_complex:
                df = pd.concat([df, df_im], axis=1, ignore_index=False)
            try:
                df.plot(kind='kde', bw_method=0.3, ax=ax)
            except Exception as e:
                print(f"Failed to generate plot due to {e}.")

        plt.grid(True)

        if var is not None:
            plt.xlabel(f'{var}')
            plt.ylabel(f'p({var})')

        plt.tight_layout()

        if filename is not None:
            plt.savefig(f'{output_path}/pdf_{algorithm}_{filename}.pdf', format='pdf', dpi=fig.dpi)

        # plt.show()
        plt.close(fig)


    def plot_constellation(self, constellation, annotate=False, filename=None):
        output_path = self.output_path

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.set_aspect('equal', 'box')
        plt.scatter(constellation['x_I'], constellation['x_Q'], c='k', marker='o', lw=2)

        if annotate:
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
        plt.tight_layout()

        if filename is not None:
            plt.savefig(f'{output_path}/constellation_{filename}.pdf', format='pdf', dpi=fig.dpi)

        # plt.show()
        plt.close(fig)


    def plot_channel(self, channel, vmin=None, vmax=None, filename=None):
        output_path = self.output_path

        N_sc, N_r, N_t = channel.shape

        # Only plot first receive antenna
        H = channel[:, 0, :]

        dB_gain = 10*np.log10(np.abs(H) ** 2 + 1e-5)

        # Create a normalization object
        norm = mcolors.Normalize(vmin=dB_gain.min(), vmax=dB_gain.max())

        # plt.rcParams['font.size'] = 36
        # plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots(figsize=(12, 6))

        plt.imshow(dB_gain, aspect='auto', norm=norm)

        plt.xlabel('TX Antennas')
        plt.ylabel('Subcarriers')

        plt.xticks(range(N_t))
        plt.tight_layout()

        if filename is not None:
            plt.savefig(f'{output_path}/channel_{filename}.pdf', format='pdf', dpi=fig.dpi)

        # plt.show()
        plt.close(fig)


    def plot_IQ(self, signal, filename=None):
        output_path = self.output_path

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        for idx in range(signal.shape[1]):
            X = signal[:, idx]
            plt.scatter(np.real(X), np.imag(X), marker='o', lw=2, label=f'TX ant. {idx}')

        plt.grid(True)
        plt.legend()
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.tight_layout()

        if filename is not None:
            plt.savefig(f'{output_path}/IQ_{filename}.pdf', format='pdf', dpi=fig.dpi)

        # plt.show()
        plt.close(fig)


    def plot_keras_learning(self, history, filename=None):
        output_path = self.output_path

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
            plt.savefig(f'{output_path}/history_keras_{filename}.pdf', format='pdf', dpi=fig.dpi)
        
        # plt.show()
        plt.close(fig)
            

    def plot_Q_learning_performance(self, values, num_episodes, is_loss=False, filename=None):
        output_path = self.output_path
        
        fig = plt.figure(figsize=(8, 5))

        y_label = 'Expected Action-Value Q' if not is_loss else r'Expected Loss'
        plt.xlabel('Episode')
        plt.ylabel(y_label)

        color = 'b' if not is_loss else 'r'
        plt.plot(1 + np.arange(num_episodes), values, linestyle='-', color=color)

        # These are integer actions.
        fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.grid(True)
        plt.tight_layout()

        if filename is not None:
            plt.savefig(f'{output_path}/Qlearning_perf_{filename}.pdf', format='pdf', dpi=fig.dpi)

        # plt.show()
        plt.close(fig)


    def plot_environment_measurements(self, environment_measurements, time_steps, measurement=None, filename=None):
        output_path = self.output_path

        fig = plt.figure(figsize=(8, 5))
        plt.plot(1 + np.arange(time_steps), environment_measurements, color='k')
        plt.xlabel('Time step t')

        if measurement is not None:
            plt.ylabel(measurement)

        plt.grid(True)
        plt.tight_layout()

        if filename is not None:
            plt.savefig(f'{output_path}/environment_{measurement}_{filename}.pdf', format='pdf', dpi=fig.dpi)
            
        # plt.show()
        plt.close(fig)


    def plot_agent_actions(self, agent_actions, time_steps, filename=None):
        output_path = self.output_path

        fig = plt.figure(figsize=(8, 5))
        plt.step(1 + np.arange(time_steps), agent_actions, color='k')
        plt.xlabel('Time step t')
        plt.ylabel('Action')

        # These are integer actions.
        fig.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.grid(True)
        plt.tight_layout()

        if filename is not None:
            plt.savefig(f'{output_path}/actions_{filename}.pdf', format='pdf', dpi=fig.dpi)
            
        # plt.show()
        plt.close(fig)
        
