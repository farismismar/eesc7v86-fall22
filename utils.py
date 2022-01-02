# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 08:43:00 2022

@author: Faris Mismar
"""

import os
import random
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import pdb

class Utils:
    def __init__(self, cmap=None, seed=None):
        plt.rcParams['font.family'] = "Arial"
        plt.rcParams['font.size'] = "14"
        
        self.cmap = None
        
        if cmap is not None:
            self.cmap = cmap
           # c = {'AT&T': 'deepskyblue', 'T-Mobile': 'magenta', 'Verizon': 'red'} # color map
        
        if seed is None:
            self.seed = np.random.mtrand._rand
        else:
            self.seed = seed
        
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        random.seed(self.seed)
        self.np_random_state = np.random.RandomState(self.seed)
        
        self.use_average = None
        self.smoothing_alpha = None
        self.low_threshold = None
    
        
    def plotXY(self, x, y, xlabel, ylabel, logx=False, logy=False, title=None):
        fig = plt.figure(figsize=(8, 5))            
            
        plt.grid(which='both', linestyle='--')
        ax = fig.gca()    
        
        if logy == True:
            plt.yscale('log')
            ax.get_yaxis().get_major_formatter().labelOnlyBase = False
        if logx == True:
            plt.xscale('log')
            ax.get_xaxis().get_major_formatter().labelOnlyBase = False
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            plt.title(title)
            
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        
    def plotXY_comparison(self, x, y1, y2, xlabel, y1label, y2label, logx=False, logy=False, title=None):    
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        plt.plot(x, y1, lw=1.5, c='black', marker='*', label=y1label)
        plt.plot(x, y2, lw=1.5, c='red', marker='*', label=y2label)
        plt.grid(which='both', linestyle='--')
        
        plt.legend(bbox_to_anchor=(0.05, -0.02, 0.9, 1), bbox_transform=fig.transFigure, 
                   loc='lower center', ncol=2, mode="expand", borderaxespad=0.)
        
        if logy == True:
            ax.set_yscale('log')
            ax.get_yaxis().get_major_formatter().labelOnlyBase = False
            #ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        if logx == True:
            ax.set_xscale('log')
            ax.get_xaxis().get_major_formatter().labelOnlyBase = False
            #ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        if title is not None:
            plt.title(title)

        ax.set_xlabel(xlabel)
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
    
    def plot_box(self, df, measure, category, cmap=None):
        if cmap is None:
            cmap = self.cmap
            
        fig = plt.figure(figsize=(8, 5))
    
        if cmap is None:
            ax = sns.boxplot(x=category, y=measure, data=df)
        else:
            ax = sns.boxplot(x=category, y=measure, data=df, palette=dict(cmap))
        
        plt.title(measure)
        plt.grid()
        
        filename = f'box_plot_{measure}.png'
        plt.savefig(f'./{filename}', dpi=fig.dpi)
        plt.show()
        plt.close(fig)
        
    
    def _plot_cdfs_kde(self, df, measure, category, cmap=None):
        if cmap is None:
            cmap = self.cmap
            
        fig = plt.figure(figsize=(8, 5))

        if cmap is None:
            sns.displot(df, x=measure, hue=category, kind='kde', cumulative=True, common_norm=False, common_grid=True)
        else:
            sns.displot(df, x=measure, hue=category, kind='kde', palette=dict(cmap), cumulative=True, common_norm=False, common_grid=True)
            
        plt.title(measure)
        plt.grid()
        
        plt.tight_layout()
        filename = f'cdf_{measure}.png'
        plt.savefig(f'./{filename}', dpi=fig.dpi)
        plt.xlim([df[measure].min(), df[measure].max()])
        plt.show()
        plt.close(fig)
        
        
    def _plot_cdfs(self, df, measure, category, cmap=None, num_bins=200):
        if cmap is None:
            cmap = self.cmap
            
        fig = plt.figure(figsize=(8, 5))
        ax = fig.gca()
        
        for data_type in df[category].unique():
            data_ = df[df[category] == data_type][measure].dropna()
            
            counts, bin_edges = np.histogram(data_, bins=num_bins, density=True)
            cdf = np.cumsum(counts) / counts.sum()
            bin_edges = np.insert(bin_edges, 0, bin_edges[0] - (bin_edges[2] - bin_edges[1]))
    
            if cmap is None:
                ax.plot(bin_edges[2:], cdf, '-', linewidth=1.5, label=data_type)
            else:
                ax.plot(bin_edges[2:], cdf, '-', color=cmap[data_type], linewidth=1.5, label=data_type)
        
        ax.set_xlabel(measure)
        ax.set_ylabel('Probability')
        
        plt.title(measure)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        filename = f'cdf_{measure}.png'
        plt.savefig(f'./{filename}', dpi=fig.dpi)
        plt.show()
        plt.close(fig)
        
        
    def plot_cdfs(self, df, measure, category, cmap=None, is_kde=False, num_bins=200):
        if is_kde:
            self._plot_cdfs_kde(df, measure, category, cmap)
        else:
            self._plot_cdfs(df, measure, category, cmap, num_bins)
            
        
    def plot_pdfs(self, df, measure, category, cmap=None, num_bins=200):
        if cmap is None:
            cmap = self.cmap
            
        fig = plt.figure(figsize=(8, 5))        
        ax = fig.gca()
            
        for data_type in df[category].unique():
            data_ = df[df[category] == data_type][measure].dropna()
            
            counts, bin_edges = np.histogram(data_, bins=num_bins, density=True)
            pdf = counts / counts.sum()
            
            bin_edges = np.insert(bin_edges, 0, bin_edges[0] - (bin_edges[2] - bin_edges[1]))
    
            if cmap is None:
                ax.plot(bin_edges[2:], pdf, '-', linewidth=1.5, label=data_type)
            else:
                ax.plot(bin_edges[2:], pdf, '-', color=cmap[data_type], linewidth=1.5, label=data_type)
        
        ax.set_xlabel(measure)
        ax.set_ylabel('Density')
        
        plt.title(measure)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        filename = f'pdf_{measure}.png'
        plt.savefig(f'./{filename}', dpi=fig.dpi)
        plt.show()
        plt.close(fig)
    
    
    def plot_data(self, df_summarized, xlabel, ylabel, category, cmap=None, reveal=False, title=None):
        if cmap is None:
            cmap = self.cmap
            
        fig = plt.figure(figsize=(8,5))
        
        for cat_i in df_summarized[category].unique():
            df_s = df_summarized[df_summarized[category] == cat_i]
            # This line visualizes the scatter, to allow fitting the hyperparameters
            # of algorithm1
            if not reveal:
                plt.scatter(df_s['x_fit'], df_s[ylabel], alpha=0.5, color=cmap[cat_i], label=f'Scatter - {cat_i}')
            ############################################################
            plt.plot(df_s.loc[:, 'x_fit'], df_s.loc[:, 'y_fitted'], color=cmap[cat_i], label=f'Best fit - {cat_i}')
            
        plt.grid(True, ls='dashed')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title is not None:
            plt.title(f'{title}')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close(fig)
    
    
    def configure_algorithm1(self, low_threshold, use_average=True, smoothing_alpha=1):
        self.use_average = use_average
        self.smoothing_alpha = smoothing_alpha
        self.low_threshold = low_threshold
        
        return self
    
        
    def algorithm1(self, df_input, x_label, y_label, sigma=1, bin_size=2, burnoff=None):
        use_average = self.use_average
        smoothing_alpha = self.smoothing_alpha
        low_threshold = self.low_threshold
        
        if use_average is None:
            raise ValueError('Call the configure_algorithm1 method first.')            
        
        random_state = self.np_random_state
        
        df = df_input[[x_label, y_label]].dropna()
        Q1 = df[y_label].quantile(0.25)
        Q3 = df[y_label].quantile(0.75)
        IQR = Q3 - Q1
        
        # remove outliers
        df.loc[(df[y_label] > Q3 + 1.5*IQR) | (df[y_label] < Q1 - 1.5*IQR), y_label] = np.nan
        
        # Bin the data
        x = df[x_label].dropna()
        y = df[y_label].dropna()
        
        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)
            
        bins = np.arange(min_x, max_x + bin_size, bin_size)
        x_binned = pd.cut(x, bins)
          
        df['x_bin'] = x_binned.values
        
        # pdb.set_trace()
    
        df_ = df.copy()
        # now obtain the aggregate
        if use_average:
            df_summarized = df_.groupby('x_bin').agg(lambda x: x.mean(skipna=True)).reset_index()
            print('INFORMATION: Binning by the average function.')
        else:
            df_summarized = df_.groupby('x_bin').agg(lambda x: x.quantile(0.97, numeric_only=True)).reset_index()
            print('INFORMATION: Binning by the percentile function.')
        
        x_upper = df_summarized['x_bin'].apply(lambda x: x.right)
    
        # if n is small, kill the sample with NaN since the statistic is insufficient.
        df_summarized.loc[:, 'N'] = df.groupby('x_bin').agg('count').reset_index()[x_label]
        df_summarized.loc[df_summarized['N'] < low_threshold, y_label] = np.nan
        df_summarized.loc[:, 'x_fit'] = df_summarized['x_bin'].apply(lambda x: x.mid)
        
        df_summarized.drop('x_bin', axis=1, inplace=True)
    
        # Guess if decreasing through rate of change
        try:
            df_sampling = df_summarized[~df_summarized[y_label].isnull()].sample(n=2, random_state=random_state)
            df_sampling = df_sampling.iloc[:,:-1].diff(1).iloc[-1,:]
            delta_y = df_sampling[y_label]
            delta_x = df_sampling[x_label]
            decreasing = (delta_y / delta_x < 0)
            
            print(f'INFORMATION: function is decreasing: {decreasing}.')
            
            if decreasing:
                df_summarized.fillna(method='bfill', inplace=True)
                df_summarized.dropna(inplace=True)
            else:
                df_summarized.fillna(method='ffill', inplace=True)
                df_summarized.dropna(inplace=True)
        except:
            print('INFORMATION: cannot determine function direction.')
            df_summarized.dropna(inplace=True)
            
        # Remove burnoff points before running the exponential smoothing
        if burnoff is not None:
            df_summarized = df_summarized.iloc[burnoff:-burnoff, :].reset_index(drop=True)
            
        y_smooth = df_summarized[y_label].ewm(alpha=smoothing_alpha).mean()
        df_summarized.loc[:, 'y_fitted'] = gaussian_filter(y_smooth, sigma=sigma) # small sigma will cause an overfit
    
        return df_summarized, x_upper