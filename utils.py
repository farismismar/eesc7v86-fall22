# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 08:43:00 2022

@author: Faris Mismar
"""

import os
import random
import numpy as np
import pandas as pd

from scipy.interpolate import UnivariateSpline #, interp1d
#from scipy.ndimage import gaussian_filter

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
        
        plt.plot(x, y, lw=1.5, c='black', marker='*', label=ylabel)
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
        
    
    def plot_box(self, df, measure, category, title=None, cmap=None):
        # Note if you do not have a category, consider using the pd.melt() function
        if cmap is None:
            cmap = self.cmap
            
        fig = plt.figure(figsize=(8, 5))
    
        if cmap is None:
            ax = sns.boxplot(x=category, y=measure, data=df)
        else:
            ax = sns.boxplot(x=category, y=measure, data=df, palette=dict(cmap))
        
        if title is None:
            title = measure
            
        plt.title(title)
        plt.grid()
        
        filename = f'box_plot_{measure}.png'
        plt.savefig(f'./{filename}', dpi=fig.dpi)
        plt.show()
        plt.close(fig)
        
    
    def _plot_cdfs_kde(self, df, measure, category, cmap=None, title=None):
        # Note if you do not have a category, consider using the pd.melt() function
        if cmap is None:
            cmap = self.cmap
            
        fig = plt.figure(figsize=(8, 5))

        if cmap is None:
            sns.displot(df, x=measure, hue=category, kind='kde', cumulative=True, common_norm=False, common_grid=True)
        else:
            sns.displot(df, x=measure, hue=category, kind='kde', palette=dict(cmap), cumulative=True, common_norm=False, common_grid=True)
            
        if title is None:
            title = measure
            
        plt.title(title)
        plt.grid()
        
        plt.tight_layout()
        filename = f'cdf_{measure}.png'
        plt.savefig(f'./{filename}', dpi=fig.dpi)
        plt.xlim([df[measure].min(), df[measure].max()])
        plt.show()
        plt.close(fig)
        
        
    def _plot_cdfs(self, df, measure, category, cmap=None, title=None, num_bins=200):
        # Note if you do not have a category, consider using the pd.melt() function
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
                
        if title is None:
            title = measure
            
        plt.title(title)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        filename = f'cdf_{measure}.png'
        plt.savefig(f'./{filename}', dpi=fig.dpi)
        plt.show()
        plt.close(fig)
        

    def plot_cdfs(self, df, measure, category, cmap=None, title=None, is_kde=False, num_bins=200):
        if is_kde:
            self._plot_cdfs_kde(df, measure, category, cmap, title)
        else:
            self._plot_cdfs(df, measure, category, cmap, title, num_bins)
            
        
    def plot_pdfs(self, df, measure, category, cmap=None, title=None, num_bins=200):
        # Note if you do not have a category, consider using the pd.melt() function
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
        
        if title is None:
            title = measure
            
        plt.title(title)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        filename = f'pdf_{measure}.png'
        plt.savefig(f'./{filename}', dpi=fig.dpi)
        plt.show()
        plt.close(fig)
    
    
    # TODO
    def plot_joint_pdf(self, X, Y, title, num_bins=50):
        fig = plt.figure(figsize=(8, 5))

        H, X_bin_edges, Y_bin_edges = np.histogram2d(X, Y, bins=(num_bins, num_bins), normed=True)
        for y in np.arange(num_bins):
            H[y,:] = H[y,:] / sum(H[y,:])
        pdf = H / num_bins    
    
        ax = plt.gca(projection="3d")
    
        x, y = np.meshgrid(X_bin_edges, Y_bin_edges)

        surf = ax.plot_surface(x[:num_bins, :num_bins], y[:num_bins, :num_bins], pdf[:num_bins, :num_bins], antialiased=True)
        ax.view_init(5, 45) # the first param rotates the z axis inwards or outwards the screen.  The second is what we need.
    
        # No background color    
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    
        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        ax.set_xlabel('3.5 GHz')
        ax.set_ylabel('28 GHz')
        ax.set_zlabel('Joint Throughput pdf')

        ax.invert_xaxis()
        ax.invert_yaxis()
        
        ax.set_xlim(int(np.max(X)), 0)
        ax.set_ylim(int(np.max(Y)), 0)
        ax.set_zlim(np.min(pdf), np.max(pdf))
        
        ax.xaxis.labelpad=20
        ax.yaxis.labelpad=20
        ax.zaxis.labelpad=20
    
        plt.xticks([3,2,1,0])
        plt.yticks([15,10,5,0])
        
        plt.tight_layout()
    
        plt.savefig('figures/joint_throughput_pdf_{}.pdf'.format(p_randomness), format='pdf')
        matplotlib2tikz.save('figures/joint_throughput_pdf_{}.tikz'.format(p_randomness))
        plt.show()
        plt.close(fig)
    

    # TODO
    def plot_joint_cdf(self, X, Y, num_bins=100):
        fig = plt.figure(figsize=(8, 5))
    
        H, X_bin_edges, Y_bin_edges = np.histogram2d(X, Y, bins=(num_bins, num_bins), normed=True)
        for y in np.arange(num_bins):
            H[y,:] = H[y,:] / sum(H[y,:])
        pdf = H / num_bins
    
        cdf = np.zeros((num_bins, num_bins))
        for i in np.arange(num_bins):
            for j in np.arange(num_bins):
                cdf[i,j] = sum(sum(pdf[:(i+1), :(j+1)]))

        ax = plt.gca(projection="3d")
        x, y = np.meshgrid(X_bin_edges, Y_bin_edges)

        surf = ax.plot_surface(x[:num_bins, :num_bins], y[:num_bins, :num_bins], cdf[:num_bins, :num_bins], antialiased=True)
        ax.view_init(5, 45) # the first param rotates the z axis inwards or outwards the screen.  The second is what we need.    

        # No background color    
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    
        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        ax.set_xlabel('3.5 GHz')
        ax.set_ylabel('28 GHz')
        ax.set_zlabel('Joint Throughput CDF')
    
        ax.invert_xaxis()
        ax.invert_yaxis()
    
        ax.set_xlim(int(np.max(X)), 0)
        ax.set_ylim(int(np.max(Y)), 0)
        ax.set_zlim(0,1)

        ax.xaxis.labelpad=20
        ax.yaxis.labelpad=20
        ax.zaxis.labelpad=20

        plt.xticks([3,2,1,0])
        plt.yticks([15,10,5,0])
    
        plt.tight_layout()
    
        plt.savefig('figures/joint_throughput_cdf_{}.pdf'.format(p_randomness), format='pdf')
        matplotlib2tikz.save('figures/joint_throughput_cdf_{}.tikz'.format(p_randomness))
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
    
    # Latest since Feb 10, 2022
    def algorithm1(self, df_input, x_label, y_label, order=2, bin_size=2, burnoff=None):
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
        mean = df[y_label].mean()
        std = df[y_label].std()
        
        # remove outliers
        df.loc[(df[y_label] >= Q3 + 1.5*IQR) | (df[y_label] <= Q1 - 1.5*IQR), y_label] = np.nan
        #df.loc[(df[y_label] >= mean + 3*std) | df[y_label] <= mean - 3*std] = np.nan
        
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
        except Exception as e:
            print(f'INFORMATION: cannot determine function direction due to {e}.')
            df_summarized.dropna(inplace=True)
            
        # Remove burnoff points before running the exponential smoothing
        if (burnoff is not None) and (burnoff > 0):
            df_summarized = df_summarized.iloc[burnoff:-burnoff, :].reset_index(drop=True)

    #     y_smooth = df_summarized[y_label].ewm(alpha=smoothing_alpha).mean()
    #     df_summarized.loc[:, 'y_fitted'] = gaussian_filter(y_smooth, sigma=sigma) # small sigma will cause an overfit
        
        #f = interp1d(df_summarized['x_fit'], df_summarized[y_label])
        
        f = UnivariateSpline(df_summarized['x_fit'], df_summarized[y_label], k=order)
        df_summarized.loc[:, 'y_fitted'] = f(df_summarized['x_fit'])
        
        return df_summarized, x_upper
    