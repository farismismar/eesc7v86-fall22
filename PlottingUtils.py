# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 08:43:00 2022

@author: farismismar
"""

import os
import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import tikzplotlib

class PlottingUtils:
    def __init__(self, seed, results_folder=None, cmap=None):
        plt.rcParams['font.family'] = "Arial"
        plt.rcParams['font.size'] = "14"
        
        self.cmap = None
        
        if results_folder is None:
            self.results_folder = './'
        else:
            self.results_folder = results_folder
            
        if cmap is not None:
            self.cmap = cmap
        
        self.seed = seed
        
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        random.seed(self.seed)
        self.np_random_state = np.random.RandomState(self.seed)
        
        
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
        filename = 'plot_xy'
        plt.savefig(f'{self.results_folder}/{filename}.png', dpi=fig.dpi)
        tikzplotlib.save(f'{self.results_folder}/{filename}.tikz')        
        plt.show()
        plt.close(fig)
        
        
    def plotXY_comparison(self, x, y1, y2, y3, xlabel, ylabel, y1label, y2label, y3label, logx=False, logy=False, title=None):    
        results_folder = self.results_folder
        
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        plt.plot(x, y1, lw=1.5, c='black', marker='*', label=y1label)
        plt.plot(x, y2, lw=1.5, c='red', marker='+', label=y2label)
        
        if y3 is not None:
            plt.plot(x, y3, lw=1.5, c='blue', marker='^', label=y3label)
        
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
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        filename = 'plot_xy_comparison'
        plt.savefig(f'{results_folder}/{filename}.pdf', format='pdf', dpi=fig.dpi)
        tikzplotlib.save(f'{results_folder}/{filename}.tikz')     
        plt.show()
        plt.close(fig)
        
    
    def plot_box(self, df, measure, category, title=None, cmap=None):
        results_folder = self.results_folder
        
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
        
        filename = f'box_plot_{measure}'
        plt.savefig(f'{results_folder}/{filename}.png', dpi=fig.dpi)
        tikzplotlib.save(f'{results_folder}/{filename}.tikz')
        plt.show()
        plt.close(fig)
        
    
    def _plot_cdfs_kde(self, df, measure, category, cmap=None, title=None):
        results_folder = self.results_folder
        
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
        filename = f'cdf_{measure}'
        plt.savefig(f'{results_folder}/{filename}.png', dpi=fig.dpi)
        tikzplotlib.save(f'{results_folder}/{filename}.tikz')
        plt.xlim([df[measure].min(), df[measure].max()])
        plt.show()
        plt.close(fig)
        
        
    def _plot_cdfs(self, df, measure, category, cmap=None, title=None, num_bins=200):
        results_folder = self.results_folder
        
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
        filename = f'cdf_{measure}'
        plt.savefig(f'{results_folder}/{filename}.png', dpi=fig.dpi)
        tikzplotlib.save(f'{results_folder}/{filename}.tikz')
        plt.show()
        plt.close(fig)
        

    def plot_cdfs(self, df, measure, category, cmap=None, title=None, is_kde=False, num_bins=200):
        if is_kde:
            self._plot_cdfs_kde(df, measure, category, cmap, title)
        else:
            self._plot_cdfs(df, measure, category, cmap, title, num_bins)
            
        
    def plot_pdfs(self, df, measure, category, cmap=None, title=None, num_bins=200):
        results_folder = self.results_folder
        
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
        filename = f'pdf_{measure}'
        plt.savefig(f'{results_folder}/{filename}.png', dpi=fig.dpi)
        tikzplotlib.save(f'{results_folder}/{filename}.tikz')
        plt.show()
        plt.close(fig)
    
    
    def plot_joint_pdf(self, X, Y, x_label, y_label, title=None, num_bins=50):
        results_folder = self.results_folder
        
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

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel('Joint pdf fxy(x,y)')

        ax.invert_xaxis()
        ax.invert_yaxis()
        
        ax.set_xlim(int(np.max(X)), 0)
        ax.set_ylim(int(np.max(Y)), 0)
        ax.set_zlim(np.min(pdf), np.max(pdf))
        
        ax.xaxis.labelpad=20
        ax.yaxis.labelpad=20
        ax.zaxis.labelpad=20
        
        if title is not None:
            plt.title(title)
    
        plt.tight_layout()
    
        plt.savefig(f'{results_folder}/joint_throughput_pdf.pdf', format='pdf')
        tikzplotlib.save(f'{results_folder}/joint_throughput_pdf.tikz')
        plt.show()
        plt.close(fig)
    

    def plot_joint_cdf(self, X, Y, x_label, y_label, title=None, num_bins=100):
        results_folder = self.results_folder
        
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

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel('Joint CDF Fxy(x,y)')
    
        ax.invert_xaxis()
        ax.invert_yaxis()
    
        ax.set_xlim(int(np.max(X)), 0)
        ax.set_ylim(int(np.max(Y)), 0)
        ax.set_zlim(0,1)

        ax.xaxis.labelpad=20
        ax.yaxis.labelpad=20
        ax.zaxis.labelpad=20

        if title is not None:
            plt.title(title)
    
        plt.tight_layout()
    
        plt.savefig(f'{results_folder}/joint_throughput_cdf.pdf', format='pdf')
        tikzplotlib.save(f'{results_folder}/joint_throughput_cdf.tikz')
        plt.show()
        plt.close(fig)
    

    # Only create probability density and cumulative distribution functions, without plotting.
    def create_cdf(self, X, num_bins=200):
        X_ = X[~np.isnan(X)]
        
        counts, bin_edges = np.histogram(X_, bins=num_bins, density=True)
        cdf = np.cumsum(counts) / counts.sum()
        bin_edges = np.insert(bin_edges, 0, bin_edges[0] - (bin_edges[2] - bin_edges[1]))

        return bin_edges[2:], cdf
    
    def create_ccdf(self, X, num_bins=200):
        X_ = X[~np.isnan(X)]
        
        counts, bin_edges = np.histogram(X_, bins=num_bins, density=True)
        ccdf = 1 - np.cumsum(counts) / counts.sum()
        ccdf = np.insert(ccdf, 0, 1)
        bin_edges = np.insert(bin_edges[1:], 0, bin_edges[0] - (bin_edges[2] - bin_edges[1]))

        return bin_edges, ccdf
    
    def create_pdf(self, X, num_bins=200):
        X_ = X[~np.isnan(X)]
        
        counts, bin_edges = np.histogram(X_, bins=num_bins, density=True)
        pdf = counts / counts.sum()
        bin_edges = np.insert(bin_edges, 0, bin_edges[0] - (bin_edges[2] - bin_edges[1]))
        
        return bin_edges, pdf
    