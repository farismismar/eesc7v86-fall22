# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 20:37:33 2021

@author: Faris Mismar
"""

import os
import random
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import class_weight

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

import pdb
import time

class EnsembleClassifier:
    def __init__(self, seed=None, prefer_gpu=True):
    
        # use_cuda = len(tf.config.list_physical_devices('GPU')) > 0 and prefer_gpu
        # self.device = "/gpu:0" if use_cuda else "/cpu:0"
        
        if seed is None:
            self.seed = np.random.mtrand._rand
        else:
            self.seed = seed
        
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        random.seed(self.seed)
        self.np_random_state = np.random.RandomState(self.seed)
        
        plt.rcParams['font.family'] = "Arial"
        plt.rcParams['font.size'] = "14"
 

    def train(self, X, y, train_size, K_fold=3, plotting=True):
        
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, train_size=train_size,
                        random_state=self.np_random_state)
        
        X_train = X_train.values
        y_train = y_train.values.ravel()
        
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(y_train),
                                                  y=y_train)
        class_weights = dict(enumerate(class_weights))
                  
        base_estimator = RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight=class_weights, random_state=self.np_random_state)
        hyperparameters = {'criterion': ['entropy', 'gini'],
                           #'min_impurity_decrease': [0.1, 0.2],
                  #'min_weight_fraction_leaf': [0.1, 0.3],
                  'min_samples_split': [2, 10]}
        start_time = time.time()
        clf = GridSearchCV(base_estimator, param_grid=hyperparameters, 
                           cv=K_fold, n_jobs=-1, scoring='roc_auc_ovr_weighted', verbose=1)
        clf.fit(X_train, y_train)
        
        end_time = time.time()
        
        print('Training time: {:.2f} mins.'.format((end_time - start_time) / 60.))
        
        self.model = clf.best_estimator_
        self.model.fit(X_train, y_train)
        
        if plotting:
            # Plot the losses vs epoch here
            fig = plt.figure(figsize=(8, 5))

            # plot1, = plt.plot(history.epoch, history.history['loss'], c='blue')
            # plot2, = plt.plot(history.epoch, history.history['val_loss'], lw=1.5, ls='--', c='blue')
            
            # plt.grid(which='both', linestyle='--')
            
            # ax = fig.gca()    
            # ax_sec = ax.twinx()
            # plot3, = ax_sec.plot(history.epoch, history.history['accuracy'], lw=2, c='red')       
            # plot4, = ax_sec.plot(history.epoch, history.history['val_accuracy'], lw=1.5, ls='--', c='red')       

            # ax.set_xlabel(r'Epoch')
            # ax.set_ylabel(r'Loss')
            # ax_sec.set_ylabel(r'Accuracy')
            # plt.legend([plot1, plot2, plot3, plot4], [r'Loss', r'Val Loss', r'Accuracy', r'Val_Accuracy'],
            #            bbox_to_anchor=(0.2, -0.04, 0.6, 1), bbox_transform=fig.transFigure, 
            #            loc='lower center', ncol=2, mode="expand", borderaxespad=0.)
            
            plt.tight_layout()
            plt.show()
            plt.close(fig)
        
        return X_test, y_test
        
    
    def test(self, X_test, y_test):
        # Testing and inference

        X_test = X_test.values
        
        model = self.model
        y_pred = model.predict(X_test)
        
        # Scoring
        accuracy = model.score(X_test, y_test)
        print('Test: Acc: {:.4f}'.format(accuracy))
        
        return accuracy, y_pred