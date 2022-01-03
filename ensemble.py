# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 20:37:33 2021

@author: Faris Mismar
"""

import os
import random
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.utils import class_weight

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt

import pdb
import time

class EnsembleClassifier:
    def __init__(self, seed=None, prefer_gpu=True, is_booster=True):
    
        # use_cuda = len(tf.config.list_physical_devices('GPU')) > 0 and prefer_gpu
        # self.device = "/gpu:0" if use_cuda else "/cpu:0"
        
        if seed is None:
            self.seed = np.random.mtrand._rand
        else:
            self.seed = seed
        
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        random.seed(self.seed)
        self.np_random_state = np.random.RandomState(self.seed)
        
        self.is_booster = is_booster
        self.prefer_gpu = prefer_gpu
        
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
        
        if self.is_booster == False:
            base_estimator = RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight=class_weights, random_state=self.np_random_state)
            hyperparameters = {'criterion': ['entropy', 'gini'],
                                #'min_impurity_decrease': [0.1, 0.2],
                      #'min_weight_fraction_leaf': [0.1, 0.3],
                      'min_samples_split': [2, 10]}
        else:
            tree_method = 'gpu_hist' if self.prefer_gpu else 'hist'
            
            base_estimator = xgb.XGBClassifier(tree_method=tree_method, seed=self.seed)
            hyperparameters = {'reg_lambda': np.linspace(0,1,3),
                               'reg_alpha': np.linspace(0,1,3),
                               'colsample_bytree': [0,0.5,1]
                               }
        
        start_time = time.time()
        clf = GridSearchCV(base_estimator, param_grid=hyperparameters, 
                           cv=K_fold, n_jobs=-1, scoring='roc_auc_ovr_weighted', verbose=1)
        clf.fit(X_train, y_train)
        
        end_time = time.time()
        
        print('Training time: {:.2f} mins.'.format((end_time - start_time) / 60.))
        
        self.model = clf.best_estimator_
        self.model.fit(X_train, y_train) # to be ready for the inference
    
        if plotting:
            train_sizes, train_scores, test_scores = \
                learning_curve(self.model, X_train, y_train, cv=K_fold,
                               n_jobs=-1, return_times=False)
            
            # Code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            # Plot the losses of training and cross validation data
            fig = plt.figure(figsize=(8, 5))
            ax = fig.gca()
            
            # Code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
            ax.grid()
            ax.fill_between(
                train_sizes,
                train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std,
                alpha=0.1,
                color="r",
            )
            ax.fill_between(
                train_sizes,
                test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std,
                alpha=0.1,
                color="g",
            )
            ax.plot(
                train_sizes, train_scores_mean, "o-", color="r", label="Training score"
            )
            ax.plot(
                train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
            )
            ax.legend(loc="best")
            
            ax.set_xlabel(r'Training size')
            ax.set_ylabel(r'Score')

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