# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 21:13:45 2021

@author: Faris Mismar
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if os.name == 'nt':
    os.add_dll_directory("/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")

import tensorflow as tf
#print(tf.config.list_physical_devices('GPU'))

# The GPU ID to use, usually either "0" or "1" based on previous line.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # My NVIDIA GeForce RTX 3050 Ti GPU output from line 15

import random
import numpy as np
from tensorflow.compat.v1 import set_random_seed

from tensorflow import keras
from tensorflow.keras import layers, initializers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import class_weight

import matplotlib.pyplot as plt

import time

class DeepNeuralNetworkClassifier:
    def __init__(self, seed=None, prefer_gpu=True):
    
        use_cuda = len(tf.config.list_physical_devices('GPU')) > 0 and prefer_gpu
        self.device = "/gpu:0" if use_cuda else "/cpu:0"
        
        if seed is None:
            self.seed = np.random.mtrand._rand
        else:
            self.seed = seed
        
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        random.seed(self.seed)
        self.np_random_state = np.random.RandomState(self.seed)
        
        set_random_seed(self.seed)
        
        self.le = None
        self.learning_rate = 0.1
        
        plt.rcParams['font.family'] = "Arial"
        plt.rcParams['font.size'] = "14"
 
        
    def _create_mlp(self, input_dimension, hidden_dimension, depth, output_dimension):
        learning_rate = self.learning_rate
        
        model = keras.Sequential()
        model.add(layers.Dense(units=hidden_dimension, 
                               input_dim=input_dimension, use_bias=True, 
                               activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(layers.Dropout(rate=0.1, seed=self.seed)) # reduces training overfitting

        for d in range(depth):
            model.add(layers.Dense(units=hidden_dimension, use_bias=True, 
                                   activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros'))
            
        model.add(layers.Dense(units=output_dimension, use_bias=True, activation="softmax", kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.compile(loss='categorical_crossentropy', 
                      optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
#                      keras.optimizers.Adam(learning_rate=learning_rate), 
                      metrics=['accuracy', 'categorical_crossentropy'])
        
        return model


    def train(self, X, y, n_epochs, batch_size, train_size, scale=True, plotting=True):
        
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, train_size=train_size,
                        random_state=self.np_random_state)

        if scale:
           # Scale X features
           self.sc = MinMaxScaler()
           X_train = self.sc.fit_transform(X_train.values)
        else:
            # Convert input to numpy array (if not scaling)
            X_train = X_train.values
        
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
        
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(y_train),
                                                  y=y_train)
        class_weights = dict(enumerate(class_weights))
                  
        device = self.device
        
        self.le = LabelEncoder()
        self.le.fit(y_train)
        encoded_y = self.le.transform(y_train)     
        Y_train = keras.utils.to_categorical(encoded_y)

        mX, nX = X_train.shape
        _, nY = Y_train.shape
        
        # Early stopping condition
        es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, min_delta=1e-4, patience=4)
        
        model = KerasClassifier(build_fn=self._create_mlp, 
                                input_dimension=nX,
                                hidden_dimension=5,
                                depth=0,
                                output_dimension=nY,
                                verbose=1,# callbacks=[es],
                                epochs=n_epochs, batch_size=batch_size)
  
        start_time = time.time()
        with tf.device(device):
            history = model.fit(X_train, Y_train, class_weight=class_weights, validation_split=0.2) 
        end_time = time.time()
        
        print('Training time: {:.2f} mins.'.format((end_time - start_time) / 60.))
        
        self.model = model.model
        
        # Reporting the number of parameters
        num_params = model.model.count_params()
        print('Number of parameters: {}'.format(num_params))

        
        if plotting:
            # Plot the losses vs epoch here
            fig = plt.figure(figsize=(8, 5))

            plot1, = plt.plot(history.epoch, history.history['loss'], c='blue')
            plot2, = plt.plot(history.epoch, history.history['val_loss'], lw=1.5, ls='--', c='blue')
            
            plt.grid(which='both', linestyle='--')
            
            ax = fig.gca()    
            ax_sec = ax.twinx()
            plot3, = ax_sec.plot(history.epoch, history.history['accuracy'], lw=2, c='red')       
            plot4, = ax_sec.plot(history.epoch, history.history['val_accuracy'], lw=1.5, ls='--', c='red')       

            ax.set_xlabel(r'Epoch')
            ax.set_ylabel(r'Loss')
            ax_sec.set_ylabel(r'Accuracy')
            plt.legend([plot1, plot2, plot3, plot4], [r'Loss', r'Val Loss', r'Accuracy', r'Val_Accuracy'],
                       bbox_to_anchor=(0.2, -0.04, 0.6, 1), bbox_transform=fig.transFigure, 
                       loc='lower center', ncol=2, mode="expand", borderaxespad=0.)
            
            plt.tight_layout()
            plt.show()
            plt.close(fig)
        
        return X_test, y_test
        
    
    def test(self, X_test, y_test, scale=True):
        # Testing and inference
        if scale:
           # Scale X features
           X_test = self.sc.transform(X_test.values)
        else:
            # Convert input to numpy array (if not scaling)
            X_test = X_test.values
        
        device = self.device
        model = self.model
        le = self.le
        
        encoded_y = le.transform(y_test)
        Y_test = keras.utils.to_categorical(encoded_y)

        with tf.device(device):
            Y_pred = model.predict(X_test)
            loss, acc, _ = model.evaluate(X_test, Y_test)
        print('Test: Loss {:.4f}, Acc: {:.4f}'.format(loss, acc))
  
        # Y_pred not coming out as a matrix?
        assert(Y_pred.shape[1] == Y_test.shape[1])
        
        # Reverse the encoded categories
        y_test = le.inverse_transform(np.argmax(Y_test, axis=1))
        y_pred = le.inverse_transform(np.argmax(Y_pred, axis=1))
        
        # Scoring
        accuracy = (y_test == y_pred).sum() / y_test.shape[0]

        return accuracy, y_pred