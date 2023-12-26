# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 18:53:59 2023

@author: farismismar
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if os.name == 'nt':
    os.add_dll_directory("/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# The GPU ID to use, usually either "0" or "1" based on previous line.
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import random
import numpy as np
from tensorflow.keras import Input, layers, losses, optimizers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import time

import pdb

# https://www.tensorflow.org/tutorials/generative/autoencoder

class Autoencoder(Model):
    def __init__(self, latent_dim, shape, seed=None):
        # Reproducibility not working well.
        # os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        random_state = np.random.RandomState(seed)
        tf.random.set_seed(seed) # sets global random seed
        # tf.keras.utils.set_random_seed(seed)
        
        super(Autoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.shape = shape
        
        # Compresses
        self.encoder = tf.keras.Sequential(name='encoder')
        self.encoder.add(layers.Flatten())
        self.encoder.add(layers.Dense(8, activation='relu'))
        self.encoder.add(layers.Dense(16, activation='relu'))
        self.encoder.add(layers.Dense(latent_dim, activation='relu'))
        
        # self.encoder.add(layers.Conv2D(32, (2, 2), input_shape=shape,
        #                                strides=(2,2), activation='relu', padding='same'))
        # self.encoder.add(layers.MaxPooling2D((2,2), padding='same'))
        # self.encoder.add(layers.Conv2D(32, (latent_dim, latent_dim), strides=(2,2), activation='relu', padding='same'))
        # self.encoder.add(layers.MaxPooling2D((2,2), padding='same'))
        
        # Decompresses
        self.decoder = tf.keras.Sequential(name='decoder')
        self.decoder.add(layers.Dense(32, activation='relu'))
        self.decoder.add(layers.Dense(tf.math.reduce_prod(shape), 
                                      activation='sigmoid'))
        self.decoder.add(layers.Reshape(shape))
        
        # self.decoder.add(layers.Conv2DTranspose(32, (2, 2), strides=(2,2), activation='relu', padding='same'))
        # self.decoder.add(layers.Conv2DTranspose(32, (2, 2), strides=(2,2), activation='relu', padding='same'))
        # self.decoder.add(layers.Conv2D(1, (2, 2), activation='sigmoid'))
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded