# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 18:53:59 2023

@author: farismismar
"""

import tensorflow as tf

import random
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# https://www.tensorflow.org/tutorials/generative/autoencoder

class Autoencoder(Model):
    def __init__(self, latent_dim, shape, seed=None):
        random.seed(seed)        
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
        
        # Decompresses
        self.decoder = tf.keras.Sequential(name='decoder')
        self.decoder.add(layers.Dense(32, activation='relu'))
        self.decoder.add(layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'))
        self.decoder.add(layers.Reshape(shape))
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded