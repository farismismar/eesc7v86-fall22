#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:02:40 2019

@author: farismismar
"""

import random
import numpy as np
from collections import deque

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


class DQNLearningAgent:
    def __init__(self, state_size, action_size, seed,
                 discount_factor=0.995, learning_rate=1e-4, prefer_gpu=True,
                 exploration_rate=1.0,
                 exploration_decay_rate=0.9995):
                               
        self.memory = deque(maxlen=2000) # replay buffer.
        self.gamma = discount_factor  # discount rate
        self.exploration_rate = exploration_rate / exploration_decay_rate # exploration rate
        self.exploration_rate_min = 0.1
        self.exploration_rate_decay = exploration_decay_rate
        self.learning_rate = learning_rate

        self._state_size = state_size
        self._action_size = action_size 
                  
        # Add a few lines to caputre the seed for reproducibility.
        self.seed = seed
        random.seed(self.seed)
        self.np_random = np.random.RandomState(seed=seed)
        tf.random.set_seed(seed) # sets global random seed
        
        use_cuda = len(tf.config.list_physical_devices('GPU')) > 0 and prefer_gpu
        self.device = "/gpu:0" if use_cuda else "/cpu:0"
        
        if self.device == '/cpu:0':
            print("WARNING: CPU is used for training.")

        self.model = self._build_model()

        
    def begin_episode(self, observation):
        # Reduce exploration over time.
        self.exploration_rate *= self.exploration_rate_decay
        if (self.exploration_rate < self.exploration_rate_min):
            self.exploration_rate = self.exploration_rate_min
            
        # return an action at random
        action = random.randrange(self._action_size)

        return action


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        # This is a state-to-largest Q converter to find best action, basically
        model = tf.keras.Sequential()
        model.add(layers.Dense(24, input_dim=self._state_size, activation='relu'))
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(self._action_size, activation='relu'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        
        return model

    
    def _construct_training_set(self, replay):
        # Select states and next states from replay memory
        # which has the structure, state,action, reward, next state, and done.
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])

        # Predict the expected Q of current state and new state using DQN
        with tf.device(self.device):
            Q = self.model.predict(states, verbose=False)
            Q_new = self.model.predict(new_states, verbose=False)

        replay_size = len(replay)
        X = np.empty((replay_size, self._state_size))
        y = np.empty((replay_size, self._action_size))
        
        # Construct training set
        for i in range(replay_size):
            state_r, action_r, reward_r, new_state_r, done_r = replay[i]

            target = Q[i]
            target[action_r] = reward_r
            
            if not done_r:
                target[action_r] += self.gamma * np.amax(Q_new[i])

            X[i] = state_r
            y[i] = target

        return X, y

    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # Make sure we restrict memory size to specified limit
        if len(self.memory) > 2000:
            self.memory.pop(0)


    def act(self, state):
        # Exploration/exploitation: choose a random action or select the best one.
        if self.np_random.uniform(0, 1) <= self.exploration_rate:
            return random.randrange(self._action_size)
        
        state = np.reshape(state, [1, self._state_size])
        with tf.device(self.device):
            act_values = self.model.predict(state, verbose=False)
            
        return np.argmax(act_values[0])  # returns action

    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        X, y = self._construct_training_set(minibatch)
        with tf.device(self.device):
            loss = self.model.train_on_batch(X, y)
        
        _q = np.mean(y)
        return loss, _q

                
    # def update_target_model(self):
    #     # copy weights from model to target_model
    #     self.target_model.set_weights(self.model.get_weights())
    #     return


    # def load(self, name):
    #     self.model.load_weights(name)

        
    # def save(self, name):
    #     self.model.save_weights(name)