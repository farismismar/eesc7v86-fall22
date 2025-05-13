#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 20:46:19 2019

@author: farismismar
"""

import random
import numpy as np


class QLearningAgent:
    def __init__(self, state_size, action_size, seed,
                 discount_factor=0.995, learning_rate=0.1,
                 exploration_rate=1.0,
                 exploration_decay_rate=0.9995):

        self.losses = []
        self.gamma = discount_factor  # discount rate
        self.exploration_rate = exploration_rate / exploration_decay_rate # exploration rate
        self.exploration_rate_min = 0.1
        self.exploration_decay_rate = exploration_decay_rate
        self.learning_rate = learning_rate
        
        self._state_size = state_size
        self._action_size = action_size 
        
        self.state = None
        self.action = None
        
        self.oversampling = 1 # for discretization.  Increasing may cause memory exhaust.
        
        # Add a few lines to caputre the seed for reproducibility.
        self.seed = seed
        random.seed(self.seed)
        self.np_random = np.random.RandomState(seed=seed)
        
        # Discretize the continuous state space for each of the features.
        num_discretization_bins = self._state_size * self.oversampling
        
        # check the site distance configuration in the environment
        self._state_bins = [
            # Current SINR
            self._discretize_range(-3, 30, num_discretization_bins),
            # Received SINR
            self._discretize_range(-3, 30, num_discretization_bins),
            # Power Command index
            np.arange(0, 4)
        ]
        
        # Create a clean Q-Table.
        self._max_bins = max(len(bin_i) for bin_i in self._state_bins)
        num_states = (self._max_bins + 1) ** len(self._state_bins)
        self.q = np.zeros(shape=(num_states, self._action_size))

        
    def begin_episode(self, observation):
        # Reduce exploration over time.
        self.exploration_rate *= self.exploration_decay_rate
        if (self.exploration_rate < self.exploration_rate_min):
            self.exploration_rate = self.exploration_rate_min
    
        self.state = self._build_state(observation)
        
        return np.argmax(self.q[self.state, :]) # returns the action with largest Q

    
    def act(self, observation, reward):
        next_state =  self._build_state(observation)
        state = self.state

        if self.np_random.uniform(0, 1) < self.exploration_rate:
            # Explore: random action
            next_action = random.randrange(self._action_size)
        else:
            # Exploit: action with the highest Q-value
            next_action = np.argmax(self.q[next_state])
        
        # Learn: update Q-learning table based on current reward and future action.
        loss = reward + self.gamma * max(self.q[next_state, :]) - self.q[state, self.action]
        self.q[state, self.action] += self.learning_rate * loss
    
        self.losses.append(np.mean(loss))
        
        self.state = next_state
        self.action = next_action
        return next_action


    def get_performance(self):
        return self.losses, self.q.mean()


    # Private members:
    def _build_state(self, observation):
        # Discretize the observation features and reduce them to a single integer.
        state = sum(
            self._discretize_value(feature, self._state_bins[i]) * ((self._max_bins + 1) ** i)
            for i, feature in enumerate(observation)
        )
        return state

    
    def _discretize_value(self, value, bins):
        return np.digitize(x=value, bins=bins)

    
    def _discretize_range(self, lower_bound, upper_bound, num_bins):
        return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]
