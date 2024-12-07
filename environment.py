#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:31:50 2019

@author: farismismar
"""

from gym import spaces
from gym.utils import seeding
import numpy as np


class radio_environment:
    '''    
        Observation: 
            Type: Box (or Discrete)
            Num Observation                                    Min      Max
            0   Received SINR                                  min      max
            1   Current SINR (before change)                   min      max
            2   Power control (-3, -1, 1, 3)                    0       3
    '''                                
    def __init__(self, action_size, min_reward, max_reward, target, max_step_count, seed):
        self.seed(seed=seed)        
        self._step_count = 0 # which step
        self._max_step_count = max_step_count

        # These change based on the use case.
        self.num_actions = action_size
        self.min_sinr = -3 # in dB
        self.max_sinr = 30 # in dB        
        self.power_control = [-3, -1, 1, 3] # in dB
        self.sinr_target = target
        
        # For Reinforcement Learning
        # Initialize observation space.
        observability_lower_bounds = np.array([self.min_sinr, self.min_sinr, 0])
        observability_upper_bounds = np.array([self.max_sinr, self.max_sinr, len(self.power_control) - 1])
        self.observation_space = spaces.Box(observability_lower_bounds, observability_upper_bounds, dtype=np.float32) # spaces.Discrete(2) # state size is here 
        self.state = None
        
        # Initialize action space.
        self.action_space = spaces.Discrete(self.num_actions) # action size is here
        self.action = None
        
        # Initialize rewards
        self.min_reward = min_reward
        self.max_reward = max_reward
        
      
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    
    def reset(self):
        # Initialize power control to any of the commands, at random
        power_control_index = self.np_random.integers(len(self.power_control))
        
        # Initialize the action
        self.action = -1
        
        # Initialize the observation.
        self.state = [0, 0, power_control_index] 
        
        # Initialize the number of steps.
        self._step_count = 0

        return np.array(self.state)

    
    def step(self, action):
        # This is the most elaborate method in this class.
        self.action = action
        state = self.state
        
        _, current_received_sinr, power_control_index = state
        
        if (action != -1):
            self._step_count += 1
        
        if action == 0:
            power_control_index = (power_control_index + 1) % len(self.power_control)
        
        if action == 1:
            power_control_index = (power_control_index - 1) % len(self.power_control)
            
        if action == 2:
            True # do nothing
            
        # This should never happen.
        if (action > self.num_actions - 1):
            print('WARNING: Invalid action played!')
            reward = 0
            return [], 0, False, True    
        
        received_sinr = self._compute_rf(current_received_sinr, power_control_index)
        
        # Did we find a FEASIBLE NON-DEGENERATE solution?
        done = (received_sinr >= self.min_sinr) and (received_sinr <= self.max_sinr) and (received_sinr >= self.sinr_target)
        abort = (done and (self._step_count < self._max_step_count - 1)) or (received_sinr > self.max_sinr) or (received_sinr < self.min_sinr) # premature solution or infeasible one is not desired.
                
        # the reward is the gain in the SINR
        reward = (received_sinr - current_received_sinr)

        if abort == True:
            done = False
            reward = self.min_reward

        if done and not abort:
            reward += self.max_reward
                
        # Update the state.        
        self.state = [current_received_sinr, received_sinr, power_control_index]
     
        if (action == -1):    
            # This action is only a transient.
            return np.array(self.state), reward, False, False
                
        return np.array(self.state), reward, done, abort
    

    # This is a very, very, simplified function in the environment.
    # It changes the quantities that directly impact the observability space.
    def _compute_rf(self, current_received_sinr, power_control_index):
        received_sinr = current_received_sinr + self.power_control[power_control_index]

        return received_sinr
