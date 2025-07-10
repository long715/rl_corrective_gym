'''
Author: Lee Violet Ong
Date: 09/07/25

General Notes: 
- Currently doesn't support rendering as it is deemed unnecessary and impacts the computational complexity of the training
'''
import random

import gymnasium as gym
from gymnasium import spaces
import pandas as pd 
import numpy as np

class CorrectiveTransferEnvironment(gym.Env):
    def __init__(self, filename:str):
        super().__init__()

        # define universal parameters

        # define required information from SCP data
        self.nominal_traj:np.ndarray = pd.read_csv(f"nominal_trajectory/{filename}").to_numpy()
        self.num_timesteps: int = len(self.nominal_traj) - 2 # remove the final state

        # task config 
        # dynamics uncertainties config (doi: 10.1016/j.actaastro.2023.10.018)
        self.dyn_pos_sd: float = 1.0
        self.dyn_vel_sd: float = 0.05
        self.dyn_m_sd: float = 1.0

        # define information for task
        self.observation_space = None
        self.action_space = None 

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        # for now, randomly choose the perturbed state with uniform probability
        chosen_timestamp: int = random.randint(0, self.num_timesteps)
        chosen_state: np.ndarray = self.nominal_traj[chosen_timestamp, :]

        # choose the gaussian noise for the chosen state 
        pos_noise: np.ndarray = np.random.normal(0, self.dyn_pos_sd, (3))
        vel_noise: np.ndarray = np.random.normal(0, self.dyn_vel_sd, (3))
        m_noise: float = np.random.normal(0, self.dyn_m_sd, (1))

        noise: np.ndarray = np.concatenate((pos_noise, vel_noise, m_noise))
        perturbed_state: np.ndarray = chosen_state + noise; 






    

        