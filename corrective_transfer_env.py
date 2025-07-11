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
        self.sun_mu: float = 1.32712440018e11
        self.au: float = 1.49597870691e8 # km
        self.ve: float = np.sqrt(self.sun_mu/self.au) # orbital velocity of earth

        # define required information from SCP data
        self.nominal_traj:np.ndarray = pd.read_csv(f"nominal_trajectory/{filename}").to_numpy()
        self.num_timesteps: int = len(self.nominal_traj) - 2 # remove the final state
        self.max_m: float = self.nominal_traj[0, -1]

        # task config 
        # dynamics uncertainties config (doi: 10.1016/j.actaastro.2023.10.018)
        self.dyn_pos_sd: float = 1.0
        self.dyn_vel_sd: float = 0.05
        self.dyn_m_sd: float = 1.0

        # define the spaces ie. all possible range of obs and action
        # NOTE: 2*au should be sufficient for the application of mars transfer
        # 2*ve at any point of the orbit results to an unbounded trajectory, which means it is unlikely for s/c have a vel beyond that
        # as interplanetary transfers in heliocentric frame would always be elliptical
        earth_constraints: np.ndarray = np.array([self.au, self.au, self.au, self.ve, self.ve, self.ve])
        self.observation_space: spaces.Box = spaces.Box(
            low= np.concatenate((-2*earth_constraints, [0.0])),
            high= np.concatenate((2*earth_constraints, [self.max_m])),
            dtype= np.float32
        )
        # NOTE: ideally mag = [0,1] but the range chosen is [-1,1] for standardised distribution so its easier to learn? 
        self.action_space: spaces.Box = spaces.Box(
            low= np.array(4*[-1.0]),
            high= np.array(4*[1.0]),
            dtype= np.float32 # TODO: will resolution impact results
        )
        
        # init state is the first state by default (no noise)
        self.state: np.ndarray = self.nominal_traj[0,:]

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
        self.state = perturbed_state

        # obs, info
        return self.state, {}






    

        