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
    def __init__(self, traj_filename:str, impulse_filname:str):
        super().__init__()

        # define universal parameters
        self.sun_mu: float = 1.32712440018e11
        self.au: float = 1.49597870691e8 # km
        self.ve: float = np.sqrt(self.sun_mu/self.au) # orbital velocity of earth

        # define required information from SCP data
        self.nominal_traj: np.ndarray = pd.read_csv(f"nominal_trajectory/{traj_filename}").to_numpy()
        self.num_timesteps: int = len(self.nominal_traj) - 2 # remove the final state
        self.max_m: float = self.nominal_traj[0, -1]
        self.nominal_imp: np.ndarray =  pd.read_csv(f"nominal_trajectory/{impulse_filname}").to_numpy()

        # TODO: move to a config file
        # task config (doi: 10.1016/j.actaastro.2023.10.018)
        self.tof: float = 348.79 # in days
        self.timestep: float = (self.tof/self.num_timesteps)*24*60*60 # in seconds

        # dynamics uncertainties config
        self.dyn_pos_sd: float = 1.0
        self.dyn_vel_sd: float = 0.05
        self.dyn_m_sd: float = 1.0

        # thruster config
        self.max_thrust: float = 0.5 


        # define the spaces ie. all possible range of obs and action
        # NOTE: 2*au should be sufficient for the application of mars transfer
        # 2*ve at any point of the orbit results to an unbounded trajectory, which means it is unlikely for s/c have a vel beyond that
        # as interplanetary transfers in heliocentric frame would always be elliptical
        # [rx, ry, rz, vx, vy, vz, m]
        earth_constraints: np.ndarray = np.array([self.au, self.au, self.au, self.ve, self.ve, self.ve])
        self.observation_space: spaces.Box = spaces.Box(
            low= np.concatenate((-2*earth_constraints, [0.0])),
            high= np.concatenate((2*earth_constraints, [self.max_m])),
            dtype= np.float32
        )
        # NOTE: ideally mag = [0,1] but the range chosen is [-1,1] for standardised distribution so its easier to learn?
        # [vmag, vx, vy, vz] 
        self.action_space: spaces.Box = spaces.Box(
            low= np.array(4*[-1.0]),
            high= np.array(4*[1.0]),
            dtype= np.float32 # TODO: will resolution impact results
        )
        
        # init state is the first state by default (no noise)
        self.state: np.ndarray = self.nominal_traj[0,:]
        self.chosen_timestamp: int = 0

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        # for now, randomly choose the perturbed state with uniform probability
        self.chosen_timestamp = random.randint(0, self.num_timesteps)
        chosen_state: np.ndarray = self.nominal_traj[self.chosen_timestamp, :]

        # choose the gaussian noise for the chosen state 
        pos_noise: np.ndarray = np.random.normal(0, self.dyn_pos_sd, (3))
        vel_noise: np.ndarray = np.random.normal(0, self.dyn_vel_sd, (3))
        m_noise: float = np.random.normal(0, self.dyn_m_sd, (1))

        noise: np.ndarray = np.concatenate((pos_noise, vel_noise, m_noise))
        perturbed_state: np.ndarray = chosen_state + noise; 
        self.state = perturbed_state

        # obs, info
        return self.state, {}

    def step(self, action):
        # compute the vmax based on the mass before impulse 
        vmax: float = self.max_thrust*self.timestep/self.state[-1]

        # compute the corrective impulse vector
        action_dir: np.ndarray = action[1:-1]
        corrective_impulse: np.ndarray = vmax * (1+action[0])/2 * action_dir/np.linalg.norm(action_dir)

        # add the corrective impulse to the current state
        current_state: np.ndarray = self.state
        current_state[3:5] += corrective_impulse

        # propagate to the final timestamp 
        # NOTE: could use pykep propagate_lagrangian function (ref: https://esa.github.io/pykep/documentation/core.html#pykep.propagate_lagrangian)

        # compute error and compute reward

        # terminal state, reward, done, truncated, info (unused)
        return super().step(action)





    

        