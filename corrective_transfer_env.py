"""
Author: Lee Violet Ong
Date: 09/07/25

General Notes:
- Currently doesn't support rendering as it is deemed unnecessary and impacts the computational complexity of the training
"""

import random

import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import pykep as pk


class CorrectiveTransferEnvironment(gym.Env):
    def __init__(self, traj_filename: str, impulse_filname: str):
        super().__init__()

        # define universal parameters
        self.sun_mu: float = 1.32712440018e11
        self.au: float = 1.49597870691e8  # km
        self.ve: float = np.sqrt(self.sun_mu / self.au)  # orbital velocity of earth

        # define required information from SCP data
        self.nominal_traj: np.ndarray = pd.read_csv(
            f"nominal_trajectory/{traj_filename}"
        ).to_numpy()
        self.num_timesteps: int = len(self.nominal_traj) - 1
        self.max_m: float = self.nominal_traj[0, -1]
        self.nominal_imp: np.ndarray = pd.read_csv(
            f"nominal_trajectory/{impulse_filname}"
        ).to_numpy()

        # TODO: move to a config file
        # task config (doi: 10.1016/j.actaastro.2023.10.018)
        self.tof: float = 348.79  # in days
        self.timestep: float = (
            (self.tof / self.num_timesteps) * 24 * 60 * 60
        )  # in seconds

        # dynamics uncertainties config (in km)
        self.dyn_pos_sd: float = 1.0
        self.dyn_vel_sd: float = 0.05
        self.dyn_m_sd: float = 1.0

        # thruster config
        self.max_thrust: float = 0.5
        self.exhaust_vel: float = 19.6133

        # define the spaces ie. all possible range of obs and action
        # NOTE: 2*au should be sufficient for the application of mars transfer
        # 2*ve at any point of the orbit results to an unbounded trajectory, which means it is unlikely for s/c have a vel beyond that
        # as interplanetary transfers in heliocentric frame would always be elliptical
        # [rx, ry, rz, vx, vy, vz, m]
        earth_constraints: np.ndarray = np.array(
            [self.au, self.au, self.au, self.ve, self.ve, self.ve]
        )
        self.observation_space: spaces.Box = spaces.Box(
            low=np.concatenate((-2 * earth_constraints, [0.0])),
            high=np.concatenate((2 * earth_constraints, [self.max_m])),
            dtype=np.float64,
        )
        # NOTE: ideally mag = [0,1] but the range chosen is [-1,1] for standardised distribution so its easier to learn?
        # [vmag, vx, vy, vz]
        self.action_space: spaces.Box = spaces.Box(
            low=np.array(4 * [-1.0]),
            high=np.array(4 * [1.0]),
            dtype=np.float64,
        )

        # init state is the first state by default (no noise)
        self.state: np.ndarray = self.nominal_traj[0, :]
        self.chosen_timestamp: int = 0
        self.noise: np.ndarray = np.array([0] * 4)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # for now, randomly choose the perturbed state with uniform probability
        self.chosen_timestamp = random.randint(0, self.num_timesteps - 1)
        chosen_state: np.ndarray = self.nominal_traj[self.chosen_timestamp, :]

        # covariance matrix set up
        pos_var: float = np.pow(self.dyn_pos_sd, 2)
        vel_var: float = np.pow(self.dyn_vel_sd, 2)
        m_var: float = np.pow(self.dyn_m_sd, 2)
        cov: np.ndarray = np.diag(
            [pos_var, pos_var, pos_var, vel_var, vel_var, vel_var, m_var]
        )
        mean: np.ndarray = np.array([0] * 7)

        # choose the gaussian noise for the chosen state
        self.noise = np.random.multivariate_normal(mean, cov)
        self.state = chosen_state + self.noise

        # obs, info
        return self.state, {}

    def step(self, action):
        # TODO: constrain the total impulse to vmax
        # compute the vmax based on the mass before impulse
        vmax: float = self.max_thrust * self.timestep / self.state[-1]

        # compute the corrective impulse vector
        action_dir: np.ndarray = np.array(action[1:4])
        corrective_impulse: np.ndarray = (
            vmax * (1 + action[0]) / 2 * action_dir / np.linalg.norm(action_dir)
        )

        # add the corrective impulse to the current state
        # creates a symbolic link rather than a copy
        current_pos: np.ndarray = self.state[0:3]
        current_vel: np.ndarray = self.state[3:6]
        current_m: float = self.state[-1]

        current_vel += corrective_impulse

        # propagate to the final timestamp
        # NOTE: could use pykep propagate_lagrangian function (ref: https://esa.github.io/pykep/documentation/core.html#pykep.propagate_lagrangian)
        total_impulse: np.ndarray = (
            corrective_impulse + self.nominal_imp[self.chosen_timestamp]
        )

        r_next, v_next = np.array(
            pk.propagate_lagrangian(
                r0=current_pos,
                v0=current_vel,
                tof=self.timestep,
                mu=self.sun_mu,
            )
        )
        m_next = self._mass_update(current_m, total_impulse)

        # for logging purposes
        log_pos: np.ndarray = np.array([current_pos, r_next])
        log_vel: np.ndarray = np.array([current_vel])
        log_m: np.ndarray = np.array([current_m, m_next])

        # continue propagating until the final state, incorporating the nominal impulse
        for i in range(self.chosen_timestamp + 1, self.num_timesteps + 1):
            applied_impulse: np.ndarray = self.nominal_imp[i]
            v_next += applied_impulse
            log_vel = np.append(log_vel, [v_next], axis=0)

            # final state reached, no need propagation
            if i == self.num_timesteps:
                break

            # propagate next step
            r_next, v_next = np.array(
                pk.propagate_lagrangian(
                    r0=r_next,
                    v0=v_next,
                    tof=self.timestep,
                    mu=self.sun_mu,
                )
            )
            m_next = self._mass_update(m_next, applied_impulse)

            # update logging
            log_pos = np.append(log_pos, [r_next], axis=0)
            log_m = np.append(log_m, m_next)

        # compute error and compute reward
        # reward structure: penalty = - |r-rf| - |v-vf|
        r_final: np.ndarray = self.nominal_traj[-1, 0:3]
        v_final: np.ndarray = self.nominal_traj[-1, 3:6]

        # TODO: consider other forms of reward functions
        reward: float = -np.linalg.norm(r_next - r_final) - np.linalg.norm(
            v_next - v_final
        )

        # terminal state, reward, done, truncated, info (unused)
        return np.concatenate((r_next, v_next, [m_next])), reward, 1, 0, {}

    def _mass_update(self, m0: float, impulse: np.ndarray) -> float:
        """
        Implements the Tsiolkovsky Rocket Equation for the mass update.

        Arguments:
        - m0: the current mass at t before impulse
        - impulse: the total impulse vector at t
        """
        return m0 * np.exp(-np.linalg.norm(impulse) / self.exhaust_vel)
