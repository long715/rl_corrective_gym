"""
Author: Lee Violet Ong
Date: 09/07/25

General Notes:
- Currently doesn't support rendering as it is deemed unnecessary and impacts the computational complexity of the training
"""

from functools import cached_property
import random
import copy

import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import pykep as pk

from rl_corrective_gym.gym_env_setup.space_env_config import SpaceEnvironmentConfig


class CorrectiveTransferEnvironment(gym.Env):
    def __init__(
        self,
        config: SpaceEnvironmentConfig,
    ):
        super().__init__()

        traj_filename: str = config.traj_filename
        impulse_filename: str = config.impulse_filename

        # define universal parameters
        self.sun_mu: float = 1.32712440018e11
        self.au: float = 1.49597870691e8  # km
        self.ve: float = np.sqrt(self.sun_mu / self.au)  # orbital velocity of earth

        # define required information from SCP data
        self.nominal_traj: np.ndarray = pd.read_csv(
            f"../../rl_corrective_gym/rl_corrective_gym/nominal_trajectory/{traj_filename}"
        ).to_numpy()

        self.num_timesteps: int = len(self.nominal_traj) - 1
        self.max_m: float = self.nominal_traj[0, -1]
        self.nominal_imp: np.ndarray = pd.read_csv(
            f"../../rl_corrective_gym/rl_corrective_gym/nominal_trajectory/{impulse_filename}"
        ).to_numpy()

        # task config (doi: 10.1016/j.actaastro.2023.10.018)
        self.tof: float = config.tof  # in days
        self.timestep: float = (
            (self.tof / self.num_timesteps) * 24 * 60 * 60
        )  # in seconds

        # dynamics uncertainties config (in km)
        self.dyn_pos_sd: float = config.dyn_pos_sd
        self.dyn_vel_sd: float = config.dyn_vel_sd

        # thruster config
        self.max_thrust: float = config.max_thrust
        self.exhaust_vel: float = config.exhaust_vel

        # reward
        self.penalty_scale_control: float = 100.0
        self.penalty_scale_dynamics: float = 10.0
        self.penalty_scale_effort: float = 10.0

        # define the spaces ie. all possible range of obs and action
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

    @cached_property
    def max_action_value(self) -> float:
        return self.action_space.high[0]

    @cached_property
    def min_action_value(self) -> float:
        return self.action_space.low[0]

    @cached_property
    def observation_space_shape(self) -> int:
        return self.observation_space.shape[0]

    @cached_property
    def action_num(self) -> int:
        if isinstance(self.action_space, spaces.Box):
            action_num = self.action_space.shape[0]
        elif isinstance(self.action_space, spaces.Discrete):
            action_num = self.action_space.n
        else:
            raise ValueError(f"Unhandled action space type: {type(self.action_space)}")
        return action_num

    def sample_action(self) -> int:
        return self.action_space.sample()

    def set_seed(self, seed: int) -> None:
        _ = self.reset(seed=seed)
        # Note issues: https://github.com/rail-berkeley/softlearning/issues/75
        self.action_space.seed(seed)

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        return np.ndarray([])

    def get_overlay_info(self) -> dict:
        return {}

    def reset(self, *, seed=None, options=None) -> np.ndarray:
        super().reset(seed=seed, options=options)

        # for now, randomly choose the perturbed state with uniform probability
        self.chosen_timestamp = random.randint(0, self.num_timesteps - 1)
        chosen_state: np.ndarray = self.nominal_traj[self.chosen_timestamp, :]

        # covariance matrix set up
        pos_var: float = np.power(self.dyn_pos_sd, 2)
        vel_var: float = np.power(self.dyn_vel_sd, 2)
        cov: np.ndarray = np.diag(
            [pos_var, pos_var, pos_var, vel_var, vel_var, vel_var]
        )
        mean: np.ndarray = np.array([0] * 6)

        # choose the gaussian noise for the chosen state
        self.noise = np.concatenate(
            (np.random.multivariate_normal(mean, cov), np.array([0]))
        )
        self.state = chosen_state + self.noise

        # obs
        return self.state

    def step(self, action) -> tuple:
        # compute the vmax based on the mass before impulse
        vmax: float = self.max_thrust * self.timestep / self.state[-1]
        nominal_impulse: np.ndarray = self.nominal_imp[self.chosen_timestamp]

        corrective_impulse: np.ndarray = self._get_control_input(vmax, action)
        total_impulse: np.ndarray = corrective_impulse + nominal_impulse

        # add the corrective impulse to the current state
        # creates a symbolic link rather than a copy
        no_guid_pos: np.ndarray = copy.deepcopy(self.state[0:3])
        no_guid_vel: np.ndarray = copy.deepcopy(self.state[3:6])
        no_guid_m: float = copy.deepcopy(self.state[-1])

        guid_pos: np.ndarray = copy.deepcopy(self.state[0:3])
        guid_vel: np.ndarray = copy.deepcopy(self.state[3:6]) + corrective_impulse
        guid_m: float = copy.deepcopy(self.state[-1])

        # propagate to the final timestamp
        # NOTE: could use pykep propagate_lagrangian function (ref: https://esa.github.io/pykep/documentation/core.html#pykep.propagate_lagrangian)

        no_guid_m = self._mass_update(no_guid_m, nominal_impulse)
        guid_m = self._mass_update(guid_m, total_impulse)

        no_guid_pos, no_guid_vel = np.array(
            pk.propagate_lagrangian(
                r0=no_guid_pos,
                v0=no_guid_vel,
                tof=self.timestep,
                mu=self.sun_mu,
            )
        )
        guid_pos, guid_vel = np.array(
            pk.propagate_lagrangian(
                r0=guid_pos,
                v0=guid_vel,
                tof=self.timestep,
                mu=self.sun_mu,
            )
        )

        # continue propagating until the final state, incorporating the nominal impulse
        for i in range(self.chosen_timestamp + 1, self.num_timesteps + 1):
            nominal_impulse = self.nominal_imp[i]

            no_guid_vel += nominal_impulse
            guid_vel += nominal_impulse

            # final state reached, no need propagation
            if i == self.num_timesteps:
                break

            no_guid_pos, no_guid_vel = np.array(
                pk.propagate_lagrangian(
                    r0=no_guid_pos,
                    v0=no_guid_vel,
                    tof=self.timestep,
                    mu=self.sun_mu,
                )
            )
            guid_pos, guid_vel = np.array(
                pk.propagate_lagrangian(
                    r0=guid_pos,
                    v0=guid_vel,
                    tof=self.timestep,
                    mu=self.sun_mu,
                )
            )

            no_guid_m = self._mass_update(no_guid_m, nominal_impulse)
            guid_m = self._mass_update(guid_m, nominal_impulse)

        # terminal state, reward, done, truncated, info (unused)
        no_gui_xf: np.ndarray = np.concatenate((no_guid_pos, no_guid_vel, [no_guid_m]))
        gui_xf: np.ndarray = np.concatenate((guid_pos, guid_vel, [guid_m]))
        total_reward, reward_effort, reward_control_penalty, reward_dyn = (
            self._reward_function(vmax, corrective_impulse, gui_xf, no_gui_xf)
        )
        info: dict = {
            "reward_effort": reward_effort,
            "reward_control_penalty": reward_control_penalty,
            "reward_dyn": reward_dyn,
            "timestep": self.chosen_timestamp,
            "noise": self.noise,
            "corrective_impulse": corrective_impulse,
            "terminal_state": gui_xf,
        }
        return gui_xf, total_reward, True, False, info

    def _mass_update(self, m0: float, impulse: np.ndarray) -> float:
        """
        Implements the Tsiolkovsky Rocket Equation for the mass update.

        Arguments:
        - m0: the current mass at t before impulse
        - impulse: the total impulse vector at t
        """
        return m0 * np.exp(-np.linalg.norm(impulse) / self.exhaust_vel)

    def _reward_function(
        self,
        vmax: float,
        control_imp: np.ndarray,
        guid_xf: np.ndarray,
        no_guid_xf: np.ndarray,
    ):
        nominal_imp: np.ndarray = self.nominal_imp[self.chosen_timestamp]
        total_corrective_imp: np.ndarray = nominal_imp + control_imp

        # reward/penalty for effort
        nominal_imp_mag: float = np.linalg.norm(nominal_imp)
        reward_effort: float = (
            (nominal_imp_mag - np.linalg.norm(total_corrective_imp))
            / nominal_imp_mag
            * self.penalty_scale_effort
        )

        # penalty for exceeding the control limits
        # NOTE: with constraints, this should be zero
        control_diff: float = vmax - np.linalg.norm(total_corrective_imp)
        reward_control_penalty: float = 0
        if control_diff < 0:
            reward_control_penalty = (control_diff / vmax) * self.penalty_scale_control

        # reward/penalty for dynamics
        nom_rv_final: np.ndarray = self.nominal_traj[-1, :]
        error_no_guid: np.ndarray = no_guid_xf - nom_rv_final
        error_guid: np.ndarray = guid_xf - nom_rv_final

        # NOTE: for now euclidean, can change into weighted norm
        error_no_guid_mag: float = np.linalg.norm(error_no_guid[0:6])
        reward_dyn = (
            (error_no_guid_mag - np.linalg.norm(error_guid[0:6]))
            / error_no_guid_mag
            * self.penalty_scale_dynamics
        )

        total_reward: float = reward_effort + reward_control_penalty + reward_dyn
        return total_reward, reward_effort, reward_control_penalty, reward_dyn

    def _get_control_input(self, vmax: float, action) -> np.ndarray:
        """
        As the mass is unchanged, chosen control input will always be bounded.
        We can find the vmax at a given direction by solving for u in the following:
        || v_norm + u*control_dir_unit|| = vmax
        ||v||^2 + u^2 + 2u v.i = vmax

        which can be rearraged to a quadratic formula:
        u^2 + Au + B = 0
        A = 2 v.i
        B = ||v||^2 - vmax^2

        chosen u will be max of the roots
        """
        nominal_imp: np.ndarray = self.nominal_imp[self.chosen_timestamp]
        action_dir: np.ndarray = np.array(action[1:4])
        action_unit: np.ndarray = action_dir / np.linalg.norm(action_dir)

        A: float = 2 * np.dot(nominal_imp, action_unit)
        B: float = np.power(np.linalg.norm(nominal_imp), 2) - np.power(vmax, 2)
        roots: np.ndarray = np.roots([1, A, B])

        corrective_mag = np.max(roots)
        return corrective_mag * (1 + action[0]) / 2 * action_unit

    def _law_of_cosine(self, theta: float, a: float, c: float):
        """
        law of cosine: c^2 = a^2 + b^2 - 2ab cos(theta')

        b^2 + Ab + B = 0
        where:
            A = -2a cos(theta')
            B = a^2 - c^2
            theta' = 180 - theta

        return the positive root

        Arguments:
        - theta: in degrees
        - a, c: adjacent and opposite sides
        """
        A: float = -2 * a * np.cos(np.radians(180 - theta))
        B: float = np.power(a, 2) - np.power(c, 2)
        roots: np.ndarray = np.roots([1, A, B])

        return np.max(roots)
