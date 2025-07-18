"""
Author: Lee Violet Ong
Date: 15/07/25

This file defines the test suite for the corrective transfer gym, build to validate the logical implementation of the gym.
This also contains test training scripts for the gym.
"""

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common import env_checker

from corrective_transfer_env import CorrectiveTransferEnvironment


def test_prop():
    """
    Tests if the propagator works without perturbation and control
    - guided and unguided should be the same in this case so expected reward is ZERO
    - the propagated state should be close to the desired final state since no perturbations
    """
    env: gym.Env = CorrectiveTransferEnvironment("SCP_impulsive_traj.csv", "SCP_dV.csv")
    final_state, reward, _, _, _ = env.step([-1, 1, 1, 1])
    state_diff: np.ndarray = env.nominal_traj[-1, :] - final_state

    # error ~1e-7 which could be attributed to rounding within propagator
    assert (
        np.linalg.norm(state_diff) < 1e-6
    ), "Propagator error exceeds acceptable threshold"
    assert reward == 0, "Error between guided and unguided trajectory"

    print("PASS")


def test_max_control():
    """
    Checks for the maximum state values, for defining the observation space. This was as a result of an assertion error
    (exceeded obs space) from `test_sb3_integration()`.

    2*au should be sufficient for the application of mars transfer
    2*ve at any point of the orbit results to an unbounded trajectory, which means it is unlikely for s/c have a
    vel beyond that as interplanetary transfers in heliocentric frame would always be elliptical
    """
    env: gym.Env = CorrectiveTransferEnvironment("SCP_impulsive_traj.csv", "SCP_dV.csv")
    nominal_control: np.ndarray = env.nominal_imp[0]
    nominal_dir: np.ndarray = nominal_control / np.linalg.norm(nominal_control)
    final_state, reward, _, _, _ = env.step(np.concatenate(([1], nominal_dir)))

    print(final_state, env.au * 2)


def test_sb3_integration():
    env: gym.Env = CorrectiveTransferEnvironment("SCP_impulsive_traj.csv", "SCP_dV.csv")
    env_checker.check_env(env, warn=False, skip_render_check=True)


def test_script():
    env: gym.Env = CorrectiveTransferEnvironment("SCP_impulsive_traj.csv", "SCP_dV.csv")
    model: sb3.PPO = sb3.PPO("MlpPolicy", env, verbose=1, n_steps=2)
    model.learn(25000)
    model.save("corrective_env_test")


if __name__ == "__main__":
    test_script()
