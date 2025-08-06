"""
Author: Lee Violet Ong
Date: 15/07/25

This file defines the test suite for the corrective transfer gym, build to validate the logical implementation of the gym.
This also contains test training scripts for the gym.
"""

import argparse
import json

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common import env_checker
from stable_baselines3.common.evaluation import evaluate_policy

# TODO: was used in the initial testing for preliminary validation, would like to
# eventually update the test functions to use the new one
from rl_corrective_gym.corrective_transfer_env import OldCorrectiveTransferEnvironment

#  use for integration testing for actual training runsS
from rl_corrective_gym.gym_env_setup.corrective_transfer_env import (
    CorrectiveTransferEnvironment,
)
from rl_corrective_gym.gym_env_setup.space_env_config import SpaceEnvironmentConfig


def test_prop():
    """
    Tests if the propagator works without perturbation and control
    - guided and unguided should be the same in this case so expected reward is ZERO
    - the propagated state should be close to the desired final state since no perturbations
    """
    env: gym.Env = OldCorrectiveTransferEnvironment(
        "SCP_impulsive_traj.csv", "SCP_dV.csv"
    )
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
    env: gym.Env = OldCorrectiveTransferEnvironment(
        "SCP_impulsive_traj.csv", "SCP_dV.csv"
    )
    nominal_control: np.ndarray = env.nominal_imp[0]
    nominal_dir: np.ndarray = nominal_control / np.linalg.norm(nominal_control)
    final_state, reward, _, _, _ = env.step(np.concatenate(([1], nominal_dir)))

    print(final_state, env.au * 2)


def test_sb3_integration():
    env: gym.Env = OldCorrectiveTransferEnvironment(
        "SCP_impulsive_traj.csv", "SCP_dV.csv"
    )
    env_checker.check_env(env, warn=False, skip_render_check=True)


def test_train(algo: str):
    env: gym.Env = OldCorrectiveTransferEnvironment(
        "SCP_impulsive_traj.csv", "SCP_dV.csv"
    )

    if algo == "PPO":
        model: sb3.PPO = sb3.PPO("MlpPolicy", env, verbose=1, n_steps=2)
    else:
        model: sb3.SAC = sb3.SAC("MlpPolicy", env, verbose=1)

    model.learn(25000)
    model.save("corrective_env_test_" + algo)


def test_eval(algo: str):
    env: gym.Env = OldCorrectiveTransferEnvironment(
        "SCP_impulsive_traj.csv", "SCP_dV.csv"
    )

    if algo == "PPO":
        model: sb3.PPO = sb3.PPO.load("corrective_env_test_PPO", env=env)
    else:
        model: sb3.SAC = sb3.SAC.load("corrective_env_test_SAC", env=env)

    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=100
    )
    print(mean_reward, std_reward)


def test_loc():
    # read in json data and initialise env config
    with open("space_configs/env_config.json") as f:
        data: dict = json.load(f)

    config: SpaceEnvironmentConfig = SpaceEnvironmentConfig()
    for k, v in data.items():
        setattr(config, k, v)

    env: CorrectiveTransferEnvironment = CorrectiveTransferEnvironment(config)

    # test the law of cosine function
    # roots: 6.39, 11.74
    # paper validated, does return the two possible lengths
    print(env._law_of_cosine(155, 10, 5))
    print(env._law_of_cosine(126.87, 3, 4))


def test_control_input():
    # read in json data and initialise env config
    with open("space_configs/env_config.json") as f:
        data: dict = json.load(f)

    config: SpaceEnvironmentConfig = SpaceEnvironmentConfig()
    for k, v in data.items():
        setattr(config, k, v)

    env: CorrectiveTransferEnvironment = CorrectiveTransferEnvironment(config)

    # test the control input
    vmax: float = env.max_thrust * env.timestep / env.state[-1]
    corrective_impulse: np.ndarray = env._get_control_input(vmax, [1, 1, 0, 0])

    assert (
        np.linalg.norm(corrective_impulse + env.nominal_imp[0]) <= vmax
    ), "Control limits exceeded."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo", required=False, type=str, default="PPO", choices=["PPO", "SAC"]
    )
    parser.add_argument(
        "--task",
        required=True,
        type=str,
        default="train",
        choices=[
            "prop",
            "max_control",
            "sb3_integration",
            "train",
            "eval",
            "loc",
            "control_input",
        ],
    )
    args = parser.parse_args()

    if args.task == "prop":
        test_prop()
    elif args.task == "max_control":
        test_max_control()
    elif args.task == "sb3_integration":
        test_sb3_integration()
    elif args.task == "train":
        test_train(args.algo)
    elif args.task == "loc":
        test_loc()
    elif args.task == "control_input":
        test_control_input()
    else:
        test_eval(args.algo)
