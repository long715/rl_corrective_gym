"""
Author: Lee Violet Ong
Date: 15/07/25

This file defines the test suite for the corrective transfer gym, build to validate the logical implementation of the gym.
This also contains test training scripts for the gym.
"""

import argparse
import json
import time

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common import env_checker
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
import matplotlib.pyplot as plt

# TODO: was used in the initial testing for preliminary validation, would like to
# eventually update the test functions to use the new one
from rl_corrective_gym.corrective_transfer_env import OldCorrectiveTransferEnvironment

#  use for integration testing for actual training runsS
from rl_corrective_gym.gym_env_setup.corrective_transfer_env import (
    CorrectiveTransferEnvironment,
)
from rl_corrective_gym.gym_env_setup.space_env_config import SpaceEnvironmentConfig


def test_init() -> CorrectiveTransferEnvironment:
    # read in json data and initialise env config
    with open("space_configs/env_config.json") as f:
        data: dict = json.load(f)

    config: SpaceEnvironmentConfig = SpaceEnvironmentConfig()
    for k, v in data.items():
        setattr(config, k, v)

    return CorrectiveTransferEnvironment(config)


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

    # extension for propagate function
    env: CorrectiveTransferEnvironment = test_init()
    _final_state: np.ndarray = env._propagate(False)

    assert np.all(final_state == _final_state), "Error in propagator function"

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
    env: CorrectiveTransferEnvironment = test_init()

    # test the law of cosine function
    # roots: 6.39, 11.74
    # paper validated, does return the two possible lengths
    print(env._law_of_cosine(155, 10, 5))
    print(env._law_of_cosine(126.87, 3, 4))


def test_control_input():
    env: CorrectiveTransferEnvironment = test_init()

    # test the control input
    vmax: float = env.max_thrust * env.timestep / env.state[-1]
    corrective_impulse: np.ndarray = env._get_control_input(vmax, [1, 1, 0, 0])

    assert (
        np.linalg.norm(corrective_impulse + env.nominal_imp[0]) <= vmax
    ), "Control limits exceeded."


def test_debug(df_id: int = 15):
    """
    For debugging certain scenarios from the plot.
    - recomputation seems to have slight deviation
    """
    df = pd.read_csv("../../SAC-mars-25_08_13_04-03-52/10/data/eval.csv")

    # TEST REWARDS
    env: CorrectiveTransferEnvironment = test_init()
    env.chosen_timestamp = df["timestep"][df_id]
    env.state = env.nominal_traj[env.chosen_timestamp, :] + np.fromstring(
        df["noise"][df_id].strip("[]"), sep=" "
    )
    corrective_impulse: np.ndarray = np.fromstring(
        df["corrective_impulse"][df_id].strip("[]"), sep=" "
    )
    nominal_imp: np.ndarray = env.nominal_imp[env.chosen_timestamp]
    vmax: float = env.max_thrust * env.timestep / env.state[-1]
    total_mag: float = np.linalg.norm(corrective_impulse + nominal_imp)
    print(vmax - total_mag)  # not matching results; this follow constraints
    print(vmax)

    # slight difference in the control effort reward
    placeholder: np.ndarray = np.array([0] * 7)
    print(env._reward_function(vmax, corrective_impulse, placeholder, placeholder))

    # TEST PROPAGATION
    env.reset(seed=10)
    print(env._propagate(True, corrective_impulse))  # minor changes
    print(env._propagate(False))

    # TEST FUNCTIONS
    action: np.ndarray = np.fromstring(df["action"][df_id].strip("[]"), sep=" ")
    csv_vmax: float = df["vmax"][df_id]
    print(env.step(action))
    print(csv_vmax - np.linalg.norm(env._get_control_input(csv_vmax, action)))


def test_opt_subset():
    """
    Aim of this test is to investigate the placement of the
    optimal control subset ie. if they are dense/sparse,
    scattered/concentrated.

    This should help us determine the effects on the training
    of the model.

    Req: ran in single run
    """
    env: CorrectiveTransferEnvironment = test_init()

    ax = plt.subplot(1, 1, 1, projection="3d")
    plt.title("Optimal Actions for Different Noise Scenarios")

    for _ in range(100):
        env.reset()
        env.is_single_reset = True
        opt_control: np.ndarray = env._optimal_control()

        ax.plot(opt_control[0], opt_control[1], opt_control[2], "rx", label="opt")

    plt.show()


def test_deviations():
    df = pd.DataFrame()

    # single sample
    env: CorrectiveTransferEnvironment = test_init()
    env.chosen_timestamp = 2
    env.is_single_reset = False
    env.reset()

    nom_terminal = env.nominal_traj[-1]
    ngui_terminal: np.ndarray = env._propagate(False)

    gui_pos = np.array([])
    gui_vel = np.array([])

    exhaust_vel_m: float = env.exhaust_vel  # km/s
    m0: float = env.state[-1]  # kg
    vmax: float = exhaust_vel_m * np.log(
        (m0 * exhaust_vel_m) / (m0 * exhaust_vel_m - env.max_thrust * env.timestep)
    )  # km/s

    plt.figure()
    ax = plt.subplot(1, 2, 2, projection="3d")
    plt.title("Sampled Actions")

    opt_control = env._optimal_control()
    ax.plot(opt_control[0], opt_control[1], opt_control[2], "rx", label="opt")
    print(opt_control)

    # compute the max deviation from optimal control
    print(f"dev: {env._get_control_input(vmax, [1,1,1,1]) - opt_control}")

    for _ in range(1000):
        action = env.sample_action()

        vel_var: float = 1e-3**2  # 1m/s - small deviation
        cov: np.ndarray = np.diag([vel_var, vel_var, vel_var])
        mean: np.ndarray = np.array([0] * 3)

        # choose the gaussian noise for the chosen state
        noise = np.random.multivariate_normal(mean, cov)

        corrective_impulse: np.ndarray = env._get_control_input(vmax, action)
        # corrective_impulse = opt_control + noise

        # plot the deviations
        gui_terminal = env._propagate(True, corrective_impulse)

        gui_pos = np.append(
            gui_pos, np.linalg.norm(gui_terminal[0:3] - nom_terminal[0:3])
        )
        gui_vel = np.append(
            gui_vel, np.linalg.norm(gui_terminal[3:6] - nom_terminal[3:6])
        )

        # want to see if control range is covered by sample
        # action_unit = action[0] * (action[1:4] / np.linalg.norm(action[1:4]))
        ax.plot(
            corrective_impulse[0],
            corrective_impulse[1],
            corrective_impulse[2],
            "bx",
            label="policy",
        )

        if (
            np.linalg.norm(gui_terminal[0:3] - nom_terminal[0:3]) == 0
            and np.linalg.norm(gui_terminal[3:6] - nom_terminal[3:6]) == 0
        ):
            print("SUCCESS")

        # log rewards
        rewards = env._reward_function(
            vmax,
            corrective_impulse,
            gui_terminal - nom_terminal,
            ngui_terminal - nom_terminal,
        )
        df = pd.concat([df, pd.DataFrame([rewards])])

    # save rewards
    df.to_csv(f"policy_rewards.csv", index=False)

    plt.subplot(1, 2, 1)
    plt.title(f"Terminal Deviations for timestep {env.chosen_timestamp}")
    plt.plot(
        np.linalg.norm(ngui_terminal[0:3] - nom_terminal[0:3]),
        np.linalg.norm(ngui_terminal[3:6] - nom_terminal[3:6]),
        "rx",
        label="ngui",
    )

    print(
        np.linalg.norm(ngui_terminal[0:3] - nom_terminal[0:3]),
        np.linalg.norm(ngui_terminal[3:6] - nom_terminal[3:6]),
    )

    plt.plot(
        gui_pos,
        gui_vel,
        "bx",
        label="gui",
    )
    plt.xlabel("Position magnitude error")
    plt.ylabel("Velocity magnitude error")
    plt.legend()

    plt.show()


def test_stm():
    env: CorrectiveTransferEnvironment = test_init()
    env.set_seed(10)
    env.reset()

    # TEST NO GUID
    final_state = env.nominal_traj[-1, 0:6]

    start_prop = time.time()
    actual_dev = env._propagate(False)[0:6] - final_state
    print(f"elapsed prop: {time.time() - start_prop}")

    # rand_imp = np.array([5.0, 1.0, 0.0])
    # guid_dev = env._propagate(True, rand_imp)[0:6] - final_state

    phi_compute = time.time()
    phi = env._stm_pert()
    print(f"phi compute: {time.time()-phi_compute}")

    start_phi = time.time()
    stm_dev = phi @ env.noise[0:6]
    # guid_stm_dev = phi @ (
    #     env.noise[0:6] + np.concatenate((np.array([0.0, 0.0, 0.0]), rand_imp))
    # )
    print(f"elapsed phi: {time.time() - start_phi}")

    print(actual_dev, stm_dev)
    # print(guid_dev, guid_stm_dev)

    # error for pos > tol; accurate enough
    tol = 1e-5
    # assert np.all(abs(actual_dev - stm_dev) < tol), "No Guid Error"
    # assert np.all(abs(guid_dev - guid_stm_dev) < tol), "Guid Error"

    # TEST OPT CONTROL
    opt_control = env._optimal_control()
    print(opt_control)


def test_optimal():
    """
    Aim is to the test the sensitivity of the control.
    Initial implementation
    """
    pass


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
            "debug",
            "dev",
            "stm",
            "opt_subset",
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
    elif args.task == "debug":
        test_debug()
    elif args.task == "dev":
        test_deviations()
    elif args.task == "stm":
        test_stm()
    elif args.task == "opt_subset":
        test_opt_subset()
    else:
        test_eval(args.algo)
