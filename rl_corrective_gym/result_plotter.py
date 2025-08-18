"""
Author: Lee Violet Ong
Date: 07/08/25

p lots we want (for both train and eval):
- plot of each of the rewards
- plot fo the terminal state
- trajectory plots (need to propagate)
"""

import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits import mplot3d

from rl_corrective_gym.env_test_script import test_init
from rl_corrective_gym.gym_env_setup.corrective_transfer_env import (
    CorrectiveTransferEnvironment,
)

df = pd.read_csv(
    "../../SAC-mars-25_08_13_04-03-52/10/data/eval.csv",
    on_bad_lines="skip",
    engine="python",
)


def plot_rewards():
    plt.figure(1)

    plt.subplot(2, 2, 1)
    plt.title("Reward Effort")
    plt.ylabel("Reward")
    df_rew_effort: pd.Series = df["reward_effort"]
    plt.plot(range(len(df_rew_effort)), df_rew_effort, ".")

    # TODO: check on unconstrained controls
    # unsure where the source of error is, seems to be giving a slightly diff mass
    plt.subplot(2, 2, 2)
    plt.title("Reward Control Penalty")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    df_rew_control_penalty: pd.Series = df["reward_control_penalty"]
    plt.plot(range(len(df_rew_control_penalty)), df_rew_control_penalty, ".")

    plt.subplot(2, 2, 3)
    plt.title("Reward Dynamics")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    df_rew_dyn: pd.Series = df["reward_dyn"]
    plt.plot(range(len(df_rew_dyn)), df_rew_dyn, ".")

    plt.show()


def plot_terminal():
    """
    Plots the terminal pos magnitude in the x axis and terminal
    vel magnitude in the y axis.
    """
    # desired terminal state
    nominal_terminal_state: np.ndarray = pd.read_csv(
        "nominal_trajectory/SCP_impulsive_traj.csv"
    ).to_numpy()[-1, :]
    nominal_pos: float = np.linalg.norm(nominal_terminal_state[0:3])
    nominal_vel: float = np.linalg.norm(nominal_terminal_state[3:6])

    # eval terminal state
    state_pos: np.ndarray = np.array([])
    state_vel: np.ndarray = np.array([])

    for state in df["terminal_state"]:
        # ignore the mass for now
        state_numpy: np.ndarray = np.fromstring(state.strip("[]"), sep=" ")
        pos_mag: float = np.linalg.norm(state_numpy[0:3])
        vel_mag: float = np.linalg.norm(state_numpy[3:6])

        state_pos = np.append(state_pos, pos_mag)
        state_vel = np.append(state_vel, vel_mag)

        if pos_mag == nominal_pos and vel_mag == nominal_vel:
            print(state_numpy[6])

    plt.figure()
    plt.title("Terminal State Deviation")
    plt.xlabel("Position magnitude")
    plt.ylabel("Velocity magnitude")

    plt.plot(nominal_pos, nominal_vel, "k+")
    plt.plot(state_pos[0:40], state_vel[0:40], "rx")
    plt.plot(state_pos[40:], state_vel[40:], "bx")

    plt.show()


def plot_control():
    env: CorrectiveTransferEnvironment = test_init()
    print(env.max_thrust, env.timestep)

    pass


def plot_trajectory():
    """
    Use env propagator to obtain whole trajectory states, plot the position
    to show the deviation of both guided and unguided.
    """
    env: CorrectiveTransferEnvironment = test_init()

    plt.figure()
    ax = plt.axes(projection="3d")

    # plot the desired trajectory

    ax.plot(env.nominal_traj[:, 0], env.nominal_traj[:, 1], env.nominal_traj[:, 2])

    # need to manually do reset setup ie. state, chosen_timestep
    for i in range(len(df)):
        env.chosen_timestamp = df["timestep"][i]
        env.noise = np.fromstring(df["noise"][i].strip("[]"), sep=" ")
        env.state = env.nominal_traj[env.chosen_timestamp] + env.noise

        corrective_impulse: np.ndarray = np.fromstring(
            df["corrective_impulse"][i].strip("[]"), sep=" "
        )

        env._init_logs()
        env._propagate(True, corrective_impulse)
        env._propagate(False)
        ax.plot(
            env.gui_log_pos[:, 0], env.gui_log_pos[:, 1], env.gui_log_pos[:, 2], "r"
        )
        ax.plot(
            env.nogui_log_pos[:, 0],
            env.nogui_log_pos[:, 1],
            env.nogui_log_pos[:, 2],
            "g",
        )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot",
        required=False,
        type=str,
        default="terminal",
        choices=["reward", "terminal", "control", "trajectory"],
    )

    args = parser.parse_args()
    if args.plot == "reward":
        plot_rewards()
    elif args.plot == "terminal":
        plot_terminal()
    elif args.plot == "control":
        plot_control()
    elif args.plot == "trajectory":
        plot_trajectory()
